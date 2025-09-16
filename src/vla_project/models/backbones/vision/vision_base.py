from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol, cast

import timm
import torch
import torchvision.transforms.functional as tvf
from PIL.Image import Image
from timm import data as tdata
from timm.models.vision_transformer import Block, VisionTransformer
from torch import nn
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Resize


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], tuple[Any]]) -> Callable[[Any], Any]:
    """Unpack tuple returns from functions.

    This wrapper function takes a function that returns a tuple and modifies it
    to return only the first element of the tuple, or the original result if
    it's not a tuple.

    Args:
        fn (Callable[[Any], tuple[Any]]): Function that returns a tuple.

    Returns:
        Callable[[Any], Any]: Wrapped function that returns the first element
            of the tuple or the original result.

    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


class ImageTransform(Protocol):
    """Protocol for image transformation functions.

    This protocol defines the interface for image transformation functions
    that can be used with vision backbones. Implementations should accept
    an image and optional keyword arguments, returning a transformed image
    or dictionary of transformed images.
    """

    def __call__[T](self, img: T, **kwargs: str | None) -> T | dict[str, T]:
        """Transform an input image.

        Args:
            img (T): Input image to transform.
            **kwargs (str | None): Optional keyword arguments for transformation.

        Returns:
            T | dict[str, T]: Transformed image or dictionary of transformed images.

        """
        ...


# A concrete class that wraps Compose and conforms to the protocol
class ComposeWrapper(ImageTransform):
    """Wrapper class for torchvision Compose transforms.

    This class wraps a torchvision Compose object to conform to the ImageTransform
    protocol, providing a consistent interface for image transformations.

    Attributes:
        _transforms (Compose): The underlying torchvision Compose transforms.

    """

    def __init__(self, transforms: Compose):
        """Initialize the ComposeWrapper.

        Args:
            transforms (Compose): Torchvision Compose object containing transforms.

        """
        self._transforms = transforms

    def __call__[T](self, img: T, **_: str | None) -> T | dict[str, T]:
        """Apply the composed transforms to an image.

        Args:
            img (T): Input image to transform.
            **kwargs (str | None): Optional keyword arguments (currently ignored).

        Returns:
            T | dict[str, T]: Transformed image.

        """
        # Perform the actual transformation
        transformed_img = self._transforms(img)

        # Here, you can decide how to handle the extra kwargs.
        # For simplicity, we just ignore them for now.
        # If any transform inside _transforms needed a specific kwarg, you'd handle it here.

        # The protocol signature requires a specific return type.
        # Assuming the final transform returns a Tensor, this is safe.
        return transformed_img  # noqa: RET504


@dataclass
class LetterboxPad:
    """Letterbox padding transformation for images.

    This transform pads an image to make it square by adding symmetric borders
    around the height and width dimensions. Useful for maintaining aspect ratio
    while ensuring consistent input dimensions.

    Attributes:
        padding_fill_value (tuple[int, int, int]): RGB values for padding fill.

    """

    padding_fill_value: tuple[int, int, int]

    def __call__(self, image: Image) -> Image:
        """Apply letterbox padding to make the image square.

        Pads the image with symmetric borders to create a square image while
        maintaining the original aspect ratio.

        Args:
            image (Image): PIL Image to be padded.

        Returns:
            Image: Square PIL Image with letterbox padding applied.

        """
        """Given a PIL.Image, pad to square by adding a symmetric border around the height/width."""
        (w, h) = image.size
        max_wh = max(image.size)
        horizontal_pad = (max_wh - w) // 2
        vertical_pad = (max_wh - h) // 2
        padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)
        return tvf.pad(image, padding, fill=self.padding_fill_value, padding_mode="constant")  # pyright: ignore[reportReturnType, reportArgumentType]


# === Abstract Base Class for arbitrary Vision Backbones ===
class VisionBackbone[T: nn.Module](nn.Module, ABC):
    """Abstract base class for vision backbone networks.

    This class defines the interface for vision backbone models used in
    vision-language models. It provides common functionality and enforces
    implementation of essential methods for feature extraction and processing.

    Type Parameters:
        T: Type of the featurizer module, must inherit from nn.Module.

    Attributes:
        identifier (str): Unique identifier for the vision backbone.
        image_resize_strategy (str): Strategy for resizing input images.
        default_image_size (int): Default size for input images.
        featurizer (T): The underlying feature extraction model.
        image_transform (ImageTransform): Transform pipeline for input images.

    """

    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        """Initialize the vision backbone.

        Args:
            vision_backbone_id (str): Unique identifier for this vision backbone.
            image_resize_strategy (str): Strategy for resizing input images.
                Options include "resize-naive", "resize-crop", "letterbox".
            default_image_size (int, optional): Default size for input images.
                Defaults to 224.

        """
        super().__init__()
        self.identifier: str = vision_backbone_id
        self.image_resize_strategy: str = image_resize_strategy
        self.default_image_size: int = default_image_size

        # Instance attributes for a Vision Backbone
        self.featurizer: T
        self.image_transform: ImageTransform

    def get_image_transform(self) -> ImageTransform:
        """Get the image transformation pipeline.

        Returns:
            ImageTransform: The configured image transformation pipeline.

        """
        return self.image_transform

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable:
        """Get the FSDP (Fully Sharded Data Parallel) wrapping policy.

        Returns:
            Callable: FSDP wrapping policy function for distributed training.

        """
        ...

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the featurizer.

        Args:
            pixel_values (torch.Tensor): Processed image tensor.

        Returns:
            torch.Tensor: Extracted patch/grid features.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def default_image_resolution(self) -> tuple[int, int, int]:
        """Get the default image resolution (channels, height, width).

        Returns:
            tuple[int, int, int]: Default image resolution as (C, H, W).

        """
        ...

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Get the embedding dimension of extracted features.

        Returns:
            int: Dimension of the feature embeddings.

        """
        ...

    @property
    @abstractmethod
    def num_patches(self) -> int:
        """Get the number of patches/tokens extracted from images.

        Returns:
            int: Number of patches or spatial tokens.

        """
        ...

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype:
        """Get the preferred half-precision dtype for this backbone.

        Returns:
            torch.dtype: Preferred half-precision data type.

        """
        ...

    # @property
    # def dtype(self) -> torch.dtype:
    #     """Get the preferred half-precision dtype for this backbone.

    #     Returns:
    #         torch.dtype: Preferred half-precision data type.

    #     """
    #     # TODO: this is a workaround, remove when all backbones have this property properly implemented
    #     return self.featurizer.parameters().__next__().dtype

    # @dtype.setter
    # def dtype(self, dtype: torch.dtype) -> None:
    #     """Set the dtype for the featurizer.

    #     Args:
    #         dtype (torch.dtype): The desired data type.

    #     """
    #     self.featurizer.to(dtype=dtype)


# === Abstract Base Class for Arbitrary TIMM Vision Transformer Backbones ===
class TimmViTBackbone(VisionBackbone[VisionTransformer], ABC):
    """TIMM Vision Transformer backbone implementation.

    This class provides a concrete implementation of VisionBackbone specifically
    for TIMM (PyTorch Image Models) Vision Transformer models. It handles model
    loading, configuration, and provides various image resize strategies.

    Attributes:
        timm_path_or_url (str): Path or URL to the TIMM model.
        override_act_layer (str | None): Optional activation layer override.
        dtype (torch.dtype): Data type for the model (torch.bfloat16).
        data_cfg (dict): TIMM data configuration for the model.

    """

    def __init__(
        self,
        vision_backbone_id: str,
        timm_path_or_url: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        override_act_layer: str | None = None,
    ) -> None:
        """Initialize TIMM Vision Transformer backbone.

        Sets up a TIMM ViT model with specified configuration and image processing
        strategy. Handles model downloading, monkey-patching for FSDP compatibility,
        and configures appropriate image transformations.

        Args:
            vision_backbone_id (str): Unique identifier for this vision backbone.
            timm_path_or_url (str): Path or URL to the TIMM model.
            image_resize_strategy (str): Strategy for resizing input images.
                Options: "resize-naive", "resize-crop", "letterbox".
            default_image_size (int, optional): Default size for input images.
                Defaults to 224.
            override_act_layer (str | None, optional): Override activation layer.
                Defaults to None.

        Raises:
            TypeError: If the loaded model is not a TIMM VisionTransformer.
            ValueError: If image_resize_strategy is not supported.
            KeyError: If required configuration keys are missing.

        """
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        self.timm_path_or_url = timm_path_or_url
        self.override_act_layer = override_act_layer
        self.dtype = torch.bfloat16

        # Initialize Featurizer (ViT) by downloading from HF / TIMM Hub if necessary
        if self.override_act_layer is None:
            self.featurizer: VisionTransformer = cast(
                "VisionTransformer",
                timm.create_model(
                    self.timm_path_or_url,
                    pretrained=True,
                    num_classes=0,
                    img_size=self.default_image_size,
                ),
            )
        else:
            self.featurizer: VisionTransformer = cast(
                "VisionTransformer",
                timm.create_model(
                    self.timm_path_or_url,
                    pretrained=True,
                    num_classes=0,
                    img_size=self.default_image_size,
                    act_layer=self.override_act_layer,
                ),
            )
        self.featurizer.eval()

        # Monkey-Patch the `forward()` function of the featurizer to ensure FSDP-compatibility
        #   => Note: By default set `get_intermediate_layers` to return the *SECOND-TO-LAST* layer patches!
        #   => TODO (siddk) Remove after resolution of https://github.com/pytorch/pytorch/issues/109385
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2}),  # pyright: ignore  # noqa: PGH003
        )

        # Validation =>> for now, this class *only* supports TIMM Vision Transformers (but can be extended!)
        if not isinstance(self.featurizer, VisionTransformer):
            msg = (
                f"Featurizer {type(self.featurizer)} is not a TIMM VisionTransformer; "
                "if you would like to support a new visual representation, "
                f"file an issue or implement the requisite logic (see `{__file__}`)!"
            )
            raise TypeError(msg)

        # Get Config =>> Note :: Override default image size to ensure correct image transform
        self.data_cfg = tdata.resolve_model_data_config(self.featurizer)  # pyright: ignore[reportPrivateImportUsage]
        self.data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        # Initialize Default Image Transform --> Modified by `self.image_resize_strategy`
        default_image_transform = cast("Compose", tdata.create_transform(**self.data_cfg, is_training=False))  # pyright: ignore[reportPrivateImportUsage]
        # Fix =>> SigLIP & IN1K default transforms resize to *larger* than `self.default_image_size` (crops image)!
        if "siglip" in self.timm_path_or_url or "in1k" in self.timm_path_or_url:
            if not isinstance(default_image_transform, Compose):
                msg = "Unexpected `default_image_transform`!"
                raise TypeError(msg)
            if not isinstance(default_image_transform.transforms[0], Resize):
                msg = "First transform in `default_image_transform` is not a Resize operation!"
                raise TypeError(msg)
            default_image_transform = Compose(
                [
                    Resize(self.default_image_size, interpolation=default_image_transform.transforms[0].interpolation),
                    *default_image_transform.transforms[1:],
                ],
            )

        # Switch on `image_resize_strategy`
        if self.image_resize_strategy == "resize-naive":
            if not isinstance(default_image_transform, Compose):
                msg = f"Unexpected `default_image_transform` {type(default_image_transform)}!"
                raise TypeError(msg)
            if not isinstance(default_image_transform.transforms[0], Resize):
                msg = "First transform in `default_image_transform` is not a Resize operation!"
                raise TypeError(msg)

            target_size = (self.default_image_size, self.default_image_size)
            self.image_transform = ComposeWrapper(
                Compose(
                    [
                        Resize(target_size, interpolation=default_image_transform.transforms[0].interpolation),
                        *default_image_transform.transforms[1:],
                    ],
                ),
            )

        elif self.image_resize_strategy == "resize-crop":
            self.image_transform = ComposeWrapper(default_image_transform)

        elif self.image_resize_strategy == "letterbox":
            if not isinstance(default_image_transform, Compose):
                msg = "Unexpected `default_image_transform`!"
                raise TypeError(msg)
            if "mean" not in self.data_cfg:
                msg = "TIMM `data_cfg` missing image normalization mean!"
                raise KeyError(msg)

            # Compute Padding Fill Value (rescaled normalization mean if applicable)
            fill = tuple([int(x * 255) for x in self.data_cfg["mean"]])
            if len(fill) != 3:
                msg = "Expected 3 channels for fill value"
                raise ValueError(msg)
            # Build New Transform
            self.image_transform = ComposeWrapper(
                Compose([LetterboxPad(fill), *default_image_transform.transforms]),
            )

        else:
            msg = f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!"
            raise ValueError(msg)

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return FSDP wrapping policy for Vision Transformer models.

        Creates a wrapping policy that handles both individual ViT blocks and
        the entire featurizer module for efficient distributed training.

        Returns:
            Callable: FSDP wrapping policy combining VisionTransformer and
                Block-level policies.

        """
        """Return a simple FSDP policy that wraps each ViT block and then the _entire_ featurizer."""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, pixel_values: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        """Run forward pass through the Vision Transformer.

        Processes the input image tensor through the ViT featurizer to extract
        patch-level features from all spatial locations.

        Args:
            pixel_values (torch.Tensor | dict[str, torch.Tensor]): Input image
                tensor or dictionary of tensors.

        Returns:
            torch.Tensor: Extracted patch features from the vision transformer.

        """
        """Run transformed image/pixel tensor through vision backbone, returning _all_ patch features."""
        return self.featurizer(pixel_values)

    @property
    def default_image_resolution(self) -> tuple[int, int, int]:
        """Get the default image resolution from TIMM config.

        Returns:
            tuple[int, int, int]: Image resolution as (channels, height, width).

        """
        return self.data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        """Get the embedding dimension of the Vision Transformer.

        Returns:
            int: Embedding dimension of the ViT featurizer.

        """
        return self.featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        """Get the number of patches from the patch embedding layer.

        Returns:
            int: Number of patches extracted by the patch embedding layer.

        """
        return self.featurizer.patch_embed.num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """Get the preferred half-precision dtype for TIMM ViT models.

        Returns:
            torch.dtype: torch.bfloat16 as the preferred half-precision type.

        """
        return self.dtype


if __name__ == "__main__":
    import PIL
    from torchvision import transforms

    img = PIL.Image.open(
        "/home/mintinson/casualCode/pythonProject/uv_projects/vla_project/tests/resources/images/test_img_1.png"
    ).convert("RGB")
    print(img.size)
    tv_trans = transforms.ToTensor()
    tv_resize = transforms.Resize((224, 334))
    tsor = tv_resize(img)
    print(tsor.size)
