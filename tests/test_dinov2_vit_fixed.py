"""Tests for DinoV2ViTBackbone class."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
from torchvision.transforms import Compose, Resize, ToTensor

from vla_project.models.backbones.vision.dinov2_vit import DINOv2_VISION_BACKBONES, DinoV2ViTBackbone
from vla_project.models.backbones.vision.vision_base import TimmViTBackbone, VisionBackbone

# Test constants
DEFAULT_IMAGE_SIZE = 224
CUSTOM_IMAGE_SIZE = 384
EMBED_DIM = 1024
NUM_PATCHES = 196


class TestDinoV2ViTBackbone:
    """Test suite for DinoV2ViTBackbone functionality."""

    @pytest.fixture
    def mock_timm_model(self):
        """Create a mock TIMM VisionTransformer model."""
        mock_model = MagicMock()

        # Make the mock identify as a VisionTransformer by setting the class
        mock_model.__class__ = VisionTransformer

        mock_model.embed_dim = EMBED_DIM
        mock_model.patch_embed.num_patches = NUM_PATCHES
        mock_model.blocks = [MagicMock() for _ in range(24)]  # Simulate 24 transformer blocks
        mock_model.eval.return_value = mock_model

        # Mock parameters for dtype property
        mock_param = MagicMock()
        mock_param.dtype = torch.float32
        mock_model.parameters.return_value = iter([mock_param])

        # Mock get_intermediate_layers method
        mock_model.get_intermediate_layers = MagicMock()

        return mock_model

    @pytest.fixture
    def mock_data_config(self):
        """Create a mock TIMM data config."""
        return {
            "input_size": (3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "interpolation": "bicubic",
        }

    @pytest.fixture
    def test_image(self):
        """Create a test PIL image."""
        return Image.new("RGB", (326, 326), color="red")

    def test_vision_backbone_registry(self):
        """Test that DINOv2 vision backbone registry is properly defined."""
        expected_registry = {
            "dinov2-vit-l": "vit_large_patch14_reg4_dinov2.lvd142m",
        }

        # Test registry structure and content
        if not isinstance(DINOv2_VISION_BACKBONES, dict):
            raise TypeError("Expected DINOv2_VISION_BACKBONES to be a dict")
        if "dinov2-vit-l" not in DINOv2_VISION_BACKBONES:
            raise KeyError("Expected 'dinov2-vit-l' in DINOv2_VISION_BACKBONES")
        if DINOv2_VISION_BACKBONES["dinov2-vit-l"] != expected_registry["dinov2-vit-l"]:
            raise ValueError("Registry value mismatch")

    @patch("vla_project.models.backbones.vision.vision_base.timm.create_model")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.resolve_model_data_config")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.create_transform")
    def test_initialization_default_params(
        self, mock_create_transform, mock_resolve_config, mock_create_model, mock_timm_model, mock_data_config,
    ):
        """Test DinoV2ViTBackbone initialization with default parameters."""
        # Setup mocks
        mock_create_model.return_value = mock_timm_model
        mock_resolve_config.return_value = mock_data_config
        mock_transform = MagicMock()
        mock_create_transform.return_value = mock_transform

        # Create backbone
        backbone = DinoV2ViTBackbone(
            vision_backbone_id="dinov2-vit-l",
            image_resize_strategy="resize-naive",
        )

        # Verify initialization
        if backbone.identifier != "dinov2-vit-l":
            raise ValueError("Identifier mismatch")
        if backbone.image_resize_strategy != "resize-naive":
            raise ValueError("Image resize strategy mismatch")
        if backbone.default_image_size != DEFAULT_IMAGE_SIZE:
            raise ValueError("Default image size mismatch")
        if backbone.featurizer != mock_timm_model:
            raise ValueError("Featurizer mismatch")

        # Verify TIMM model was created correctly
        mock_create_model.assert_called_once_with(
            "vit_large_patch14_reg4_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            img_size=DEFAULT_IMAGE_SIZE,
        )

    @patch("vla_project.models.backbones.vision.vision_base.timm.create_model")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.resolve_model_data_config")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.create_transform")
    def test_initialization_custom_image_size(
        self, mock_create_transform, mock_resolve_config, mock_create_model, mock_timm_model, mock_data_config,
    ):
        """Test DinoV2ViTBackbone initialization with custom image size."""
        # Setup mocks
        mock_create_model.return_value = mock_timm_model
        mock_resolve_config.return_value = mock_data_config
        mock_transform = MagicMock()
        mock_create_transform.return_value = mock_transform

        # Create backbone with custom image size
        backbone = DinoV2ViTBackbone(
            vision_backbone_id="dinov2-vit-l",
            image_resize_strategy="resize-crop",
            default_image_size=CUSTOM_IMAGE_SIZE,
        )

        # Verify initialization
        if backbone.default_image_size != CUSTOM_IMAGE_SIZE:
            raise ValueError("Custom image size mismatch")

        # Verify TIMM model was created with custom image size
        mock_create_model.assert_called_once_with(
            "vit_large_patch14_reg4_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            img_size=CUSTOM_IMAGE_SIZE,
        )

    def test_initialization_invalid_backbone_id(self):
        """Test DinoV2ViTBackbone initialization with invalid backbone ID."""
        with pytest.raises(KeyError):
            DinoV2ViTBackbone(
                vision_backbone_id="invalid-backbone",
                image_resize_strategy="resize-naive",
            )

    @patch("vla_project.models.backbones.vision.vision_base.timm.create_model")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.resolve_model_data_config")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.create_transform")
    def test_properties(
        self, mock_create_transform, mock_resolve_config, mock_create_model, mock_timm_model, mock_data_config,
    ):
        """Test DinoV2ViTBackbone properties."""
        # Setup mocks
        mock_create_model.return_value = mock_timm_model
        mock_resolve_config.return_value = mock_data_config
        mock_transform = MagicMock()
        mock_create_transform.return_value = mock_transform

        # Create backbone
        backbone = DinoV2ViTBackbone(
            vision_backbone_id="dinov2-vit-l",
            image_resize_strategy="resize-naive",
        )

        # Test properties
        expected_resolution = (3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
        if backbone.default_image_resolution != expected_resolution:
            raise ValueError("Default image resolution mismatch")
        if backbone.embed_dim != EMBED_DIM:
            raise ValueError("Embed dim mismatch")
        if backbone.num_patches != NUM_PATCHES:
            raise ValueError("Number of patches mismatch")
        if backbone.half_precision_dtype != torch.bfloat16:
            raise ValueError("Half precision dtype mismatch")

    @patch("vla_project.models.backbones.vision.vision_base.timm.create_model")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.resolve_model_data_config")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.create_transform")
    def test_forward_pass(
        self, mock_create_transform, mock_resolve_config, mock_create_model, mock_timm_model, mock_data_config,
    ):
        """Test DinoV2ViTBackbone forward pass."""
        # Setup mocks
        mock_create_model.return_value = mock_timm_model
        mock_resolve_config.return_value = mock_data_config
        mock_transform = MagicMock()
        mock_create_transform.return_value = mock_transform

        # Mock forward pass output
        expected_output = torch.randn(1, NUM_PATCHES, EMBED_DIM)
        mock_timm_model.return_value = expected_output

        # Create backbone
        backbone = DinoV2ViTBackbone(
            vision_backbone_id="dinov2-vit-l",
            image_resize_strategy="resize-naive",
        )

        # Test forward pass
        input_tensor = torch.randn(1, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
        output = backbone(input_tensor)

        if not torch.equal(output, expected_output):
            raise ValueError("Forward pass output mismatch")
        mock_timm_model.assert_called_once_with(input_tensor)

    @patch("vla_project.models.backbones.vision.vision_base.timm.create_model")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.resolve_model_data_config")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.create_transform")
    def test_get_fsdp_wrapping_policy(
        self, mock_create_transform, mock_resolve_config, mock_create_model, mock_timm_model, mock_data_config,
    ):
        """Test DinoV2ViTBackbone FSDP wrapping policy."""
        # Setup mocks
        mock_create_model.return_value = mock_timm_model
        mock_resolve_config.return_value = mock_data_config
        mock_transform = MagicMock()
        mock_create_transform.return_value = mock_transform

        # Create backbone
        backbone = DinoV2ViTBackbone(
            vision_backbone_id="dinov2-vit-l",
            image_resize_strategy="resize-naive",
        )

        # Test FSDP policy
        policy = backbone.get_fsdp_wrapping_policy()
        if not callable(policy):
            raise ValueError("FSDP policy should be callable")

    @patch("vla_project.models.backbones.vision.vision_base.timm.create_model")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.resolve_model_data_config")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.create_transform")
    def test_get_image_transform(
        self, mock_create_transform, mock_resolve_config, mock_create_model, mock_timm_model, mock_data_config,
    ):
        """Test DinoV2ViTBackbone image transform getter."""
        # Setup mocks
        mock_create_model.return_value = mock_timm_model
        mock_resolve_config.return_value = mock_data_config
        mock_transform = MagicMock()
        mock_create_transform.return_value = mock_transform

        # Create backbone
        backbone = DinoV2ViTBackbone(
            vision_backbone_id="dinov2-vit-l",
            image_resize_strategy="resize-naive",
        )

        # Test image transform getter
        transform = backbone.get_image_transform()
        if transform is None:
            raise ValueError("Transform should not be None")
        if not callable(transform):
            raise ValueError("Transform should be callable")

    @patch("vla_project.models.backbones.vision.vision_base.timm.create_model")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.resolve_model_data_config")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.create_transform")
    def test_dtype_property_and_setter(
        self, mock_create_transform, mock_resolve_config, mock_create_model, mock_timm_model, mock_data_config,
    ):
        """Test DinoV2ViTBackbone dtype property and setter."""
        # Setup mocks
        mock_create_model.return_value = mock_timm_model
        mock_resolve_config.return_value = mock_data_config
        mock_transform = MagicMock()
        mock_create_transform.return_value = mock_transform

        # Create backbone
        backbone = DinoV2ViTBackbone(
            vision_backbone_id="dinov2-vit-l",
            image_resize_strategy="resize-naive",
        )

        # Test initial dtype
        if backbone.dtype != torch.float32:  # From mock parameter
            raise ValueError("Initial dtype mismatch")

        # Test dtype setter
        backbone.dtype = torch.float16
        mock_timm_model.to.assert_called_with(dtype=torch.float16)

    @pytest.mark.parametrize("strategy", ["resize-naive", "resize-crop", "letterbox"])
    @patch("vla_project.models.backbones.vision.vision_base.timm.create_model")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.resolve_model_data_config")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.create_transform")
    def test_different_resize_strategies(
        self, mock_create_transform, mock_resolve_config, mock_create_model, mock_timm_model, mock_data_config, strategy,
    ):
        """Test DinoV2ViTBackbone with different image resize strategies."""
        # Setup mocks
        mock_create_model.return_value = mock_timm_model
        mock_resolve_config.return_value = mock_data_config

        # Create mock compose transform
        mock_transform = Compose([Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)), ToTensor()])
        mock_create_transform.return_value = mock_transform

        # Create backbone with different strategies
        backbone = DinoV2ViTBackbone(
            vision_backbone_id="dinov2-vit-l",
            image_resize_strategy=strategy,
        )

        # Verify that backbone was created successfully
        if backbone.image_resize_strategy != strategy:
            raise ValueError("Image resize strategy mismatch")
        if backbone.image_transform is None:
            raise ValueError("Image transform should not be None")

    @patch("vla_project.models.backbones.vision.vision_base.timm.create_model")
    def test_invalid_resize_strategy(self, mock_create_model, mock_timm_model):
        """Test DinoV2ViTBackbone with invalid resize strategy."""
        mock_create_model.return_value = mock_timm_model

        with pytest.raises(ValueError, match=r"Image Resize Strategy.*is not supported"):
            DinoV2ViTBackbone(
                vision_backbone_id="dinov2-vit-l",
                image_resize_strategy="invalid-strategy",
            )

    @patch("vla_project.models.backbones.vision.vision_base.timm.create_model")
    def test_non_vit_model_error(self, mock_create_model):
        """Test DinoV2ViTBackbone raises error with non-ViT model."""
        # Create a mock non-ViT model
        mock_non_vit = MagicMock()
        mock_non_vit.__class__ = type("NotViT", (), {})
        mock_create_model.return_value = mock_non_vit

        with pytest.raises(TypeError, match="Featurizer is not a TIMM VisionTransformer"):
            DinoV2ViTBackbone(
                vision_backbone_id="dinov2-vit-l",
                image_resize_strategy="resize-naive",
            )

    @patch("vla_project.models.backbones.vision.vision_base.timm.create_model")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.resolve_model_data_config")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.create_transform")
    def test_inheritance_from_timm_vit_backbone(
        self, mock_create_transform, mock_resolve_config, mock_create_model, mock_timm_model, mock_data_config,
    ):
        """Test that DinoV2ViTBackbone properly inherits from TimmViTBackbone."""
        # Setup mocks
        mock_create_model.return_value = mock_timm_model
        mock_resolve_config.return_value = mock_data_config
        mock_transform = MagicMock()
        mock_create_transform.return_value = mock_transform

        # Create backbone
        backbone = DinoV2ViTBackbone(
            vision_backbone_id="dinov2-vit-l",
            image_resize_strategy="resize-naive",
        )

        # Verify it inherits from TimmViTBackbone which inherits from VisionBackbone
        if not isinstance(backbone, TimmViTBackbone):
            raise TypeError("Should inherit from TimmViTBackbone")
        if not isinstance(backbone, VisionBackbone):
            raise TypeError("Should inherit from VisionBackbone")

    @patch("vla_project.models.backbones.vision.vision_base.timm.create_model")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.resolve_model_data_config")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.create_transform")
    def test_model_evaluation_mode(
        self, mock_create_transform, mock_resolve_config, mock_create_model, mock_timm_model, mock_data_config,
    ):
        """Test that the model is set to evaluation mode during initialization."""
        # Setup mocks
        mock_create_model.return_value = mock_timm_model
        mock_resolve_config.return_value = mock_data_config
        mock_transform = MagicMock()
        mock_create_transform.return_value = mock_transform

        # Create backbone
        DinoV2ViTBackbone(
            vision_backbone_id="dinov2-vit-l",
            image_resize_strategy="resize-naive",
        )

        # Verify eval() was called
        mock_timm_model.eval.assert_called_once()

    @patch("vla_project.models.backbones.vision.vision_base.timm.create_model")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.resolve_model_data_config")
    @patch("vla_project.models.backbones.vision.vision_base.tdata.create_transform")
    def test_monkey_patch_forward_function(
        self, mock_create_transform, mock_resolve_config, mock_create_model, mock_timm_model, mock_data_config,
    ):
        """Test that the forward function is monkey-patched for FSDP compatibility."""
        # Setup mocks
        mock_create_model.return_value = mock_timm_model
        mock_resolve_config.return_value = mock_data_config
        mock_transform = MagicMock()
        mock_create_transform.return_value = mock_transform

        # Mock get_intermediate_layers method
        mock_timm_model.get_intermediate_layers = MagicMock()

        # Create backbone
        backbone = DinoV2ViTBackbone(
            vision_backbone_id="dinov2-vit-l",
            image_resize_strategy="resize-naive",
        )

        # Test that forward function was replaced
        if not hasattr(backbone.featurizer, "forward"):
            raise AttributeError("Forward method should exist")
        # The forward method should now be the monkey-patched version
        if not callable(backbone.featurizer.forward):
            raise ValueError("Forward method should be callable")
