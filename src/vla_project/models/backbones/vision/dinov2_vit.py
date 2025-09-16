# Registry =>> Supported DINOv2 Vision Backbones (from TIMM) =>> Note:: Using DINOv2 w/ Registers!
#   => Reference: https://arxiv.org/abs/2309.16588
from vla_project.models.backbones.vision.vision_base import TimmViTBackbone

DINOv2_VISION_BACKBONES = {
    "dinov2-vit-l": "vit_large_patch14_reg4_dinov2.lvd142m",
    "dinov2-vit-s": "vit_small_patch14_reg4_dinov2.lvd142m",
}


class DinoV2ViTBackbone(TimmViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        """Initialize the DinoV2ViTBackbone.

        Args:
            vision_backbone_id: The ID of the DINOv2 vision backbone to use.
            image_resize_strategy: The strategy for resizing images.
            default_image_size: The default size to resize images to.

        """
        super().__init__(
            vision_backbone_id,
            DINOv2_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
        )


if __name__ == "__main__":
    vision_backbone = DinoV2ViTBackbone("dinov2-vit-s", "resize-naive")
    print(vision_backbone)
    print(vision_backbone.image_transform)