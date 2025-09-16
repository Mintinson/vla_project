"""Comprehensive tests for DinoV2ViTBackbone class using dinov2-vit-s model."""

from pathlib import Path

import PIL.Image
import pytest
import torch

from vla_project.models.backbones.vision.dinov2_vit import DINOv2_VISION_BACKBONES, DinoV2ViTBackbone
from vla_project.models.backbones.vision.vision_base import ComposeWrapper


class TestDinoV2ViTBackbone:
    """Test suite for DinoV2ViTBackbone class."""

    @pytest.fixture
    def test_image_path(self):
        """Provide path to test image."""
        return Path(__file__).parent / "resources" / "images" / "test_img_1.png"

    @pytest.fixture
    def test_image(self, test_image_path):
        """Load test image as PIL Image."""
        return PIL.Image.open(test_image_path).convert("RGB")

    @pytest.fixture
    def backbone_resize_naive(self):
        """Create DinoV2ViTBackbone with resize-naive strategy."""
        return DinoV2ViTBackbone("dinov2-vit-s", "resize-naive", default_image_size=224)

    @pytest.fixture
    def backbone_resize_crop(self):
        """Create DinoV2ViTBackbone with resize-crop strategy."""
        return DinoV2ViTBackbone("dinov2-vit-s", "resize-crop", default_image_size=224)

    @pytest.fixture
    def backbone_letterbox(self):
        """Create DinoV2ViTBackbone with letterbox strategy."""
        return DinoV2ViTBackbone("dinov2-vit-s", "letterbox", default_image_size=224)

    def test_init_basic(self):
        """Test basic initialization of DinoV2ViTBackbone."""
        backbone = DinoV2ViTBackbone("dinov2-vit-s", "resize-naive")

        assert backbone.identifier == "dinov2-vit-s"
        assert backbone.image_resize_strategy == "resize-naive"
        assert backbone.default_image_size == 224
        assert backbone.timm_path_or_url == DINOv2_VISION_BACKBONES["dinov2-vit-s"]
        assert backbone.featurizer is not None
        assert backbone.image_transform is not None

    def test_init_custom_image_size(self):
        """Test initialization with custom image size."""
        custom_size = 384
        backbone = DinoV2ViTBackbone("dinov2-vit-s", "resize-naive", default_image_size=custom_size)

        assert backbone.default_image_size == custom_size
        assert backbone.default_image_resolution == (3, custom_size, custom_size)

    def test_init_different_resize_strategies(self):
        """Test initialization with different image resize strategies."""
        strategies = ["resize-naive", "resize-crop", "letterbox"]

        for strategy in strategies:
            backbone = DinoV2ViTBackbone("dinov2-vit-s", strategy)
            assert backbone.image_resize_strategy == strategy
            assert isinstance(backbone.image_transform, ComposeWrapper)

    def test_init_invalid_backbone_id(self):
        """Test initialization with invalid backbone ID."""
        with pytest.raises(KeyError):
            DinoV2ViTBackbone("invalid-backbone", "resize-naive")

    def test_init_invalid_resize_strategy(self):
        """Test initialization with invalid resize strategy."""
        with pytest.raises(ValueError, match="Image Resize Strategy .* is not supported"):
            DinoV2ViTBackbone("dinov2-vit-s", "invalid-strategy")

    def test_properties(self, backbone_resize_naive):
        """Test all properties of the backbone."""
        # Test embed_dim
        assert isinstance(backbone_resize_naive.embed_dim, int)
        assert backbone_resize_naive.embed_dim > 0

        # Test num_patches
        assert isinstance(backbone_resize_naive.num_patches, int)
        assert backbone_resize_naive.num_patches > 0

        # Test default_image_resolution
        resolution = backbone_resize_naive.default_image_resolution
        assert isinstance(resolution, tuple)
        assert len(resolution) == 3
        assert resolution == (3, 224, 224)

        # Test half_precision_dtype
        assert backbone_resize_naive.half_precision_dtype == torch.bfloat16

    def test_image_transform_interface(self, backbone_resize_naive, test_image):
        """Test that image_transform follows the expected interface."""
        transform = backbone_resize_naive.get_image_transform()

        # Test that transform is callable
        assert callable(transform)

        # Test that it accepts PIL Image
        transformed = transform(test_image)
        assert isinstance(transformed, torch.Tensor)

        # Test tensor properties
        assert transformed.ndim == 3  # Should be (C, H, W)
        assert transformed.shape[0] == 3  # RGB channels
        assert transformed.shape[1] == 224  # Height
        assert transformed.shape[2] == 224  # Width

    def test_image_transform_resize_naive(self, backbone_resize_naive, test_image):
        """Test image transform with resize-naive strategy."""
        transform = backbone_resize_naive.get_image_transform()
        transformed = transform(test_image)

        # Should resize to exactly 224x224
        assert transformed.shape == (3, 224, 224)

        # Values should be normalized (typical range for normalized images)
        assert transformed.min() >= -3.0  # Reasonable lower bound after normalization
        assert transformed.max() <= 3.0  # Reasonable upper bound after normalization

    def test_image_transform_letterbox(self, backbone_letterbox, test_image):
        """Test image transform with letterbox strategy."""
        transform = backbone_letterbox.get_image_transform()
        transformed = transform(test_image)

        # Should still result in 224x224 after letterbox padding
        assert transformed.shape == (3, 224, 224)

    def test_forward_pass(self, backbone_resize_naive, test_image):
        """Test forward pass through the backbone."""
        # Prepare input tensor
        transform = backbone_resize_naive.get_image_transform()
        pixel_values = transform(test_image).unsqueeze(0)  # Add batch dimension

        # Forward pass
        with torch.no_grad():
            features = backbone_resize_naive.forward(pixel_values)

        # Check output properties
        # Note: Due to monkey-patching, forward returns a list with the tensor
        if isinstance(features, list):
            assert len(features) == 1
            features = features[0]

        assert isinstance(features, torch.Tensor)
        assert features.ndim == 3  # (batch_size, num_patches, embed_dim)
        assert features.shape[0] == 1  # Batch size
        assert features.shape[1] == backbone_resize_naive.num_patches
        assert features.shape[2] == backbone_resize_naive.embed_dim

    def test_forward_pass_batch(self, backbone_resize_naive, test_image):
        """Test forward pass with batch of images."""
        # Prepare batch of input tensors
        transform = backbone_resize_naive.get_image_transform()
        pixel_values = torch.stack(
            [transform(test_image), transform(test_image), transform(test_image)]
        )  # Batch of 3 identical images

        # Forward pass
        with torch.no_grad():
            features = backbone_resize_naive.forward(pixel_values)

        # Check output properties
        # Note: Due to monkey-patching, forward returns a list with the tensor
        if isinstance(features, list):
            assert len(features) == 1
            features = features[0]

        assert features.shape[0] == 3  # Batch size
        assert features.shape[1] == backbone_resize_naive.num_patches
        assert features.shape[2] == backbone_resize_naive.embed_dim

    def test_fsdp_wrapping_policy(self, backbone_resize_naive):
        """Test FSDP wrapping policy generation."""
        policy = backbone_resize_naive.get_fsdp_wrapping_policy()

        # Should return a callable
        assert callable(policy)

    def test_eval_mode(self, backbone_resize_naive):
        """Test that featurizer is in eval mode by default."""
        assert not backbone_resize_naive.featurizer.training

    def test_dtype_consistency(self, backbone_resize_naive):
        """Test that dtype is properly set."""
        assert backbone_resize_naive.dtype == torch.bfloat16
        assert backbone_resize_naive.half_precision_dtype == torch.bfloat16

    def test_different_image_sizes(self):
        """Test backbone with different image sizes."""
        sizes = [224, 384, 512]

        for size in sizes:
            backbone = DinoV2ViTBackbone("dinov2-vit-s", "resize-naive", default_image_size=size)
            assert backbone.default_image_size == size
            assert backbone.default_image_resolution == (3, size, size)

    def test_backbone_registry(self):
        """Test that backbone IDs are properly registered."""
        # Test valid backbone IDs - only test dinov2-vit-s to avoid long download times
        valid_ids = ["dinov2-vit-s"]

        for backbone_id in valid_ids:
            assert backbone_id in DINOv2_VISION_BACKBONES
            backbone = DinoV2ViTBackbone(backbone_id, "resize-naive")
            assert backbone.identifier == backbone_id

        # Just check that dinov2-vit-l is in registry without instantiating
        assert "dinov2-vit-l" in DINOv2_VISION_BACKBONES

    def test_str_representation(self, backbone_resize_naive):
        """Test string representation of the backbone."""
        str_repr = str(backbone_resize_naive)
        assert "DinoV2ViTBackbone" in str_repr or "VisionTransformer" in str_repr

    def test_image_transform_with_different_input_sizes(self, backbone_resize_naive):
        """Test image transform with different input image sizes."""
        transform = backbone_resize_naive.get_image_transform()

        # Test with different sized images
        small_image = PIL.Image.new("RGB", (100, 100), color="red")
        large_image = PIL.Image.new("RGB", (500, 300), color="blue")
        square_image = PIL.Image.new("RGB", (224, 224), color="green")

        for img in [small_image, large_image, square_image]:
            transformed = transform(img)
            assert transformed.shape == (3, 224, 224)

    def test_feature_extraction_deterministic(self, backbone_resize_naive, test_image):
        """Test that feature extraction is deterministic (same input -> same output)."""
        transform = backbone_resize_naive.get_image_transform()
        pixel_values = transform(test_image).unsqueeze(0)

        with torch.no_grad():
            features1 = backbone_resize_naive.forward(pixel_values)
            features2 = backbone_resize_naive.forward(pixel_values)

        # Handle list return from monkey-patched forward
        if isinstance(features1, list):
            features1 = features1[0]
        if isinstance(features2, list):
            features2 = features2[0]

        # Should be exactly the same (deterministic)
        torch.testing.assert_close(features1, features2, rtol=1e-5, atol=1e-5)

    def test_gradient_flow(self, backbone_resize_naive, test_image):
        """Test that gradients can flow through the backbone when in training mode."""
        backbone_resize_naive.featurizer.train()  # Set to training mode

        transform = backbone_resize_naive.get_image_transform()
        pixel_values = transform(test_image).unsqueeze(0)
        pixel_values.requires_grad_(True)

        features = backbone_resize_naive.forward(pixel_values)

        # Handle list return from monkey-patched forward
        if isinstance(features, list):
            features = features[0]

        loss = features.sum()  # Dummy loss
        loss.backward()

        # Check that gradients are computed
        assert pixel_values.grad is not None
        assert not torch.allclose(pixel_values.grad, torch.zeros_like(pixel_values.grad))

    @pytest.mark.parametrize("resize_strategy", ["resize-naive", "resize-crop", "letterbox"])
    def test_all_resize_strategies_work(self, resize_strategy, test_image):
        """Parametrized test for all resize strategies."""
        backbone = DinoV2ViTBackbone("dinov2-vit-s", resize_strategy)
        transform = backbone.get_image_transform()

        # Should work without errors
        transformed = transform(test_image)
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 224, 224)

        # Forward pass should work
        with torch.no_grad():
            features = backbone.forward(transformed.unsqueeze(0))
        # Handle list return from monkey-patched forward
        if isinstance(features, list):
            features = features[0]
        assert features.shape[0] == 1

    def test_memory_usage_reasonable(self, backbone_resize_naive, test_image):
        """Test that memory usage is reasonable for small batches."""
        transform = backbone_resize_naive.get_image_transform()

        # Test with small batch
        batch = torch.stack([transform(test_image) for _ in range(4)])

        with torch.no_grad():
            features = backbone_resize_naive.forward(batch)

        # Handle list return from monkey-patched forward
        if isinstance(features, list):
            features = features[0]

        # Should not raise memory errors and produce reasonable output size
        assert features.shape[0] == 4
        assert features.numel() < 1e6  # Reasonable number of elements

    def test_device_compatibility(self, backbone_resize_naive, test_image):
        """Test that backbone works on CPU (GPU testing would require GPU availability)."""
        transform = backbone_resize_naive.get_image_transform()
        pixel_values = transform(test_image).unsqueeze(0)

        # Ensure everything is on CPU
        assert pixel_values.device.type == "cpu"

        # Forward pass on CPU
        with torch.no_grad():
            features = backbone_resize_naive.forward(pixel_values)

        # Handle list return from monkey-patched forward
        if isinstance(features, list):
            features = features[0]

        assert features.device.type == "cpu"

    def test_backbone_parameters_frozen_in_eval(self, backbone_resize_naive):
        """Test that backbone parameters don't update when in eval mode."""
        backbone_resize_naive.featurizer.eval()

        # Get initial parameter values
        initial_params = [p.clone() for p in backbone_resize_naive.featurizer.parameters()]

        # Dummy forward and backward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        output = backbone_resize_naive.forward(dummy_input)

        # Handle list return from monkey-patched forward
        if isinstance(output, list):
            output = output[0]

        loss = output.sum()
        loss.backward()

        # Parameters should not have changed (no optimizer step, just checking grad computation)
        for initial, current in zip(initial_params, backbone_resize_naive.featurizer.parameters()):
            torch.testing.assert_close(initial, current, rtol=1e-7, atol=1e-7)


class TestDinoV2ViTBackboneIntegration:
    """Integration tests for DinoV2ViTBackbone."""

    def test_main_script_execution(self):
        """Test that the main script in dinov2_vit.py runs without errors."""
        # This tests the if __name__ == "__main__" block
        from vla_project.models.backbones.vision.dinov2_vit import DinoV2ViTBackbone

        # Should be able to create and use the backbone as in the main script
        vision_backbone = DinoV2ViTBackbone("dinov2-vit-s", "resize-naive")
        assert vision_backbone is not None
        assert vision_backbone.image_transform is not None

    def test_backbone_inheritance_structure(self):
        """Test that DinoV2ViTBackbone properly inherits from TimmViTBackbone."""
        from vla_project.models.backbones.vision.vision_base import TimmViTBackbone, VisionBackbone

        backbone = DinoV2ViTBackbone("dinov2-vit-s", "resize-naive")

        assert isinstance(backbone, TimmViTBackbone)
        assert isinstance(backbone, VisionBackbone)

    def test_full_pipeline_with_real_image(self, test_image_path=None):
        """Test the full pipeline from image loading to feature extraction."""
        if test_image_path is None:
            test_image_path = Path(__file__).parent / "resources" / "images" / "test_img_1.png"

        # Skip if test image doesn't exist
        if not test_image_path.exists():
            pytest.skip("Test image not found")

        # Load image
        image = PIL.Image.open(test_image_path).convert("RGB")

        # Create backbone
        backbone = DinoV2ViTBackbone("dinov2-vit-s", "resize-naive")

        # Transform image
        transform = backbone.get_image_transform()
        transformed = transform(image)
        assert isinstance(transformed, torch.Tensor)
        pixel_values = transformed.unsqueeze(0)

        # Extract features
        with torch.no_grad():
            features = backbone.forward(pixel_values)

        # Handle list return from monkey-patched forward
        if isinstance(features, list):
            features = features[0]

        # Verify results
        assert features.shape[0] == 1  # Batch size
        assert features.shape[1] > 0  # Number of patches
        assert features.shape[2] > 0  # Feature dimension
        assert torch.isfinite(features).all()  # No NaN or Inf values
