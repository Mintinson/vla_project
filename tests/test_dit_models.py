"""Comprehensive tests for DiT (Diffusion Transformer) model components.

This module tests the core DiT model architecture including embedding layers,
transformer blocks, and the main DiT model class.
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from vla_project.action_model.models import (
    ActionEmbedder,
    DiT,
    DiTBlock,
    FinalLayer,
    HistoryEmbedder,
    LabelEmbedder,
    TimestepEmbedder,
    modulate,
)


class TestModulateFunction:
    """Tests for the modulate function used in adaptive layer normalization."""

    def test_modulate_basic(self):
        """Test basic modulation functionality."""
        x = torch.randn(2, 10, 512)
        shift = torch.randn(2, 10, 512)
        scale = torch.randn(2, 10, 512)

        result = modulate(x, shift, scale)
        expected = x * (1 + scale) + shift

        assert torch.allclose(result, expected, rtol=1e-5)

    def test_modulate_shapes(self):
        """Test modulate with different tensor shapes."""
        batch_size, seq_len, hidden_size = 4, 16, 768

        x = torch.randn(batch_size, seq_len, hidden_size)
        shift = torch.randn(batch_size, seq_len, hidden_size)
        scale = torch.randn(batch_size, seq_len, hidden_size)

        result = modulate(x, shift, scale)

        assert result.shape == (batch_size, seq_len, hidden_size)
        assert result.dtype == x.dtype

    def test_modulate_broadcasting(self):
        """Test modulate with broadcast-compatible shapes."""
        x = torch.randn(2, 10, 512)
        shift = torch.randn(1, 1, 512)  # Broadcast-compatible
        scale = torch.randn(1, 1, 512)  # Broadcast-compatible

        result = modulate(x, shift, scale)
        expected = x * (1 + scale) + shift

        assert torch.allclose(result, expected, rtol=1e-5)


class TestTimestepEmbedder:
    """Tests for TimestepEmbedder class."""

    @pytest.fixture
    def embedder(self):
        """Create a TimestepEmbedder for testing."""
        return TimestepEmbedder(hidden_size=512)

    def test_timestep_embedder_init(self, embedder):
        """Test TimestepEmbedder initialization."""
        assert embedder.hidden_size == 512
        assert isinstance(embedder.mlp, nn.Sequential)
        assert len(embedder.mlp) == 3  # Linear -> SiLU -> Linear

    def test_timestep_embedder_forward(self, embedder):
        """Test TimestepEmbedder forward pass."""
        batch_size = 4
        t = torch.randint(0, 1000, (batch_size,))

        result = embedder(t)

        assert result.shape == (batch_size, 512)
        assert result.dtype == torch.float32

    def test_timestep_embedder_different_timesteps(self, embedder):
        """Test embedder with different timestep values."""
        t1 = torch.tensor([0, 100, 500, 999])
        t2 = torch.tensor([0, 100, 500, 999])

        result1 = embedder(t1)
        result2 = embedder(t2)

        # Same timesteps should produce same embeddings
        assert torch.allclose(result1, result2, rtol=1e-5)

    def test_timestep_embedder_gradient_flow(self, embedder):
        """Test that gradients flow through the embedder."""
        t = torch.randint(0, 1000, (2,), dtype=torch.float, requires_grad=True)
        result = embedder(t)
        loss = result.sum()
        loss.backward()

        # Check that gradients exist
        for param in embedder.parameters():
            assert param.grad is not None


class TestLabelEmbedder:
    """Tests for LabelEmbedder class."""

    @pytest.fixture
    def embedder(self):
        """Create a LabelEmbedder for testing."""
        return LabelEmbedder(in_size=4096, hidden_size=768, dropout_prob=0.1)

    def test_label_embedder_init(self, embedder):
        """Test LabelEmbedder initialization."""
        assert embedder.dropout_prob == 0.1
        assert isinstance(embedder.linear, nn.Linear)
        assert embedder.linear.in_features == 4096
        assert embedder.linear.out_features == 768

    def test_label_embedder_forward_train(self, embedder):
        """Test LabelEmbedder forward pass in training mode."""
        batch_size = 2
        z = torch.randn(batch_size, 1, 4096)

        embedder.train()
        result = embedder(z, training=True)

        assert result.shape == (batch_size, 1, 768)
        assert result.dtype == torch.float32

    def test_label_embedder_forward_eval(self, embedder):
        """Test LabelEmbedder forward pass in evaluation mode."""
        batch_size = 2
        z = torch.randn(batch_size, 1, 4096)

        embedder.eval()
        result = embedder(z, training=False)

        assert result.shape == (batch_size, 1, 768)

    def test_label_embedder_dropout_consistency(self, embedder):
        """Test dropout behavior consistency."""
        z = torch.randn(2, 1, 4096)

        # In eval mode, should be deterministic
        embedder.eval()
        result1 = embedder(z, training=False)
        result2 = embedder(z, training=False)

        assert torch.allclose(result1, result2, rtol=1e-5)

    def test_label_embedder_class_dropout(self, embedder):
        """Test class dropout functionality."""
        z = torch.randn(2, 1, 4096)

        embedder.train()
        # With class dropout, some samples might use unconditional embedding
        with patch("torch.rand", return_value=torch.tensor(0.05)):  # Below dropout threshold
            result = embedder(z, training=True)
            assert result.shape == (2, 1, 768)


class TestActionEmbedder:
    """Tests for ActionEmbedder class."""

    @pytest.fixture
    def embedder(self):
        """Create an ActionEmbedder for testing."""
        return ActionEmbedder(action_size=7, hidden_size=512)

    def test_action_embedder_init(self, embedder):
        """Test ActionEmbedder initialization."""
        assert isinstance(embedder.linear, nn.Linear)
        assert embedder.linear.in_features == 7
        assert embedder.linear.out_features == 512

    def test_action_embedder_forward(self, embedder):
        """Test ActionEmbedder forward pass."""
        batch_size, seq_len = 4, 10
        actions = torch.randn(batch_size, seq_len, 7)

        result = embedder(actions)

        assert result.shape == (batch_size, seq_len, 512)
        assert result.dtype == torch.float32

    def test_action_embedder_different_shapes(self, embedder):
        """Test ActionEmbedder with different input shapes."""
        # Test different sequence lengths
        actions1 = torch.randn(2, 5, 7)
        actions2 = torch.randn(2, 15, 7)

        result1 = embedder(actions1)
        result2 = embedder(actions2)

        assert result1.shape == (2, 5, 512)
        assert result2.shape == (2, 15, 512)


class TestHistoryEmbedder:
    """Tests for HistoryEmbedder class."""

    @pytest.fixture
    def embedder(self):
        """Create a HistoryEmbedder for testing."""
        return HistoryEmbedder(action_size=7, hidden_size=512)

    def test_history_embedder_init(self, embedder):
        """Test HistoryEmbedder initialization."""
        assert isinstance(embedder.linear, nn.Linear)
        assert embedder.linear.in_features == 7
        assert embedder.linear.out_features == 512

    def test_history_embedder_forward(self, embedder):
        """Test HistoryEmbedder forward pass."""
        batch_size, history_len = 3, 8
        history = torch.randn(batch_size, history_len, 7)

        result = embedder(history)

        assert result.shape == (batch_size, history_len, 512)
        assert result.dtype == torch.float32


class TestDiTBlock:
    """Tests for DiTBlock transformer block."""

    @pytest.fixture
    def block(self):
        """Create a DiTBlock for testing."""
        return DiTBlock(hidden_size=768, num_heads=12, mlp_ratio=4.0)

    def test_dit_block_init(self, block):
        """Test DiTBlock initialization."""
        assert isinstance(block.norm1, nn.LayerNorm)
        assert isinstance(block.norm2, nn.LayerNorm)
        assert hasattr(block, "attn")
        assert hasattr(block, "mlp")

    def test_dit_block_forward(self, block):
        """Test DiTBlock forward pass."""
        batch_size, seq_len, hidden_size = 2, 16, 768
        x = torch.randn(batch_size, seq_len, hidden_size)

        result = block(x)

        assert result.shape == (batch_size, seq_len, hidden_size)
        assert result.dtype == x.dtype

    def test_dit_block_residual_connection(self, block):
        """Test that residual connections work properly."""
        x = torch.randn(2, 16, 768)

        # The output should be different from input due to transformations
        result = block(x)
        assert not torch.allclose(result, x, rtol=1e-3)

    def test_dit_block_gradient_flow(self, block):
        """Test gradient flow through the block."""
        x = torch.randn(2, 16, 768, requires_grad=True)
        result = block(x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        for param in block.parameters():
            assert param.grad is not None


class TestFinalLayer:
    """Tests for FinalLayer output layer."""

    @pytest.fixture
    def final_layer(self):
        """Create a FinalLayer for testing."""
        return FinalLayer(hidden_size=768, out_channels=7)

    def test_final_layer_init(self, final_layer):
        """Test FinalLayer initialization."""
        assert isinstance(final_layer.norm_final, nn.LayerNorm)
        assert isinstance(final_layer.linear, nn.Linear)
        assert final_layer.linear.in_features == 768
        assert final_layer.linear.out_features == 7

    def test_final_layer_forward(self, final_layer):
        """Test FinalLayer forward pass."""
        batch_size, seq_len, hidden_size = 2, 10, 768
        x = torch.randn(batch_size, seq_len, hidden_size)

        result = final_layer(x)

        assert result.shape == (batch_size, seq_len, 7)
        assert result.dtype == x.dtype

    def test_final_layer_zero_initialization(self, final_layer):
        """Test that final layer is properly zero-initialized."""
        # The final layer should be initialized to zero
        assert torch.allclose(final_layer.linear.weight, torch.zeros_like(final_layer.linear.weight))
        assert torch.allclose(final_layer.linear.bias, torch.zeros_like(final_layer.linear.bias))


class TestDiT:
    """Tests for the main DiT model."""

    @pytest.fixture
    def dit_small(self):
        """Create a small DiT model for testing."""
        return DiT(
            in_channels=7,
            hidden_size=384,
            depth=6,
            num_heads=4,
            token_size=4096,
            future_action_window_size=10,
            past_action_window_size=0,
        )

    @pytest.fixture
    def dit_base(self):
        """Create a base DiT model for testing."""
        return DiT(
            in_channels=7,
            hidden_size=768,
            depth=12,
            num_heads=12,
            token_size=4096,
            future_action_window_size=15,
            past_action_window_size=0,
        )

    def test_dit_init_small(self, dit_small):
        """Test DiT initialization with small configuration."""
        assert dit_small.in_channels == 7
        assert dit_small.out_channels == 7  # learn_sigma=False
        assert dit_small.num_heads == 4
        assert dit_small.future_action_window_size == 10
        assert dit_small.past_action_window_size == 0
        assert len(dit_small.blocks) == 6

    def test_dit_init_with_learn_sigma(self):
        """Test DiT initialization with learn_sigma=True."""
        dit = DiT(
            in_channels=7,
            hidden_size=384,
            depth=6,
            num_heads=4,
            learn_sigma=True,
        )
        assert dit.out_channels == 14  # 2 * in_channels when learn_sigma=True

    def test_dit_init_past_action_error(self):
        """Test that DiT raises error for non-zero past_action_window_size."""
        with pytest.raises(ValueError, match="Error: action_history is not used now"):
            DiT(past_action_window_size=5)

    def test_dit_forward(self, dit_small):
        """Test DiT forward pass."""
        batch_size = 2
        seq_len = dit_small.future_action_window_size

        x = torch.randn(batch_size, seq_len, 7)
        t = torch.randint(0, 1000, (batch_size,))
        z = torch.randn(batch_size, 1, 4096)

        result = dit_small(x, t, z)

        assert result.shape == (batch_size, seq_len, 7)
        assert result.dtype == x.dtype

    def test_dit_forward_with_cfg(self, dit_small):
        """Test DiT forward pass with classifier-free guidance."""
        batch_size = 2
        seq_len = dit_small.future_action_window_size

        x = torch.randn(batch_size, seq_len, 7)
        t = torch.randint(0, 1000, (batch_size,))
        z = torch.randn(batch_size, 1, 4096)
        cfg_scale = 2.0

        result = dit_small.forward_with_cfg(x, t, z, cfg_scale)

        assert result.shape == (batch_size, seq_len, 7)
        assert result.dtype == x.dtype

    def test_dit_forward_cfg_scale_1(self, dit_small):
        """Test that CFG with scale=1.0 equals normal forward."""
        batch_size = 2
        seq_len = dit_small.future_action_window_size

        x = torch.randn(batch_size, seq_len, 7)
        t = torch.randint(0, 1000, (batch_size,))
        z = torch.randn(batch_size, 1, 4096)

        normal_result = dit_small(x, t, z)
        cfg_result = dit_small.forward_with_cfg(x, t, z, cfg_scale=1.0)

        assert torch.allclose(normal_result, cfg_result, rtol=1e-4)

    def test_dit_different_batch_sizes(self, dit_small):
        """Test DiT with different batch sizes."""
        seq_len = dit_small.future_action_window_size

        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, seq_len, 7)
            t = torch.randint(0, 1000, (batch_size,))
            z = torch.randn(batch_size, 1, 4096)

            result = dit_small(x, t, z)
            assert result.shape == (batch_size, seq_len, 7)

    def test_dit_positional_embedding_shape(self, dit_small):
        """Test positional embedding has correct shape."""
        # +2 for conditional token and current action prediction
        expected_pos_emb_len = dit_small.future_action_window_size + dit_small.past_action_window_size + 2
        assert dit_small.positional_embedding.shape == (
            expected_pos_emb_len,
            dit_small.blocks[0].norm1.normalized_shape[0],
        )

    def test_dit_gradient_flow(self, dit_small):
        """Test gradient flow through the entire model."""
        batch_size = 2
        seq_len = dit_small.future_action_window_size

        x = torch.randn(batch_size, seq_len, 7, requires_grad=True)
        t = torch.randint(0, 1000, (batch_size,))
        z = torch.randn(batch_size, 1, 4096, requires_grad=True)

        result = dit_small(x, t, z)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert z.grad is not None
        for param in dit_small.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_dit_eval_mode(self, dit_small):
        """Test DiT behavior in evaluation mode."""
        dit_small.eval()

        batch_size = 2
        seq_len = dit_small.future_action_window_size

        x = torch.randn(batch_size, seq_len, 7)
        t = torch.randint(0, 1000, (batch_size,))
        z = torch.randn(batch_size, 1, 4096)

        with torch.no_grad():
            result1 = dit_small(x, t, z)
            result2 = dit_small(x, t, z)

        # Should be deterministic in eval mode
        assert torch.allclose(result1, result2, rtol=1e-5)

    def test_dit_weight_initialization(self, dit_small):
        """Test that weights are properly initialized."""
        # Check that weights are not all zeros (indicating proper initialization)
        for name, param in dit_small.named_parameters():
            if param.requires_grad and param.numel() > 1:
                assert not torch.allclose(param, torch.zeros_like(param)), f"Parameter {name} is all zeros"

    def test_dit_parameter_count(self, dit_small, dit_base):
        """Test parameter count scaling with model size."""
        small_params = sum(p.numel() for p in dit_small.parameters())
        base_params = sum(p.numel() for p in dit_base.parameters())

        # Base model should have significantly more parameters than small
        assert base_params > small_params * 2

    def test_dit_memory_efficient_attention(self, dit_small):
        """Test model works with different sequence lengths efficiently."""
        batch_size = 1

        # Test with different sequence lengths
        for seq_len in [5, 10, 20]:
            dit = DiT(
                in_channels=7,
                hidden_size=384,
                depth=6,
                num_heads=4,
                future_action_window_size=seq_len,
            )

            x = torch.randn(batch_size, seq_len, 7)
            t = torch.randint(0, 1000, (batch_size,))
            z = torch.randn(batch_size, 1, 4096)

            result = dit(x, t, z)
            assert result.shape == (batch_size, seq_len, 7)

    def test_dit_numerical_stability(self, dit_small):
        """Test numerical stability with extreme inputs."""
        batch_size = 2
        seq_len = dit_small.future_action_window_size

        # Test with very small values
        x_small = torch.randn(batch_size, seq_len, 7) * 1e-6
        t = torch.randint(0, 1000, (batch_size,))
        z_small = torch.randn(batch_size, 1, 4096) * 1e-6

        result_small = dit_small(x_small, t, z_small)
        assert torch.isfinite(result_small).all()

        # Test with larger values
        x_large = torch.randn(batch_size, seq_len, 7) * 10
        z_large = torch.randn(batch_size, 1, 4096) * 10

        result_large = dit_small(x_large, t, z_large)
        assert torch.isfinite(result_large).all()


class TestDiTIntegration:
    """Integration tests for DiT model components."""

    def test_dit_component_compatibility(self):
        """Test that all DiT components work together."""
        # Create model
        dit = DiT(
            in_channels=7,
            hidden_size=512,
            depth=8,
            num_heads=8,
            future_action_window_size=12,
        )

        # Test full forward pass
        batch_size = 3
        x = torch.randn(batch_size, 12, 7)
        t = torch.randint(0, 1000, (batch_size,))
        z = torch.randn(batch_size, 1, 4096)

        # Normal forward
        result = dit(x, t, z)
        assert result.shape == (batch_size, 12, 7)

        # Forward with CFG
        cfg_result = dit.forward_with_cfg(x, t, z, cfg_scale=1.5)
        assert cfg_result.shape == (batch_size, 12, 7)

        # Check that results are different (CFG should modify output)
        assert not torch.allclose(result, cfg_result, rtol=1e-3)

    def test_dit_training_loop_simulation(self):
        """Simulate a training loop to test end-to-end functionality."""
        dit = DiT(
            in_channels=7,
            hidden_size=256,
            depth=4,
            num_heads=4,
            future_action_window_size=8,
        )

        optimizer = torch.optim.Adam(dit.parameters(), lr=1e-4)

        # Simulate training steps
        for step in range(3):
            batch_size = 2
            x = torch.randn(batch_size, 8, 7)
            t = torch.randint(0, 1000, (batch_size,))
            z = torch.randn(batch_size, 1, 4096)
            target = torch.randn(batch_size, 8, 7)

            # Forward pass
            pred = dit(x, t, z)
            loss = torch.nn.functional.mse_loss(pred, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            assert torch.isfinite(loss)
            assert loss.item() >= 0
