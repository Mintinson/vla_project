"""Comprehensive test suite for the action_model sub-module.

This module provides extensive testing for all components of the action_model package,
including diffusion processes, transformer models, embedding layers, and utility functions.
The tests cover both unit testing of individual components and integration testing
of the complete action modeling pipeline.

Test coverage includes:
    - Factory functions for creating diffusion processes
    - TimestepEmbedder and LabelEmbedder functionality
    - DiT model architecture and forward passes
    - Gaussian diffusion processes and noise schedules
    - Timestep respacing and DDIM sampling
    - ActionModel integration and end-to-end functionality
    - Error handling and edge cases
"""

from typing import Literal
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vla_project.action_model import create_diffusion
from vla_project.action_model.actionmodel import ActionModel, get_dit_b, get_dit_l, get_dit_s
from vla_project.action_model.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
    mean_flat,
)
from vla_project.action_model.models import DiT, LabelEmbedder, TimestepEmbedder, modulate
from vla_project.action_model.respace import SpacedDiffusion, space_timesteps

# Constants for testing
ACTION_DIM = 7
TOKEN_SIZE = 4096
DIFFUSION_STEPS = 100
BATCH_SIZE = 4


class TestFactoryFunctions:
    """Test suite for action_model factory functions."""

    def test_create_diffusion_basic(self) -> None:
        """Test basic diffusion creation with default parameters."""
        diffusion = create_diffusion(
            timestep_respacing=None,
            noise_schedule="linear",
        )

        assert isinstance(diffusion, SpacedDiffusion)
        assert diffusion.num_timesteps == 1000
        assert len(diffusion.use_timesteps) == 1000

    def test_create_diffusion_ddim(self) -> None:
        """Test DDIM diffusion creation with respacing."""
        diffusion = create_diffusion(
            timestep_respacing="ddim50",
            noise_schedule="squaredcos_cap_v2",
        )

        assert isinstance(diffusion, SpacedDiffusion)
        assert len(diffusion.use_timesteps) == 50

    def test_create_diffusion_kl_loss(self) -> None:
        """Test diffusion creation with KL divergence loss."""
        diffusion = create_diffusion(
            timestep_respacing=None,
            noise_schedule="linear",
            use_kl=True,
        )

        assert diffusion.loss_type == LossType.RESCALED_KL

    def test_create_diffusion_learned_sigma(self) -> None:
        """Test diffusion creation with learned variance."""
        diffusion = create_diffusion(
            timestep_respacing=None,
            noise_schedule="linear",
            learn_sigma=True,
        )

        assert diffusion.model_var_type == ModelVarType.LEARNED_RANGE

    def test_create_diffusion_predict_xstart(self) -> None:
        """Test diffusion creation with x0 prediction."""
        diffusion = create_diffusion(
            timestep_respacing=None,
            noise_schedule="linear",
            predict_xstart=True,
        )

        assert diffusion.model_mean_type == ModelMeanType.START_X


class TestTimestepEmbedder:
    """Test suite for TimestepEmbedder class."""

    @pytest.fixture
    def embedder(self) -> TimestepEmbedder:
        """Create a TimestepEmbedder for testing."""
        return TimestepEmbedder(hidden_size=128, frequency_embedding_size=64)

    def test_init(self, embedder: TimestepEmbedder) -> None:
        """Test TimestepEmbedder initialization."""
        assert embedder.frequency_embedding_size == 64
        assert isinstance(embedder.mlp, torch.nn.Sequential)
        assert len(embedder.mlp) == 3  # Linear, SiLU, Linear

    def test_timestep_embedding_basic(self) -> None:
        """Test basic timestep embedding functionality."""
        t = torch.tensor([0, 100, 500])
        embedding = TimestepEmbedder.timestep_embedding(t, dim=128)

        assert embedding.shape == (3, 128)
        assert torch.isfinite(embedding).all()

    def test_forward_pass(self, embedder: TimestepEmbedder) -> None:
        """Test forward pass through TimestepEmbedder."""
        t = torch.tensor([0, 100, 500, 999])

        with torch.no_grad():
            output = embedder(t)

        assert output.shape == (4, 128)
        assert torch.isfinite(output).all()

    def test_gradient_flow(self, embedder: TimestepEmbedder) -> None:
        """Test that gradients flow through the embedder."""
        t = torch.tensor([100.0], requires_grad=True)
        output = embedder(t)
        loss = output.sum()
        loss.backward()

        assert t.grad is not None
        assert not torch.allclose(t.grad, torch.zeros_like(t.grad))


class TestLabelEmbedder:
    """Test suite for LabelEmbedder class."""

    @pytest.fixture
    def embedder(self) -> LabelEmbedder:
        """Create a LabelEmbedder for testing."""
        return LabelEmbedder(
            in_size=TOKEN_SIZE,
            hidden_size=512,
            dropout_prob=0.1,
            conditions_shape=(1, 1, TOKEN_SIZE),
        )

    @pytest.fixture
    def embedder_no_dropout(self) -> LabelEmbedder:
        """Create a LabelEmbedder without dropout for testing."""
        return LabelEmbedder(
            in_size=TOKEN_SIZE,
            hidden_size=512,
            dropout_prob=0.0,
        )

    def test_init_with_dropout(self, embedder: LabelEmbedder) -> None:
        """Test LabelEmbedder initialization with dropout."""
        assert embedder.dropout_prob == 0.1
        assert hasattr(embedder, "uncondition")
        assert isinstance(embedder.linear, torch.nn.Linear)

    def test_init_without_dropout(self, embedder_no_dropout: LabelEmbedder) -> None:
        """Test LabelEmbedder initialization without dropout."""
        assert embedder_no_dropout.dropout_prob == 0.0
        assert not hasattr(embedder_no_dropout, "uncondition")

    def test_forward_no_dropout(self, embedder_no_dropout: LabelEmbedder) -> None:
        """Test forward pass without dropout."""
        conditions = torch.randn(BATCH_SIZE, 1, TOKEN_SIZE)

        output = embedder_no_dropout(conditions, train=True)

        assert output.shape == (BATCH_SIZE, 1, 512)
        assert torch.isfinite(output).all()

    def test_token_drop_functionality(self, embedder: LabelEmbedder) -> None:
        """Test token_drop method directly."""
        conditions = torch.randn(BATCH_SIZE, 1, TOKEN_SIZE)
        force_drop_ids = torch.tensor([1, 0, 0, 1])

        dropped_conditions = embedder.token_drop(conditions, force_drop_ids)

        assert dropped_conditions.shape == conditions.shape
        # Verify that specified positions were replaced with unconditional embeddings
        assert torch.allclose(dropped_conditions[0, 0], embedder.uncondition)
        assert torch.allclose(dropped_conditions[3, 0], embedder.uncondition)


class TestDiTModel:
    """Test suite for DiT (Diffusion Transformer) model."""

    @pytest.fixture
    def dit_small(self) -> DiT:
        """Create a small DiT model for testing."""
        return get_dit_s(
            in_channels=ACTION_DIM,
            token_size=TOKEN_SIZE,
            future_action_window_size=1,
            past_action_window_size=0,
        )

    def test_dit_factory_functions(self) -> None:
        """Test DiT factory functions create correct architectures."""
        dit_s = get_dit_s(
            in_channels=ACTION_DIM,
            token_size=TOKEN_SIZE,
            future_action_window_size=1,
            past_action_window_size=0,
        )
        dit_b = get_dit_b(
            in_channels=ACTION_DIM,
            token_size=TOKEN_SIZE,
            future_action_window_size=1,
            past_action_window_size=0,
        )
        dit_l = get_dit_l(
            in_channels=ACTION_DIM,
            token_size=TOKEN_SIZE,
            future_action_window_size=1,
            past_action_window_size=0,
        )

        # Check model sizes
        assert dit_s.hidden_size == 384
        assert dit_s.depth == 6
        assert dit_s.num_heads == 4

        assert dit_b.hidden_size == 768
        assert dit_b.depth == 12
        assert dit_b.num_heads == 12

        assert dit_l.hidden_size == 1024
        assert dit_l.depth == 24
        assert dit_l.num_heads == 16

    def test_dit_forward_pass(self, dit_small: DiT) -> None:
        """Test DiT forward pass."""
        # Create inputs with correct shapes
        x_t = torch.randn(BATCH_SIZE, 1, ACTION_DIM)  # Noisy actions
        t = torch.randint(0, 1000, (BATCH_SIZE,))  # Timesteps
        z = torch.randn(BATCH_SIZE, 1, TOKEN_SIZE)  # Conditioning

        with torch.no_grad():
            output = dit_small(x_t, t, z)

        assert output.shape == (BATCH_SIZE, 1, ACTION_DIM)
        assert torch.isfinite(output).all()

    def test_dit_gradient_flow(self, dit_small: DiT) -> None:
        """Test gradient flow through DiT model."""
        x_t = torch.randn(2, 1, ACTION_DIM, requires_grad=True)
        t = torch.randint(0, 1000, (2,))
        z = torch.randn(2, 1, TOKEN_SIZE, requires_grad=True)

        output = dit_small(x_t, t, z)
        loss = output.sum()
        loss.backward()

        assert x_t.grad is not None
        assert z.grad is not None
        assert not torch.allclose(x_t.grad, torch.zeros_like(x_t.grad))

    def test_modulate_function(self) -> None:
        """Test the modulate function used in DiT."""
        x = torch.randn(2, 4, 128)
        shift = torch.randn(2, 4, 128)
        scale = torch.randn(2, 4, 128)

        result = modulate(x, shift, scale)
        expected = x * (1 + scale) + shift

        assert torch.allclose(result, expected)
        assert result.shape == x.shape


class TestGaussianDiffusion:
    """Test suite for GaussianDiffusion class."""

    @pytest.fixture
    def diffusion(self) -> GaussianDiffusion:
        """Create a GaussianDiffusion for testing."""
        betas = get_named_beta_schedule("linear", 1000)
        return GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
        )

    def test_beta_schedules(self) -> None:
        """Test different beta schedules."""
        schedules: list[Literal["linear", "squaredcos_cap_v2"]] = ["linear", "squaredcos_cap_v2"]

        for schedule in schedules:
            betas = get_named_beta_schedule(schedule, 1000)
            assert isinstance(betas, np.ndarray)
            assert len(betas) == 1000
            assert np.all(betas > 0)
            assert np.all(betas < 1)

    def test_diffusion_initialization(self, diffusion: GaussianDiffusion) -> None:
        """Test GaussianDiffusion initialization."""
        assert diffusion.num_timesteps == 1000
        assert diffusion.model_mean_type == ModelMeanType.EPSILON
        assert diffusion.model_var_type == ModelVarType.FIXED_SMALL
        assert diffusion.loss_type == LossType.MSE

    def test_q_sample(self, diffusion: GaussianDiffusion) -> None:
        """Test forward diffusion (q_sample)."""
        x_start = torch.randn(BATCH_SIZE, ACTION_DIM)
        t = torch.randint(0, 1000, (BATCH_SIZE,))

        x_t = diffusion.q_sample(x_start, t)

        assert x_t.shape == x_start.shape
        assert torch.isfinite(x_t).all()

    def test_training_losses(self, diffusion: GaussianDiffusion) -> None:
        """Test training loss computation."""
        # Mock model
        model = MagicMock()
        model.return_value = torch.randn(BATCH_SIZE, ACTION_DIM)

        x_start = torch.randn(BATCH_SIZE, ACTION_DIM)
        t = torch.randint(0, 1000, (BATCH_SIZE,))

        losses = diffusion.training_losses(model, x_start, t)

        assert "loss" in losses
        assert isinstance(losses["loss"], torch.Tensor)
        assert losses["loss"].shape == (BATCH_SIZE,)

    def test_mean_flat_function(self) -> None:
        """Test mean_flat utility function."""
        x = torch.randn(BATCH_SIZE, ACTION_DIM, 3)
        result = mean_flat(x)

        assert result.shape == (BATCH_SIZE,)
        expected = x.view(BATCH_SIZE, -1).mean(dim=1)
        assert torch.allclose(result, expected)


class TestActionModel:
    """Test suite for ActionModel class."""

    @pytest.fixture
    def action_model(self) -> ActionModel:
        """Create an ActionModel for testing."""
        return ActionModel(
            token_size=TOKEN_SIZE,
            model_type="DiT-S",
            in_channels=ACTION_DIM,
            future_action_window_size=1,
            past_action_window_size=0,
            diffusion_steps=DIFFUSION_STEPS,
            noise_schedule="linear",
        )

    def test_action_model_init(self, action_model: ActionModel) -> None:
        """Test ActionModel initialization."""
        assert action_model.in_channels == ACTION_DIM
        assert action_model.diffusion_steps == DIFFUSION_STEPS
        assert hasattr(action_model, "net")
        assert hasattr(action_model, "diffusion")

    def test_action_model_loss_computation(self, action_model: ActionModel) -> None:
        """Test ActionModel loss computation."""
        # Create test data
        x = torch.randn(BATCH_SIZE, 1, ACTION_DIM)  # Ground truth actions
        z = torch.randn(BATCH_SIZE, 1, TOKEN_SIZE)  # Conditioning

        loss = action_model.loss(x, z)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert torch.isfinite(loss).all()

    def test_action_model_ddim_creation(self, action_model: ActionModel) -> None:
        """Test DDIM diffusion creation."""
        ddim_diffusion = action_model.create_ddim(ddim_step=20)

        assert isinstance(ddim_diffusion, SpacedDiffusion)
        assert len(ddim_diffusion.use_timesteps) == 20
        assert action_model.ddim_diffusion is not None

    def test_action_model_training_mode(self, action_model: ActionModel) -> None:
        """Test ActionModel in training mode."""
        action_model.train()

        x = torch.randn(2, 1, ACTION_DIM, requires_grad=True)
        z = torch.randn(2, 1, TOKEN_SIZE, requires_grad=True)

        loss = action_model.loss(x, z)
        loss.backward()

        assert x.grad is not None
        assert z.grad is not None

    def test_action_model_net_forward(self, action_model: ActionModel) -> None:
        """Test direct forward pass through the network."""
        action_model.eval()

        x_t = torch.randn(2, 1, ACTION_DIM)  # Noisy actions
        timesteps = torch.randint(0, DIFFUSION_STEPS, (2,))  # Timesteps
        z = torch.randn(2, 1, TOKEN_SIZE)  # Conditioning

        with torch.no_grad():
            output = action_model.net(x_t, timesteps, z)

        assert output.shape == (2, 1, ACTION_DIM)
        assert torch.isfinite(output).all()


class TestTimestepRespacing:
    """Test suite for timestep respacing functionality."""

    def test_space_timesteps_basic(self) -> None:
        """Test basic timestep spacing."""
        timesteps = space_timesteps(100, [50])
        assert len(timesteps) == 50
        assert all(isinstance(t, int) for t in timesteps)

    def test_space_timesteps_ddim(self) -> None:
        """Test DDIM timestep spacing."""
        timesteps = space_timesteps(1000, "ddim50")
        assert len(timesteps) == 50

    def test_space_timesteps_multiple_sections(self) -> None:
        """Test timestep spacing with multiple sections."""
        timesteps = space_timesteps(300, [10, 15, 20])
        assert len(timesteps) == 45  # 10 + 15 + 20

    def test_space_timesteps_invalid_ddim(self) -> None:
        """Test invalid DDIM configuration."""
        with pytest.raises(ValueError):
            space_timesteps(100, "ddim101")  # More steps than available


class TestSpacedDiffusion:
    """Test suite for SpacedDiffusion class."""

    @pytest.fixture
    def spaced_diffusion(self) -> SpacedDiffusion:
        """Create a SpacedDiffusion for testing."""
        betas = get_named_beta_schedule("linear", 1000)
        use_timesteps = space_timesteps(1000, "ddim50")

        return SpacedDiffusion(
            use_timesteps=use_timesteps,
            betas=betas,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
        )

    def test_spaced_diffusion_init(self, spaced_diffusion: SpacedDiffusion) -> None:
        """Test SpacedDiffusion initialization."""
        assert spaced_diffusion.num_timesteps == 50
        assert spaced_diffusion.original_num_steps == 1000
        assert len(spaced_diffusion.timestep_map) == 50


class TestIntegration:
    """Integration tests for the complete action_model package."""

    def test_end_to_end_pipeline(self) -> None:
        """Test complete end-to-end action modeling pipeline."""
        # Create action model
        action_model = ActionModel(
            token_size=TOKEN_SIZE,
            model_type="DiT-S",
            in_channels=ACTION_DIM,
            future_action_window_size=1,
            past_action_window_size=0,
            diffusion_steps=DIFFUSION_STEPS,
        )

        # Test training step
        x = torch.randn(BATCH_SIZE, 1, ACTION_DIM)  # Ground truth actions
        z = torch.randn(BATCH_SIZE, 1, TOKEN_SIZE)  # Conditioning

        # Compute loss
        loss = action_model.loss(x, z)
        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss).all()

        # Test DDIM sampling
        ddim_diffusion = action_model.create_ddim(ddim_step=10)
        assert isinstance(ddim_diffusion, SpacedDiffusion)

    def test_device_compatibility(self) -> None:
        """Test device compatibility across components."""
        device = torch.device("cpu")

        # Create model on device
        model = get_dit_s(
            in_channels=ACTION_DIM,
            token_size=TOKEN_SIZE,
            future_action_window_size=1,
            past_action_window_size=0,
        ).to(device)

        # Create inputs on device
        x_t = torch.randn(2, 1, ACTION_DIM, device=device)
        z = torch.randn(2, 1, TOKEN_SIZE, device=device)
        timesteps = torch.randint(0, 100, (2,), device=device)

        # Forward pass
        with torch.no_grad():
            output = model(x_t, timesteps, z)

        assert output.device == device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
