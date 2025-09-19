"""Comprehensive tests for Gaussian diffusion processes and utilities.

This module tests the Gaussian diffusion implementation, diffusion utilities,
timestep sampling, and respacing functionality.
"""

from typing import Literal
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vla_project.action_model import create_diffusion
from vla_project.action_model.diffusion_utils import (
    approx_standard_normal_cdf,
    discretized_gaussian_log_likelihood,
    normal_kl,
)
from vla_project.action_model.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelOutput,
    ModelVarType,
    mean_flat,
)
from vla_project.action_model.respace import (
    SpacedDiffusion,
    space_timesteps,
)
from vla_project.action_model.timestep_sampler import (
    UniformSampler,
    create_named_schedule_sampler,
)


class TestDiffusionUtils:
    """Tests for diffusion utility functions."""

    def test_normal_kl_basic(self):
        """Test normal_kl with basic inputs."""
        mean1 = torch.tensor([0.0, 1.0])
        logvar1 = torch.tensor([0.0, 0.0])  # var = 1
        mean2 = torch.tensor([0.0, 0.0])
        logvar2 = torch.tensor([0.0, 0.0])  # var = 1

        kl = normal_kl(mean1, logvar1, mean2, logvar2)

        assert isinstance(kl, torch.Tensor)
        assert kl.shape == mean1.shape
        assert torch.isfinite(kl).all()

    def test_normal_kl_shapes(self):
        """Test normal_kl with different tensor shapes."""
        batch_size = 4
        feature_dim = 10

        mean1 = torch.randn(batch_size, feature_dim)
        logvar1 = torch.randn(batch_size, feature_dim)
        mean2 = torch.randn(batch_size, feature_dim)
        logvar2 = torch.randn(batch_size, feature_dim)

        kl = normal_kl(mean1, logvar1, mean2, logvar2)

        assert kl.shape == (batch_size, feature_dim)
        assert torch.isfinite(kl).all()

    def test_normal_kl_broadcasting(self):
        """Test normal_kl with broadcasting."""
        mean1 = torch.randn(4, 10)
        logvar1 = torch.randn(4, 10)
        mean2 = torch.randn(1, 10)  # Broadcast-compatible
        logvar2 = torch.randn(1, 10)  # Broadcast-compatible

        kl = normal_kl(mean1, logvar1, mean2, logvar2)

        assert kl.shape == (4, 10)
        assert torch.isfinite(kl).all()

    def test_normal_kl_numpy_inputs(self):
        """Test normal_kl with numpy inputs."""
        mean1 = np.array([0.0, 1.0])
        logvar1 = np.array([0.0, 0.0])
        mean2 = np.array([0.0, 0.0])
        logvar2 = np.array([0.0, 0.0])
        with pytest.raises(ValueError, match="at least one argument must be a Tensor"):
            kl = normal_kl(mean1, logvar1, mean2, logvar2)

    def test_discretized_gaussian_log_likelihood(self):
        """Test discretized Gaussian log likelihood computation."""
        x = torch.randn(2, 5, 10)
        means = torch.randn(2, 5, 10)
        log_scales = torch.randn(2, 5, 10)

        log_likelihood = discretized_gaussian_log_likelihood(x, means=means, log_scales=log_scales)

        if not isinstance(log_likelihood, torch.Tensor):
            raise TypeError("Expected torch.Tensor output")
        if log_likelihood.shape != x.shape:
            raise ValueError("Output shape doesn't match input shape")
        if not torch.isfinite(log_likelihood).all():
            raise ValueError("Output contains non-finite values")

    def test_approx_standard_normal_cdf(self):
        """Test approximate standard normal CDF."""
        x = torch.randn(3, 4, 5)

        cdf = approx_standard_normal_cdf(x)

        assert isinstance(cdf, torch.Tensor)
        assert cdf.shape == x.shape
        assert torch.all(cdf >= 0.0)
        assert torch.all(cdf <= 1.0)
        assert torch.isfinite(cdf).all()

    def test_mean_flat(self):
        """Test mean_flat utility function."""
        x = torch.randn(4, 10, 8, 6)

        result = mean_flat(x)

        assert result.shape == (4,)  # Only batch dimension preserved
        assert torch.isfinite(result).all()

    def test_mean_flat_2d(self):
        """Test mean_flat with 2D input."""
        x = torch.randn(5, 20)

        result = mean_flat(x)

        assert result.shape == (5,)
        assert torch.isfinite(result).all()


class TestTimestepRespacing:
    """Tests for timestep respacing functionality."""

    def test_space_timesteps_string_input(self):
        """Test space_timesteps with string input."""
        timesteps = space_timesteps(100, "10")

        assert isinstance(timesteps, set)
        assert len(timesteps) == 10
        assert all(isinstance(t, int) for t in timesteps)
        assert all(0 <= t < 100 for t in timesteps)

    def test_space_timesteps_list_input(self):
        """Test space_timesteps with list input."""
        timesteps = space_timesteps(100, [5, 5, 5])

        assert isinstance(timesteps, set)
        assert len(timesteps) == 15  # 5 + 5 + 5
        assert all(isinstance(t, int) for t in timesteps)

    def test_space_timesteps_ddim_format(self):
        """Test space_timesteps with DDIM format."""
        timesteps = space_timesteps(100, "ddim10")

        assert isinstance(timesteps, set)
        assert len(timesteps) == 10
        assert all(isinstance(t, int) for t in timesteps)

    def test_space_timesteps_edge_cases(self):
        """Test space_timesteps edge cases."""
        # Single timestep
        timesteps = space_timesteps(100, "1")
        assert len(timesteps) == 1

        # All timesteps
        timesteps = space_timesteps(10, "10")
        assert len(timesteps) == 10

    def test_space_timesteps_invalid_input(self):
        """Test space_timesteps with invalid input."""
        with pytest.raises((ValueError, AssertionError)):
            space_timesteps(100, "invalid")


class TestGaussianDiffusion:
    """Tests for GaussianDiffusion class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.return_value = torch.randn(2, 10, 7)  # Mock output
        return model

    @pytest.fixture
    def gaussian_diffusion(self, mock_model):
        """Create a GaussianDiffusion instance for testing."""
        return GaussianDiffusion(
            betas=np.linspace(0.0001, 0.02, 100),
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
        )

    def test_gaussian_diffusion_init(self, gaussian_diffusion):
        """Test GaussianDiffusion initialization."""
        assert gaussian_diffusion.num_timesteps == 100
        assert gaussian_diffusion.model_mean_type == ModelMeanType.EPSILON
        assert gaussian_diffusion.model_var_type == ModelVarType.FIXED_SMALL
        assert gaussian_diffusion.loss_type == LossType.MSE

        # Check that beta schedule is properly set
        assert hasattr(gaussian_diffusion, "betas")
        assert hasattr(gaussian_diffusion, "alphas_cumprod_prev")
        assert hasattr(gaussian_diffusion, "alphas_cumprod_next")
        assert hasattr(gaussian_diffusion, "alphas_cumprod")

    def test_gaussian_diffusion_q_sample(self, gaussian_diffusion):
        """Test forward diffusion sampling (q_sample)."""
        x_start = torch.randn(2, 10, 7)
        t = torch.randint(0, 100, (2,))

        x_t = gaussian_diffusion.q_sample(x_start, t)

        assert x_t.shape == x_start.shape
        assert torch.isfinite(x_t).all()

    def test_gaussian_diffusion_q_sample_with_noise(self, gaussian_diffusion):
        """Test q_sample with provided noise."""
        x_start = torch.randn(2, 10, 7)
        t = torch.randint(0, 100, (2,))
        noise = torch.randn_like(x_start)

        x_t = gaussian_diffusion.q_sample(x_start, t, noise=noise)

        assert x_t.shape == x_start.shape
        assert torch.isfinite(x_t).all()

    def test_gaussian_diffusion_q_posterior_mean_variance(self, gaussian_diffusion):
        """Test posterior mean and variance computation."""
        x_start = torch.randn(2, 10, 7)
        x_t = torch.randn(2, 10, 7)
        t = torch.randint(0, 100, (2,))

        posterior_mean, posterior_variance, posterior_log_variance = gaussian_diffusion.q_posterior_mean_variance(
            x_start, x_t, t,
        )

        assert posterior_mean.shape == x_start.shape
        assert posterior_variance.shape == x_start.shape
        assert posterior_log_variance.shape == x_start.shape
        assert torch.isfinite(posterior_mean).all()
        assert torch.isfinite(posterior_variance).all()
        assert torch.isfinite(posterior_log_variance).all()

    def test_gaussian_diffusion_training_losses(self, gaussian_diffusion, mock_model):
        """Test training loss computation."""
        x_start = torch.randn(2, 10, 7)
        t = torch.randint(0, 100, (2,))

        losses = gaussian_diffusion.training_losses(mock_model, x_start, t)

        assert isinstance(losses, dict)
        assert "loss" in losses
        assert isinstance(losses["loss"], torch.Tensor)
        assert torch.isfinite(losses["loss"]).all()

    def test_gaussian_diffusion_training_losses_with_model_kwargs(self, gaussian_diffusion, mock_model):
        """Test training losses with model kwargs."""
        x_start = torch.randn(2, 10, 7)
        t = torch.randint(0, 100, (2,))
        model_kwargs = {"condition": torch.randn(2, 1, 4096)}

        losses = gaussian_diffusion.training_losses(mock_model, x_start, t, model_kwargs=model_kwargs)

        assert isinstance(losses, dict)
        assert "loss" in losses

    # def test_gaussian_diffusion_p_losses(self, gaussian_diffusion, mock_model):
    #     """Test p_losses method."""
    #     x_start = torch.randn(2, 10, 7)
    #     t = torch.randint(0, 100, (2,))

    #     losses = gaussian_diffusion.p_losses(mock_model, x_start, t)

    #     assert isinstance(losses, dict)
    #     assert "loss" in losses
    #     assert torch.isfinite(losses["loss"]).all()

    def test_gaussian_diffusion_different_loss_types(self, mock_model):
        """Test GaussianDiffusion with different loss types."""
        loss_types = [LossType.MSE, LossType.RESCALED_MSE, LossType.KL, LossType.RESCALED_KL]

        for loss_type in loss_types:
            diffusion = GaussianDiffusion(
                betas=np.linspace(0.0001, 0.02, 50),
                model_mean_type=ModelMeanType.EPSILON,
                model_var_type=ModelVarType.FIXED_SMALL,
                loss_type=loss_type,
            )

            x_start = torch.randn(2, 5, 7)
            t = torch.randint(0, 50, (2,))

            losses = diffusion.training_losses(mock_model, x_start, t)
            assert torch.isfinite(losses["loss"]).all()

    def test_gaussian_diffusion_different_model_var_types(self, mock_model):
        """Test GaussianDiffusion with different model variance types."""
        var_types = [
            ModelVarType.LEARNED,
            ModelVarType.FIXED_SMALL,
            ModelVarType.FIXED_LARGE,
            ModelVarType.LEARNED_RANGE,
        ]

        for var_type in var_types:
            if var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                # Mock model should return double the channels for learned variance
                mock_model.return_value = torch.randn(2, 5, 14)  # 7 * 2
            else:
                mock_model.return_value = torch.randn(2, 5, 7)

            diffusion = GaussianDiffusion(
                betas=np.linspace(0.0001, 0.02, 50),
                model_mean_type=ModelMeanType.EPSILON,
                model_var_type=var_type,
                loss_type=LossType.MSE,
            )

            x_start = torch.randn(2, 5, 7)
            t = torch.randint(0, 50, (2,))

            losses = diffusion.training_losses(mock_model, x_start, t)
            assert torch.isfinite(losses["loss"]).all()


class TestSpacedDiffusion:
    """Tests for SpacedDiffusion class."""

    @pytest.fixture
    def original_diffusion(self):
        """Create an original GaussianDiffusion for testing."""
        return GaussianDiffusion(
            betas=np.linspace(0.0001, 0.02, 100),
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
        )

    @pytest.fixture
    def spaced_diffusion(self, original_diffusion):
        """Create a SpacedDiffusion instance."""
        use_timesteps = space_timesteps(100, "10")
        return SpacedDiffusion(use_timesteps=use_timesteps, **original_diffusion.get_original_args())

    def test_spaced_diffusion_init(self, spaced_diffusion):
        """Test SpacedDiffusion initialization."""
        assert isinstance(spaced_diffusion, SpacedDiffusion)
        assert spaced_diffusion.num_timesteps == 10  # Reduced from 100
        assert hasattr(spaced_diffusion, "timestep_map")

    def test_spaced_diffusion_q_sample(self, spaced_diffusion):
        """Test q_sample in spaced diffusion."""
        x_start = torch.randn(2, 8, 7)
        t = torch.randint(0, 10, (2,))  # Use spaced timesteps

        x_t = spaced_diffusion.q_sample(x_start, t)

        assert x_t.shape == x_start.shape
        assert torch.isfinite(x_t).all()

    def test_spaced_diffusion_training_losses(self, spaced_diffusion):
        """Test training losses with spaced diffusion."""
        mock_model = Mock()
        mock_model.return_value = torch.randn(2, 8, 7)

        x_start = torch.randn(2, 8, 7)
        t = torch.randint(0, 10, (2,))

        losses = spaced_diffusion.training_losses(mock_model, x_start, t)

        assert isinstance(losses, dict)
        assert "loss" in losses
        assert torch.isfinite(losses["loss"]).all()

    def test_spaced_diffusion_timestep_mapping(self, spaced_diffusion):
        """Test timestep mapping in spaced diffusion."""
        # Check that timestep mapping exists and is correct
        assert hasattr(spaced_diffusion, "timestep_map")
        assert len(spaced_diffusion.timestep_map) == spaced_diffusion.num_timesteps

    def test_spaced_diffusion_consistency_with_original(self, original_diffusion):
        """Test that spaced diffusion is consistent with original."""
        # Create spaced diffusion that uses all timesteps
        use_timesteps = set(range(100))
        spaced_diffusion = SpacedDiffusion(use_timesteps=use_timesteps, **original_diffusion.get_original_args())

        # Should be equivalent to original diffusion
        assert spaced_diffusion.num_timesteps == original_diffusion.num_timesteps


class TestTimestepSampling:
    """Tests for timestep sampling strategies."""

    @pytest.fixture
    def mock_diffusion(self):
        """Create a mock diffusion object."""
        diffusion = Mock()
        diffusion.num_timesteps = 100
        return diffusion

    def test_create_named_schedule_sampler_uniform(self, mock_diffusion):
        """Test creation of uniform schedule sampler."""
        sampler = create_named_schedule_sampler("uniform", mock_diffusion)

        assert isinstance(sampler, UniformSampler)

    def test_create_named_schedule_sampler_loss_aware(self, mock_diffusion):
        """Test creation of loss-aware schedule sampler."""
        sampler = create_named_schedule_sampler("loss-second-moment", mock_diffusion)

        assert isinstance(sampler, LossAwareSampler)

    def test_create_named_schedule_sampler_invalid(self, mock_diffusion):
        """Test creation with invalid sampler name."""
        with pytest.raises(NotImplementedError):
            create_named_schedule_sampler("invalid", mock_diffusion)

    def test_uniform_sampler(self, mock_diffusion):
        """Test UniformSampler functionality."""
        sampler = UniformSampler(mock_diffusion)

        batch_size = 4
        indices, weights = sampler.sample(batch_size, device=torch.device("cpu"))

        assert indices.shape == (batch_size,)
        assert weights.shape == (batch_size,)
        assert torch.all(indices >= 0)
        assert torch.all(indices < mock_diffusion.num_timesteps)
        assert torch.allclose(weights, torch.ones_like(weights))

    def test_loss_aware_sampler(self, mock_diffusion):
        """Test LossAwareSampler functionality."""
        sampler = LossAwareSampler(mock_diffusion)

        batch_size = 4
        indices, weights = sampler.sample(batch_size, device=torch.device("cpu"))

        assert indices.shape == (batch_size,)
        assert weights.shape == (batch_size,)
        assert torch.all(indices >= 0)
        assert torch.all(indices < mock_diffusion.num_timesteps)

    def test_loss_aware_sampler_update(self, mock_diffusion):
        """Test LossAwareSampler update functionality."""
        sampler = LossAwareSampler(mock_diffusion)

        # Simulate updating with losses
        timesteps = torch.randint(0, 100, (4,))
        losses = torch.randn(4).abs()  # Positive losses

        sampler.update_with_all_losses(timesteps, losses)

        # Should not raise any errors
        indices, weights = sampler.sample(4, device=torch.device("cpu"))
        assert indices.shape == (4,)
        assert weights.shape == (4,)


class TestCreateDiffusion:
    """Tests for the create_diffusion factory function."""

    def test_create_diffusion_basic(self):
        """Test basic diffusion creation."""
        diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule="linear",
            diffusion_steps=100,
        )

        assert isinstance(diffusion, GaussianDiffusion)
        assert diffusion.num_timesteps == 100

    def test_create_diffusion_with_respacing(self):
        """Test diffusion creation with timestep respacing."""
        diffusion = create_diffusion(
            timestep_respacing="10",
            noise_schedule="squaredcos_cap_v2",
            diffusion_steps=100,
        )

        assert isinstance(diffusion, SpacedDiffusion)
        assert diffusion.num_timesteps == 10

    def test_create_diffusion_ddim(self):
        """Test DDIM diffusion creation."""
        diffusion = create_diffusion(
            timestep_respacing="ddim10",
            noise_schedule="linear",
            diffusion_steps=100,
        )

        assert isinstance(diffusion, SpacedDiffusion)
        assert diffusion.num_timesteps == 10

    def test_create_diffusion_different_schedules(self):
        """Test diffusion creation with different noise schedules."""
        schedules: list[Literal["linear", "squaredcos_cap_v2"]] = ["linear", "squaredcos_cap_v2"]

        for schedule in schedules:
            diffusion = create_diffusion(
                timestep_respacing="",
                noise_schedule=schedule,
                diffusion_steps=50,
            )

            assert isinstance(diffusion, GaussianDiffusion)
            assert diffusion.num_timesteps == 50

    def test_create_diffusion_with_options(self):
        """Test diffusion creation with various options."""
        diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule="linear",
            diffusion_steps=100,
            use_kl=True,
            sigma_small=True,
            predict_xstart=True,
            learn_sigma=True,
        )

        assert isinstance(diffusion, GaussianDiffusion)
        assert diffusion.loss_type in [LossType.KL, LossType.RESCALED_KL]
        assert diffusion.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]

    def test_create_diffusion_rescale_learned_sigmas(self):
        """Test diffusion creation with rescaled learned sigmas."""
        diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule="linear",
            diffusion_steps=100,
            learn_sigma=True,
            rescale_learned_sigmas=True,
        )

        assert isinstance(diffusion, GaussianDiffusion)
        assert diffusion.model_var_type == ModelVarType.LEARNED_RANGE


class TestDiffusionIntegration:
    """Integration tests for diffusion components."""

    def test_full_diffusion_pipeline(self):
        """Test the complete diffusion pipeline."""
        # Create diffusion process
        diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule="squaredcos_cap_v2",
            diffusion_steps=50,
        )

        # Mock model
        mock_model = Mock()
        mock_model.return_value = torch.randn(2, 8, 7)

        # Test data
        x_start = torch.randn(2, 8, 7)
        t = torch.randint(0, 50, (2,))

        # Forward process (add noise)
        x_t = diffusion.q_sample(x_start, t)
        assert x_t.shape == x_start.shape

        # Training loss computation
        losses = diffusion.training_losses(mock_model, x_start, t)
        assert torch.isfinite(losses["loss"]).all()

    def test_diffusion_with_spaced_timesteps(self):
        """Test diffusion with spaced timesteps."""
        # Create spaced diffusion
        diffusion = create_diffusion(
            timestep_respacing="ddim5",
            noise_schedule="linear",
            diffusion_steps=100,
        )

        assert isinstance(diffusion, SpacedDiffusion)
        assert diffusion.num_timesteps == 5

        # Test sampling and loss computation
        mock_model = Mock()
        mock_model.return_value = torch.randn(2, 10, 7)

        x_start = torch.randn(2, 10, 7)
        t = torch.randint(0, 5, (2,))

        x_t = diffusion.q_sample(x_start, t)
        losses = diffusion.training_losses(mock_model, x_start, t)

        assert torch.isfinite(losses["loss"]).all()

    def test_timestep_sampler_integration(self):
        """Test integration of timestep samplers with diffusion."""
        diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule="linear",
            diffusion_steps=100,
        )

        # Test different samplers
        samplers = [
            create_named_schedule_sampler("uniform", diffusion),
            create_named_schedule_sampler("loss-second-moment", diffusion),
        ]

        for sampler in samplers:
            indices, weights = sampler.sample(4, device=torch.device("cpu"))
            assert indices.shape == (4,)
            assert weights.shape == (4,)
            assert torch.all(indices >= 0)
            assert torch.all(indices < diffusion.num_timesteps)
