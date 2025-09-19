"""Tests for timestep sampling and basic diffusion functionality.

This module tests timestep samplers and the create_diffusion factory function.
"""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vla_project.action_model import create_diffusion
from vla_project.action_model.respace import space_timesteps
from vla_project.action_model.timestep_sampler import (
    UniformSampler,
    create_named_schedule_sampler,
)


class TestTimestepSampling:
    """Tests for timestep sampling strategies."""

    @pytest.fixture
    def mock_diffusion(self) -> Mock:
        """Create a mock diffusion object."""
        diffusion = Mock()
        diffusion.num_timesteps = 100
        return diffusion

    def test_create_named_schedule_sampler_uniform(self, mock_diffusion: Mock) -> None:
        """Test creation of uniform schedule sampler."""
        sampler = create_named_schedule_sampler("uniform", mock_diffusion)

        if not isinstance(sampler, UniformSampler):
            msg = "Expected UniformSampler instance"
            raise TypeError(msg)

    def test_create_named_schedule_sampler_invalid(self, mock_diffusion: Mock) -> None:
        """Test creation with invalid sampler name."""
        with pytest.raises(NotImplementedError):
            create_named_schedule_sampler("invalid", mock_diffusion)

    def test_uniform_sampler(self, mock_diffusion: Mock) -> None:
        """Test UniformSampler functionality."""
        sampler = UniformSampler(mock_diffusion)

        batch_size = 4
        indices, weights = sampler.sample(batch_size, device=torch.device("cpu"))

        expected_indices_shape = (batch_size,)
        expected_weights_shape = (batch_size,)

        if indices.shape != expected_indices_shape:
            msg = f"Expected indices shape {expected_indices_shape}, got {indices.shape}"
            raise ValueError(msg)
        if weights.shape != expected_weights_shape:
            msg = f"Expected weights shape {expected_weights_shape}, got {weights.shape}"
            raise ValueError(msg)
        if not torch.all(indices >= 0):
            msg = "All indices should be non-negative"
            raise ValueError(msg)
        if not torch.all(indices < mock_diffusion.num_timesteps):
            msg = f"All indices should be less than {mock_diffusion.num_timesteps}"
            raise ValueError(msg)
        if not torch.allclose(weights, torch.ones_like(weights)):
            msg = "Uniform sampler should produce equal weights"
            raise ValueError(msg)

    def test_uniform_sampler_weights(self, mock_diffusion: Mock) -> None:
        """Test UniformSampler weights property."""
        sampler = UniformSampler(mock_diffusion)
        weights = sampler.weights()

        expected_length = mock_diffusion.num_timesteps
        if len(weights) != expected_length:
            msg = f"Expected weights length {expected_length}, got {len(weights)}"
            raise ValueError(msg)
        if not np.allclose(weights, np.ones_like(weights)):
            msg = "Uniform sampler weights should all be equal"
            raise ValueError(msg)


class TestTimestepRespacing:
    """Tests for timestep respacing functionality."""

    def test_space_timesteps_string_input(self) -> None:
        """Test space_timesteps with string input."""
        total_timesteps = 100
        respacing = "10"
        timesteps = space_timesteps(total_timesteps, respacing)

        if not isinstance(timesteps, set):
            msg = "Expected set output"
            raise TypeError(msg)
        expected_length = 10
        if len(timesteps) != expected_length:
            msg = f"Expected {expected_length} timesteps, got {len(timesteps)}"
            raise ValueError(msg)
        if not all(isinstance(t, int) for t in timesteps):
            msg = "All timesteps should be integers"
            raise TypeError(msg)
        if not all(0 <= t < total_timesteps for t in timesteps):
            msg = f"All timesteps should be in range [0, {total_timesteps})"
            raise ValueError(msg)

    def test_space_timesteps_list_input(self) -> None:
        """Test space_timesteps with list input."""
        total_timesteps = 100
        respacing = [5, 5, 5]
        timesteps = space_timesteps(total_timesteps, respacing)

        if not isinstance(timesteps, set):
            msg = "Expected set output"
            raise TypeError(msg)
        expected_length = 15  # 5 + 5 + 5
        if len(timesteps) != expected_length:
            msg = f"Expected {expected_length} timesteps, got {len(timesteps)}"
            raise ValueError(msg)
        if not all(isinstance(t, int) for t in timesteps):
            msg = "All timesteps should be integers"
            raise TypeError(msg)

    def test_space_timesteps_ddim_format(self) -> None:
        """Test space_timesteps with DDIM format."""
        total_timesteps = 100
        respacing = "ddim10"
        timesteps = space_timesteps(total_timesteps, respacing)

        if not isinstance(timesteps, set):
            msg = "Expected set output"
            raise TypeError(msg)
        expected_length = 10
        if len(timesteps) != expected_length:
            msg = f"Expected {expected_length} timesteps, got {len(timesteps)}"
            raise ValueError(msg)
        if not all(isinstance(t, int) for t in timesteps):
            msg = "All timesteps should be integers"
            raise TypeError(msg)

    def test_space_timesteps_edge_cases(self) -> None:
        """Test space_timesteps edge cases."""
        # Single timestep
        timesteps = space_timesteps(100, "1")
        if len(timesteps) != 1:
            msg = "Expected single timestep"
            raise ValueError(msg)

        # All timesteps
        timesteps = space_timesteps(10, "10")
        if len(timesteps) != 10:
            msg = "Expected all 10 timesteps"
            raise ValueError(msg)

    def test_space_timesteps_invalid_input(self) -> None:
        """Test space_timesteps with invalid input."""
        with pytest.raises((ValueError, AssertionError)):
            space_timesteps(100, "invalid")


class TestCreateDiffusion:
    """Tests for the create_diffusion factory function."""

    def test_create_diffusion_basic(self) -> None:
        """Test basic diffusion creation."""
        diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule="linear",
            diffusion_steps=100,
        )

        if diffusion.num_timesteps != 100:
            msg = f"Expected 100 timesteps, got {diffusion.num_timesteps}"
            raise ValueError(msg)

    def test_create_diffusion_with_respacing(self) -> None:
        """Test diffusion creation with timestep respacing."""
        diffusion = create_diffusion(
            timestep_respacing="10",
            noise_schedule="squaredcos_cap_v2",
            diffusion_steps=100,
        )

        expected_timesteps = 10
        if diffusion.num_timesteps != expected_timesteps:
            msg = f"Expected {expected_timesteps} timesteps, got {diffusion.num_timesteps}"
            raise ValueError(msg)

    def test_create_diffusion_ddim(self) -> None:
        """Test DDIM diffusion creation."""
        diffusion = create_diffusion(
            timestep_respacing="ddim10",
            noise_schedule="linear",
            diffusion_steps=100,
        )

        expected_timesteps = 10
        if diffusion.num_timesteps != expected_timesteps:
            msg = f"Expected {expected_timesteps} timesteps, got {diffusion.num_timesteps}"
            raise ValueError(msg)

    def test_create_diffusion_different_schedules(self) -> None:
        """Test diffusion creation with different noise schedules."""
        schedules = ["linear", "squaredcos_cap_v2"]

        for schedule in schedules:
            diffusion = create_diffusion(
                timestep_respacing="",
                noise_schedule=schedule,
                diffusion_steps=50,
            )

            expected_timesteps = 50
            if diffusion.num_timesteps != expected_timesteps:
                msg = f"Expected {expected_timesteps} timesteps, got {diffusion.num_timesteps}"
                raise ValueError(msg)

    def test_create_diffusion_with_options(self) -> None:
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

        if diffusion.num_timesteps != 100:
            msg = f"Expected 100 timesteps, got {diffusion.num_timesteps}"
            raise ValueError(msg)
        # Additional checks for configuration could be added here

    def test_create_diffusion_rescale_learned_sigmas(self) -> None:
        """Test diffusion creation with rescaled learned sigmas."""
        diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule="linear",
            diffusion_steps=100,
            learn_sigma=True,
            rescale_learned_sigmas=True,
        )

        if diffusion.num_timesteps != 100:
            msg = f"Expected 100 timesteps, got {diffusion.num_timesteps}"
            raise ValueError(msg)

    def test_create_diffusion_integration(self) -> None:
        """Test integration of different diffusion configurations."""
        # Test that different configurations work together
        configs = [
            {
                "timestep_respacing": "",
                "noise_schedule": "linear",
                "diffusion_steps": 50,
            },
            {
                "timestep_respacing": "10",
                "noise_schedule": "squaredcos_cap_v2",
                "diffusion_steps": 100,
            },
            {
                "timestep_respacing": "ddim5",
                "noise_schedule": "linear",
                "diffusion_steps": 100,
                "use_kl": True,
            },
        ]

        for config in configs:
            diffusion = create_diffusion(**config)
            # Basic functionality check
            if not hasattr(diffusion, "num_timesteps"):
                msg = "Diffusion should have num_timesteps attribute"
                raise AttributeError(msg)
            if diffusion.num_timesteps <= 0:
                msg = "Number of timesteps should be positive"
                raise ValueError(msg)
