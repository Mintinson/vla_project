"""Tests for diffusion utilities and helper functions.

This module tests utility functions used by the diffusion process, including
KL divergence computation, CDF approximations, and other mathematical utilities.
"""

import numpy as np
import pytest
import torch

from vla_project.action_model.diffusion_utils import (
    approx_standard_normal_cdf,
    discretized_gaussian_log_likelihood,
    normal_kl,
)
from vla_project.action_model.gaussian_diffusion import mean_flat


class TestDiffusionUtils:
    """Tests for diffusion utility functions."""

    def test_normal_kl_basic(self):
        """Test normal_kl with basic inputs."""
        mean1 = torch.tensor([0.0, 1.0])
        logvar1 = torch.tensor([0.0, 0.0])  # var = 1
        mean2 = torch.tensor([0.0, 0.0])
        logvar2 = torch.tensor([0.0, 0.0])  # var = 1

        kl = normal_kl(mean1, logvar1, mean2, logvar2)

        if not isinstance(kl, torch.Tensor):
            msg = "Expected torch.Tensor output"
            raise TypeError(msg)
        if kl.shape != mean1.shape:
            msg = "Output shape doesn't match input shape"
            raise ValueError(msg)
        if not torch.isfinite(kl).all():
            msg = "Output contains non-finite values"
            raise ValueError(msg)

    def test_normal_kl_shapes(self):
        """Test normal_kl with different tensor shapes."""
        batch_size = 4
        feature_dim = 10

        mean1 = torch.randn(batch_size, feature_dim)
        logvar1 = torch.randn(batch_size, feature_dim)
        mean2 = torch.randn(batch_size, feature_dim)
        logvar2 = torch.randn(batch_size, feature_dim)

        kl = normal_kl(mean1, logvar1, mean2, logvar2)

        expected_shape = (batch_size, feature_dim)
        if kl.shape != expected_shape:
            msg = f"Expected shape {expected_shape}, got {kl.shape}"
            raise ValueError(msg)
        if not torch.isfinite(kl).all():
            msg = "Output contains non-finite values"
            raise ValueError(msg)

    def test_normal_kl_broadcasting(self):
        """Test normal_kl with broadcasting."""
        mean1 = torch.randn(4, 10)
        logvar1 = torch.randn(4, 10)
        mean2 = torch.randn(1, 10)  # Broadcast-compatible
        logvar2 = torch.randn(1, 10)  # Broadcast-compatible

        kl = normal_kl(mean1, logvar1, mean2, logvar2)

        expected_shape = (4, 10)
        if kl.shape != expected_shape:
            msg = f"Expected shape {expected_shape}, got {kl.shape}"
            raise ValueError(msg)
        if not torch.isfinite(kl).all():
            msg = "Output contains non-finite values"
            raise ValueError(msg)

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
            msg = "Expected torch.Tensor output"
            raise TypeError(msg)
        if log_likelihood.shape != x.shape:
            msg = "Output shape doesn't match input shape"
            raise ValueError(msg)
        if not torch.isfinite(log_likelihood).all():
            msg = "Output contains non-finite values"
            raise ValueError(msg)

    def test_approx_standard_normal_cdf(self):
        """Test approximate standard normal CDF."""
        x = torch.randn(3, 4, 5)

        cdf = approx_standard_normal_cdf(x)

        if not isinstance(cdf, torch.Tensor):
            msg = "Expected torch.Tensor output"
            raise TypeError(msg)
        if cdf.shape != x.shape:
            msg = "Output shape doesn't match input shape"
            raise ValueError(msg)
        if not torch.all(cdf >= 0.0):
            msg = "CDF values should be non-negative"
            raise ValueError(msg)
        if not torch.all(cdf <= 1.0):
            msg = "CDF values should be at most 1.0"
            raise ValueError(msg)
        if not torch.isfinite(cdf).all():
            msg = "Output contains non-finite values"
            raise ValueError(msg)

    def test_mean_flat(self):
        """Test mean_flat utility function."""
        x = torch.randn(4, 10, 8, 6)

        result = mean_flat(x)

        expected_shape = (4,)  # Only batch dimension preserved
        if result.shape != expected_shape:
            msg = f"Expected shape {expected_shape}, got {result.shape}"
            raise ValueError(msg)
        if not torch.isfinite(result).all():
            msg = "Output contains non-finite values"
            raise ValueError(msg)

    def test_mean_flat_2d(self):
        """Test mean_flat with 2D input."""
        x = torch.randn(5, 20)

        result = mean_flat(x)

        expected_shape = (5,)
        if result.shape != expected_shape:
            msg = f"Expected shape {expected_shape}, got {result.shape}"
            raise ValueError(msg)
        if not torch.isfinite(result).all():
            msg = "Output contains non-finite values"
            raise ValueError(msg)

    def test_discretized_gaussian_log_likelihood_edge_cases(self):
        """Test discretized Gaussian log likelihood with edge cases."""
        # Test with extreme values
        x = torch.tensor([[-0.999], [0.999]])
        means = torch.zeros_like(x)
        log_scales = torch.ones_like(x)

        log_likelihood = discretized_gaussian_log_likelihood(x, means=means, log_scales=log_scales)

        if not torch.isfinite(log_likelihood).all():
            msg = "Output contains non-finite values for edge cases"
            raise ValueError(msg)

    def test_approx_standard_normal_cdf_extreme_values(self):
        """Test approximate standard normal CDF with extreme values."""
        x = torch.tensor([-10.0, 0.0, 10.0])

        cdf = approx_standard_normal_cdf(x)

        # Should be close to 0, 0.5, 1 respectively
        expected_values = torch.tensor([0.0, 0.5, 1.0])
        tolerance = 0.1
        if not torch.allclose(cdf, expected_values, atol=tolerance):
            msg = f"CDF values {cdf} not close to expected {expected_values}"
            raise ValueError(msg)

    def test_normal_kl_identical_distributions(self):
        """Test normal_kl with identical distributions (should be zero)."""
        mean = torch.randn(3, 4)
        logvar = torch.randn(3, 4)

        kl = normal_kl(mean, logvar, mean, logvar)

        if not torch.allclose(kl, torch.zeros_like(kl), atol=1e-6):
            msg = "KL divergence between identical distributions should be zero"
            raise ValueError(msg)

    def test_normal_kl_value_error_case(self):
        """Test normal_kl error case with no tensors."""
        mean1 = 0.0
        logvar1 = 0.0
        mean2 = 0.0
        logvar2 = 0.0

        with pytest.raises(ValueError, match="at least one argument must be a Tensor"):
            normal_kl(mean1, logvar1, mean2, logvar2)

    def test_discretized_gaussian_log_likelihood_shape_mismatch(self):
        """Test discretized Gaussian log likelihood with shape mismatch."""
        x = torch.randn(2, 5)
        means = torch.randn(2, 6)  # Different shape
        log_scales = torch.randn(2, 5)

        with pytest.raises(ValueError, match="Input tensors must have the same shape"):
            discretized_gaussian_log_likelihood(x, means=means, log_scales=log_scales)

    def test_mean_flat_single_dimension(self):
        """Test mean_flat with single dimension input."""
        x = torch.randn(10)

        result = mean_flat(x)

        # For single batch dimension, should return scalar
        if result.shape != torch.Size([]):
            msg = f"Expected scalar output, got shape {result.shape}"
            raise ValueError(msg)

    def test_normal_kl_gradient_flow(self):
        """Test that normal_kl preserves gradients."""
        mean1 = torch.randn(2, 3, requires_grad=True)
        logvar1 = torch.randn(2, 3, requires_grad=True)
        mean2 = torch.randn(2, 3)
        logvar2 = torch.randn(2, 3)

        kl = normal_kl(mean1, logvar1, mean2, logvar2)
        loss = kl.sum()
        loss.backward()

        if mean1.grad is None or logvar1.grad is None:
            msg = "Gradients should be computed for input tensors"
            raise ValueError(msg)
        if not torch.isfinite(mean1.grad).all() or not torch.isfinite(logvar1.grad).all():
            msg = "Gradients contain non-finite values"
            raise ValueError(msg)
