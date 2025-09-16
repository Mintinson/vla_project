"""Utility functions for diffusion models.

Modified from OpenAI's diffusion repositories:
    GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
    ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
    IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
"""

import numpy as np
import torch


def normal_kl(
    mean1: torch.Tensor | np.ndarray | float,
    logvar1: torch.Tensor | np.ndarray | float,
    mean2: torch.Tensor | np.ndarray | float,
    logvar2: torch.Tensor | np.ndarray | float,
) -> torch.Tensor:
    """Compute the KL divergence between two Gaussian distributions.

    Calculates KL(N(mean1, exp(logvar1)) || N(mean2, exp(logvar2))).
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.

    Args:
        mean1: Mean of the first Gaussian distribution.
        logvar1: Log variance of the first Gaussian distribution.
        mean2: Mean of the second Gaussian distribution.
        logvar2: Log variance of the second Gaussian distribution.

    Returns:
        torch.Tensor: KL divergence values with shape determined by broadcasting.

    Raises:
        ValueError: If none of the arguments are torch.Tensor objects.

    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    if tensor is None:
        msg = "at least one argument must be a Tensor"
        raise ValueError(msg)

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor) for x in (logvar1, logvar2)]

    return 0.5 * (
        -1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    ) # pyright: ignore[reportReturnType]


def approx_standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Fast approximation of the cumulative distribution function of standard normal.

    Uses a tanh-based approximation to compute the CDF of the standard normal
    distribution N(0,1). This is faster than torch.distributions implementations
    but less accurate for extreme values.

    Args:
        x (torch.Tensor): Input values at which to evaluate the CDF.

    Returns:
        torch.Tensor: Approximate CDF values, same shape as input.

    Note:
        The approximation formula is: 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def continuous_gaussian_log_likelihood(
    x: torch.Tensor,
    *,
    means: torch.Tensor,
    log_scales: torch.Tensor,
) -> torch.Tensor:
    """Compute the log-likelihood of a continuous Gaussian distribution.

    Calculates the log probability density of target values under a Gaussian
    distribution with specified means and log standard deviations.

    Args:
        x (torch.Tensor): Target values for which to compute log-likelihood.
        means (torch.Tensor): Gaussian mean parameters.
        log_scales (torch.Tensor): Gaussian log standard deviation parameters.

    Returns:
        torch.Tensor: Log probabilities (in nats) with same shape as x.

    Note:
        This function computes log p(x) where x ~ N(means, exp(log_scales)²).

    """
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    normalized_x = centered_x * inv_stdv
    return torch.distributions.Normal(torch.zeros_like(x), torch.ones_like(x)).log_prob(normalized_x)


def discretized_gaussian_log_likelihood(
    x: torch.Tensor, *, means: torch.Tensor, log_scales: torch.Tensor,
) -> torch.Tensor:
    """Compute log-likelihood of a Gaussian distribution discretized to given values.

    This function is designed for image data that was originally uint8 values
    rescaled to [-1, 1]. It computes the probability mass for discrete buckets
    around each target value by integrating the Gaussian density.

    Args:
        x (torch.Tensor): Target values, assumed to be uint8 values rescaled to [-1, 1].
        means (torch.Tensor): Gaussian mean parameters, same shape as x.
        log_scales (torch.Tensor): Gaussian log standard deviation parameters, same shape as x.

    Returns:
        torch.Tensor: Log probabilities (in nats) with same shape as x.

    Raises:
        ValueError: If input tensors don't have the same shape.

    Note:
        For each target value x, this computes the integral of the Gaussian density
        over the interval [x - 1/255, x + 1/255], which corresponds to the
        discretization bucket for that pixel value.

    """
    if x.shape != means.shape or x.shape != log_scales.shape:
        msg = "Input tensors must have the same shape"
        raise ValueError(msg)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs
