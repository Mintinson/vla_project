"""Modified from OpenAI's diffusion repos.

GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
"""

import enum
from collections.abc import Callable, Generator, Sequence
from typing import Any, Literal, TypedDict, cast

import numpy as np
import torch
from torch import nn

from vla_project.action_model.diffusion_utils import discretized_gaussian_log_likelihood, normal_kl


class ModelOutput(TypedDict):
    mean: torch.Tensor
    variance: torch.Tensor
    log_variance: torch.Tensor | np.ndarray
    pred_xstart: torch.Tensor
    extra: torch.Tensor | None


def mean_flat(x: torch.Tensor) -> torch.Tensor:
    """Take the mean over all non-batch dimensions.

    Args:
        x (torch.Tensor): Input tensor with shape (N, ...).

    Returns:
        torch.Tensor: Mean value with shape (N,).

    """
    return x.mean(dim=list(range(1, len(x.shape))))


class ModelMeanType(enum.Enum):
    """Which type of output the model predicts."""

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """What is used as the model's variance (if any)."""

    LEARNED = enum.auto()  # the model outputs the variance
    FIXED_SMALL = enum.auto()  # the model uses fixed small variance
    FIXED_LARGE = enum.auto()  # the model uses fixed large variance
    LEARNED_RANGE = enum.auto()  # the model outputs a value between [0, 1] to interpolate between fixed large and small


class LossType(enum.Enum):
    """Types of loss functions for diffusion model training."""

    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = enum.auto()  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # Like KL, but rescale to estimate the full VLB

    def is_vb(self) -> bool:
        """Check if this loss type uses variational bound.

        Returns:
            bool: True if loss type is KL or RESCALED_KL.

        """
        return self in (LossType.KL, LossType.RESCALED_KL)


def _warmup_beta(beta_start: float, beta_end: float, num_diffusion_timesteps: int, warmup_frac: float) -> np.ndarray:
    """Create beta schedule with linear warmup.

    Creates a beta schedule that starts with a linear warmup phase followed by
    constant values. Used for diffusion noise schedules.

    Args:
        beta_start (float): Starting beta value for warmup.
        beta_end (float): Ending beta value (constant after warmup).
        num_diffusion_timesteps (int): Total number of diffusion timesteps.
        warmup_frac (float): Fraction of timesteps to use for warmup phase.

    Returns:
        np.ndarray: Beta schedule with warmup.

    """
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(
    beta_schedule: Literal["quad", "linear", "warmup10", "warmup50", "const", "jsd"],
    *,
    beta_start: float,
    beta_end: float,
    num_diffusion_timesteps: int,
) -> np.ndarray:
    """Generate beta schedule for diffusion process.

    Creates various types of beta schedules used in diffusion models to control
    the noise addition process during forward diffusion.

    Args:
        beta_schedule: Type of beta schedule to generate. Options:
            - "quad": Quadratic schedule
            - "linear": Linear schedule
            - "warmup10": Linear warmup for 10% of timesteps
            - "warmup50": Linear warmup for 50% of timesteps
            - "const": Constant beta values
            - "jsd": Schedule based on Jensen-Shannon divergence
        beta_start (float): Starting beta value.
        beta_end (float): Ending beta value.
        num_diffusion_timesteps (int): Total number of diffusion timesteps.

    Returns:
        np.ndarray: Beta schedule array of shape (num_diffusion_timesteps,).

    Raises:
        NotImplementedError: If unknown beta schedule is specified.
        RuntimeError: If resulting beta array has incorrect shape.

    """
    if beta_schedule == "quad":
        betas = np.linspace(beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        msg = f"unknown beta schedule: {beta_schedule}"
        raise NotImplementedError(msg)
    if betas.shape != (num_diffusion_timesteps,):
        msg = f"betas must have shape ({num_diffusion_timesteps},), but got {betas.shape}"
        raise RuntimeError(msg)
    return betas


def betas_for_alpha_bar(
    num_diffusion_timesteps: int,
    alpha_bar: Callable[[float], float],
    max_beta: float = 0.999,
) -> np.ndarray:
    """Create a beta schedule that ensures alpha_bar follows a specific function.

    This function generates betas such that the cumulative product of alphas
    (alpha_bar) follows the provided function over timesteps.

    Args:
        num_diffusion_timesteps (int): Number of diffusion timesteps.
        alpha_bar (Callable[[float], float]): Function defining alpha_bar over time.
        max_beta (float, optional): Maximum allowed beta value. Defaults to 0.999.

    Returns:
        np.ndarray: Beta schedule array.

    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_named_beta_schedule(
    schedule_name: Literal["linear", "squaredcos_cap_v2"],
    num_diffusion_timesteps: int,
) -> np.ndarray:
    """Get a predefined beta schedule by name.

    Provides access to commonly used beta schedules with standard parameter
    settings for diffusion models.

    Args:
        schedule_name: Name of the beta schedule. Options:
            - "linear": Linear schedule scaled for the number of timesteps
            - "squaredcos_cap_v2": Squared cosine schedule with cap
        num_diffusion_timesteps (int): Number of diffusion timesteps.

    Returns:
        np.ndarray: Beta schedule array.

    Raises:
        NotImplementedError: If schedule_name is not recognized.

    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    if schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            alpha_bar=lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
        )

    msg = f"unknown beta schedule: {schedule_name}"
    raise NotImplementedError(msg)


class GaussianDiffusion:
    """Utilities for training and sampling diffusion models.

    This class implements the forward and reverse diffusion processes for training
    and sampling from diffusion models. It handles noise scheduling, loss computation,
    and various sampling strategies including DDIM.

    Original ported from this codebase:
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    Attributes:
        model_mean_type (ModelMeanType): Type of model mean prediction.
        model_var_type (ModelVarType): Type of model variance prediction.
        loss_type (LossType): Type of loss function to use.
        betas (np.ndarray): Beta schedule for diffusion process.
        num_timesteps (int): Number of diffusion timesteps.
        alphas_cumprod (np.ndarray): Cumulative product of alphas.
        sqrt_alphas_cumprod (np.ndarray): Square root of cumulative alphas.
        sqrt_one_minus_alphas_cumprod (np.ndarray): Square root of (1 - cumulative alphas).
        posterior_variance (np.ndarray): Posterior variance for reverse process.

    """

    def __init__(
        self,
        *,
        betas: np.ndarray,
        model_mean_type: ModelMeanType,
        model_var_type: ModelVarType,
        loss_type: LossType,
    ) -> None:
        """Initialize the Gaussian diffusion process.

        Args:
            betas (np.ndarray): 1-D array of betas for each diffusion timestep,
                starting at T and going to 1.
            model_mean_type (ModelMeanType): How the model predicts the mean.
            model_var_type (ModelVarType): How the model predicts variance.
            loss_type (LossType): Type of loss function to use for training.

        Raises:
            ValueError: If betas array is not 1-dimensional or has invalid values.

        """
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas

        if len(betas.shape) != 1 or not (betas > 0).all() or not (betas <= 1).all():
            msg = "betas must be a 1-D array with values in (0, 1]"
            raise ValueError(msg)

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)  # noqa: S101

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = (
            np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
            if len(self.posterior_variance) > 1
            else np.array([])
        )

        self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def q_mean_variance(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the distribution q(x_t | x_0).

        Computes the mean, variance, and log variance of the forward process
        distribution q(x_t | x_0) that adds noise to clean data.

        Args:
            x_start (torch.Tensor): Clean data tensor x_0.
            t (torch.Tensor): Timestep values.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Mean, variance, and
                log variance of q(x_t | x_0).

        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Sample from q(x_t | x_0) - the forward diffusion process.

        Applies noise to clean data according to the diffusion schedule at timestep t.

        Args:
            x_start (torch.Tensor): Clean data tensor x_0.
            t (torch.Tensor): Timestep values.
            noise (torch.Tensor | None, optional): Pre-generated noise tensor.
                If None, generates random noise. Defaults to None.

        Returns:
            torch.Tensor: Noisy data tensor x_t.

        Raises:
            ValueError: If noise shape doesn't match x_start shape.

        """
        if noise is None:
            noise = torch.randn_like(x_start)
        if noise.shape != x_start.shape:
            msg = f"Noise shape {noise.shape} must match x_start shape {x_start.shape}"
            raise ValueError(msg)
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Get the distribution q(x_{t-1} | x_t, x_0).

        Computes the mean, variance, and log variance of the posterior
        distribution q(x_{t-1} | x_t, x_0) in the reverse diffusion process.

        Args:
            x_start (torch.Tensor): Clean data tensor x_0.
            x_t (torch.Tensor): Noisy data tensor x_t.
            t (torch.Tensor): Timestep values.

        Returns:
            tuple[torch.Tensor, torch.Tensor, np.ndarray]: Posterior mean,
                variance, and log variance clipped.

        Raises:
            ValueError: If x_start and x_t shapes don't match or batch dimensions differ.

        """
        # q(x_{t-1} | x_t, x_0)
        if x_start.shape != x_t.shape:
            msg = f"x_start shape {x_start.shape} must match x_t shape {x_t.shape}"
            raise ValueError(msg)
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.posterior_log_variance_clipped
        # posterior_log_variance_clipped = _extract_into_tensor(
        #     self.posterior_log_variance_clipped, t, x_t.shape
        # )
        if posterior_mean.shape[0] != posterior_variance.shape[0] or posterior_mean.shape[0] != x_start.shape[0]:
            msg = (
                f"Shapes of posterior_mean {posterior_mean.shape}, "
                f"posterior_variance {posterior_variance.shape}, "
                f"x_start {x_start.shape} do not match in batch dimension"
            )
            raise ValueError(msg)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        model: nn.Module | Callable[..., torch.Tensor],
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        clip_denoised: bool = True,
        denoised_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        model_kwargs: dict | None = None,
        # ) -> dict[str, torch.Tensor | np.ndarray | None]:
    ) -> ModelOutput:
        """Apply the model to get p(x_{t-1} | x_t).

        Applies the model to predict the mean and variance of the reverse
        diffusion step p(x_{t-1} | x_t).

        Args:
            model: The model to apply.
            x: The [N x C x ...] tensor at time t.
            t: A 1-D tensor of timesteps.
            clip_denoised: If True, clip the denoised signal to [-1, 1].
            denoised_fn: If not None, a function which applies to the
                x_start prediction before it is used to sample. Applies before
                clip_denoised.
            model_kwargs: If not None, a dict of extra keyword arguments to
                pass to the model. This can be used for conditioning.

        Returns:
            ModelOutput: A dict with the following keys:
                - 'mean': the model mean output.
                - 'variance': the model variance output.
                - 'log_variance': the log of 'variance'.
                - 'pred_xstart': the prediction for x_0.

        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)
        else:  # noqa: PLR5501
            if len(self.betas) == 1:
                model_variance, model_log_variance = {
                    ModelVarType.FIXED_SMALL: (
                        self.posterior_variance,
                        self.posterior_log_variance_clipped,
                    ),
                }[self.model_var_type]
                model_variance = _extract_into_tensor(model_variance, t, x.shape)
            else:
                model_variance, model_log_variance = {
                    # for fixedlarge, we set the initial (log-)variance like so
                    # to get a better decoder log likelihood.
                    ModelVarType.FIXED_LARGE: (
                        np.append(self.posterior_variance[1], self.betas[1:]),
                        np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                    ),
                    ModelVarType.FIXED_SMALL: (
                        self.posterior_variance,
                        self.posterior_log_variance_clipped,
                    ),
                }[self.model_var_type]
                model_variance = _extract_into_tensor(model_variance, t, x.shape)
                model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x: torch.Tensor) -> torch.Tensor:
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert model_mean.shape == pred_xstart.shape == x.shape  # == model_log_variance.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
        }

    def _predict_xstart_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from epsilon prediction.

        Given x_t and predicted noise epsilon, recovers the original x_0
        using the diffusion formula.

        Args:
            x_t (torch.Tensor): Noisy tensor at timestep t.
            t (torch.Tensor): Timestep values.
            eps (torch.Tensor): Predicted noise.

        Returns:
            torch.Tensor: Predicted x_0.

        """
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t: torch.Tensor, t: torch.Tensor, pred_xstart: torch.Tensor) -> torch.Tensor:
        """Predict epsilon from x_0 prediction.

        Given x_t and predicted x_0, recovers the noise epsilon that was
        added during the forward diffusion process.

        Args:
            x_t (torch.Tensor): Noisy tensor at timestep t.
            t (torch.Tensor): Timestep values.
            pred_xstart (torch.Tensor): Predicted x_0.

        Returns:
            torch.Tensor: Predicted noise epsilon.

        """
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def condition_mean(
        self,
        cond_fn: Callable[[torch.Tensor, torch.Tensor, dict[str, Any]], torch.Tensor] | "nn.Module",
        p_mean_var: ModelOutput,
        x: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Apply conditioning function to adjust the predicted mean.

        Modifies the predicted mean using gradients from a conditioning function,
        typically used for classifier guidance during sampling.

        Args:
            cond_fn: Conditioning function that returns gradients.
            p_mean_var: Dictionary containing mean and variance predictions.
            x: Current noisy tensor.
            t: Timestep values.
            model_kwargs: Additional model keyword arguments.

        Returns:
            torch.Tensor: Adjusted mean prediction.

        """
        if model_kwargs is None:
            model_kwargs = {}
        gradient: torch.Tensor = cond_fn(x, t, model_kwargs)
        return (
            cast("torch.Tensor", p_mean_var["mean"]).float()
            + cast("torch.Tensor", p_mean_var["variance"]) * gradient.float()
        )

    def condition_score(
        self,
        cond_fn: "Callable[[torch.Tensor, torch.Tensor, dict[str, Any]], torch.Tensor] | nn.Module",
        p_mean_var: ModelOutput,
        x: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: dict[str, Any] | None = None,
    ) -> ModelOutput:
        """Compute what the p_mean_variance output would have been, should the model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).

        Args:
            cond_fn: Conditioning function that returns gradients.
            p_mean_var: Dictionary containing mean and variance predictions.
            x: Current noisy tensor.
            t: Timestep values.
            model_kwargs: Additional model keyword arguments.

        Returns:
            ModelOutput: Updated predictions with conditioning applied.

        """
        if model_kwargs is None:
            model_kwargs = {}
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, cast("torch.Tensor", p_mean_var["pred_xstart"]))
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=cast("torch.Tensor", out["pred_xstart"]), x_t=x, t=t)
        return out

    def p_sample(
        self,
        model: nn.Module | Callable[..., torch.Tensor],
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        clip_denoised: bool = True,
        denoised_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        cond_fn: Callable[[torch.Tensor, torch.Tensor, dict[str, Any]], torch.Tensor] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample x_{t-1} from the model at the given timestep.

        Args:
            model: The model to sample from.
            x: The current tensor at x_{t-1}.
            t: The value of t, starting at 0 for the first diffusion step.
            clip_denoised: If True, clip the x_start prediction to [-1, 1].
            denoised_fn: If not None, a function which applies to the
                x_start prediction before it is used to sample.
            cond_fn: If not None, this is a gradient function that acts
                similarly to the model.
            model_kwargs: If not None, a dict of extra keyword arguments to
                pass to the model. This can be used for conditioning.

        Returns:
            dict[str, torch.Tensor]: A dict containing the following keys:
                - 'sample': a random sample from the model.
                - 'pred_xstart': a prediction of x_0.

        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = (
            cast("torch.Tensor", out["mean"])
            + nonzero_mask * torch.exp(0.5 * cast("torch.Tensor", out["log_variance"])) * noise
        )
        return {"sample": sample, "pred_xstart": cast("torch.Tensor", out["pred_xstart"])}

    def p_sample_loop(
        self,
        model: nn.Module | Callable[..., torch.Tensor],
        shape: Sequence[int],
        noise: torch.Tensor | None = None,
        *,
        clip_denoised: bool = True,
        denoised_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        cond_fn: Callable[[torch.Tensor, torch.Tensor, dict], torch.Tensor] | None = None,
        model_kwargs: dict | None = None,
        device: torch.device | None = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """Generate a final sample from the diffusion model.

        This function is a convenience wrapper around `p_sample_loop_progressive`.
        It performs the complete reverse diffusion process and returns only the
        final generated sample.

        Args:
            model: The diffusion model to use for denoising.
            shape: The shape of the sample to generate, e.g.,
                (batch_size, channels, ...).
            noise: Optional initial noise tensor. If None, noise is sampled from a
                standard normal distribution.
            clip_denoised: If True, clip the predicted `x_start` to `[-1, 1]`.
            denoised_fn: An optional function to apply to the denoised sample
                at each step.
            cond_fn: An optional function to guide the sampling process (e.g., for
                classifier-free guidance). It should take `x_t`, `t`, and
                `model_kwargs` and return a gradient.
            model_kwargs: A dictionary of additional arguments to pass to the model,
                typically for conditioning.
            device: The torch device to perform the computation on. If None, the
                model's device is used.
            progress: If True, a progress bar will be displayed.

        Returns:
            The final generated sample as a torch.Tensor.

        """
        final = {}
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model: nn.Module | Callable[..., torch.Tensor],
        shape: Sequence[int],
        noise: torch.Tensor | None = None,
        *,
        clip_denoised: bool = True,
        denoised_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        cond_fn: Callable[[torch.Tensor, torch.Tensor, dict], torch.Tensor] | None = None,
        model_kwargs: dict | None = None,
        device: torch.device | None = None,
        progress: bool = False,
    ) -> Generator[dict[str, torch.Tensor]]:
        """Generate samples from the model and yield intermediate samples from each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            if not isinstance(model, "nn.Module"):
                msg = "If model is not an nn.Module, must specify device"
                raise ValueError(msg)
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        img = noise if noise is not None else torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm  # noqa: PLC0415

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model: nn.Module | Callable[..., torch.Tensor],
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        clip_denoised: bool = True,
        denoised_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        cond_fn: Callable[[torch.Tensor, torch.Tensor, dict[str, Any]], torch.Tensor] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        eta: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Sample x_{t-1} from the model using DDIM.

        Deterministic Denoising Implicit Models (DDIM) sampling that allows
        for faster sampling with fewer timesteps while maintaining quality.

        Args:
            model: The model to sample from.
            x: The current tensor at x_{t-1}.
            t: The value of t, starting at 0 for the first diffusion step.
            clip_denoised: If True, clip the x_start prediction to [-1, 1].
            denoised_fn: If not None, a function which applies to the
                x_start prediction before it is used to sample.
            cond_fn: If not None, this is a gradient function that acts
                similarly to the model.
            model_kwargs: If not None, a dict of extra keyword arguments to
                pass to the model. This can be used for conditioning.
            eta: Controls the amount of noise added during sampling.
                eta=0 gives deterministic sampling, eta=1 gives DDPM sampling.

        Returns:
            dict[str, torch.Tensor]: A dict containing the following keys:
                - 'sample': a random sample from the model.
                - 'pred_xstart': a prediction of x_0.

        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, cast("torch.Tensor", out["pred_xstart"]))

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = out["pred_xstart"] * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        clip_denoised: bool = True,
        denoised_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        cond_fn: Callable[[torch.Tensor, torch.Tensor, dict[str, Any]], torch.Tensor] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        eta: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Sample x_{t+1} from the model using DDIM reverse ODE.

        Performs reverse DDIM sampling to go from x_t to x_{t+1}, which is
        useful for inverting the diffusion process deterministically.

        Args:
            model: The model to use for reverse sampling.
            x: The current tensor at x_t.
            t: The timestep value.
            clip_denoised: If True, clip the x_start prediction to [-1, 1].
            denoised_fn: If not None, a function which applies to the
                x_start prediction before it is used to sample.
            cond_fn: If not None, this is a gradient function that acts
                similarly to the model.
            model_kwargs: If not None, a dict of extra keyword arguments to
                pass to the model. This can be used for conditioning.
            eta: Must be 0.0 for deterministic reverse ODE.

        Returns:
            dict[str, torch.Tensor]: A dict containing the following keys:
                - 'sample': the reverse sample x_{t+1}.
                - 'pred_xstart': a prediction of x_0.

        Raises:
            AssertionError: If eta is not 0.0 (only deterministic path allowed).

        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = out["pred_xstart"] * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * eps

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model: nn.Module | Callable[..., torch.Tensor],
        shape: Sequence[int],
        noise: torch.Tensor | None = None,
        *,
        clip_denoised: bool = True,
        denoised_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        cond_fn: Callable[[torch.Tensor, torch.Tensor, dict[str, Any]], torch.Tensor] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        device: torch.device | None = None,
        progress: bool = False,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Generate samples from the model using DDIM.

        Same usage as p_sample_loop().

        Args:
            model: The model to sample from.
            shape: The shape of the samples, (N, C, H, W).
            noise: If specified, the noise from the encoder to sample.
                Should be of the same shape as `shape`.
            clip_denoised: If True, clip x_start predictions to [-1, 1].
            denoised_fn: If not None, a function which applies to the
                x_start prediction before it is used to sample.
            cond_fn: If not None, this is a gradient function that acts
                similarly to the model.
            model_kwargs: If not None, a dict of extra keyword arguments to
                pass to the model. This can be used for conditioning.
            device: If specified, the device to create the samples on.
                If not specified, use a model parameter's device.
            progress: If True, show a tqdm progress bar.
            eta: Controls the amount of noise added during sampling.

        Returns:
            torch.Tensor: A non-differentiable batch of samples.

        """
        final = {}
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model: nn.Module | Callable[..., torch.Tensor],
        shape: Sequence[int],
        noise: torch.Tensor | None = None,
        *,
        clip_denoised: bool = True,
        denoised_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        cond_fn: Callable[[torch.Tensor, torch.Tensor, dict[str, Any]], torch.Tensor] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        device: torch.device | None = None,
        progress: bool = False,
        eta: float = 0.0,
    ) -> Generator[dict[str, torch.Tensor]]:
        """Use DDIM to sample from the model and yield intermediate samples from each timestep of DDIM.

        Same usage as p_sample_loop_progressive().

        Args:
            model: The model to sample from.
            shape: The shape of the samples, (N, C, H, W).
            noise: If specified, the noise from the encoder to sample.
                Should be of the same shape as `shape`.
            clip_denoised: If True, clip x_start predictions to [-1, 1].
            denoised_fn: If not None, a function which applies to the
                x_start prediction before it is used to sample.
            cond_fn: If not None, this is a gradient function that acts
                similarly to the model.
            model_kwargs: If not None, a dict of extra keyword arguments to
                pass to the model. This can be used for conditioning.
            device: If specified, the device to create the samples on.
                If not specified, use a model parameter's device.
            progress: If True, show a tqdm progress bar.
            eta: Controls the amount of noise added during sampling.

        Yields:
            dict[str, torch.Tensor]: A dict containing sample and pred_xstart.

        """
        if device is None:
            if not isinstance(model, "nn.Module"):
                msg = "If model is not an nn.Module, must specify device"
                raise ValueError(msg)
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        img = noise if noise is not None else torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm  # noqa: PLC0415

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self,
        model: nn.Module | Callable,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        clip_denoised: bool = True,
        model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        Args:
            model: The model to evaluate.
            x_start: The clean input tensor.
            x_t: The noisy input tensor.
            t: Timestep values.
            clip_denoised: Whether to clip denoised predictions.
            model_kwargs: Additional model arguments.

        Returns:
            dict[str, torch.Tensor]: A dict with the following keys:
                - 'output': a shape [N] tensor of NLLs or KLs.
                - 'pred_xstart': the x_0 predictions.

        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        out = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)  # pyright: ignore[reportArgumentType]
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start,
            means=out["mean"],
            log_scales=0.5 * torch.tensor(out["log_variance"]),
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: dict[str, Any] | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute training losses for a single timestep.

        Args:
            model: The model to evaluate loss on.
            x_start: The [N x C x ...] tensor of inputs.
            t: A batch of timestep indices.
            model_kwargs: If not None, a dict of extra keyword arguments to
                pass to the model. This can be used for conditioning.
            noise: If specified, the specific Gaussian noise to try to remove.

        Returns:
            dict[str, torch.Tensor]: A dict with the key "loss" containing a tensor of shape [N].
                Some mean or variance settings may also have other keys.

        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type in (LossType.KL, LossType.RESCALED_KL):
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type in (LossType.MSE, LossType.RESCALED_MSE):
            model_output = model(x_t, t, **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start: torch.Tensor) -> torch.Tensor:
        """Get the prior KL term for the variational lower-bound, measured in bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        Args:
            x_start: The [N x C x ...] tensor of inputs.

        Returns:
            torch.Tensor: A batch of [N] KL values (in bits), one per batch element.

        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        *,
        clip_denoised: bool = True,
        model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the entire variational lower-bound, measured in bits-per-dim.

        Computes the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        Args:
            model: The model to evaluate loss on.
            x_start: The [N x C x ...] tensor of inputs.
            clip_denoised: If True, clip denoised samples.
            model_kwargs: If not None, a dict of extra keyword arguments to
                pass to the model. This can be used for conditioning.

        Returns:
            dict[str, torch.Tensor]: A dict containing the following keys:
                - total_bpd: the total variational lower-bound, per batch element.
                - prior_bpd: the prior term in the lower-bound.
                - vb: an [N x T] tensor of terms in the lower-bound.
                - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                - mse: an [N x T] tensor of epsilon MSEs for each timestep.

        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = torch.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with torch.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = torch.stack(vb, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr: np.ndarray, timesteps: torch.Tensor, broadcast_shape: Sequence[int]) -> torch.Tensor:
    """Extract values from a 1-D numpy array for a batch of indices.

    This function extracts values from a numpy array based on timestep indices and
    broadcasts them to match the desired tensor shape for batch operations.

    Args:
        arr (np.ndarray): 1-D numpy array to extract values from.
        timesteps (torch.Tensor): Tensor of timestep indices.
        broadcast_shape (Sequence[int]): Target shape for broadcasting.

    Returns:
        torch.Tensor: Extracted values broadcasted to the target shape.

    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)
