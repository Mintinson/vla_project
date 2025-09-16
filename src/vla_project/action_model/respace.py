"""Utilities for creating respaced diffusion processes.

This module provides functionality to create diffusion processes that skip steps from
the original process, enabling faster sampling while maintaining quality. It implements
the timestep respacing technique commonly used in diffusion models.

Modified from OpenAI's diffusion repositories:
    GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
    ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
    IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn

from .gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelOutput, ModelVarType


def space_timesteps(num_timesteps: int, section_counts: str | list[int]) -> set[int]:
    """Create a list of timesteps to use from an original diffusion process.

    Given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    Args:
        num_timesteps (int): The number of diffusion steps in the original
            process to divide up.
        section_counts (str | list[int]): Either a list of numbers, or a string containing
            comma-separated numbers, indicating the step count per section. As a special
            case, use "ddimN" where N is a number of steps to use the striding from the
            DDIM paper.

    Returns:
        set[int]: A set of diffusion steps from the original process to use.

    Raises:
        ValueError: If cannot create exactly the desired number of steps with an integer stride,
            or if cannot divide section of steps into the specified section count.

    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            if desired_count == 1:
                return {50}
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            msg = f"cannot create exactly {num_timesteps} steps with an integer stride"
            raise ValueError(msg)
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            msg = f"cannot divide section of {size} steps into {section_count}"
            raise ValueError(msg)
        frac_stride = 1 if section_count <= 1 else (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """A diffusion process which can skip steps in a base diffusion process.

    This class creates a new diffusion process that uses only a subset of timesteps
    from the original process, enabling faster sampling while maintaining quality.
    The selected timesteps are mapped to new consecutive indices.

    Attributes:
        use_timesteps (set[int]): Set of timesteps from the original process to retain.
        timestep_map (list[int]): Mapping from new timestep indices to original indices.
        original_num_steps (int): Number of timesteps in the original diffusion process.

    """

    def __init__(
        self,
        use_timesteps: set[int] | list[int],
        *,
        betas: np.ndarray,
        model_mean_type: ModelMeanType,
        model_var_type: ModelVarType,
        loss_type: LossType,
    ) -> None:
        """Initialize the spaced diffusion process.

        Args:
            use_timesteps: A collection (sequence or set) of timesteps from the
                original diffusion process to retain.
            **kwargs: Additional keyword arguments to create the base diffusion process.
                Must include 'betas' parameter.

        """
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(betas)

        base_diffusion = GaussianDiffusion(
            betas=betas, model_mean_type=model_mean_type, model_var_type=model_var_type, loss_type=loss_type
        )
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        betas = np.array(new_betas)
        super().__init__(
            betas=betas, model_mean_type=model_mean_type, model_var_type=model_var_type, loss_type=loss_type
        )

    def p_mean_variance(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        model: nn.Module,
        *args,
        **kwargs,
    ) -> ModelOutput:
        """Compute the mean and variance of the posterior distribution.

        This method wraps the model to handle timestep mapping and delegates
        to the parent class implementation.

        Args:
            model: The neural network model to use for prediction.
            *args: Variable length argument list passed to parent method.
            **kwargs: Arbitrary keyword arguments passed to parent method.

        Returns:
            dict[str, torch.Tensor | np.ndarray | None]: Dictionary containing mean,
                variance, and other prediction outputs.

        """
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self,
        model: nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute training losses for the spaced diffusion process.

        This method wraps the model to handle timestep mapping and delegates
        to the parent class implementation.

        Args:
            model: The neural network model to compute losses for.
            *args: Variable length argument list passed to parent method.
            **kwargs: Arbitrary keyword arguments passed to parent method.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing computed losses.

        """
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(
        self,
        cond_fn: Callable[[torch.Tensor, torch.Tensor, dict[str, Any]], torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply conditioning to the predicted mean.

        This method wraps the conditioning function to handle timestep mapping
        and delegates to the parent class implementation.

        Args:
            cond_fn: Conditioning function that takes (x, t, kwargs) and returns gradients.
            *args: Variable length argument list passed to parent method.
            **kwargs: Arbitrary keyword arguments passed to parent method.

        Returns:
            torch.Tensor: Conditioned mean prediction.

        """
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        cond_fn: Callable[[torch.Tensor, torch.Tensor, dict[str, Any]], torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> ModelOutput:
        """Apply score-based conditioning to predictions.

        This method wraps the conditioning function to handle timestep mapping
        and delegates to the parent class implementation.

        Args:
            cond_fn: Conditioning function that takes (x, t, kwargs) and returns gradients.
            *args: Variable length argument list passed to parent method.
            **kwargs: Arbitrary keyword arguments passed to parent method.

        Returns:
            dict[str, torch.Tensor | np.ndarray | None]: Dictionary containing conditioned predictions.

        """
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model: nn.Module | Callable) -> "_WrappedModel":
        """Wrap a model or function to handle timestep mapping.

        Args:
            model: The model or function to wrap.

        Returns:
            _WrappedModel: Wrapped model that handles timestep remapping.

        """
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model, self.timestep_map, self.original_num_steps)

    def _scale_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        """Scale timesteps for the spaced diffusion process.

        In spaced diffusion, scaling is handled by the wrapped model, so this
        method returns the timesteps unchanged.

        Args:
            t: Timestep tensor.

        Returns:
            torch.Tensor: Unchanged timestep tensor.

        """
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel(nn.Module):
    """Wrapper for models in spaced diffusion to handle timestep remapping.

    This internal helper class wraps neural network models or callable functions
    to automatically remap timestep indices from the spaced diffusion process
    back to the original timestep indices that the model expects.

    Attributes:
        model: The wrapped model or callable function.
        timestep_map (list[int]): Mapping from spaced timesteps to original timesteps.
        original_num_steps (int): Number of steps in the original diffusion process.

    """

    def __init__(
        self,
        model: nn.Module | Callable,
        timestep_map: list[int],
        original_num_steps: int,
    ) -> None:
        """Initialize the wrapped model.

        Args:
            model: The model or callable to wrap.
            timestep_map: List mapping spaced timestep indices to original indices.
            original_num_steps: Total number of steps in original diffusion process.

        """
        super().__init__()
        self.model = model
        self.timestep_map = timestep_map
        # self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x: torch.Tensor, ts: torch.Tensor, **kwargs: Any) -> Any:
        """Call the wrapped model with remapped timesteps.

        Args:
            x: Input tensor.
            ts: Timestep tensor with spaced indices.
            **kwargs: Additional keyword arguments passed to the model.

        Returns:
            Any: Output from the wrapped model.

        """
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        # if self.rescale_timesteps:
        #     new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
