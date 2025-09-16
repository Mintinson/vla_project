"""Action model package initialization and factory functions.

This module provides factory functions for creating diffusion processes with various
configurations. It serves as the main entry point for the action model package and
provides convenient abstractions for setting up diffusion models with different
noise schedules, timestep respacing, and loss types.
"""

from typing import Literal

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


def create_diffusion(
    timestep_respacing: str | list[int] | None,
    noise_schedule: Literal["linear", "squaredcos_cap_v2"],
    *,
    use_kl: bool = False,
    sigma_small: bool = False,
    predict_xstart: bool = False,
    learn_sigma: bool = True,
    rescale_learned_sigmas: bool = False,
    diffusion_steps: int = 1000,
) -> SpacedDiffusion:
    """Create a diffusion process with specified configuration.

    This factory function creates a SpacedDiffusion object with the given parameters,
    automatically configuring the appropriate noise schedule, loss type, model
    parameterization, and timestep respacing.

    Args:
        timestep_respacing (str | list[int] | None): Controls which timesteps to use
            for sampling. Options:
            - None or "": Use all diffusion_steps timesteps
            - "ddimN": Use N timesteps with DDIM spacing (e.g., "ddim50")
            - list[int]: Explicit list of timestep counts per section
            - str: Comma-separated timestep counts (e.g., "10,15,20")
        noise_schedule (Literal["linear", "squaredcos_cap_v2"]): Type of noise schedule.
            - "linear": Linear beta schedule
            - "squaredcos_cap_v2": Squared cosine schedule with cap
        use_kl (bool, optional): Whether to use KL divergence loss instead of MSE.
            When True, uses RESCALED_KL loss. Defaults to False.
        sigma_small (bool, optional): Whether to use small fixed variance instead of
            large fixed variance when not learning sigma. Only applies when
            learn_sigma=False. Defaults to False.
        predict_xstart (bool, optional): Whether the model predicts x_0 directly
            instead of predicting noise (epsilon). Defaults to False.
        learn_sigma (bool, optional): Whether the model should learn to predict
            variance parameters. When True, uses LEARNED_RANGE variance type.
            Defaults to True.
        rescale_learned_sigmas (bool, optional): Whether to use rescaled MSE loss
            instead of standard MSE. Only applies when use_kl=False. Defaults to False.
        diffusion_steps (int, optional): Total number of diffusion timesteps in the
            forward process. Defaults to 1000.

    Returns:
        SpacedDiffusion: Configured diffusion process ready for training or sampling.

    Example:
        Create a standard diffusion process with 1000 steps:

        >>> diffusion = create_diffusion(
        ...     timestep_respacing=None,
        ...     noise_schedule="linear"
        ... )

        Create a DDIM sampler with 50 steps:

        >>> ddim_diffusion = create_diffusion(
        ...     timestep_respacing="ddim50",
        ...     noise_schedule="squaredcos_cap_v2"
        ... )

    Note:
        The function automatically configures loss types and model parameterizations
        based on the provided flags:
        - Loss type: KL > RESCALED_MSE > MSE (in order of priority)
        - Model mean type: START_X if predict_xstart=True, else EPSILON
        - Model variance type: LEARNED_RANGE if learn_sigma=True, else FIXED_SMALL/LARGE

    """
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X),
        model_var_type=(
            (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        # rescale_timesteps=rescale_timesteps,
    )
