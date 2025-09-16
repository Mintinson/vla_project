"""Action model implementation using Diffusion Transformers (DiT).

This module provides the ActionModel class which combines diffusion processes with
transformer architectures for action sequence modeling. It supports various model
sizes and includes both standard diffusion and DDIM sampling capabilities.
"""

from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor, nn

from vla_project.action_model import create_diffusion

from . import gaussian_diffusion as gd
from .models import DiT
from .respace import SpacedDiffusion


def get_dit_s(**kwargs) -> DiT:
    """Create a small DiT model.

    Args:
        **kwargs: Additional keyword arguments passed to DiT constructor.

    Returns:
        DiT: Small DiT model with 6 layers, 384 hidden size, and 4 attention heads.

    """
    return DiT(depth=6, hidden_size=384, num_heads=4, **kwargs)


def get_dit_b(**kwargs) -> DiT:
    """Create a base DiT model.

    Args:
        **kwargs: Additional keyword arguments passed to DiT constructor.

    Returns:
        DiT: Base DiT model with 12 layers, 768 hidden size, and 12 attention heads.

    """
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)


def get_dit_l(**kwargs) -> DiT:
    """Create a large DiT model.

    Args:
        **kwargs: Additional keyword arguments passed to DiT constructor.

    Returns:
        DiT: Large DiT model with 24 layers, 1024 hidden size, and 16 attention heads.

    """
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)


# Model size configurations
DiT_models = {"DiT-S": get_dit_s, "DiT-B": get_dit_b, "DiT-L": get_dit_l}


class ActionModel(nn.Module):
    """Action model implementation using Diffusion Transformers (DiT).

    This class combines diffusion processes with transformer architectures for
    action sequence modeling. It supports various model sizes and includes both
    standard diffusion and DDIM sampling capabilities.

    The model uses a DiT (Diffusion Transformer) backbone to predict noise in the
    diffusion process, conditioned on past actions and other context information.

    Attributes:
        in_channels (int): Number of input channels.
        noise_schedule (str): Type of noise schedule used for diffusion.
        diffusion_steps (int): Number of diffusion timesteps.
        diffusion (GaussianDiffusion): Main diffusion process.
        ddim_diffusion (SpacedDiffusion | None): DDIM diffusion process for fast sampling.
        past_action_window_size (int): Size of past action context window.
        future_action_window_size (int): Size of future action prediction window.
        net (DiT): The underlying DiT model.

    """

    def __init__(
        self,
        token_size: int,
        model_type: Literal["DiT-S", "DiT-B", "DiT-L"],
        in_channels: int,
        future_action_window_size: int,
        past_action_window_size: int,
        diffusion_steps: int = 100,
        noise_schedule: Literal["linear", "squaredcos_cap_v2"] = "squaredcos_cap_v2",
    ) -> None:
        """Initialize the ActionModel.

        Args:
            token_size (int): Size of each token in the sequence.
            model_type (Literal["DiT-S", "DiT-B", "DiT-L"]): Type of DiT model to use.
                - "DiT-S": Small model (6 layers, 384 hidden size, 4 heads)
                - "DiT-B": Base model (12 layers, 768 hidden size, 12 heads)
                - "DiT-L": Large model (24 layers, 1024 hidden size, 16 heads)
            in_channels (int): Number of input channels for the model.
            future_action_window_size (int): Number of future action steps to predict.
            past_action_window_size (int): Number of past action steps to use as context.
            diffusion_steps (int): Number of diffusion timesteps for the process.
            noise_schedule (Literal["linear", "squaredcos_cap_v2"]): Type of noise schedule.
                - "linear": Linear beta schedule
                - "squaredcos_cap_v2": Squared cosine schedule with cap

        """
        super().__init__()
        self.in_channels = in_channels
        self.noise_schedule: Literal["linear", "squaredcos_cap_v2"] = noise_schedule

        # Gaussian diffusion offers forward and backward functions q_sample and p_sample.
        self.diffusion_steps = diffusion_steps
        self.diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule=noise_schedule,
            diffusion_steps=self.diffusion_steps,
            sigma_small=True,
            learn_sigma=False,
        )
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            lean_sigma = True
        else:
            lean_sigma = False

        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.net = DiT_models[model_type](
            token_size=token_size,
            in_channels=in_channels,
            class_dropout_prob=0.1,
            learn_sigma=lean_sigma,
            future_action_window_size=future_action_window_size,
            past_action_window_size=past_action_window_size,
        )

    # Given condition z and ground truth token x, compute loss
    def loss(self, x: Float[Tensor, "b t c"], z: Float[Tensor, "b t c"]) -> Tensor:
        """Compute the diffusion loss for action sequence prediction.

        This method computes the L2 loss between the predicted noise and the actual
        noise added during the forward diffusion process. The model is trained to
        predict the noise that was added to the clean action sequence.

        Args:
            x (Float[Tensor, "b t c"]): Ground truth action sequence tensor with shape
                (batch_size, sequence_length, channels).
            z (Float[Tensor, "b t c"]): Conditioning tensor (e.g., past actions) with shape
                (batch_size, sequence_length, channels).

        Returns:
            Tensor: Mean L2 loss between predicted and actual noise.

        Raises:
            RuntimeError: If tensor shapes don't match expected dimensions.

        """
        noise = torch.randn_like(x)
        timestep = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device=x.device)
        # sample x_t from x
        x_t = self.diffusion.q_sample(x_start=x, t=timestep, noise=noise)
        # predict noise from x_t
        noise_pred: Tensor = self.net(x_t, timestep, z)
        if noise.shape != noise_pred.shape or noise.shape != x.shape:
            msg = f"Shape mismatch: noise {noise.shape}, noise_pred {noise_pred.shape}, x {x.shape}"
            raise RuntimeError(msg)
        # Compute L2 loss
        return ((noise_pred - noise) ** 2).mean()

    # Create DDIM sampler
    def create_ddim(self, ddim_step: int = 10) -> SpacedDiffusion:
        """Create a DDIM (Denoising Diffusion Implicit Models) sampler for fast sampling.

        DDIM allows for faster sampling by using a subset of the original diffusion
        timesteps, enabling high-quality generation with fewer denoising steps.

        Args:
            ddim_step (int, optional): Number of denoising steps to use in DDIM sampling.
                Defaults to 10. Lower values result in faster sampling but potentially
                lower quality.

        Returns:
            SpacedDiffusion: DDIM diffusion process configured with the specified
                number of timesteps.

        Note:
            This method creates and stores the DDIM diffusion process in the
            `ddim_diffusion` attribute for later use.

        """
        self.ddim_diffusion = create_diffusion(
            timestep_respacing="ddim" + str(ddim_step),
            noise_schedule=self.noise_schedule,
            diffusion_steps=self.diffusion_steps,
            sigma_small=True,
            learn_sigma=False,
        )
        return self.ddim_diffusion
