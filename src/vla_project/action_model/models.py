# Modified from facebookresearch's DiT repos
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
from functools import partial
from typing import cast

import torch
from jaxtyping import Float
from timm.models.vision_transformer import Attention, Mlp  # pyright: ignore[reportPrivateImportUsage]
from torch import Tensor, nn


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply adaptive layer normalization modulation.

    Args:
        x: Input tensor to modulate.
        shift: Shift parameter for modulation.
        scale: Scale parameter for modulation.

    Returns:
        Modulated tensor computed as x * (1 + scale) + shift.

    """
    return x * (1 + scale) + shift


#################################################################################
#               Embedding Layers for Timesteps and conditions                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations.

    This module converts scalar timestep values into high-dimensional embeddings
    using sinusoidal position encoding followed by an MLP projection.

    Attributes:
        mlp (nn.Sequential): MLP for projecting frequency embeddings to hidden size.
        frequency_embedding_size (int): Dimension of the frequency embedding.

    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        """Initialize the timestep embedder.

        Args:
            hidden_size (int): Output embedding dimension.
            frequency_embedding_size (int, optional): Intermediate frequency embedding
                dimension. Defaults to 256.

        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Float[Tensor, "n d"]:
        """Create sinusoidal timestep embeddings.

        Generates sinusoidal position encodings for timestep values, similar to
        the positional encodings used in transformers.

        Args:
            t (Tensor): Timestep values to embed, with shape [n].
            dim (int): Embedding dimension.
            max_period (int, optional): Maximum period for the sinusoidal encoding.
                Defaults to 10000.

        Returns:
            Float[Tensor, "n d"]: Sinusoidal embeddings for the input timesteps.

        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device,
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        """Forward pass through the timestep embedder.

        Args:
            t (Tensor): Input timestep values.

        Returns:
            Tensor: Timestep embeddings of shape [batch_size, hidden_size].

        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(next(self.mlp.parameters()).dtype)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Embeds conditions into vector representations with classifier-free guidance support.

    This module handles conditional embeddings and supports label dropout for
    classifier-free guidance during diffusion model training and inference.

    Attributes:
        linear (nn.Linear): Linear projection layer.
        dropout_prob (float): Probability of dropping conditions.
        uncondition (nn.Parameter): Learnable unconditional embedding.

    """

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        dropout_prob: float = 0.1,
        conditions_shape: tuple[int, int, int] = (1, 1, 4096),
    ) -> None:
        """Initialize the label embedder.

        Args:
            in_size (int): Input condition dimension.
            hidden_size (int): Output embedding dimension.
            dropout_prob (float, optional): Probability of condition dropout for
                classifier-free guidance. Defaults to 0.1.
            conditions_shape (tuple[int, int, int], optional): Shape of condition
                tensors for unconditional embedding. Defaults to (1, 1, 4096).

        """
        super().__init__()
        self.linear = nn.Linear(in_size, hidden_size)
        self.dropout_prob = dropout_prob
        if dropout_prob > 0:
            self.uncondition = nn.Parameter(torch.empty(conditions_shape[1:]))

    def token_drop(self, conditions: Tensor, force_drop_ids: Tensor | None = None) -> Tensor:
        """Apply condition dropout for classifier-free guidance.

        Randomly or forcibly replaces condition embeddings with unconditional
        embeddings to enable classifier-free guidance.

        Args:
            conditions (Tensor): Input condition embeddings.
            force_drop_ids (Tensor | None, optional): Explicit dropout mask.
                If None, uses random dropout. Defaults to None.

        Returns:
            Tensor: Conditions with dropout applied.

        """
        if force_drop_ids is None:
            drop_ids = torch.rand(conditions.shape[0], device=conditions.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(
            drop_ids.unsqueeze(1).unsqueeze(1).expand(conditions.shape[0], *self.uncondition.shape),
            self.uncondition,
            conditions,
        )

    def forward(self, conditions: Tensor, *, train: bool, force_drop_ids: Tensor | None = None) -> Tensor:
        """Forward pass through the label embedder.

        Args:
            conditions (Tensor): Input condition tensors.
            train (bool): Whether in training mode (affects dropout).
            force_drop_ids (Tensor | None, optional): Explicit dropout mask.
                Defaults to None.

        Returns:
            Tensor: Embedded condition representations.

        """
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            conditions = self.token_drop(conditions, force_drop_ids)
        return self.linear(conditions)


#################################################################################
#                      Embedding Layers for Actions and                         #
#################################################################################
class ActionEmbedder(nn.Module):
    """Embeds action vectors into higher-dimensional representations.

    Simple linear projection layer for converting raw action vectors into
    embeddings suitable for the transformer backbone.

    Attributes:
        linear (nn.Linear): Linear projection layer.

    """

    def __init__(self, action_size: int, hidden_size: int) -> None:
        """Initialize the action embedder.

        Args:
            action_size (int): Dimension of input action vectors.
            hidden_size (int): Output embedding dimension.

        """
        super().__init__()
        self.linear = nn.Linear(action_size, hidden_size)

    def forward(self, x: Float[Tensor, "b seq action_size"]) -> Float[Tensor, "b seq hidden_size"]:
        """Forward pass through the action embedder.

        Args:
            x (Tensor): Input action tensors of shape [batch, seq_len, action_size].

        Returns:
            Tensor: Action embeddings of shape [batch, seq_len, hidden_size].

        """
        return self.linear(x)


# Action_History is not used now
class HistoryEmbedder(nn.Module):
    """Embeds action history vectors (currently unused).

    This module was designed for embedding action history but is not currently
    used in the model. Kept for potential future use.

    Attributes:
        linear (nn.Linear): Linear projection layer.

    """

    def __init__(self, action_size: int, hidden_size: int) -> None:
        """Initialize the history embedder.

        Args:
            action_size (int): Dimension of input action vectors.
            hidden_size (int): Output embedding dimension.

        """
        super().__init__()
        self.linear = nn.Linear(action_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the history embedder.

        Args:
            x (Tensor): Input action history tensors.

        Returns:
            Tensor: Action history embeddings.

        """
        return self.linear(x)


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """Diffusion Transformer block with self-attention and MLP.

    A standard transformer block consisting of layer normalization, multi-head
    self-attention, and an MLP, with residual connections.

    Attributes:
        norm1 (nn.LayerNorm): Layer normalization before attention.
        attn (Attention): Multi-head self-attention module.
        norm2 (nn.LayerNorm): Layer normalization before MLP.
        mlp (Mlp): Multi-layer perceptron module.

    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs) -> None:  # noqa: ANN003
        """Initialize the DiT block.

        Args:
            hidden_size (int): Hidden dimension of the transformer.
            num_heads (int): Number of attention heads.
            mlp_ratio (float, optional): Ratio of MLP hidden size to embedding size.
                Defaults to 4.0.
            **block_kwargs: Additional arguments passed to attention module.

        """
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = partial(nn.GELU, approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)  # pyright: ignore[reportArgumentType]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the DiT block.

        Args:
            x (Tensor): Input tensor of shape [batch, seq_len, hidden_size].

        Returns:
            Tensor: Output tensor with same shape as input.

        """
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class FinalLayer(nn.Module):
    """Final output layer of the DiT model.

    Applies layer normalization followed by a linear projection to produce
    the final model outputs.

    Attributes:
        norm_final (nn.LayerNorm): Final layer normalization.
        linear (nn.Linear): Output projection layer.

    """

    def __init__(self, hidden_size: int, out_channels: int) -> None:
        """Initialize the final layer.

        Args:
            hidden_size (int): Input hidden dimension.
            out_channels (int): Number of output channels.

        """
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the final layer.

        Args:
            x (Tensor): Input tensor of shape [batch, seq_len, hidden_size].

        Returns:
            Tensor: Output tensor of shape [batch, seq_len, out_channels].

        """
        x = self.norm_final(x)
        return self.linear(x)


class DiT(nn.Module):
    """Diffusion Transformer for action prediction.

    A transformer-based diffusion model that predicts future actions conditioned
    on visual features and timestep information. The model uses a sequence of
    DiT blocks with positional embeddings and supports classifier-free guidance.

    Attributes:
        learn_sigma (bool): Whether to predict noise variance.
        in_channels (int): Input action dimension.
        out_channels (int): Output action dimension.
        class_dropout_prob (float): Condition dropout probability.
        num_heads (int): Number of attention heads.
        past_action_window_size (int): Size of action history window.
        future_action_window_size (int): Size of future action prediction window.
        history_embedder (HistoryEmbedder): Embedder for action history.
        x_embedder (ActionEmbedder): Embedder for current actions.
        t_embedder (TimestepEmbedder): Embedder for timesteps.
        z_embedder (LabelEmbedder): Embedder for visual conditions.
        positional_embedding (nn.Parameter): Learnable positional embeddings.
        blocks (nn.ModuleList): List of DiT transformer blocks.
        final_layer (FinalLayer): Final output layer.

    """

    def __init__(
        self,
        in_channels: int = 7,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        token_size: int = 4096,
        future_action_window_size: int = 1,
        past_action_window_size: int = 0,
        *,
        learn_sigma: bool = False,
    ) -> None:
        """Initialize the DiT model.

        Args:
            in_channels (int, optional): Dimension of input actions. Defaults to 7.
            hidden_size (int, optional): Hidden dimension of transformer. Defaults to 1152.
            depth (int, optional): Number of transformer layers. Defaults to 28.
            num_heads (int, optional): Number of attention heads. Defaults to 16.
            mlp_ratio (float, optional): MLP expansion ratio. Defaults to 4.0.
            class_dropout_prob (float, optional): Condition dropout probability.
                Defaults to 0.1.
            token_size (int, optional): Size of visual condition tokens. Defaults to 4096.
            future_action_window_size (int, optional): Number of future actions to predict.
                Defaults to 1.
            past_action_window_size (int, optional): Number of past actions to condition on.
                Defaults to 0. Currently must be 0.
            learn_sigma (bool, optional): Whether to predict noise variance. Defaults to False.

        Raises:
            ValueError: If past_action_window_size is not 0.

        """
        super().__init__()

        if past_action_window_size != 0:
            msg = "Error: action_history is not used now"
            raise ValueError(msg)

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.class_dropout_prob = class_dropout_prob
        self.num_heads = num_heads
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size

        # Action history is not used now.
        self.history_embedder = HistoryEmbedder(action_size=in_channels, hidden_size=hidden_size)

        self.x_embedder = ActionEmbedder(action_size=in_channels, hidden_size=hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.z_embedder = LabelEmbedder(in_size=token_size, hidden_size=hidden_size, dropout_prob=class_dropout_prob)
        scale = hidden_size**-0.5

        # Learnable positional embeddings
        # +2, one for the conditional token, and one for the current action prediction
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(future_action_window_size + past_action_window_size + 2, hidden_size),
        )

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize model weights using appropriate initialization schemes.

        Applies Xavier uniform initialization to most linear layers, with specific
        initialization for embeddings and the final layer set to zero for stable
        training start.
        """

        # Initialize transformer layers:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # # Initialize token_embed like nn.Linear
        nn.init.normal_(self.x_embedder.linear.weight, std=0.02)
        nn.init.constant_(self.x_embedder.linear.bias, 0)

        nn.init.normal_(self.history_embedder.linear.weight, std=0.02)
        nn.init.constant_(self.history_embedder.linear.bias, 0)

        # Initialize label embedding table:
        if self.class_dropout_prob > 0:
            nn.init.normal_(self.z_embedder.uncondition, std=0.02)
        nn.init.normal_(self.z_embedder.linear.weight, std=0.02)
        nn.init.constant_(self.z_embedder.linear.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(cast("nn.Linear", self.t_embedder.mlp[0]).weight, std=0.02)
        nn.init.normal_(cast("nn.Linear", self.t_embedder.mlp[2]).weight, std=0.02)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: Float[Tensor, "n t d"], t: Tensor, z: Float[Tensor, "n 1 d"]) -> Float[Tensor, "n t c"]:
        """Forward pass through the DiT model.

        Processes action sequences conditioned on visual features and timesteps
        to predict noise or future actions.

        Args:
            x (Float[Tensor, "n t d"]): Input action sequences of shape
                [batch_size, sequence_length, action_dim].
            t (Tensor): Timestep values of shape [batch_size].
            z (Float[Tensor, "n 1 d"]): Visual condition features of shape
                [batch_size, 1, feature_dim].

        Returns:
            Float[Tensor, "n t c"]: Model predictions of shape
                [batch_size, sequence_length, out_channels].

        """
        x = self.x_embedder(x)  # (N, T, D)
        t = self.t_embedder(t)  # (N, D)
        z = self.z_embedder(z, self.training)  # (N, 1, D)
        c = t.unsqueeze(1) + z  # (N, 1, D)
        x = torch.cat((c, x), dim=1)  # (N, T+1, D)
        x = x + self.positional_embedding  # (N, T+1, D)
        for block in self.blocks:
            x = block(x)  # (N, T+1, D)
        x = self.final_layer(x)  # (N, T+1, out_channels)
        return x[:, 1:, :]  # (N, T, C)

    def forward_with_cfg(
        self,
        x: Float[Tensor, "n t d"],
        t: Tensor,
        z: Float[Tensor, "n 1 d"],
        cfg_scale: float,
    ) -> Float[Tensor, "n t c"]:
        """Forward pass with classifier-free guidance.

        Performs a forward pass that combines conditional and unconditional
        predictions using classifier-free guidance for improved sample quality.

        Args:
            x (Float[Tensor, "n t d"]): Input action sequences of shape
                [batch_size, sequence_length, action_dim].
            t (Tensor): Timestep values of shape [batch_size].
            z (Float[Tensor, "n 1 d"]): Visual condition features of shape
                [batch_size, 1, feature_dim].
            cfg_scale (float): Classifier-free guidance scale factor.

        Returns:
            Model predictions with classifier-free guidance applied.

        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]  # (N/2, T, D)
        combined = torch.cat([half, half], dim=0).to(next(self.x_embedder.parameters()).dtype)
        model_out = self.forward(combined, t, z)  # (N, T, C)
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps = model_out[:, :, : self.in_channels]  # (N, T, in_channels)
        rest = model_out[:, :, self.in_channels :]  # (N, T, C - in_channels)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)  # (N//2, T, in_channels)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)  # (N, T, in_channels)
        # return torch.cat([eps, rest], dim=1)
        return torch.cat([eps, rest], dim=2)  # (N, T, C)
