"""Neural network utility modules for vision-language model projectors.

This module provides various projector architectures for connecting vision encoders
to language models in vision-language architectures. The projectors transform vision
features to match the dimensionality and representation space of language model embeddings.

Key projector types:
    - LinearProjector: Simple linear transformation for basic projection
    - MLPProjector: Multi-layer perceptron with nonlinear transformations
    - FusedMLPProjector: Advanced projector for fused multi-modal features

These projectors are commonly used in vision-language models to bridge the gap between
visual feature representations and textual embeddings, enabling effective multimodal
understanding and generation.

Example:
    Creating and using different projector types:

    ```python
    from vla_project.utils.nn_utils import LinearProjector, MLPProjector

    # Simple linear projection
    linear_proj = LinearProjector(vision_dim=768, llm_dim=1024)

    # MLP projection with GELU activation
    mlp_proj = MLPProjector(vision_dim=768, llm_dim=1024, mlp_type="gelu-mlp")

    # Forward pass
    vision_features = torch.randn(batch_size, seq_len, 768)
    projected_features = mlp_proj(vision_features)
    ```

"""

from typing import Literal

import torch
from torch import nn


class LinearProjector(nn.Module):
    """Simple linear projector for vision-to-language feature transformation.

    This projector implements a basic linear transformation to map vision features
    from the vision encoder output dimension to the language model input dimension.
    It provides the simplest form of cross-modal feature alignment.

    The projector applies a single linear layer with bias to transform features:
        output = W * input + b

    Attributes:
        projector: The linear transformation layer.

    Example:
        Basic usage for projecting vision features:

        ```python
        projector = LinearProjector(vision_dim=768, llm_dim=1024)
        vision_features = torch.randn(32, 196, 768)  # [batch, patches, vision_dim]
        projected = projector(vision_features)  # [32, 196, 1024]
        ```

    """

    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        """Initialize the linear projector.

        Args:
            vision_dim: Dimensionality of input vision features.
            llm_dim: Dimensionality of output language model features.

        """
        super().__init__()
        self.projector = nn.Linear(vision_dim, llm_dim, bias=True)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        """Project vision features to language model dimension.

        Args:
            img_patches: Input vision features with shape [batch_size, seq_len, vision_dim].

        Returns:
            Projected features with shape [batch_size, seq_len, llm_dim].

        """
        return self.projector(img_patches)


class MLPProjector(nn.Module):
    """Multi-layer perceptron projector for advanced vision-to-language feature transformation.

    This projector implements a multi-layer perceptron with nonlinear activation functions
    to map vision features from the vision encoder output dimension to the language model
    input dimension. It provides more sophisticated feature transformation compared to
    the simple linear projector.

    The projector architecture consists of:
        - First linear layer: vision_dim -> llm_dim
        - Activation function (GELU)
        - Second linear layer: llm_dim -> llm_dim

    This design allows for more complex feature transformations and better alignment
    between vision and language representations.

    Attributes:
        projector: The sequential MLP transformation layers.

    Example:
        Creating and using an MLP projector:

        ```python
        projector = MLPProjector(vision_dim=768, llm_dim=1024, mlp_type="gelu-mlp")
        vision_features = torch.randn(32, 196, 768)  # [batch, patches, vision_dim]
        projected = projector(vision_features)  # [32, 196, 1024]
        ```

    """

    def __init__(self, vision_dim: int, llm_dim: int, mlp_type: Literal["gelu-mlp"] = "gelu-mlp") -> None:
        """Initialize the MLP projector.

        Args:
            vision_dim: Dimensionality of input vision features.
            llm_dim: Dimensionality of output language model features.
            mlp_type: Type of MLP architecture. Currently only supports "gelu-mlp".
                Defaults to "gelu-mlp".

        Raises:
            ValueError: If an unsupported mlp_type is specified.

        """
        super().__init__()
        if mlp_type == "gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            msg = f"Projector with `{mlp_type = }` is not supported yet!"
            raise ValueError(msg)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        """Project vision features to language model dimension using MLP.

        Args:
            img_patches: Input vision features with shape [batch_size, seq_len, vision_dim].

        Returns:
            Projected features with shape [batch_size, seq_len, llm_dim].

        """
        return self.projector(img_patches)


class FusedMLPProjector(nn.Module):
    """Advanced fused multi-layer perceptron projector for complex multi-modal feature transformation.

    This projector implements a sophisticated multi-layer perceptron designed for processing
    fused multi-modal features (typically combining multiple vision encoders or other modalities).
    It uses a deeper architecture with multiple GELU activations to enable more complex
    transformations of high-dimensional fused features.

    The projector architecture consists of:
        - First linear layer: fused_vision_dim -> (fused_vision_dim * 4)
        - First GELU activation
        - Second linear layer: (fused_vision_dim * 4) -> llm_dim
        - Second GELU activation
        - Third linear layer: llm_dim -> llm_dim

    This design provides enhanced capacity for transforming complex fused features from
    multiple vision encoders or modalities into the language model representation space.

    Attributes:
        initial_projection_dim: Intermediate dimension (fused_vision_dim * 4) for expansion.
        projector: The sequential MLP transformation layers.

    Example:
        Creating and using a fused MLP projector:

        ```python
        projector = FusedMLPProjector(
            fused_vision_dim=1536,
            llm_dim=1024,
            mlp_type="fused-gelu-mlp"
        )
        fused_features = torch.randn(32, 196, 1536)  # [batch, patches, fused_dim]
        projected = projector(fused_features)  # [32, 196, 1024]
        ```

    """

    def __init__(
        self,
        fused_vision_dim: int,
        llm_dim: int,
        mlp_type: Literal["fused-gelu-mlp"] = "fused-gelu-mlp",
    ) -> None:
        """Initialize the fused MLP projector.

        Args:
            fused_vision_dim: Dimensionality of input fused vision features.
            llm_dim: Dimensionality of output language model features.
            mlp_type: Type of fused MLP architecture. Currently only supports
                "fused-gelu-mlp". Defaults to "fused-gelu-mlp".

        Raises:
            ValueError: If an unsupported mlp_type is specified.

        """
        super().__init__()
        self.initial_projection_dim = fused_vision_dim * 4
        if mlp_type == "fused-gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(fused_vision_dim, self.initial_projection_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.initial_projection_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            msg = f"Fused Projector with `{mlp_type = }` is not supported!"
            raise ValueError(msg)

    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        """Project fused vision features to language model dimension using deep MLP.

        Args:
            fused_img_patches: Input fused vision features with shape
                [batch_size, seq_len, fused_vision_dim].

        Returns:
            Projected features with shape [batch_size, seq_len, llm_dim].

        """
        return self.projector(fused_img_patches)
