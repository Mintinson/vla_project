from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import override

import torch
from torch import nn
from transformers import GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from vla_project.models.backbones.llm import LLMBackbone
from vla_project.models.backbones.llm.prompting import PromptBuilder
from vla_project.models.backbones.vision import VisionBackbone


# === Abstract Base Class for arbitrary Vision-Language Models ===
class VLM(nn.Module, GenerationMixin, ABC):
    """Abstract base class for Vision-Language Models.

    This class defines the interface for multimodal models that combine vision
    and language understanding. It inherits from both nn.Module for standard
    PyTorch functionality and GenerationMixin for text generation capabilities.

    Attributes:
        model_family (str): Family name of the model (e.g., "prismatic", "llava").
        model_id (str): Specific model identifier within the family.
        vision_backbone (VisionBackbone): Vision encoder for processing images.
        llm_backbone (LLMBackbone): Language model backbone for text processing.
        enable_mixed_precision_training (bool): Whether to enable mixed precision.
        all_module_keys (list | None): Keys for all modules in the model.
        trainable_module_keys (list | None): Keys for trainable modules.
        generation_config: Configuration for text generation.
        main_input_name (str): Name of the main input for generation.

    """

    def __init__(
        self,
        model_family: str,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        *,
        enable_mixed_precision_training: bool = True,
    ) -> None:
        """Initialize the Vision-Language Model.

        Args:
            model_family (str): Family name of the model.
            model_id (str): Specific model identifier within the family.
            vision_backbone (VisionBackbone): Vision encoder for processing images.
            llm_backbone (LLMBackbone): Language model backbone for text processing.
            enable_mixed_precision_training (bool, optional): Whether to enable
                mixed precision training. Defaults to True.

        """
        super().__init__()
        self.model_family = model_family
        self.model_id = model_id
        self.vision_backbone = vision_backbone
        self.llm_backbone = llm_backbone
        self.enable_mixed_precision_training = enable_mixed_precision_training

        # Instance Attributes for a generic VLM
        self.all_module_keys = None
        self.trainable_module_keys = None

        # === GenerationMixin Expected Attributes =>> *DO NOT MODIFY* ===
        self.generation_config = self.llm_backbone.llm.generation_config
        self.main_input_name = "input_ids"

    @property
    def device(self) -> torch.device:
        """Get the device where the model parameters are located.

        Borrowed from `transformers.modeling_utils.py` -- checks parameter device
        and assumes the model is on a single device.

        Returns:
            torch.device: The device where model parameters are located.

        """
        return next(self.parameters()).device

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_family: str,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        **kwargs: str,
    ) -> "VLM":
        """Load a VLM from a pretrained checkpoint.

        Args:
            pretrained_checkpoint (Path): Path to the pretrained model checkpoint.
            model_family (str): Family name of the model.
            model_id (str): Specific model identifier within the family.
            vision_backbone (VisionBackbone): Vision encoder for processing images.
            llm_backbone (LLMBackbone): Language model backbone for text processing.
            **kwargs (str): Additional keyword arguments for model loading.

        Returns:
            VLM: Loaded Vision-Language Model instance.

        """
        ...

    @abstractmethod
    def freeze_backbones(self, stage: str) -> None:
        """Freeze specific model components based on training stage.

        Args:
            stage (str): Training stage identifier (e.g., "align", "finetune").

        """
        ...

    @abstractmethod
    def load_from_checkpoint(self, stage: str, run_dir: Path, pretrained_checkpoint: Path | None = None) -> None:
        """Load model weights from a training checkpoint.

        Args:
            stage (str): Training stage identifier.
            run_dir (Path): Directory containing training run artifacts.
            pretrained_checkpoint (Path | None, optional): Path to specific
                checkpoint file. Defaults to None.

        """
        ...

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable:
        """Get the FSDP (Fully Sharded Data Parallel) wrapping policy.

        Returns:
            Callable: FSDP wrapping policy function for distributed training.

        """
        ...

    @abstractmethod
    def get_prompt_builder(self, system_prompt: str | None = None) -> PromptBuilder:
        """Get a prompt builder for formatting conversations.

        Args:
            system_prompt (str | None, optional): Optional system prompt to
                initialize conversations. Defaults to None.

        Returns:
            PromptBuilder: Configured prompt builder for this model.

        """
        ...

    # @override
    # @abstractmethod
    # def generate(self, image: torch.Tensor, prompt_text: str, **kwargs) -> str: # pyright: ignore[reportIncompatibleMethodOverride]
    #     """Generate text response for a single image-text pair.

    #     Args:
    #         image (torch.Tensor): Input image tensor.
    #         prompt_text (str): Text prompt for generation.
    #         **kwargs: Additional generation parameters.

    #     Returns:
    #         str: Generated text response.

    #     """
    #     ...

    @abstractmethod
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        *,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        multimodal_indices: torch.LongTensor | None = None,
    ) -> CausalLMOutputWithPast:
        """Forward pass through the Vision-Language Model.

        Args:
            input_ids (torch.LongTensor | None, optional): Token IDs for text input.
            attention_mask (torch.Tensor | None, optional): Attention mask for input.
            pixel_values (torch.FloatTensor | None, optional): Pixel values for images.
            labels (torch.LongTensor | None, optional): Target labels for training.
            inputs_embeds (torch.FloatTensor | None, optional): Input embeddings.
            past_key_values (list[torch.FloatTensor] | None, optional): Cached
                key-value pairs for generation.
            use_cache (bool | None, optional): Whether to use caching.
            output_attentions (bool | None, optional): Whether to return attention weights.
            output_hidden_states (bool | None, optional): Whether to return hidden states.
            return_dict (bool | None, optional): Whether to return a dict or tuple.
            multimodal_indices (torch.LongTensor | None, optional): Indices for
                multimodal token positions.

        Returns:
            CausalLMOutputWithPast: Model output including loss, logits, and optional states.

        """
        ...

    # === GenerationMixin Expected Properties & Methods (DO NOT MODIFY) ===
    @staticmethod
    def can_generate() -> bool:
        """Check if the model can generate text.

        Returns:
            bool: Always True for VLM models.

        """
        return True

    @property
    def config(self) -> PretrainedConfig:
        """Get the model configuration from the language model backbone.

        Returns:
            PretrainedConfig: Configuration object from the LLM backbone.

        """
        return self.llm_backbone.llm.config

    # => Beam Search Utility
    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder cached key-value pairs for beam search.

        Delegates to the underlying language model's cache reordering method
        to support beam search generation.

        Args:
            past_key_values: Cached key-value pairs to reorder.
            beam_idx: Beam indices for reordering.

        Returns:
            Reordered cached key-value pairs.

        """
        return self.llm_backbone.llm._reorder_cache(past_key_values, beam_idx)
