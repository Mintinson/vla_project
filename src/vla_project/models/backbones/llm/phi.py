from collections.abc import Sequence
from typing import cast

import torch
from torch import nn
from transformers import PhiForCausalLM
from transformers.models.phi.modeling_phi import PhiDecoderLayer

from vla_project.models.backbones.llm.llm_base import HFCausalLLMBackbone
from vla_project.models.backbones.llm.prompting import PhiPromptBuilder, PromptBuilder

PHI_MODELS = {
    # === Phi-2 ===
    "phi-2-3b": {
        "llm_family": "phi",
        "llm_cls": PhiForCausalLM,
        "hf_hub_path": "microsoft/phi-2",
    },
}


class PhiLLMBackbone(HFCausalLLMBackbone):
    """Phi-series language model backbone implementation.

    This class provides a concrete implementation of HFCausalLLMBackbone specifically
    for Microsoft's Phi-series models (e.g., Phi-2). It handles Phi-specific
    configurations including special token handling and prompt formatting.

    The Phi models use CodeGenTokenizer which requires special handling for BOS/EOS
    tokens and padding tokens. This implementation adds a custom pad token and
    resizes the token embeddings accordingly.

    Attributes:
        All attributes inherited from HFCausalLLMBackbone plus Phi-specific
        tokenizer modifications.

    """

    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: str | None = None,
        *,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
    ) -> None:
        """Initialize Phi LLM backbone.

        Sets up the Phi model with proper tokenizer configuration including
        adding a custom pad token and resizing embeddings. Phi models require
        special handling due to their use of CodeGenTokenizer.

        Args:
            llm_backbone_id (str): Identifier for the specific Phi model variant.
                Must be a key in PHI_MODELS dictionary.
            llm_max_length (int, optional): Maximum sequence length for the model.
                Defaults to 2048.
            hf_token (str | None, optional): HuggingFace authentication token.
                Defaults to None.
            inference_mode (bool, optional): Whether to initialize for inference only.
                Defaults to False.
            use_flash_attention_2 (bool, optional): Whether to use Flash Attention 2.
                Defaults to True.

        Raises:
            KeyError: If llm_backbone_id is not found in PHI_MODELS.

        """
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **PHI_MODELS[llm_backbone_id],
        )

        # [Special Case] Phi PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.llm.config.pad_token_id = cast("int", self.tokenizer.pad_token_id)
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> type[PromptBuilder]:
        """Get the appropriate prompt builder class for this Phi model.

        Returns the PhiPromptBuilder for Phi-2 models, which handles the specific
        formatting requirements including Input/Output style prompts and proper
        BOS/EOS token placement.

        Returns:
            type[PromptBuilder]: PhiPromptBuilder class for Phi-2 models.

        Raises:
            ValueError: If no prompt builder is defined for the given model identifier.

        """
        if self.identifier.startswith("phi-2"):
            return PhiPromptBuilder

        msg = f"No PromptBuilder defined for LLM Backbone `{self.identifier}`"
        raise ValueError(msg)

    @property
    def transformer_layer_cls(self) -> type[nn.Module]:
        """Get the transformer layer class for FSDP wrapping.

        Returns:
            type[nn.Module]: PhiDecoderLayer class used in Phi models.

        """
        return PhiDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """Get the preferred half-precision dtype for Phi models.

        Returns:
            torch.dtype: torch.bfloat16 as the preferred half-precision type for Phi.

        """
        return torch.bfloat16

    @property
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        """Get modules for last-layer fine-tuning in Phi models.

        Returns the language modeling head and final layer normalization
        modules that are typically fine-tuned in the last layer.

        Returns:
            Sequence[nn.Module]: List containing the LM head and final norm modules.

        """
        return [self.llm.lm_head, self.llm.model.final_layernorm]
