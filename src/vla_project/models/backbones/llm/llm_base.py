import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial

import torch
from torch import nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

from vla_project.overwatch import initialize_overwatch

from .prompting import PromptBuilder

# Suppress HF Deprecation Warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class LLMBackbone(ABC, nn.Module):
    """Abstract base class for Large Language Model backbones.

    This class defines the interface for LLM backbones used in Vision-Language Models.
    It provides common functionality and enforces implementation of essential methods
    for model initialization, tokenization, and forward passes.

    Attributes:
        identifier (str): Unique identifier for the LLM backbone.
        llm (PreTrainedModel): The underlying language model instance.
        tokenizer (PreTrainedTokenizerBase): Tokenizer for text processing.

    """

    def __init__(self, llm_backbone_id: str) -> None:
        """Initialize the LLM backbone with an identifier.

        Args:
            llm_backbone_id (str): Unique identifier for this LLM backbone.

        """
        super().__init__()
        self.identifier = llm_backbone_id

        # Instance attributes for an LLM Backbone
        self.llm: PreTrainedModel
        self.tokenizer: PreTrainedTokenizerBase

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        """Get the tokenizer instance.

        Returns:
            PreTrainedTokenizerBase: The tokenizer for text processing.

        Raises:
            ValueError: If tokenizer has not been initialized.

        """
        if self.tokenizer is None:
            msg = "Tokenizer has not been initialized."
            raise ValueError(msg)
        return self.tokenizer

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable:
        """Get the FSDP (Fully Sharded Data Parallel) wrapping policy.

        Returns:
            Callable: FSDP wrapping policy function for distributed training.

        """
        ...

    @abstractmethod
    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to reduce memory usage during training."""
        ...

    @abstractmethod
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        *,  # key-word only arguments
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the LLM.

        Args:
            input_ids (torch.LongTensor | None, optional): Token IDs for input sequences.
            attention_mask (torch.Tensor | None, optional): Mask to avoid attention on padding tokens.
            position_ids (torch.LongTensor | None, optional): Position indices for tokens.
            past_key_values (list[torch.FloatTensor] | None, optional): Cached key-value pairs.
            inputs_embeds (torch.FloatTensor | None, optional): Input embeddings.
            labels (torch.LongTensor | None, optional): Target labels for loss computation.
            use_cache (bool | None, optional): Whether to use cached key-value pairs.
            output_attentions (bool | None, optional): Whether to return attention weights.
            output_hidden_states (bool | None, optional): Whether to return hidden states.
            return_dict (bool | None, optional): Whether to return a dict or tuple.

        Returns:
            CausalLMOutputWithPast: Model output including loss, logits, and optional states.

        """
        raise NotImplementedError

    @abstractmethod
    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Convert input token IDs to embeddings.

        Args:
            input_ids (torch.LongTensor): Token IDs to embed.

        Returns:
            torch.Tensor: Token embeddings.

        """
        ...

    @property
    @abstractmethod
    def prompt_builder_fn(self) -> type[PromptBuilder]:
        """Get the prompt builder class for this LLM.

        Returns:
            type[PromptBuilder]: Prompt builder class for formatting conversations.

        """
        ...

    @property
    @abstractmethod
    def transformer_layer_cls(self) -> type[nn.Module]:
        """Get the transformer layer class for FSDP wrapping.

        Returns:
            type[nn.Module]: Transformer layer class used in this LLM.

        """
        ...

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype:
        """Get the preferred half-precision dtype for this LLM.

        Returns:
            torch.dtype: Preferred half-precision data type (e.g., torch.bfloat16).

        """
        ...

    @property
    @abstractmethod
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        """Get modules for last-layer fine-tuning.

        Returns:
            Sequence[nn.Module]: Modules to be fine-tuned in the last layer.

        """
        ...

    @property
    def embed_dim(self) -> int:
        """Get the embedding dimension of the LLM.

        Returns:
            int: Hidden size/embedding dimension of the model.

        Raises:
            ValueError: If LLM has not been initialized.

        """
        if self.llm is None:
            msg = "LLM has not been initialized."
            raise ValueError(msg)
        return self.llm.config.hidden_size

    @property
    def pad_token_id(self) -> int:
        """Get the padding token ID.

        Returns:
            int: Token ID used for padding sequences.

        Raises:
            ValueError: If tokenizer has not been initialized or pad token is not set.

        """
        if self.tokenizer is None or not isinstance(self.tokenizer.pad_token_id, int):
            msg = "Tokenizer has not been initialized or pad token is not set."
            raise ValueError(msg)
        return self.tokenizer.pad_token_id


# === Abstract Base Class for Arbitrary HF Causal LLMs ===
class HFCausalLLMBackbone(LLMBackbone, ABC):
    """Abstract base class for HuggingFace Causal Language Model backbones.

    This class provides a concrete implementation of LLMBackbone for HuggingFace
    causal language models. It handles model loading, tokenizer initialization,
    and provides standard implementations for common operations.

    Attributes:
        llm_family (str): Family name of the LLM (e.g., "llama", "mistral").
        llm_max_length (int): Maximum sequence length for the model.
        inference_mode (bool): Whether the model is in inference mode.

    """

    def __init__(
        self,
        llm_backbone_id: str,
        llm_family: str,
        llm_cls: type[PreTrainedModel],
        hf_hub_path: str,
        llm_max_length: int = 2048,
        hf_token: str | None = None,
        *,
        inference_mode: bool = False,
        use_flash_attention_2: bool = False,
    ) -> None:
        """Initialize HuggingFace Causal LLM backbone.

        Args:
            llm_backbone_id (str): Unique identifier for this LLM backbone.
            llm_family (str): Family name of the LLM (e.g., "llama", "mistral").
            llm_cls (type[PreTrainedModel]): HuggingFace model class to instantiate.
            hf_hub_path (str): Path to the model on HuggingFace Hub.
            llm_max_length (int, optional): Maximum sequence length. Defaults to 2048.
            hf_token (str | None, optional): HuggingFace authentication token.
            inference_mode (bool, optional): Whether to initialize for inference only.
                Defaults to False.
            use_flash_attention_2 (bool, optional): Whether to use Flash Attention 2.
                Defaults to False.

        """
        super().__init__(llm_backbone_id)
        self.llm_family = llm_family
        self.llm_max_length = llm_max_length
        self.inference_mode = inference_mode

        # Initialize LLM (downloading from HF Hub if necessary) --> `llm_cls` is the actual {Model}ForCausalLM class!
        #   => Note: We're eschewing use of the AutoModel API so that we can be more explicit about LLM-specific details
        if not self.inference_mode:
            overwatch.info(
                f"Loading [bold]{llm_family}[/] LLM from [underline]`{hf_hub_path}`[/]",
                ctx_level=1,
            )
            self.llm = llm_cls.from_pretrained(
                hf_hub_path,
                token=hf_token,
                use_flash_attention_2=use_flash_attention_2 if not self.inference_mode else False,
                # The following parameters are set to prevent `UserWarnings` from HF; we want greedy decoding!
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )

        # [Contract] `inference_mode` means we're loading from a pretrained checkpoint; no need to load base weights!
        else:
            overwatch.info(
                f"Building empty [bold]{llm_family}[/] LLM from [underline]`{hf_hub_path}`[/]",
                ctx_level=1,
            )
            llm_config = AutoConfig.from_pretrained(hf_hub_path, token=hf_token)
            self.llm = llm_cls._from_config(llm_config)  # noqa: SLF001

        # Lightweight Handling (with extended explanation) for setting some LLM Parameters
        #   => Set `decoder.use_cache = False` --> incompatible with gradient checkpointing (+ training in general)
        #
        #      Reference: https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958
        self.llm.config.use_cache = self.inference_mode

        #   => Turns out that when gradient checkpointing is on and the underlying LLM has no "trainable" parameters
        #      (requires_grad is False), backprop will fail; setting `enable_input_requires_grad()` registers a new
        #      forward hook that fixes this =>> also totally safe for the "full finetuning" setting!
        if not self.inference_mode:
            self.llm.enable_input_require_grads()

        # Load (Fast) Tokenizer
        overwatch.info(
            f"Loading [bold]{llm_family}[/] (Fast) Tokenizer via the AutoTokenizer API",
            ctx_level=1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_hub_path,
            model_max_length=self.llm_max_length,
            token=hf_token,
            padding_side="right",
        )

        # Validation =>> Our VLM logic currently operates under the assumption that the tokenization of a new input
        #                starts with a <BOS> token unless `add_special_tokens = False`; for these models, we empirically
        #                find that adding image patches *after* the BOS leads to much better performance.
        #
        # As a result we explicitly validate that a tokenizer conforms to the expected behavior; if you're reading this
        # line, it's probably because you're adding a new LLM with a different tokenizer behavior. If so, feel free to
        # override the `SPECIAL_CASES` set below, but make sure to make the appropriate changes in the `datasets.py`
        # and VLM `forward()` logic!
        SPECIAL_CASES = {  # noqa: N806
            # Phi-2 Tokenizer doesn't add any BOS tokens by default, and sets BOS == EOS == "<|endoftext|>"
            #   =>> We'll prepend BOS to first input (to play nicely with image token insertion logic; verified that
            #       this works well with base LLM generation.
            #   =>> Like Llama-2 Tokenizers -- we'll add a special PAD token for training purposes.
            "phi-2-3b",
        }
        if self.identifier in SPECIAL_CASES:
            return

        # Note =>> this assert should hold for all Llama-derived tokenizers (`LlamaTokenizerFast` ==> includes Mistral!
        if (
            self.tokenizer("Test 123", add_special_tokens=True).input_ids[0] != self.tokenizer.bos_token_id
            or self.tokenizer("Test 123", add_special_tokens=False).input_ids[0] == self.tokenizer.bos_token_id
        ):
            msg = (
                f"Default Tokenizer of type `{type(self.tokenizer)}` does not "
                "automatically prefix inputs with BOS token!\n"
                f"Please read the comment in `{__file__}` for more information!",
            )
            raise RuntimeError(msg)

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return FSDP wrapping policy for transformer layers.

        Creates a wrapping policy that wraps each instance of the transformer
        layer class for efficient distributed training with FSDP.

        Returns:
            Callable: FSDP transformer auto-wrap policy function.

        """
        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={self.transformer_layer_cls},
        )


    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on the underlying LLM.

        Activates gradient checkpointing to trade compute for memory during
        training by recomputing activations during backward pass.
        """
        self.llm.gradient_checkpointing_enable()

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Convert input token IDs to embeddings using the LLM's embedding layer.

        Args:
            input_ids (torch.LongTensor): Token IDs to be embedded.

        Returns:
            torch.Tensor: Token embeddings from the LLM's input embedding layer.

        """
        return self.llm.get_input_embeddings()(input_ids)

    # [Contract] Should match the `forward` call of the underlying `llm` instance!
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        *,  # key-word only arguments
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> CausalLMOutputWithPast:
        """Forward pass through the underlying HuggingFace LLM.

        Delegates the forward call to the underlying LLM instance with all
        provided arguments. This maintains compatibility with the HuggingFace
        transformers API.

        Args:
            input_ids (torch.LongTensor | None, optional): Token IDs for input sequences.
            attention_mask (torch.Tensor | None, optional): Mask to avoid attention on padding.
            position_ids (torch.LongTensor | None, optional): Position indices for tokens.
            past_key_values (list[torch.FloatTensor] | None, optional): Cached key-value pairs.
            inputs_embeds (torch.FloatTensor | None, optional): Input embeddings.
            labels (torch.LongTensor | None, optional): Target labels for loss computation.
            use_cache (bool | None, optional): Whether to use cached key-value pairs.
            output_attentions (bool | None, optional): Whether to return attention weights.
            output_hidden_states (bool | None, optional): Whether to return hidden states.
            return_dict (bool | None, optional): Whether to return a dict or tuple.

        Returns:
            CausalLMOutputWithPast: Model output including loss, logits, past key values,
                and optional attention weights and hidden states.

        """
        output: CausalLMOutputWithPast = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return output
