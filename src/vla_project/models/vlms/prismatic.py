from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast, override

import torch
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from vla_project.models.backbones import LLMBackbone, VisionBackbone
from vla_project.models.backbones.llm.prompting.base_prompter import PromptBuilder
from vla_project.overwatch import initialize_overwatch
from vla_project.utils.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

from .vlm_base import VLM

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class PrismaticVLM(VLM):
    """Prismatic Vision-Language Model implementation.

    This class implements a Prismatic-style VLM that combines a vision backbone
    with a language model backbone through a learned projection layer. It supports
    different projection architectures including linear, MLP, and fused MLP variants.

    Attributes:
        arch_specifier (str): Architecture type for the projection layer.
        projector (nn.Module): Projection layer between vision and language modalities.
        vision_backbone_requires_grad (bool): Whether vision backbone requires gradients.
        all_module_keys (list[str]): Keys for all model modules.
        trainable_module_keys (list[str]): Keys for trainable modules.
        string2idx (dict[str, int]): Mapping from trigger strings to token indices.

    """

    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        *,
        enable_mixed_precision_training: bool = True,
        arch_specifier: Literal["linear", "fused-gelu-mlp", "gelu-mlp"] = "gelu-mlp",
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the Prismatic Vision-Language Model.

        Args:
            model_id (str): Unique identifier for this model instance.
            vision_backbone (VisionBackbone): Pre-trained vision encoder.
            llm_backbone (LLMBackbone): Pre-trained language model backbone.
            enable_mixed_precision_training (bool, optional): Whether to enable
                mixed precision training. Defaults to True.
            arch_specifier (Literal["linear", "fused-gelu-mlp", "gelu-mlp"], optional):
                Architecture type for the projection layer. Defaults to "gelu-mlp".
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Raises:
            ValueError: If arch_specifier is not supported.
            RuntimeError: If trigger strings are tokenized as multiple tokens.

        """
        super().__init__(
            "prismatic",
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
        )

        # Set Weight Initialization Seed for Projector Consistency
        torch.manual_seed(vision_backbone.embed_dim)

        # Initialize Projection (Adapter) based on `arch_specifier`
        self.arch_specifier = arch_specifier
        if arch_specifier == "linear":
            self.projector = LinearProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("fused-gelu-mlp"):
            self.projector = FusedMLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("gelu-mlp"):
            self.projector = MLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        else:
            # {arch_specifier = } will outputs the variable name and its value
            msg = f"PrismaticVLM with `{arch_specifier = }` is not supported!"
            raise ValueError(msg)

        # Trackers
        self.vision_backbone_requires_grad = False

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vision_backbone", "llm_backbone", "projector"]
        self.trainable_module_keys = []

        # === Generation Utilities ===
        #   => For computing likelihoods --> get tokens corresponding to "True", "False" and "Yes", "No"
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.llm_backbone.tokenizer.encode(trigger_string, add_special_tokens=False)
            if len(token_idx_list) != 1:
                msg = f'String "{trigger_string}" is tokenized as more than one token!'
                raise RuntimeError(msg)
            self.string2idx[trigger_string] = token_idx_list[0]

    @classmethod
    def from_pretrained(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        *,
        enable_mixed_precision_training: bool = True,
        arch_specifier: Literal["linear", "fused-gelu-mlp", "gelu-mlp"] = "gelu-mlp",
        freeze_weights: bool = True,
        **kwargs,
    ) -> "PrismaticVLM":
        """Initialize a PrismaticVLM from a pretrained checkpoint.

        Loads model weights from a checkpoint file and optionally freezes all weights
        for inference. The checkpoint is loaded to CPU first to avoid device conflicts.

        Args:
            pretrained_checkpoint (Path): Path to the pretrained model checkpoint.
            model_id (str): Unique identifier for this model instance.
            vision_backbone (VisionBackbone): Pre-trained vision encoder.
            llm_backbone (LLMBackbone): Pre-trained language model backbone.
            enable_mixed_precision_training (bool, optional): Whether to enable
                mixed precision training. Defaults to True.
            arch_specifier (Literal["linear", "fused-gelu-mlp", "gelu-mlp"], optional):
                Architecture type for the projection layer. Defaults to "gelu-mlp".
            freeze_weights (bool, optional): Whether to freeze all model weights
                after loading. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            PrismaticVLM: Initialized model with loaded pretrained weights.

        Raises:
            ValueError: If required checkpoint keys are missing.

        """
        vlm = cls(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )

        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        if "projector" not in model_state_dict or "llm_backbone" not in model_state_dict:
            msg = "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"
            raise ValueError(msg)

        vlm.projector.load_state_dict(model_state_dict["projector"])
        vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
        if "vision_backbone" in model_state_dict:
            vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(requires_grad=False)
            vlm.eval()

        return vlm

    def load_from_checkpoint(self, stage: str, run_dir: Path, pretrained_checkpoint: Path | None = None) -> None:
        """Load weights from checkpoint based on training stage.

        Handles stage-specific checkpoint loading logic. For alignment stage, no
        pretrained weights are needed. For fine-tuning stages, loads projector
        weights from alignment stage checkpoints.

        Args:
            stage (str): Training stage ("align", "finetune", or "full-finetune").
            run_dir (Path): Directory containing training run artifacts.
            pretrained_checkpoint (Path | None, optional): Specific checkpoint path.
                If None, automatically discovers checkpoint from run directory.

        Raises:
            ValueError: If stage is not supported.
            RuntimeError: If multiple or no valid pretrained directories exist.
            ValueError: If required checkpoint cannot be found.

        """
        if stage not in {"align", "finetune", "full-finetune"}:
            msg = f"Stage {stage} is not supported!"
            raise ValueError(msg)

        # If we're running a `no-align` architecture, we're good!
        if self.arch_specifier.startswith("no-align"):
            overwatch.info(
                f"PrismaticVLM with `{self.arch_specifier = }` does not require pretrained weights!",
                ctx_level=1,
            )
            return

        # Otherwise, handle stage-specific logic!
        if stage == "align":
            overwatch.info("Stage `align` does not require pretrained weights =>> Starting Training", ctx_level=1)
            return

        # Otherwise, load from `pretrained_checkpoint` or match on `run_dir` (s/+stage-finetune/+stage-align/g)
        overwatch.info("Stage `finetune` requires `align` pretrained weights", ctx_level=1)

        # Config specifies path to a checkpoint to load
        if pretrained_checkpoint is not None:
            overwatch.info(f"Loading from Provided Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projector.load_state_dict(model_state_dict["projector"])

            return

        # [Contract] If no `pretrained_checkpoint`, assume `align` lives in the run directory; string substitution!
        model, scale, _, seed = run_dir.name.split("+")
        align_dirs = [
            d
            for d in run_dir.parent.iterdir()
            if (d.name.startswith(f"{model}+{scale}") and d.name.endswith(f"+stage-align+{seed}"))
        ]
        if len(align_dirs) != 1:
            msg = "Multiple or No Valid Pretrained Directories Exist -- Double Check `runs`!"
            raise RuntimeError(msg)
        if (pretrained_checkpoint := (align_dirs[0] / "checkpoints" / "latest-checkpoint.pt")).exists():
            overwatch.info(f"Loading from Discovered Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projector.load_state_dict(model_state_dict["projector"])
        else:
            msg = f"Could not find valid `align` checkpoint at {pretrained_checkpoint}!"
            raise ValueError(msg)

    def freeze_backbones(self, stage: str) -> None:  # noqa: PLR0915
        """Set gradient requirements for model components based on training stage.

        Configures which parts of the model should be trainable depending on the
        training stage. Supports multiple training strategies from alignment to
        full fine-tuning.

        Args:
            stage (str): Training stage identifier. Supported stages:
                - "align": Only projector is trainable
                - "finetune"/"vla-train": LLM backbone and projector are trainable
                - "full-finetune"/"vla-full-train": All components are trainable
                - "last-layer-finetune"/"vla-last-layer-train": Only LLM last layer
                - "vla-sandwich-train": Vision backbone, projector, and LLM last layer

        Raises:
            ValueError: If the specified stage is not supported.

        """
        if stage == "align":
            self.vision_backbone.requires_grad_(requires_grad=False)
            self.llm_backbone.requires_grad_(requires_grad=False)
            self.projector.requires_grad_(requires_grad=True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage in {"finetune", "vla-train"}:
            self.vision_backbone.requires_grad_(requires_grad=False)
            self.llm_backbone.requires_grad_(requires_grad=True)
            self.projector.requires_grad_(requires_grad=True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector", "llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage in {"full-finetune", "vla-full-train"}:
            self.vision_backbone.dtype = torch.float32
            self.vision_backbone.requires_grad_(requires_grad=True)
            self.llm_backbone.requires_grad_(requires_grad=True)
            self.projector.requires_grad_(requires_grad=True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_backbone", "projector", "llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[TRAINABLE] 🔥 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage in {"last-layer-finetune", "vla-last-layer-train"}:
            self.vision_backbone.requires_grad_(requires_grad=False)
            self.projector.requires_grad_(requires_grad=False)
            self.llm_backbone.requires_grad_(requires_grad=False)

            # Unfreeze final LLM layer
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(requires_grad=True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            # fmt: off
            overwatch.info(f"[Frozen]                    🥶   =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen, except last layer] 🥶🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen]                    🥶   =>> Projector `{self.arch_specifier}`", ctx_level=1)
            # fmt: on

        elif stage in {"vla-sandwich-train"}:
            self.vision_backbone.dtype = torch.float32
            self.vision_backbone.requires_grad_(requires_grad=True)
            self.projector.requires_grad_(requires_grad=True)
            self.llm_backbone.requires_grad_(requires_grad=False)

            # Unfreeze final LLM layer
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(requires_grad=True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_backbone", "projector", "llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            # fmt: off
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen, except last layer] 🥶🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Projector `{self.arch_specifier}`", ctx_level=1)
            # fmt: on

        else:
            msg = f"Stage `{stage}` is not supported for LLaVa! Try < align | finetune >"
            raise ValueError(msg)

        overwatch.debug("##################################################")
        overwatch.debug("#####      Trainable Network Parameters:     #####")
        overwatch.debug("##################################################")
        for name, param in self.named_parameters():
            if param.requires_grad:
                overwatch.debug(name)

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
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward pass through the Prismatic VLM.

        Processes multimodal inputs by combining vision and language features.
        Handles both training and inference scenarios, including cached generation.

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
            multimodal_indices (torch.LongTensor | None, optional): Indices indicating
                which samples contain both image and text.

        Returns:
            CausalLMOutputWithPast: Model output including loss, logits, and optional states.

        Raises:
            ValueError: If required inputs are missing.
            RuntimeError: If forward call is invalid.

        """
        # Handle Inference (leverage cache, short-circuit on just LLM forward)
        if input_ids is None or labels is None or attention_mask is None:
            msg = "`input_ids`, `labels`, and `attention_mask` must be provided!"
            raise ValueError(msg)
        if input_ids.shape[1] == 1 and past_key_values is not None:
            # We're leveraging the cache, so just redirect to `self.llm_backbone` with `input_ids` and `past_key_values`
            output = self.llm_backbone(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return output
        if input_ids.shape[1] == 1 or pixel_values is None:
            msg = "Invalid `forward()` call!"
            raise RuntimeError(msg)
        # Handle Multimodal Indices is None --> pretend like the batch is fully multimodal (always image + text)!
        if multimodal_indices is None:
            multimodal_indices = cast(
                "torch.LongTensor",
                torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device),
            )
        # Handle Multimodal Indices is Empty (len == 0) --> simple unimodal forward
        elif len(multimodal_indices) == 0:
            return self.llm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # Run Visual Feature Extraction
        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(pixel_values, dict):
                patch_features = self.vision_backbone({k: pixel_values[k][multimodal_indices] for k in pixel_values})
            else:
                patch_features = self.vision_backbone(pixel_values[multimodal_indices])

        # Projection Logic :: [bsz, num_patches, llm_embed_dim] =>> num_patches = (2 *) (256 + 1) for ViT-L + CLS
        projected_patch_embeddings: torch.Tensor = self.projector(patch_features)
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # Get Input Embeddings from LLM Backbone :: [bsz, input_seq_len, llm_embed_dim]
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)

        # Build Multimodal Embeddings (and build resulting attention mask)
        multimodal_embeddings = torch.cat(
            [
                input_embeddings[multimodal_indices, :1, :],
                projected_patch_embeddings,
                input_embeddings[multimodal_indices, 1:, :],
            ],
            dim=1,
        )
        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [
                    attention_mask[multimodal_indices, :1],
                    cast("torch.Tensor", projected_patch_attention_mask),
                    attention_mask[multimodal_indices, 1:],
                ],
                dim=1,
            )

        # [Contract] We assume the first token of `labels` (associated with <BOS>) is already marked as "IGNORE"
        #   => We'll ignore the per-token outputs for each of the patch embeddings as well!
        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat(
                [labels[multimodal_indices, :1], projected_patch_labels, labels[multimodal_indices, 1:]],
                dim=1,
            )

        # === Add Unimodal Handling ===

        # Create Fused Embeddings, Attention Mask, and Labels by Merging with "unimodal" Inputs (if applicable)
        unimodal_indices = torch.tensor(
            [idx for idx in range(len(input_ids)) if idx not in multimodal_indices],
            dtype=torch.long,
            device=multimodal_indices.device,
        )

        # No "unimodal" data --> Fused == Multimodal
        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_attention_mask = multimodal_attention_mask
            fused_labels = multimodal_labels

        else:
            # Otherwise --> Merge w/ unimodal data

            # This doesn't matter --> but in the "normal" case this is the embedding of the <PAD> token
            #   => NOTE :: Verified that `zeros/randn/empty/<PAD> embedding` all return the same result!
            unimodal_embeddings_pad = torch.zeros(
                (len(unimodal_indices), projected_patch_embeddings.shape[1], input_embeddings.shape[2]),
                dtype=input_embeddings.dtype,
                device=input_embeddings.device,
            )
            unimodal_attention_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                False,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            unimodal_labels_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )

            unimodal_embeddings = torch.cat([input_embeddings[unimodal_indices], unimodal_embeddings_pad], dim=1)
            unimodal_attention_mask = torch.cat([attention_mask[unimodal_indices], unimodal_attention_pad], dim=1)
            unimodal_labels = torch.cat([labels[unimodal_indices], unimodal_labels_pad], dim=1)

            # Create "Fused" Tensors by Stacking Multimodal & Unimodal
            fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])
            fused_attention_mask = torch.vstack([multimodal_attention_mask, unimodal_attention_mask])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])

        # Run LLM Forward --> returns CausalLMOutputWithPast!
        return self.llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # === GenerationMixin Methods ===
    #   => Note: The following methods override the functionality of `transformers.GenerationMixin`; these expect the
    #            contract in each of the function signatures, and also expect our `forward` function to roughly take
    #            the same arguments as the underlying LLM (see `LlamaModelForCausalLM` as an example)
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        **kwargs: torch.Tensor,
    ) -> dict:
        """Prepare inputs for text generation.

        Formats inputs for the GenerationMixin interface, handling caching and
        multimodal inputs appropriately for incremental generation.

        Args:
            input_ids (torch.LongTensor): Token IDs for text input.
            attention_mask (torch.Tensor | None, optional): Attention mask for input.
            pixel_values (torch.FloatTensor | None, optional): Pixel values for images.
            inputs_embeds (torch.FloatTensor | None, optional): Input embeddings.
            past_key_values (list[torch.FloatTensor] | None, optional): Cached
                key-value pairs from previous generation steps.
            use_cache (bool | None, optional): Whether to use caching.
            **kwargs (torch.Tensor): Additional keyword arguments.

        Returns:
            dict: Formatted model inputs ready for generation.

        """
        if past_key_values:
            input_ids = input_ids[:, -1:]
        # model_inputs: dict[str, torch.Tensor | list[torch.Tensor] | None] = {}
        model_inputs = {}
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs.update({"inputs_embeds": inputs_embeds})
        else:
            model_inputs.update({"input_ids": input_ids})

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            },
        )

        return model_inputs

    @torch.inference_mode()
    def generate_batch(
        self,
        pixel_values: torch.Tensor | dict[str, torch.Tensor],
        texts: list[str],
        return_string_probabilities: list[str] | None = None,
        **kwargs,
    ) -> list[str] | list[list[float]]:
        """Generate text responses for a batch of multimodal inputs.

        Processes a batch of images and text prompts to generate text responses.
        Optionally returns probabilities for specific output strings.

        Args:
            pixel_values (torch.Tensor | dict[str, torch.Tensor]): Image pixel values
                or dictionary of pixel values for different image types.
            texts (list[str]): List of text prompts to process.
            return_string_probabilities (list[str] | None, optional): List of strings
                for which to return generation probabilities. If None, returns
                generated text. Defaults to None.
            **kwargs: Additional generation parameters.

        Returns:
            list[str] | list[list[float]]: Generated text strings if
                return_string_probabilities is None, otherwise lists of probabilities
                for each specified string.

        Raises:
            TypeError: If pixel_values type is not supported.

        """
        # For now, only support generation with a batch size of 1 for simplicity
        tokenizer = self.llm_backbone.tokenizer

        # Prepare Inputs
        batch_input_ids = [
            tokenizer(text, truncation=True, return_tensors="pt").input_ids.to(self.device) for text in texts
        ]
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            msg = f"Unsupported `pixel_values` type = {type(pixel_values)}"
            raise TypeError(msg)

        # Create Output Lists
        gen_texts, gen_probabilities = [], []

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            for idx, input_ids in enumerate(batch_input_ids):
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values[idx]
                elif isinstance(pixel_values, dict):
                    pixel_values = {k: pixel_values[k][idx] for k in pixel_values}
                else:
                    msg = f"Unsupported `pixel_values` type = {type(pixel_values)}"
                    raise TypeError(msg)

                # Handle `return_string_probabilities`
                if return_string_probabilities is None:
                    full_out_ids = super().generate(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
                    gen_ids = full_out_ids[0, input_ids.shape[1] :]

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                else:
                    full_out_dict = cast(
                        "GenerateOutput",
                        super().generate(
                            input_ids=input_ids,
                            pixel_values=pixel_values,
                            output_scores=True,
                            return_dict_in_generate=True,
                            **kwargs,
                        ),
                    )

                    # Generation pattern should usually be [TOKEN] <EOS> for True/False and Yes/No Generations
                    gen_ids = full_out_dict.sequences[0, input_ids.shape[1] :]

                    # [Debug] Verify that the first token generated is in `self.string2idx.values()`
                    # assert gen_ids[0] in self.string2idx.values(), "Generated ID not in mapping!"

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                    # Get all token probabilities --> softmax over logits
                    token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)  # pyright: ignore[reportOptionalSubscript]

                    # Get *normalized* probabilities for all values in `return_token_probabilities`
                    slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                    string_probs_unnormalized = token_probs[slice_idxs]
                    string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                    gen_probabilities.append(string_probs.cpu().numpy().tolist())

        return gen_texts if return_string_probabilities is None else gen_probabilities

    @override
    @torch.inference_mode()
    def generate(self, image: torch.Tensor, prompt_text: str, **kwargs) -> str:
        """Generate text response for a single image-text pair.

        Processes a single image and text prompt to generate a text response.
        This method overrides the base VLM generate method for Prismatic-specific
        implementation.

        Args:
            image (torch.Tensor): Input image tensor.
            prompt_text (str): Text prompt for generation.
            **kwargs: Additional generation parameters passed to the underlying
                generation method.

        Returns:
            str: Generated text response.

        Raises:
            TypeError: If pixel_values type is not supported.

        """
        # For now, only support generation with a batch size of 1 for simplicity
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            msg = f"Unsupported `pixel_values` type = {type(pixel_values)}"
            raise TypeError(msg)

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super().generate(
                input_ids=input_ids,            # Shape: [1, seq]
                pixel_values=pixel_values,      # Shape: [1, 3, res, res] or Dict[str, Shape[1, 3, res, res]]
                **kwargs,
            )
            # fmt: on

        generated_text = tokenizer.decode(generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()

        return generated_text

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector`
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector},
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    def get_prompt_builder(self, system_prompt: str | None = None) -> PromptBuilder:
        prompt_initializer: type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)
