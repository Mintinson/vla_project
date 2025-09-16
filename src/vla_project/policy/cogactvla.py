import os
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from einops import einsum
from jaxtyping import Float
from torch import Tensor, nn
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from torch.nn.utils.rnn import pad_sequence
from transformers import LlamaTokenizerFast

from vla_project.action_model.actionmodel import ActionModel
from vla_project.action_model.models import DiT
from vla_project.models.backbones.llm.llm_base import LLMBackbone
from vla_project.models.backbones.vision.vision_base import VisionBackbone
from vla_project.models.vlms.prismatic import PrismaticVLM
from vla_project.overwatch.overwatch import initialize_overwatch
from vla_project.utils.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

# from torch.nn.attention.flex_attention import flex_attention
mse_weight = float(os.environ["MSE"].upper())
balance_weight = float(os.environ["BALANCE"].upper())
KD_weight = float(os.environ["KD"].upper())

print(f"mse_weight : {mse_weight}")
print(f"balance_weight : {balance_weight}")
print(f"KD_weight : {KD_weight}")

overwatch = initialize_overwatch(__name__)


def reverse_kl_loss(
    tensor1: Float[Tensor, "b l d"],
    tensor2: Float[Tensor, "b l d"],
    dim: int = -1,
) -> Float[Tensor, "b l d"]:
    """Compute reverse KL divergence between two tensors.

    Calculates KL(q||p) where p = softmax(tensor1) and q = softmax(tensor2).
    This is used for knowledge distillation between teacher and student models.

    Args:
        tensor1 (Float[Tensor, "b l d"]): First tensor (teacher logits).
        tensor2 (Float[Tensor, "b l d"]): Second tensor (student logits).
        dim (int, optional): Dimension to apply softmax. Defaults to -1.

    Returns:
        Float[Tensor, "b l d"]: Reverse KL divergence values.

    Raises:
        AssertionError: If tensor shapes don't match.

    """
    assert tensor1.shape == tensor2.shape, "两个输入张量的形状必须一致"

    p = F.softmax(tensor1, dim=dim)
    q = F.softmax(tensor2, dim=dim)

    return q * (torch.log(q + 1e-8) - torch.log(p + 1e-8))


def cross_attention(
    query: Float[Tensor, "b lq d"],
    key: Float[Tensor, "b lk d"],
    value: Float[Tensor, "b lk d"],
) -> tuple[Float[Tensor, "b lq d"], Float[Tensor, "b lq lk"]]:
    """Perform cross-attention between query, key, and value tensors.

    Computes attention weights using scaled dot-product attention and applies
    them to the value tensor to produce attended outputs.

    Args:
        query (Float[Tensor, "b lq d"]): Query tensor of shape [batch, query_len, embed_dim].
        key (Float[Tensor, "b lk d"]): Key tensor of shape [batch, key_len, embed_dim].
        value (Float[Tensor, "b lk d"]): Value tensor of shape [batch, key_len, embed_dim].

    Returns:
        tuple[Float[Tensor, "b lq d"], Float[Tensor, "b lq lk"]]:
            - Attention output of shape [batch, query_len, embed_dim]
            - Attention weights of shape [batch, query_len, key_len]

    """
    d_k = query.shape[-1]  # 获取 embed_dim
    # QK^T / sqrt(d_k)
    attn_scores = einsum(query, key, "b lq d, b lk d -> b lq lk") / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # oftmax normalized attention weights
    attn_weights = F.softmax(attn_scores, dim=-1)
    # print(f"attn_weights: {attn_weights.shape}")
    # attention output
    attn_output = einsum(attn_weights, value, "b lq lk, b lk d -> b lq d")

    return attn_output, attn_weights


class CogACT(nn.Module):
    """Cognitive Action Transformer for Vision-Language-Action learning.

    This model combines a Vision-Language Model (VLM) with a diffusion-based action
    prediction model. It learns to predict robot actions by first extracting cognitive
    features from visual and language inputs, then using these features to condition
    a diffusion model for action generation.

    Attributes:
        action_model (ActionModel): Diffusion model for action prediction.
        vlm (PrismaticVLM): Vision-language model for feature extraction.
        future_action_window_size (int): Number of future actions to predict.
        past_action_window_size (int): Number of past actions to condition on.
        use_ema (bool): Whether to use exponential moving averages.
        ema_diffusion (ActionModel): EMA version of action model (if enabled).
        all_module_keys (list[str]): Keys for all model modules.
        norm_stats (dict): Normalization statistics for actions.

    """

    def __init__(
        self,
        *,
        vlm: PrismaticVLM,
        action_model_type: Literal["DiT-S", "DiT-B", "DiT-L"] = "DiT-B",
        norm_stats: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        token_size: int = 4096,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        use_ema: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the CogACT model.

        Args:
            vlm (PrismaticVLM): Pre-trained vision-language model.
            action_model_type (Literal["DiT-S", "DiT-B", "DiT-L"], optional):
                Size of the diffusion transformer. Defaults to "DiT-B".
            token_size (int, optional): Size of feature tokens. Defaults to 4096.
            action_dim (int, optional): Dimension of action space. Defaults to 7.
            future_action_window_size (int, optional): Number of future actions
                to predict. Defaults to 15.
            past_action_window_size (int, optional): Number of past actions
                to condition on. Defaults to 0.
            use_ema (bool, optional): Whether to use exponential moving averages.
                Defaults to False.
            norm_stats (dict | None, optional): Action normalization statistics.
                Defaults to None.
            **kwargs: Additional keyword arguments.

        """
        super().__init__()
        self.action_model = ActionModel(
            model_type=action_model_type,
            token_size=token_size,
            in_channels=action_dim,
            future_action_window_size=future_action_window_size,
            past_action_window_size=past_action_window_size,
        )
        self.vlm = vlm
        self.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size
        self.use_ema = use_ema
        if self.use_ema:
            self.ema_diffusion = deepcopy(self.action_model)
            self.ema_diffusion.requires_grad_(requires_grad=False)
            self.all_module_keys = ["action_model", "ema_diffusion"]
        else:
            self.all_module_keys = ["action_model"]
        # for module_keys in self.vlm.all_module_keys:
        #     self.all_module_keys.append("vlm." + module_keys)
        self.all_module_keys.extend(["vlm." + key for key in self.vlm.all_module_keys])

        # Diffusion head is always trainable
        self._trainable_module_keys = ["action_model"]
        self.norm_stats = norm_stats

    @property
    def trainable_module_keys(self) -> list[str]:
        """Get keys for trainable modules.

        Returns:
            list[str]: List of module keys that should be trained.

        """
        keys = ["vlm." + key for key in self.vlm.trainable_module_keys]
        keys += self._trainable_module_keys
        return keys

    @property
    def llm_backbone(self) -> LLMBackbone:
        """Get the language model backbone.

        Returns:
            LLMBackbone: The LLM backbone from the VLM.

        """
        return self.vlm.llm_backbone

    @property
    def vision_backbone(self) -> VisionBackbone:
        """Get the vision model backbone.

        Returns:
            VisionBackbone: The vision backbone from the VLM.

        """
        return self.vlm.vision_backbone

    def freeze_backbones(self, stage: str):
        """Freeze model backbones based on training stage.

        Args:
            stage (str): Training stage identifier (e.g., "align", "finetune").

        """
        self.vlm.freeze_backbones(stage)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        actions: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        *,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        repeated_diffusion_steps: int = 4,
        action_masks: torch.Tensor | None = None,
    ) -> tuple:
        """Forward pass through the CogACT model.

        Processes multimodal inputs through the VLM to extract cognitive features,
        then trains the action model using these features and ground truth actions.
        Combines multiple loss components including language modeling, knowledge
        distillation, and action prediction losses.

        Args:
            input_ids (torch.LongTensor | None, optional): Token IDs for text input.
            attention_mask (torch.Tensor | None, optional): Attention mask for input.
            pixel_values (torch.FloatTensor | None, optional): Image pixel values.
            labels (torch.LongTensor | None, optional): Target labels for language modeling.
            actions (torch.FloatTensor | None, optional): Ground truth action sequences.
            inputs_embeds (torch.FloatTensor | None, optional): Input embeddings.
            past_key_values (list[torch.FloatTensor] | None, optional): Cached key-values.
            use_cache (bool | None, optional): Whether to use caching.
            output_attentions (bool | None, optional): Whether to output attention weights.
            output_hidden_states (bool | None, optional): Whether to output hidden states.
            return_dict (bool | None, optional): Whether to return dict format.
            repeated_diffusion_steps (int, optional): Number of diffusion training steps.
                Defaults to 4.
            action_masks (torch.Tensor | None, optional): Masks for action sequences.

        Returns:
            tuple: Combined loss and VLM output.

        Raises:
            ValueError: If vision backbone is not found.
            AssertionError: If attention_mask is None.

        """
        output, teacher_output = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        balance_loss: torch.Tensor = output.loss
        # extract the last hidden state and the learnable EOS token feature
        last_hidden = output.hidden_states[-1]
        # import pdb; pdb.set_trace()
        teacher_last_hidden = teacher_output.hidden_states[-1]
        # print(f"teacher_last_hidden: {teacher_last_hidden.shape}")
        # print(f"last_hidden : {last_hidden.shape}")  # 按理来说是把这两个做mseloss
        MSE_loss = torch.nn.functional.mse_loss(teacher_last_hidden, last_hidden, reduction="none")
        KL_loss = reverse_kl_loss(teacher_last_hidden, last_hidden)
        KD_loss = (1 - mse_weight) * KL_loss + mse_weight * MSE_loss
        # KD_loss = torch.nn.functional.mse_loss(teacher_last_hidden, last_hidden, reduction='none')
        # print(f"KD_loss : {KD_loss}")
        # extract the visual token number
        if self.vlm.vision_backbone.featurizer is not None:
            num_patch = self.vlm.vision_backbone.featurizer.patch_embed.num_patches
        elif (
            hasattr(self.vlm.vision_backbone, "siglip_featurizer")
            and self.vlm.vision_backbone.siglip_featurizer is not None
        ):
            num_patch = self.vlm.vision_backbone.siglip_featurizer.patch_embed.num_patches
        else:
            msg = "No vision backbone found"
            raise ValueError(msg)

        last_hidden = last_hidden[:, num_patch:]
        teacher_last_hidden = teacher_last_hidden[:, num_patch:]
        # extract the cognition feature
        assert attention_mask is not None, "attention_mask cannot be None"
        cumulative_sum = attention_mask.cumsum(dim=1)
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden.size(-1))
        cognition_features = last_hidden.gather(1, expanded_indices.unsqueeze(1))  # [B, 1, D]
        teacher_cognition_features = teacher_last_hidden.gather(1, expanded_indices.unsqueeze(1))  # [B, 1, D]
        _, teacher_attn_weights = cross_attention(
            teacher_cognition_features,
            output.hidden_states[-1],
            output.hidden_states[-1],
        )
        _, student_attn_weights = cross_attention(
            cognition_features,
            output.hidden_states[-1],
            output.hidden_states[-1],
        )
        attn_weights = teacher_attn_weights * student_attn_weights  # [64, 1, 292]
        # print(f"attn_weights: {attn_weights.shape}")
        KD_loss = (attn_weights.transpose(2, 1) * KD_loss).mean()
        # okk那就是两个[B, 1, D]，然后呢，teacher 的[B, 1, D]对student的[B, 292, D]做一个attention,
        # student的[B, 1, D]对student的[B, n, D]做cross attn我得理解是[B, 1, D]做q， [B, n, D]做kv,但是这里没有linear层映射，是直接做的对吧 okk
        # 那现在有两个[B, 1, D]的attn_weight，这两个对应元素相乘，得到最终的[B, 1, D]的attn_weight.shape==[B, 1, D]将其乘到KD_loss.shape==[1]
        # 等下，乘完得到的还是[B, 1, D]的tensor, 乘到是个什么操作？view(-1)然后？那现在得到的是[B*1*D]的tensor，然后取mean？
        # 这个attn我之前写过，我先跑起来吧，明早再写也行， 可
        actions_history = actions[:, 0 : self.past_action_window_size, :]
        actions_future = actions[:, -(self.future_action_window_size + 1) :, :]

        # Repeat 'actions' 'repeated_diffusion_steps' times, resulting in [repeated_diffusion_steps*B, T, D]
        actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
        actions_history_repeated = actions_history.repeat(repeated_diffusion_steps, 1, 1)
        cognition_features_repeated = cognition_features.repeat(
            repeated_diffusion_steps,
            1,
            1,
        )  # [repeated_diffusion_steps*B, 1, D]

        # Action model forward and compute loss
        loss = self.action_model.loss(actions_repeated, cognition_features_repeated)
        return loss + balance_loss * balance_weight + KD_loss * KD_weight, output

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Get FSDP wrapping policy for distributed training.

        Returns a combined policy that wraps vision backbone, LLM backbone,
        and action model components appropriately for FSDP.

        Returns:
            Callable: FSDP wrapping policy function.

        """
        vision_fsdp_wrapping_policy = self.vlm.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.vlm.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector` and DiT
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector, DiT},
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

    def load_ema_to_weights(self):
        """Load exponential moving average weights to the action model.

        Transfers the EMA state dict to the main action model weights and
        deletes the EMA model to save memory.
        """
        if self.use_ema:
            self.action_model.load_state_dict(self.ema_diffusion.state_dict())
            del self.ema_diffusion

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: Literal["linear", "fused-gelu-mlp", "gelu-mlp"] = "gelu-mlp",
        freeze_weights: bool = True,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        action_model_type: Literal["DiT-S", "DiT-B", "DiT-L"] = "DiT-B",
        use_ema: bool = False,
        norm_stats=None,
        **kwargs,
    ):
        """Load a CogACT model from a pretrained checkpoint.

        Creates a CogACT model by first loading a PrismaticVLM from checkpoint,
        then initializing the action model component and loading its weights
        if available in the checkpoint.

        Args:
            pretrained_checkpoint (Path): Path to the pretrained model checkpoint.
            model_id (str): Model identifier.
            vision_backbone (VisionBackbone): Vision encoder backbone.
            llm_backbone (LLMBackbone): Language model backbone.
            enable_mixed_precision_training (bool, optional): Whether to enable
                mixed precision. Defaults to True.
            arch_specifier (Literal["linear", "fused-gelu-mlp", "gelu-mlp"], optional):
                Architecture type for projection. Defaults to "gelu-mlp".
            freeze_weights (bool, optional): Whether to freeze VLM weights.
                Defaults to True.
            action_dim (int, optional): Action space dimension. Defaults to 7.
            future_action_window_size (int, optional): Future action window size.
                Defaults to 15.
            past_action_window_size (int, optional): Past action window size.
                Defaults to 0.
            action_model_type (Literal["DiT-S", "DiT-B", "DiT-L"], optional):
                Action model size. Defaults to "DiT-B".
            use_ema (bool, optional): Whether to use EMA. Defaults to False.
            norm_stats: Action normalization statistics. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            CogACT: Initialized CogACT model with loaded weights.

        Raises:
            RuntimeError: If required checkpoint keys are missing.

        """
        # load VLM backbone, borrowed from PrismaticVLM
        vlm = PrismaticVLM(
            model_id=model_id,
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )

        # Load from Checkpoint (Custom --> should load project and llm weights)
        model_state_dict: dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        if "projector" not in model_state_dict and "llm_backbone" not in model_state_dict:
            msg = "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"
            raise RuntimeError(msg)
        vlm.projector.load_state_dict(model_state_dict["projector"])
        vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"], strict=False)

        if "vision_backbone" in model_state_dict:
            vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(requires_grad=False)
            vlm.eval()

        cogact = CogACT(
            vlm,
            token_size=vlm.llm_backbone.llm.lm_head.in_features,
            action_dim=action_dim,
            future_action_window_size=future_action_window_size,
            past_action_window_size=past_action_window_size,
            action_model_type=action_model_type,
            use_ema=use_ema,
            norm_stats=norm_stats,
        )
        # Load ActionModel from Checkpoint
        if "action_model" in model_state_dict:
            cogact.action_model.load_state_dict(model_state_dict["action_model"])
            if "ema_diffusion" in model_state_dict and use_ema:
                cogact.ema_diffusion.load_state_dict(model_state_dict["ema_diffusion"])
            elif use_ema:
                cogact.ema_diffusion.load_state_dict(model_state_dict["action_model"])
        else:
            overwatch.warning("No ActionModel found in the pretrained checkpoint. Initializing a new one.")
        return cogact

    @torch.inference_mode()
    def predict_action(
        self,
        image,
        instruction: str,
        unnorm_key: str | None = None,
        cfg_scale: float = 1.5,
        *,
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        **kwargs: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict robot actions for a single image-instruction pair.

        Processes an image and instruction through the VLM to extract cognitive
        features, then uses the action model to generate action predictions via
        diffusion sampling.

        Args:
            image: Input image (PIL Image or tensor).
            instruction (str): Natural language instruction.
            unnorm_key (str | None, optional): Key for action normalization stats.
                Defaults to None.
            cfg_scale (float, optional): Classifier-free guidance scale.
                Defaults to 1.5.
            use_ddim (bool, optional): Whether to use DDIM sampling. Defaults to False.
            num_ddim_steps (int, optional): Number of DDIM steps. Defaults to 5.
            **kwargs (str): Additional generation arguments.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - Unnormalized actions of shape [sequence_length, action_dim]
                - Normalized actions of shape [sequence_length, action_dim]

        Raises:
            TypeError: If tokenizer or pixel_values type is unsupported.
            ValueError: If batch size is not 1 or cognition features have wrong shape.

        """
        image_transform = self.vlm.vision_backbone.image_transform
        tokenizer = self.vlm.llm_backbone.tokenizer

        # Build VLA Prompt
        prompt_builder = self.vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            # Note: We need to add this special empty token ('') after the colon (':') token in "ASSISTANT:"
            #       insert it to match the inputs seen at training time. The empty token is at index 29871.
            #       We also need to add the special cognition token at index 2 (i.e. the EOS token).
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.tensor([29871, 2]).long(), dim=0).to(self.vlm.device)),
                dim=1,
            )
        else:
            msg = f"Unsupported tokenizer type = {type(tokenizer)}"
            raise TypeError(msg)

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.vlm.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.vlm.device) for k, v in pixel_values.items()}
        else:
            msg = f"Unsupported `pixel_values` type = {type(pixel_values)}"
            raise TypeError(msg)

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.vlm.llm_backbone.half_precision_dtype

        # Generate cognition feature through vlm
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
            output = super(PrismaticVLM, self.vlm).generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs,
            )

        cognition_features = output.hidden_states[0][-1][:, -1, :]
        if cognition_features.shape[:2] != (1, 4096):
            msg = "Batch size must be 1 for action prediction"
            raise ValueError(msg)

        using_cfg = cfg_scale > 1.0

        model_dtype = next(self.action_model.net.parameters()).dtype
        B = cognition_features.shape[0]

        cognition_features = cognition_features.unsqueeze(1).to(model_dtype)  # [B, 1, D]

        # Sample random noise
        noise = torch.randn(B, self.future_action_window_size + 1, self.action_model.in_channels)

        # Setup classifier-free guidance
        if using_cfg:
            noise = torch.cat([noise, noise], dim=0)  # (2*B, T+1, D)
            uncondition = self.action_model.net.z_embedder.uncondition  # (D,)
            uncondition = uncondition.unsqueeze(0)  # [1, D]
            uncondition = uncondition.expand(B, 1, -1)  # [B, 1, D]
            z = torch.cat([cognition_features, uncondition], 0)  # (2*B, 1, D)
            cfg_scale = cfg_scale
            model_kwargs = {"z": z, "cfg_scale": cfg_scale}
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = {"z": cognition_features}
            sample_fn = self.action_model.net.forward

        # DDIM Sampling
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(
                sample_fn,
                noise.shape,
                noise=noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=cognition_features.device,
                eta=0.0,
            )
        else:
            # DDPM Sampling
            samples = self.action_model.diffusion.p_sample_loop(
                sample_fn,
                noise.shape,
                noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=cognition_features.device,
            )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        normalized_actions = samples[0].cpu().numpy()

        # Un-normalize Actions
        action_norm_stats: dict[str, np.ndarray] = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions, normalized_actions

    @torch.inference_mode()
    def predict_action_batch(
        self,
        image,
        instruction: list[str],
        unnorm_key: str | None = None,
        cfg_scale: float = 1.5,
        use_ddim: bool = False,
        num_ddim_steps: int = 10,
        **kwargs: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict robot actions for a batch of image-instruction pairs.

        Processes multiple images and instructions through the VLM to extract
        cognitive features, then uses the action model to generate action
        predictions for the entire batch.

        Args:
            image: Batch of input images.
            instruction (list[str]): List of natural language instructions.
            unnorm_key (str | None, optional): Key for action normalization stats.
                Defaults to None.
            cfg_scale (float, optional): Classifier-free guidance scale.
                Defaults to 1.5.
            use_ddim (bool, optional): Whether to use DDIM sampling. Defaults to False.
            num_ddim_steps (int, optional): Number of DDIM steps. Defaults to 10.
            **kwargs (str): Additional generation arguments.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - Unnormalized actions of shape [batch_size, sequence_length, action_dim]
                - Normalized actions of shape [batch_size, sequence_length, action_dim]

        Raises:
            TypeError: If tokenizer or pixel_values type is unsupported.
            ValueError: If vision backbone is not found or shapes are incorrect.

        """
        image_transform, tokenizer = self.vlm.vision_backbone.image_transform, self.vlm.llm_backbone.tokenizer

        input_ids = []
        pixel_values = []

        # Build VLA Prompt
        B = len(image)

        if isinstance(tokenizer, LlamaTokenizerFast):
            pass
        else:
            msg = f"Unsupported `tokenizer` type = {type(tokenizer)}"
            raise TypeError(msg)

        for id in range(B):
            prompt_builder = self.vlm.get_prompt_builder()
            prompt_builder.add_turn(
                role="human",
                message=f"What action should the robot take to {instruction[id].lower()}?",
            )
            prompt_text = prompt_builder.get_prompt()
            # Prepare Inputs
            single_input_ids = (
                tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device).squeeze(0)
            )
            # Note: We need to add this special empty token ('') after the colon (':') token in "ASSISTANT:"
            #       insert it to match the inputs seen at training time. The empty token is at index 29871.
            #       We also need to add the special cognition token at index 2 (i.e. the EOS token).
            single_input_ids = torch.cat(
                (single_input_ids, torch.Tensor([29871, 2]).long().to(self.vlm.device)),
                dim=0,
            )  # [seq]

            input_ids.append(single_input_ids)
            # Preprocess Image
            pixel_values.append(image_transform(image[id]))

        # Padding
        padding_side = "right"
        # For now, we only support Tokenizers with `padding_side = "right"`
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert padding_side == "right", f"Invalid Tokenizer `{padding_side = }`"

        model_max_length = tokenizer.model_max_length
        pad_token_id = tokenizer.pad_token_id
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)

        # Truncate (if necessary)
        input_ids = input_ids[:, :model_max_length]
        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(pad_token_id)

        # Preprocess Image
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values).to(self.vlm.device)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]).to(self.vlm.device)
                for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.vlm.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
            # fmt: off
            output = super(PrismaticVLM, self.vlm).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                attention_mask = attention_mask,
                **kwargs,
            )
            # fmt: on

        # Extract cognition feature
        if self.vlm.vision_backbone.featurizer is not None:
            num_patch = self.vlm.vision_backbone.featurizer.patch_embed.num_patches
        elif (
            hasattr(self.vlm.vision_backbone, "siglip_featurizer")
            and self.vlm.vision_backbone.siglip_featurizer is not None
        ):
            num_patch = self.vlm.vision_backbone.siglip_featurizer.patch_embed.num_patches
        else:
            msg = "No vision backbone found"
            raise ValueError(msg)

        last_hidden = output.hidden_states[0][-1]
        last_hidden = last_hidden[:, num_patch:]

        cumulative_sum = attention_mask.cumsum(dim=1)
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden.size(-1))
        cognition_features = last_hidden.gather(1, expanded_indices.unsqueeze(1)).squeeze(1)  # [B, D]

        assert (cognition_features.shape[0], cognition_features.shape[1]) == (B, 4096), (
            "Batch size must be B for action prediction"
        )
        using_cfg = cfg_scale > 1.0

        model_dtype = next(self.action_model.net.parameters()).dtype

        B = cognition_features.shape[0]

        cognition_features = cognition_features.unsqueeze(1).to(model_dtype)  # [B, 1, D]

        # Sample random noise
        noise = torch.randn(
            B,
            self.future_action_window_size + 1,
            self.action_model.in_channels,
            device=cognition_features.device,
        ).to(model_dtype)  # [B, T, D]
        # Setup classifier-free guidance:
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  # [1, D]
            uncondition = uncondition.expand(B, 1, -1)  # [B, 1, D]
            z = torch.cat([cognition_features, uncondition], 0)
            cfg_scale = cfg_scale
            model_kwargs = {"z": z, "cfg_scale": cfg_scale}
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = {"z": cognition_features}
            sample_fn = self.action_model.net.forward

        # DDIM Sampling
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(
                sample_fn,
                noise.shape,
                noise,
                clip_denoised=False,  # False, try to set True
                model_kwargs=model_kwargs,
                progress=False,
                device=cognition_features.device,
                eta=0.0,
            )
        else:
            # DDPM Sampling
            samples = self.action_model.diffusion.p_sample_loop(
                sample_fn,
                noise.shape,
                noise,
                clip_denoised=False,  # False, try to set True
                model_kwargs=model_kwargs,
                progress=False,
                device=cognition_features.device,
            )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        normalized_actions = samples.cpu().numpy()

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, :, 6] = np.where(normalized_actions[:, :, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions, normalized_actions

    def get_action_stats(self, unnorm_key: str | None = None):
        """Get action normalization statistics for the specified dataset.

        Args:
            unnorm_key: Key identifying which dataset's statistics to use.
                If None, uses the only available dataset (if there's only one).

        Returns:
            dict: Dictionary containing action normalization statistics including
                'q01', 'q99', and optionally 'mask' for the specified dataset.

        """
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    @staticmethod
    def _check_unnorm_key(norm_stats: dict[str, Any], unnorm_key: str | None) -> str:
        """Validate and resolve the unnormalization key.

        Args:
            norm_stats: Dictionary of normalization statistics by dataset.
            unnorm_key: Key for dataset statistics, or None to auto-select.

        Returns:
            str: Validated unnormalization key.

        Raises:
            AssertionError: If no key is provided but multiple datasets exist,
                or if the provided key is not found in available statistics.

        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
