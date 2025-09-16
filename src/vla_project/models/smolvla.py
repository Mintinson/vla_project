import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Literal, Protocol, Self, cast

import torch
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel
from torch import Tensor, nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    PretrainedConfig,
    PreTrainedModel,
    SmolVLMForConditionalGeneration,
    SmolVLMModel,
)

if TYPE_CHECKING:
    from transformers.modeling_outputs import BaseModelOutput


class AttentionInterface(Protocol):
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear
    head_dim: int


class TransformerBlockLike(Protocol):
    input_layernorm: nn.Module
    post_attention_layernorm: nn.Module
    self_attn: AttentionInterface
    mlp: nn.Module


class SmolVLMWithExpertConfig(BaseModel):
    model_id: str
    load_vlm_weight: bool = True
    train_expert_only: bool = True
    freeze_vision_encoder: bool = False
    attention_mode: Literal["self_attn", "cross_attn"] = "self_attn"
    num_expert_layers: int = -1
    num_vlm_layers: int = -1
    self_attn_every_n_layers: int = -1
    expert_width_multiplier: float = 0.5


# from


class SmolVLMWithExpertModel(nn.Module):
    def __init__(self, config: SmolVLMWithExpertConfig) -> None:
        super().__init__()
        self.config = config
        if self.config.load_vlm_weight:
            logger = logging.getLogger(__name__)
            logger.info("loading weight from %s", self.config.model_id)
            self.vlm: PreTrainedModel = AutoModelForImageTextToText.from_pretrained(
                config.model_id,
                device_map="auto",
                torch_dtype="bfloat16",
                low_cpu_mem_usage=True,
            )
            new_config: PretrainedConfig = self.vlm.config
        else:
            new_config = AutoConfig.from_pretrained(config.model_id)
            self.vlm = SmolVLMForConditionalGeneration(config=new_config)

        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)

        lm_expert_config = deepcopy(new_config.get_text_config())
        hidden_size = lm_expert_config.hidden_size
        lm_expert_config.hidden_size = int(
            hidden_size * config.expert_width_multiplier,
        )  # hidden_size // 2
        lm_expert_config.num_hidden_layers = self.num_vlm_layers

        if config.num_expert_layers > 0:
            assert (
                len(self.get_vlm_model().text_model.layers) % config.num_expert_layers == 0
            ), f"Number of layers in the VLM {
                len(self.get_vlm_model().text_model.layers)
            } are not multiple of num_expert_layers {config.num_expert_layers}"
            lm_expert_config.num_hidden_layers = config.num_expert_layers
        self.lm_expert = AutoModel.from_config(lm_expert_config)

        self.num_expert_layers = len(self.lm_expert.layers)

        self.processor = AutoProcessor.from_pretrained(config.model_id)

        self.self_attn_every_n_layers = config.self_attn_every_n_layers
        # if config.num_vlm_layers > 0:
        self.freeze_vision_encoder = config.freeze_vision_encoder
        self.trains_expert_only = config.train_expert_only

        self.num_attention_heads = new_config.text_config.num_attention_heads
        self.num_key_value_heads = new_config.text_config.num_key_value_heads

    def get_vlm_model(self) -> SmolVLMModel:
        return self.vlm.model

    def set_requires_grad(self) -> None:
        """Sets the `requires_grad` attribute for model parameters based on training configuration.

        This method freezes parts of the model (vision encoder, full VLM) according to the
        `freeze_vision_encoder` and `train_expert_only` flags. It also handles partial freezing
        of the VLM's text model for distributed training to avoid issues with unused parameters.
        """
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False
        if self.trains_expert_only:
            self.vlm.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False
        else:
            last_layers = [self.num_vlm_layers - 1]
            if (
                self.num_vlm_layers != self.num_expert_layers
                and self.num_vlm_layers % self.num_expert_layers == 0
            ):
                last_layers.append(self.num_vlm_layers - 2)
            frozen_layers = ["lm_head", "text_model.model.norm.weight"]
            for layer in last_layers:
                frozen_layers.append(f"text_model.model.layers.{layer}.")  # noqa: PERF401
            for name, params in self.vlm.named_parameters():
                if any(k in name for k in frozen_layers):
                    params.requires_grad = False
        # To avoid unused params issue with distributed training
        for name, params in self.lm_expert.named_parameters():
            if "lm_head" in name:
                params.requires_grad = False

    def train(self, mode: bool = True) -> Self:
        """Set the model in training mode.

        Args:
            mode (bool): whether to set training mode (True) or evaluation
                         mode (False). Default: True.

        """
        super().train(mode)

        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
        if self.trains_expert_only:
            self.vlm.eval()
        return self

    def embed_image(self, image: Float[Tensor, "bs c h w"]) -> Float[Tensor, "bs seq_len hidden"]:
        """Embeds a batch of images into a sequence of hidden states.

        This method processes a batch of images through the vision encoder of the
        Vision-Language Model (VLM). The output hidden states from the vision model are
        then passed through a connector layer to project them into the appropriate
        embedding space.

        Args:
            image (Float[Tensor, "bs c h w"]): A tensor representing a batch of
                images, where 'bs' is batch size, 'c' is channels, 'h' is height,
                and 'w' is width.

        Returns:
            Float[Tensor, "bs seq_len hidden"]: A tensor of embedded image features,
                where 'seq_len' is the sequence length of the image patches and
                'hidden' is the dimension of the hidden states.

        """
        patch_attention_mask = None
        # Get sequence from the vision encoder
        image_hidden_states = cast(
            "BaseModelOutput",
            self.get_vlm_model().vision_model(
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                patch_attention_mask=patch_attention_mask,
            ),
        ).last_hidden_state
        return self.get_vlm_model().connector(image_hidden_states)

    def embed_language_tokens(
        self,
        tokens: Int[Tensor, "bs seq_len"],
    ) -> Float[Tensor, "bs seq_len hidden"]:
        """Embeds language token IDs into continuous vector representations.

        This method takes a batch of tokenized text sequences and uses the
        underlying Vision-Language Model's text embedding layer to convert
        the token IDs into dense floating-point vectors.

        Args:
            tokens (Int[Tensor, "bs seq_len"]): A tensor of integer token IDs,
                typically the output of a tokenizer.

        Returns:
            Float[Tensor, "bs seq_len hidden"]: The corresponding language
                embeddings for the input tokens.

        """
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: dict[int, dict[str, Tensor]] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool = False,
        fill_kv_cache: bool = False,
    ) -> tuple[list[Tensor | None], dict[int, dict[str, Tensor]] | None]:
        if inputs_embeds is None:
            inputs_embeds = []
        models = [self.get_vlm_model().text_model, self.lm_expert]
        model_layers = self.get_model_layers(models)
        batch_size = 1
        for hidden_states in inputs_embeds:
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]
        # RMSNorm
        num_layers = self.num_vlm_layers
        head_dim = self.vlm.config.get_text_config().head_dim
        for layer_idx in range(num_layers):
            if (
                fill_kv_cache
                or self.config.attention_mode != "cross_attn"
                or (
                    self.self_attn_every_n_layers > 0
                    and layer_idx % self.self_attn_every_n_layers == 0
                )
            ):
                att_outputs, past_key_values = self.forward_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            else:
                # TODO cross-attention with vision features
                att_outputs, past_key_values = self.forward_cross_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_output = (
                    att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                )  # in case of self_attn
                if hidden_states is not None:
                    if layer is None:
                        outputs_embeds.append(hidden_states)
                        continue
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    att_out = att_output[:, start:end]
                    out_emb: Tensor = layer.self_attn.o_proj(att_out)

                    # residual connection
                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)

                    start = end if len(att_outputs) == 1 else 0
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb: Tensor = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)
        return outputs_embeds, past_key_values

    def get_model_layers(
        self, models: list,
    ) -> tuple[list[TransformerBlockLike], list[TransformerBlockLike]]:
        """Retrieve and align layers from a VLM and an expert model.

        This method constructs two lists of neural network layers. The first list
        contains all layers from the primary Vision-Language Model (VLM). The second
        list contains layers from an "expert" model. The expert layers are
        distributed or interleaved among the VLM layers based on the ratio of
        `self.num_vlm_layers` to `self.num_expert_layers`. If an expert layer is not
        placed at a specific index, `None` is used as a placeholder.

        Args:
            models (list): A list containing two pre-trained models.
                - `models[0]` is expected to be the primary VLM.
                - `models[1]` is expected to be the expert model.

        Returns:
            tuple[list[nn.Module], list[nn.Module]]: A tuple containing two lists:
                - The first list (`vlm_layers`) contains the layers of the VLM.
                - The second list (`expert_layers`) contains the layers of the
                  expert model, interspersed with `None` placeholders to align
                  with the VLM layers.

        """
        vlm_layers = []
        expert_layers = []
        multiple_of = self.num_vlm_layers // self.num_expert_layers
        for i in range(self.num_vlm_layers):
            if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                expert_layer = None
            else:
                expert_layer_index = i // multiple_of if multiple_of > 0 else i
                expert_layer = models[1].layers[expert_layer_index]
            vlm_layers.append(models[0].layers[i])
            expert_layers.append(expert_layer)
        return (vlm_layers, expert_layers)

    def forward_attn_layer(
        self,
        model_layers,
        inputs_embeds: list[torch.FloatTensor],
        layer_idx: int,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        batch_size: int,
        head_dim: int,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values: dict[int, dict[str, Tensor]] | None = None,
    ) -> tuple[list[Tensor], dict[int, dict[str, Tensor]] | None]:
        query_states = []
        key_states = []
        value_states = []
        for i, hidden_states in enumerate(inputs_embeds):
            layer: TransformerBlockLike = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue
            hidden_states = layer.input_layernorm(hidden_states)  # noqa: PLW2901

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)  # noqa: PLW2901
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)

        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)

        seq_len = query_states.shape[1]
        if seq_len < position_ids.shape[1]:
            _position_ids = position_ids[:, :seq_len]
            _attention_mask = attention_mask[:, :seq_len, :seq_len]
        else:
            _position_ids = position_ids
            _attention_mask = attention_mask

        attention_mask_ = _attention_mask
        position_ids_ = _position_ids

        query_states = apply_rope(query_states, position_ids_)
        key_states = apply_rope(key_states, position_ids_)

        if use_cache:
            if past_key_values is None:
                past_key_values = {}

            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                # the max len, then we (for instance) double the cache size. This implementation already exists
                # in `transformers`. (molbap)
                key_states = torch.cat(
                    [past_key_values[layer_idx]["key_states"], key_states],
                    dim=1,
                )
                value_states = torch.cat(
                    [past_key_values[layer_idx]["value_states"], value_states],
                    dim=1,
                )

        attention_interface = self.get_attention_interface()

        att_output = attention_interface(
            attention_mask_,
            batch_size,
            head_dim,
            query_states,
            key_states,
            value_states,
        )
        return [att_output], past_key_values

    def get_attention_interface(self):
        """Return the attention function to be used."""
        return self.eager_attention_forward

    def eager_attention_forward(
        self,
        attention_mask: Bool[Tensor, "bs s_q s_kv"],
        batch_size: int,
        head_dim: int,
        query_states: Float[Tensor, "b s_q h d"],
        key_states: Float[Tensor, "b s_kv h_kv d"],
        value_states: Float[Tensor, "b s_kv h_kv d"],
    ) -> Float[Tensor, "b s_q (h d)"]:
        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        # num_key_value_groups = num_att_heads // num_key_value_heads

        # sequence_length = key_states.shape[1]
        # q: (b, s_q, h, d) -> (b, h, s_q, d)
        query_states = rearrange(query_states, "b s_q h d -> b h s_q d")

        key_states = rearrange(
            key_states,
            "b s_kv h_kv d -> b (h_kv g) s_kv d",
            g=(num_att_heads // num_key_value_heads),
        )
        value_states = rearrange(
            value_states,
            "b s_kv h_kv d -> b (h_kv g) s_kv d",
            g=(num_att_heads // num_key_value_heads),
        )

        # Attention here is upcasted to float32 to match the original eager implementation.
        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        scale = head_dim**-0.5
        att_weights = (
            einsum(
                query_states.to(torch.float32),
                key_states.to(torch.float32),
                "b h s_q d, b h s_kv d -> b h s_q s_kv",
            )
            * scale
        )

        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min  # -2.3819763e38  # See gemma/modules.py
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = einsum(probs, value_states, "b h s_q s_kv, b h s_kv d -> b h s_q d")
        att_output = rearrange(att_output, "b h s_q d -> b s_q (h d)")

        return att_output

    def forward_cross_attn_layer(
        self,
        model_layers: tuple[list[TransformerBlockLike], list[TransformerBlockLike]],
        inputs_embeds: list[torch.FloatTensor],
        layer_idx: int,
        position_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        batch_size: int,
        head_dim: int,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ) -> list[torch.Tensor]:
        attention_interface = self.get_attention_interface()

        att_outputs = []
        assert len(inputs_embeds) == 2 or (
            use_cache and past_key_values is not None and not fill_kv_cache
        ), (
            f"Both len(inputs_embeds) == {len(inputs_embeds)} and past_key_values is {past_key_values}"
        )

        if len(inputs_embeds) == 2 and not past_key_values:
            # Prefix attention
            seq_len = inputs_embeds[0].shape[1]
            position_id, expert_position_id = position_ids[:, :seq_len], position_ids[:, seq_len:]
            prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]

            layer = model_layers[0][layer_idx]

            hidden_states = layer.input_layernorm(inputs_embeds[0])

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            query_states = apply_rope(query_state, position_id)
            key_states = apply_rope(key_state, position_id)

            att_output = attention_interface(
                prefix_attention_mask,
                batch_size,
                head_dim,
                query_states,
                key_states,
                value_states,
            )
            att_outputs.append(att_output)
        else:
            expert_position_id = position_ids

        if use_cache and past_key_values is None:
            past_key_values = {}

        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                # the max len, then we (for instance) double the cache size. This implementation already exists
                # in `transformers`. (molbap)
                key_states = past_key_values[layer_idx]["key_states"]
                value_states = past_key_values[layer_idx]["value_states"]

        # Expert
        expert_layer = model_layers[1][layer_idx]
        if expert_layer is not None:
            expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])

            expert_input_shape = expert_hidden_states.shape[:-1]
            expert_hidden_shape = (*expert_input_shape, -1, expert_layer.self_attn.head_dim)

            expert_hidden_states = expert_hidden_states.to(
                dtype=expert_layer.self_attn.q_proj.weight.dtype,
            )
            expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states).view(
                expert_hidden_shape,
            )

            _key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).view(
                *key_states.shape[:2],
                -1,
            )
            expert_key_states = expert_layer.self_attn.k_proj(_key_states).view(
                *_key_states.shape[:-1],
                -1,
                expert_layer.self_attn.head_dim,
            )  # k_proj should have same dim as kv

            _value_states = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).view(
                *value_states.shape[:2],
                -1,
            )
            expert_value_states = expert_layer.self_attn.v_proj(_value_states).view(
                *_value_states.shape[:-1],
                -1,
                expert_layer.self_attn.head_dim,
            )

            expert_position_id = (
                expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values
            )  # start from 0
            expert_attention_mask = attention_mask[
                :,
                -inputs_embeds[1].shape[1] :,
                : expert_key_states.shape[1] :,
            ]  # take into account kv

            expert_query_states = apply_rope(expert_query_state, expert_position_id)

            att_output = attention_interface(
                expert_attention_mask,
                batch_size,
                head_dim,
                expert_query_states,
                expert_key_states,
                expert_value_states,
            )
            att_outputs.append(att_output)
        else:
            att_outputs.append(None)

        # att_output = att_output.to(dtype=models[i].dtype)
        return att_outputs, past_key_values


if __name__ == "__main__":
    config = SmolVLMWithExpertConfig(
        model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weight=False,
    )
    model = SmolVLMWithExpertModel(config)
    last_layers = [12 - 1]
    frozen_layers = ["lm_head", "text_model.model.norm.weight"]
    for layer in last_layers:
        frozen_layers.extend(f"text_model.model.layers.{layer}.")
    print(frozen_layers)

    x = model.embed_image(
        image=torch.randn(1, 3, 128, 128),
    )  # .to(device="cuda", dtype=torch.bfloat16))
    print(x.shape)
