"""
Author: Jonathan Pilault
This files contains adaptations to transformers.models.bart.modeling_bart.
Classes are inherited whenever changes are made.
MyBartModel covers both BartModel and MBartModel and integrates prompt generators.
"""

import torch
import torch.nn as nn

import warnings
from typing import Optional, Tuple

from transformers.utils import logging
from torch.utils.checkpoint import checkpoint
from transformers.models.t5 import (
    T5Model, T5Config,
    modeling_t5
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqModelOutput,
    BaseModelOutputWithPastAndCrossAttentions
)

# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

logger = logging.get_logger(__name__)

SEQ2SEQMODEL_REGISTRY = {}


def register_model(model_class):
    model_type = model_class.model_type().lower()
    if model_type in SEQ2SEQMODEL_REGISTRY:
        raise ValueError(f"Encoder registry already contains {model_type}")
    SEQ2SEQMODEL_REGISTRY[model_type] = model_class
    return model_class


def get_model_class(model_type):
    model_type = model_type.lower()

    if model_type not in SEQ2SEQMODEL_REGISTRY:
        raise ValueError(
            f"Invalid seq2seq model type '{model_type}'. Available models={', '.join(list(SEQ2SEQMODEL_REGISTRY.keys()))}"
        )

    return SEQ2SEQMODEL_REGISTRY[model_type]


class MyT5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)

    def forward(
        self,
        hidden_states,
        key_value_prompts,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        int_seq_length = int(seq_length)

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, key_value_prompts, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
                hidden_states = torch.cat([key_value_prompts, hidden_states], dim=2)
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))
                hidden_states = torch.cat([key_value_prompts, hidden_states], dim=2)

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, key_value_prompts[0], past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, key_value_prompts[1], past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.training and self.gradient_checkpointing:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -int_seq_length:, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class MyT5LayerSelfAttention(modeling_t5.T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.SelfAttention = MyT5Attention(config, has_relative_attention_bias=has_relative_attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_prompts: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias=None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache=False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            key_value_prompts=key_value_prompts,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class MyT5Block(modeling_t5.T5Block):
    def __init__(self, config, layer_idx: int, has_relative_attention_bias=False, prompt_generator: Optional[nn.Module] = None):
        super().__init__(config)
        self.layer = nn.ModuleList()
        self.layer.append(MyT5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(MyT5LayerSelfAttention(config))

        if getattr(config, "encoder_self_prefix_layer_ids", False):
            self.prompt_enc_layer_idx = config.encoder_self_prefix_layer_ids
        else:
            self.prompt_enc_layer_idx = []
        if getattr(config, "decoder_self_prefix_layer_ids", False):
            self.prompt_dec_layer_idx = config.decoder_self_prefix_layer_ids
        else:
            self.prompt_dec_layer_idx = []
        if getattr(config, "decoder_cross_prefix_layer_ids", False):
            self.prompt_crosslayer_idx = config.decoder_cross_prefix_layer_ids
        else:
            self.prompt_crosslayer_idx = []
        if layer_idx in self.prompt_selflayer_idx or layer_idx in self.prompt_crosslayer_idx:
            self.prompt_generator = prompt_generator
        else:
            self.prompt_generator = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        descriptors: torch.Tensor,  # new
        descriptors_attention_mask: torch.Tensor,  # new
        layer_idx: int,  # new
        attention_mask: Optional[torch.Tensor] = None,
        position_bias=None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_decoder_position_bias=None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        return_dict=True,
    ):
        if layer_idx in self.prompt_enc_layer_idx and not self.is_decoder:
            key_value_prompts = self.prompt_generator(
                layer_type="encoder_self_attn",
                layer_idx=layer_idx,
                descriptors=descriptors,
                descriptors_attention_mask=descriptors_attention_mask,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask
            )
        else:
            key_value_prompts = None

        if past_key_value is None and layer_idx in self.prompt_dec_layer_idx and self.is_decoder:
            key_value_prompts = self.prompt_generator(
                layer_type="decoder_self_attn",
                layer_idx=layer_idx,
                descriptors=descriptors,
                descriptors_attention_mask=descriptors_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )
        else:
            key_value_prompts = None

        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention' if expected_num_past_key_values == 4 else ''}."
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        # add present self-attn cache to positions 1,2 of present_key_value tuple
        self_attention_outputs = self.layer[0](
            hidden_states=hidden_states,
            key_value_prompts=key_value_prompts,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            if past_key_value is None and layer_idx in self.prompt_crosslayer_idx:
                crossattn_key_value_prompts = self.prompt_generator(
                    layer_type="decoder_cross_attn",
                    layer_idx=layer_idx,
                    descriptors=descriptors,
                    descriptors_attention_mask=descriptors_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask
                )
            else:
                crossattn_key_value_prompts = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_prompts=crossattn_key_value_prompts,  # new
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  #


class MyT5Stack(modeling_t5.T5Stack):
    def __init__(
            self,
            config,
            embed_tokens: Optional[nn.Embedding] = None,
            prompt_generator: Optional[nn.Module] = None,
    ):
        super().__init__(config, embed_tokens)
        self.num_prompts = prompt_generator.num_prompts if prompt_generator is not None else 0
        self.block = nn.ModuleList(
            [MyT5Block(config, has_relative_attention_bias=bool(idx == 0), layer_idx=idx, prompt_generator=prompt_generator) for idx in range(config.num_layers)]
        )

    def forward(
        self,
        input_ids=None,
        descriptors=None,  # new
        descriptors_attention_mask=None,  # new
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        err_msg_prefix = "decoder_" if self.is_decoder else ""
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}_input_ids and {err_msg_prefix}_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                f"You have to specify either {err_msg_prefix}_input_ids or {err_msg_prefix}_inputs_embeds"
            )

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        # prompt_key_values_length
        prompt_key_values_length = self.num_prompts
        # past_key_values_length
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
        mask_seq_length += prompt_key_values_length

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape,
                                                                   inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for idx, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[idx]
            cross_attn_layer_head_mask = cross_attn_head_mask[idx]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )

            elif getattr(self.config, 'n_dec_recurrence', None) is not None and idx not in self.config.freeze_decoder_layers:
                assert idx == len(self.layers) - 1, "only works for the last layer at the moment"
                for t in range(self.config.n_dec_recurrence):
                    past_key_value = past_key_values[idx + t] if past_key_values is not None else None
                    layer_outputs = layer_module(
                        hidden_states,
                        descriptors,  # new
                        descriptors_attention_mask,  # new
                        idx + t,  # new
                        attention_mask=extended_attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_extended_attention_mask,
                        layer_head_mask=(head_mask[idx + t] if head_mask is not None else None),
                        cross_attn_layer_head_mask=(
                            cross_attn_head_mask[idx + t] if cross_attn_head_mask is not None else None
                        ),
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                    # layer_outputs is a tuple with:
                    # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
                    if use_cache is False:
                        layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

                    hidden_states, present_key_value_state = layer_outputs[:2]

                    if use_cache and t < self.config.n_dec_recurrence - 1:
                        present_key_value_states = present_key_value_states + (present_key_value_state,)
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    descriptors,  # new
                    descriptors_attention_mask,  # new
                    idx,  # new
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if idx == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class MyT5Model(T5Model):
    def __init__(self, config, prompt_generator: Optional[nn.Module] = None):
        super().__init__(config)
        self.prompt_generator = prompt_generator
        self.encoder = MyT5Stack(config, self.shared, self.prompt_generator)
        self.decoder = MyT5Stack(config, self.shared, self.prompt_generator)

    def forward(
            self,
            input_ids=None,
            descriptors=None,  # new
            descriptors_attention_mask=None,  # new
            decoder_start_token_id=None,  # new
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        r"""
               Returns:

               Example::

                   >>> from transformers import T5Tokenizer, T5Model

                   >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
                   >>> model = T5Model.from_pretrained('t5-small')

                   >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
                   >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
                   >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

                   >>> last_hidden_states = outputs.last_hidden_state
               """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if decoder_start_token_id is not None and self.training:
            self.config.decoder_start_token_id = decoder_start_token_id.view(-1)

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                descriptors=descriptors,  # new
                descriptors_attention_mask=descriptors_attention_mask,  # new
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        descriptors2 = descriptors.clone() if descriptors is not None else None
        descriptors_attention_mask2 = descriptors_attention_mask.clone() if descriptors_attention_mask is not None else None
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            descriptors=descriptors2,  # new
            descriptors_attention_mask=descriptors_attention_mask2,  # new
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
