"""
Author: Jonathan Pilault
This files contains adaptations to transformers.models.bart.modeling_bart.
Classes are inherited whenever changes are made.
MyBartModel covers both BartModel and MBartModel and integrates prompt generators.
"""

import torch
import torch.nn as nn

import random
from typing import Optional, Tuple

from transformers.utils import logging
from transformers.models.bart import (
    BartModel, BartConfig,
    modeling_bart
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqModelOutput,
    BaseModelOutputWithPastAndCrossAttentions
)
from transformers.models.mbart import MBartModel

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


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, prompt_len: int, tgt_len: Optional[int] = None):  # new
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    prompt_mask = torch.ones_like(mask[:, :1]).expand(bsz, prompt_len)  # new
    mask = torch.cat((prompt_mask, mask), dim=1)  # new
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len + prompt_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class MyBartAttention(modeling_bart.BartAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim, num_heads, dropout=0.0, is_decoder=False, bias=True):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_prompts: Optional[torch.Tensor] = None,  # new
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
            if key_value_prompts is not None:  # new
                key_states = torch.cat([key_value_prompts[0], key_states], dim=2)  # new
                value_states = torch.cat([key_value_prompts[1], value_states], dim=2)  # new
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            if key_value_prompts is not None:  # new
                key_states = torch.cat([key_value_prompts[0], key_states], dim=2)  # new
                value_states = torch.cat([key_value_prompts[1], value_states], dim=2)  # new

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size(3) != src_len:
                assert key_value_prompts is None  # new
                attention_mask = attention_mask[:, :, :, -src_len:]
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class MyBartEncoderLayer(modeling_bart.BartEncoderLayer):
    def __init__(self, config: BartConfig, layer_idx, prompt_generator: Optional[nn.Module] = None):  # new
        super().__init__(config)
        self.self_attn = MyBartAttention(  # new
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        if getattr(config, "encoder_self_prefix_layer_ids", False):  # new
            self.prompt_layer_idx = config.encoder_self_prefix_layer_ids
        else:
            self.prompt_layer_idx = []
        if layer_idx in self.prompt_layer_idx:  # new
            self.prompt_generator = prompt_generator
        else:
            self.prompt_generator = None
        if config.model_type == 'mbart':  # new
            self.pre_layer_norm = True
            self.post_layer_norm = False
        else:
            self.pre_layer_norm = False
            self.post_layer_norm = True
        if torch.cuda.is_available():  # new
            try:
                from apex.normalization import FusedLayerNorm
                self.self_attn_layer_norm = FusedLayerNorm(self.embed_dim)
                self.final_layer_norm = FusedLayerNorm(self.embed_dim)
            except ImportError:
                pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        descriptors: torch.Tensor,  # new
        descriptors_attention_mask: torch.Tensor,  # new
        layer_idx: int,  # new
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        if layer_idx in self.prompt_layer_idx:  # new
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

        residual = hidden_states

        if self.pre_layer_norm:  # MBart (new)
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            key_value_prompts=key_value_prompts,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if self.post_layer_norm:  # Bart (new)
            hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        if self.pre_layer_norm:  # MBart (new)
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if self.post_layer_norm:  # Bart (new)
            hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class MyBartDecoderLayer(modeling_bart.BartDecoderLayer):
    def __init__(self, config: BartConfig, layer_idx: int, prompt_generator: Optional[nn.Module] = None):
        super().__init__(config)
        self.self_attn = MyBartAttention(  # new
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn = MyBartAttention(  # new
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )

        if getattr(config, "decoder_self_prefix_layer_ids", False):  # new
            self.prompt_selflayer_idx = config.decoder_self_prefix_layer_ids
        else:
            self.prompt_selflayer_idx = []
        if getattr(config, "decoder_cross_prefix_layer_ids", False):  # new
            self.prompt_crosslayer_idx = config.decoder_cross_prefix_layer_ids
        else:
            self.prompt_crosslayer_idx = []
        if layer_idx in self.prompt_selflayer_idx or layer_idx in self.prompt_crosslayer_idx:  # new
            self.prompt_generator = prompt_generator
        else:
            self.prompt_generator = None
        if config.model_type == 'mbart':  # new
            self.pre_layer_norm = True
            self.post_layer_norm = False
        else:
            self.pre_layer_norm = False
            self.post_layer_norm = True
        if torch.cuda.is_available():  # new
            try:
                from apex.normalization import FusedLayerNorm
                self.self_attn_layer_norm = FusedLayerNorm(self.embed_dim)
                self.encoder_attn_layer_norm = FusedLayerNorm(self.embed_dim)
                self.final_layer_norm = FusedLayerNorm(self.embed_dim)
            except ImportError:
                pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        descriptors: torch.Tensor,  # new
        descriptors_attention_mask: torch.Tensor,  # new
        layer_idx: int,  # new
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        if past_key_value is None and layer_idx in self.prompt_selflayer_idx:
            selfattn_key_value_prompts = self.prompt_generator(  # new
                layer_type="decoder_self_attn",
                layer_idx=layer_idx,
                descriptors=descriptors,
                descriptors_attention_mask=descriptors_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )
        else:
            selfattn_key_value_prompts = None

        residual = hidden_states
        if self.pre_layer_norm:  # MBart (new)
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # prompt self-attn cached key/values tuple are already appended
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            key_value_prompts=selfattn_key_value_prompts,  # new
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if self.post_layer_norm:  # Bart (new)
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            if past_key_value is None and layer_idx in self.prompt_crosslayer_idx:
                crossattn_key_value_prompts = self.prompt_generator(  # new
                    layer_type="decoder_cross_attn",
                    layer_idx=layer_idx,
                    descriptors=descriptors,
                    descriptors_attention_mask=descriptors_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask
                )
            else:
                crossattn_key_value_prompts = None

            residual = hidden_states
            if self.pre_layer_norm:  # MBart (new)
                hidden_states = self.encoder_attn_layer_norm(hidden_states)
            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_prompts=crossattn_key_value_prompts,  # new
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            if self.post_layer_norm:  # Bart (new)
                hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            # prompt cross-attn cached key/values tuple are already appended
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        if self.pre_layer_norm:  # MBart (new)
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if self.post_layer_norm:  # Bart (new)
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MyBartEncoder(modeling_bart.BartEncoder):
    def __init__(
            self,
            config: BartConfig,
            embed_tokens: Optional[nn.Embedding] = None,
            prompt_generator: Optional[nn.Module] = None,  # new
    ):
        super().__init__(config, embed_tokens)
        self.num_prompts = prompt_generator.num_prompts if prompt_generator is not None else 0  # new
        self.layers = nn.ModuleList([MyBartEncoderLayer(config, idx, prompt_generator) for idx in range(config.encoder_layers)])  # new
        if config.model_type == 'mbart':  # new
            self.layer_norm = nn.LayerNorm(config.d_model)
        else:
            self.layer_norm = None
        if torch.cuda.is_available():  # new
            try:
                from apex.normalization import FusedLayerNorm
                self.layernorm_embedding = FusedLayerNorm(config.d_model)
                if self.layer_norm is not None:
                    self.layer_norm = FusedLayerNorm(config.d_model)
            except ImportError:
                pass

    def forward(
            self,
            input_ids=None,
            descriptors=None,  # new
            descriptors_attention_mask=None,  # new
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        prompt_key_values_length = self.num_prompts  # new

        if inputs_embeds is None:
            assert self.embed_tokens.weight.device == input_ids.device
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, prompt_len=prompt_key_values_length, tgt_len=input_shape[-1]  # new
            )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                # new:
                elif getattr(self.config, 'n_enc_recurrence', None) is not None and idx not in self.config.freeze_encoder_layers:
                    assert idx == len(self.layers) - 1, "only works for the last layer at the moment"
                    for t in range(self.config.n_enc_recurrence):
                        layer_outputs = encoder_layer(
                            hidden_states,
                            descriptors,  # new
                            descriptors_attention_mask,  # new
                            idx + t,  # new
                            attention_mask,
                            layer_head_mask=(head_mask[idx + t] if head_mask is not None else None),
                            output_attentions=output_attentions,
                        )
                        hidden_states = layer_outputs[0]
                # new:
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        descriptors,  # new
                        descriptors_attention_mask,  # new
                        idx,  # new
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if self.layer_norm is not None:  # MBart (new)
            hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class MyBartDecoder(modeling_bart.BartDecoder):
    def __init__(
            self,
            config: BartConfig,
            embed_tokens: Optional[nn.Embedding] = None,
            prompt_generator: Optional[nn.Module] = None,  # new
    ):
        super().__init__(config, embed_tokens)
        self.num_prompts = prompt_generator.num_prompts if prompt_generator is not None else 0  # new
        self.layers = nn.ModuleList([MyBartDecoderLayer(config, idx, prompt_generator) for idx in range(config.decoder_layers)])  # new
        if config.model_type == 'mbart':  # new
            self.layer_norm = nn.LayerNorm(config.d_model)
        else:
            self.layer_norm = None
        if torch.cuda.is_available():  # new
            try:
                from apex.normalization import FusedLayerNorm
                self.layernorm_embedding = FusedLayerNorm(config.d_model)
                if self.layer_norm is not None:
                    self.layer_norm = FusedLayerNorm(config.d_model)
            except ImportError:
                pass

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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # prompt_key_values_length
        prompt_key_values_length = self.num_prompts  # new
        # past_key_values_length
        past_key_values_length = past_key_values[2][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length + prompt_key_values_length  # new
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(
                encoder_attention_mask, inputs_embeds.dtype, prompt_len=prompt_key_values_length, tgt_len=input_shape[-1]  # new
            )

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            # new:
            elif getattr(self.config, 'n_dec_recurrence', None) is not None and idx not in self.config.freeze_decoder_layers:
                assert idx == len(self.layers) - 1, "only works for the last layer at the moment"
                for t in range(self.config.n_dec_recurrence):
                    past_key_value = past_key_values[idx + t] if past_key_values is not None else None
                    layer_outputs = decoder_layer(
                        hidden_states,
                        descriptors,  # new
                        descriptors_attention_mask,  # new
                        idx + t,  # new
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        layer_head_mask=(head_mask[idx + t] if head_mask is not None else None),
                        cross_attn_layer_head_mask=(
                            cross_attn_head_mask[idx + t] if cross_attn_head_mask is not None else None
                        ),
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                    hidden_states = layer_outputs[0]

                    if use_cache and t < self.config.n_dec_recurrence - 1:
                        next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
            # new:
            else:
                layer_outputs = decoder_layer(
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
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        if self.layer_norm is not None:  # MBart (new)
            hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class MyBartModel(BartModel):
    def __init__(self, config: BartConfig, prompt_generator: Optional[nn.Module] = None):  # new
        super().__init__(config)
        self.prompt_generator = prompt_generator  # new
        self.encoder = MyBartEncoder(config, self.shared, self.prompt_generator)  # new
        self.decoder = MyBartDecoder(config, self.shared, self.prompt_generator)  # new

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

        # MBart had different decoder_start_token_id depending on the language and
        # decoder_start_token_id during training for multilanguage batches
        if decoder_start_token_id is not None and self.training:
            self.config.decoder_start_token_id = decoder_start_token_id.view(-1)
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided only during training
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = modeling_bart.shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                descriptors=descriptors,  # new
                descriptors_attention_mask=descriptors_attention_mask,  # new
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        descriptors2 = descriptors.clone() if descriptors is not None else None
        descriptors_attention_mask2 = descriptors_attention_mask.clone() if descriptors_attention_mask is not None else None
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            descriptors=descriptors2,  # new
            descriptors_attention_mask=descriptors_attention_mask2,  # new
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
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
