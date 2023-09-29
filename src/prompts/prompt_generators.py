"""
Author: Jonathan Pilault
We define LinearPrompt, TransformerPrompt, NeuralPromptProducer here.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from transformers.models.bart.modeling_bart import _expand_mask
from src.multi_task_data.multi_task_dataset import MAX_LENGTHS
from src.prompts.prompt_generator_layers import LinearPromptLayers, TransformerPromptLayer, NeuralPromptProducerLayer
from src.prompts.prompt_functional import linear

TYPES = {
    "encoder_self_attn": 0,
    "decoder_self_attn": 1,
    "decoder_cross_attn": 2,
}


def get_layer_idx_mapping(self):
    # used to align created prompt layers and transformer layer indices where prompting is required
    return {
            "encoder_self_attn": {idx: n for n, idx in enumerate(self.encoder_self_prefix_layer_ids)},
            "decoder_self_attn": {idx: n for n, idx in enumerate(self.decoder_self_prefix_layer_ids)},
            "decoder_cross_attn": {idx: n for n, idx in enumerate(self.decoder_cross_prefix_layer_ids)}
        }


def get_max_num_layers(self):
    # used to create equal amounts of prompt generator layers
    return max(
            len(self.encoder_self_prefix_layer_ids),
            len(self.decoder_self_prefix_layer_ids),
            len(self.decoder_cross_prefix_layer_ids)
        )


class LinearPrompt(nn.Module):
    """
    Unconditional prompt generator using linear layers
    Prompts are passed to past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`,
            with each tuple having 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`)
            and 2 additional tensors of shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    """
    def __init__(self, config):
        super().__init__()
        assert config.encoder_layers == config.decoder_layers, "Assumes equal encoder/decoder number of layers"
        assert config.encoder_attention_heads == config.decoder_attention_heads

        self.dropout = config.prompt_dropout
        self.prompt_embeddings = nn.Parameter(
            torch.empty(len(TYPES), config.num_prompts, config.d_model),
            requires_grad=True
        )
        self.encoder_self_prefix_layer_ids = config.encoder_self_prefix_layer_ids
        self.decoder_self_prefix_layer_ids = config.decoder_self_prefix_layer_ids
        self.decoder_cross_prefix_layer_ids = config.decoder_cross_prefix_layer_ids
        self.idx2layer = get_layer_idx_mapping(self)
        self.model_layers = get_max_num_layers(self)
        self.prompt_layers = nn.ModuleList([
            LinearPromptLayers(config, self.model_layers),
            LinearPromptLayers(config, self.model_layers),
            LinearPromptLayers(config, self.model_layers)
        ])
        self.num_heads = config.encoder_attention_heads
        self.num_prompts = config.num_prompts
        self.head_dim = config.d_model // self.num_heads
        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.prompt_embeddings)

    def forward(
            self,
            layer_type,
            layer_idx,
            descriptors,
            descriptors_attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # unconditional prompt
        type = TYPES[layer_type]
        batch_size = descriptors.size(0)
        p_layer_idx = self.idx2layer[layer_type][layer_idx]

        prompt_states = self.prompt_embeddings[type]
        prompt_layers = self.prompt_layers[type]
        prompt_states = prompt_layers(prompt_states)
        prompt_states = F.dropout(prompt_states, p=self.dropout, training=self.training)
        prompt_states = prompt_states.view(self.model_layers, 1, self.num_heads, self.num_prompts, -1)
        prompt_states = prompt_states.repeat(1, batch_size, 1, 1, 1)

        key_value_prompts = prompt_states.chunk(2, dim=-1)
        return key_value_prompts[0][p_layer_idx], key_value_prompts[1][p_layer_idx]


class TransformerPrompt(nn.Module):
    """
    Conditional prompt generator using Transformer layers.
    Prompts are passed to past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`,
        with each tuple having 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`).
        Prompts are conditioned on the descriptors, continuous embeddings.
    """
    def __init__(self, config, token_embeddings):
        super().__init__()
        assert config.encoder_layers == config.decoder_layers, "Assumes equal encoder/decoder number of layers"
        assert config.encoder_attention_heads == config.decoder_attention_heads
        assert config.max_position_embeddings >= config.max_descriptor_length
        self.num_heads = config.encoder_attention_heads
        self.prompt_num_heads = config.prompt_attention_heads
        self.num_prompts = config.num_prompts
        self.use_trf_hidden = config.prompt_hidden_condition
        self.dropout = config.prompt_dropout
        self.head_dim = config.d_model // self.num_heads
        self.prompt_head_dim = config.prompt_d_model // self.prompt_num_heads
        self.encoder_self_prefix_layer_ids = config.encoder_self_prefix_layer_ids
        self.decoder_self_prefix_layer_ids = config.decoder_self_prefix_layer_ids
        self.decoder_cross_prefix_layer_ids = config.decoder_cross_prefix_layer_ids
        self.idx2layer = get_layer_idx_mapping(self)
        self.model_layers = get_max_num_layers(self)
        self.token_embeddings = token_embeddings

        self.token_emb_proj = nn.Linear(config.d_model, config.prompt_d_model)
        self.effective_num_prompts = max(config.num_prompts//2 - config.max_descriptor_length, 0)
        self.prompt_embeddings = nn.Parameter(
            torch.empty(len(TYPES), self.model_layers, self.effective_num_prompts, config.prompt_d_model),
            requires_grad=True
        )

        self.layers = nn.ModuleList(
            [TransformerPromptLayer(config) for _ in range(config.prompt_layers)]
        )
        if config.prompt_d_model != config.d_model:
            self.expand_layer = nn.Linear(config.prompt_d_model, config.d_model)
        else:
            self.expand_layer = None
        if self.use_trf_hidden:
            self.trf_h_proj_w_types = nn.Parameter(
                torch.empty(len(TYPES), config.prompt_d_model, config.d_model//2),
                requires_grad=True
            )
            self.trf_h_proj_w_layers = nn.Parameter(
                torch.empty(self.model_layers, config.d_model//2, config.d_model),
                requires_grad=True
            )
            self.trf_h_proj_b = nn.Parameter(
                torch.empty(len(TYPES), self.model_layers, 1, config.prompt_d_model),
                requires_grad=True
            )
            #self.trf_h_up_proj = nn.Linear(config.prompt_d_model//4, config.prompt_d_model)
        self.pad_token_id = config.pad_token_id
        self.init_weights()

    def init_weights(self):
        mean = self.token_embeddings.weight.mean()
        std = self.token_embeddings.weight.std()
        self.prompt_embeddings.data.normal_(mean=float(mean), std=float(std))
        if self.use_trf_hidden:
            xavier_uniform_(self.trf_h_proj_w_types)
            xavier_uniform_(self.trf_h_proj_w_layers)
            constant_(self.trf_h_proj_b, 0.)

    def get_inputs(self,
            layer_type,
            layer_idx,
            descriptors,
            descriptors_attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            batch_size
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        type = TYPES[layer_type]
        p_layer_idx = self.idx2layer[layer_type][layer_idx]

        description_embedding = self.token_emb_proj(self.token_embeddings(descriptors))
        prompt_embedding = self.prompt_embeddings[type, p_layer_idx]
        prompt_embedding = prompt_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
        prompt_states = torch.cat((prompt_embedding, description_embedding), dim=1)

        prompt_emb_mask = torch.ones_like(prompt_embedding[:, :, 0])
        prompt_mask1 = torch.cat((prompt_emb_mask, descriptors_attention_mask), dim=1)
        prompt_mask = _expand_mask(prompt_mask1, prompt_states.dtype, prompt_mask1.size(-1)).squeeze(1)

        if self.use_trf_hidden:
            h = linear(
                input=encoder_hidden_states[:, :MAX_LENGTHS['prompt_input_h']],
                weight=torch.mm(self.trf_h_proj_w_types[type], self.trf_h_proj_w_layers[p_layer_idx]),
                bias=self.trf_h_proj_b[type, p_layer_idx]
            )
            # h = self.trf_h_up_proj(torch.relu(h))
            prompt_states = torch.cat((prompt_states, h), dim=1)
            start = self.num_prompts
            end = self.num_prompts + h.size(1)
            h_mask = encoder_attention_mask[:, 0, :1, start:end].repeat(1, h.size(1), 1)
            prompt_mask_expanded = torch.cat((prompt_mask, h_mask[:, :1, :].repeat(1, prompt_mask.size(-1), 1)), dim=-1)
            h_mask_expanded = torch.cat((prompt_mask[:, :1, :].repeat(1, h.size(1), 1), h_mask), dim=-1)
            prompt_mask = torch.cat((prompt_mask_expanded, h_mask_expanded), dim=1)

        return prompt_states, prompt_mask

    def forward(
            self,
            layer_type,
            layer_idx,
            descriptors,
            descriptors_attention_mask,
            encoder_hidden_states,
            encoder_attention_mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # conditional prompt
        batch_size = descriptors.size(0)
        prompt_states, prompt_mask = self.get_inputs(
            layer_type, layer_idx, descriptors, descriptors_attention_mask,
            encoder_hidden_states, encoder_attention_mask, batch_size
        )
        assert prompt_states.size(1) >= self.num_prompts * 2, \
            "Prompt states need to need to be twice as large as num_prompts"
        for idx, prompt_layer in enumerate(self.layers):
            prompt_states = prompt_layer(prompt_states, prompt_mask)

        prompt_states = F.dropout(prompt_states, p=self.dropout, training=self.training)
        if self.expand_layer is not None:
            prompt_states = self.expand_layer(prompt_states[:, :self.num_prompts * 2])   # one for key and one for value
        else:
            prompt_states = prompt_states[:, :self.num_prompts * 2]
        prompt_states = prompt_states.view(batch_size, self.num_heads, self.num_prompts, -1)

        key_value_prompts = prompt_states.chunk(2, dim=-1)
        return key_value_prompts[0], key_value_prompts[1]


class PromptProductionSystem(TransformerPrompt):
    """
    Conditional prompt generator using Neural Production Systems.
    Prompts are passed to past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`,
        with each tuple having 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`).
        Prompts are conditioned on the descriptors, continuous embeddings.
    """

    def __init__(self, config, token_embeddings):
        super().__init__(config, token_embeddings)
        self.layers = nn.ModuleList(
            [NeuralPromptProducerLayer(config, self.effective_num_prompts) for _ in range(config.prompt_layers)]
        )


class UniversalTransformerPrompt(TransformerPrompt):
    """
    Conditional prompt generator using Transformer layers.
    Prompts are passed to past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`,
        with each tuple having 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`).
        Prompts are conditioned on the descriptors, continuous embeddings.
    """
    def __init__(self, config, token_embeddings):
        super().__init__(config, token_embeddings)

        self.layers = nn.ModuleList(
            [TransformerPromptLayer(config)]
        )
        self.num_recurrence = config.prompt_layers

    def forward(
            self,
            layer_type,
            layer_idx,
            descriptors,
            descriptors_attention_mask,
            encoder_hidden_states,
            encoder_attention_mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # conditional prompt
        batch_size = descriptors.size(0)
        prompt_states, prompt_mask = self.get_inputs(
            layer_type, layer_idx, descriptors, descriptors_attention_mask,
            encoder_hidden_states, encoder_attention_mask, batch_size
        )
        assert prompt_states.size(1) >= self.num_prompts * 2, \
            "Prompt states need to need to be twice as large as num_prompts"
        prompt_layer = self.layers[0]
        for t in range(self.num_recurrence):
            prompt_states = prompt_layer(prompt_states, prompt_mask)

        prompt_states = F.dropout(prompt_states, p=self.dropout, training=self.training)
        if self.expand_layer is not None:
            prompt_states = self.expand_layer(prompt_states[:, :self.num_prompts * 2])  # one for key and one for value
        else:
            prompt_states = prompt_states[:, :self.num_prompts * 2]
        prompt_states = prompt_states.view(batch_size, self.num_heads, self.num_prompts, -1)

        key_value_prompts = prompt_states.chunk(2, dim=-1)
        return key_value_prompts[0], key_value_prompts[1]


class UniversalPromptProductionSystem(UniversalTransformerPrompt):
    """
    Our main model: UPPS
    Conditional prompt generator using Transformer layers.
    Prompts are passed to past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`,
        with each tuple having 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`).
        Prompts are conditioned on the descriptors, continuous embeddings.
    """
    def __init__(self, config, token_embeddings):
        super().__init__(config, token_embeddings)

        self.layers = nn.ModuleList(
            [NeuralPromptProducerLayer(config, self.effective_num_prompts)]
        )
        self.num_recurrence = config.prompt_layers
