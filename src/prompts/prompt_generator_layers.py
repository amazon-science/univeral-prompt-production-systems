import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

from transformers.activations import ACT2FN
from src.multi_task_data.multi_task_dataset import MAX_LENGTHS
from src.prompts.prompt_modules import EntityEncoder, SelectAttention, TopkArgMax, GroupMLP, GroupLinearLayer
from src.prompts.prompt_functional import topk_gumbel_softmax


class LinearPromptLayers(nn.Module):
    def __init__(self, config, model_layers):
        super().__init__()
        embed_dim = config.d_model
        mid_dim = embed_dim * 3 // 4
        kv = 2  # one for key and one for value

        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Tanh()) for _ in range(config.prompt_layers - 2)]
        )
        self.layers.insert(0, nn.Sequential(nn.Linear(embed_dim, mid_dim), nn.Tanh()))
        self.layers.append(nn.Linear(mid_dim, embed_dim * model_layers * kv))

    def forward(self, prompt_states):
        for idx, prompt_layer in enumerate(self.layers):
            prompt_states = prompt_layer(prompt_states)
        return prompt_states


class PromptLayerAttention(nn.MultiheadAttention):
    def __init__(self, **kwargs) -> None:
        super(PromptLayerAttention, self).__init__(**kwargs)


class TransformerPromptLayer(nn.TransformerEncoderLayer):
    def __init__(self, config):
        super().__init__(
            d_model=config.prompt_d_model,
            nhead=config.prompt_attention_heads,
            dim_feedforward=config.prompt_ffn_dim,
            dropout=config.prompt_dropout,
            batch_first=True,
        )
        self.prompt_num_heads = config.prompt_attention_heads

    def _sa_block(self, x, attn_mask, key_padding_mask) -> torch.Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src_mask = src_mask.unsqueeze(1).repeat(1, self.prompt_num_heads, 1, 1)
        src_mask = src_mask.view((-1,) + src_mask.size()[-2:])
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x


class NeuralPromptProducerLayer(nn.Module):
    def __init__(self, config, num_prompts):
        super().__init__()
        d_model = config.prompt_d_model
        self.head_dim = config.prompt_d_model // config.prompt_attention_heads
        self.scaling = self.head_dim ** -0.5
        select_dim = self.head_dim // 2
        self.num_rules = config.prompt_attention_heads

        self.top_k = config.tok_k_rules
        self.use_trf_hidden = config.prompt_hidden_condition
        self.dropout = config.prompt_dropout

        self.prompts_len = num_prompts
        self.sources_len = MAX_LENGTHS["sources"]
        self.domains_len = MAX_LENGTHS["domains"]
        self.descriptor_len = config.max_descriptor_length
        self.seq_entity_lengths = {
            "inductive_entity": self.prompts_len,
            #"sources_domains_entity": self.sources_len + self.domains_len,
            "descriptor_entity": self.sources_len + self.domains_len + self.descriptor_len
        }
        if self.use_trf_hidden:
            max_len = min(config.max_source_length, MAX_LENGTHS['prompt_input_h'])
            self.seq_entity_lengths["input_rep_entity1"] = max_len
            #self.seq_entity_lengths["input_rep_entity2"] = max_len - max_len // 2
            #self.seq_entity_lengths["input_rep_entity3"] = max_len - max_len // 3 * 2
        self.num_entities = len(self.seq_entity_lengths)
        self.ctx_size = self.num_entities//2
        self.entity_encoder = EntityEncoder(select_dim, d_model, self.num_entities)
        self.rules_embed = nn.Parameter(torch.empty(self.top_k, self.num_rules, self.head_dim))

        self.entity_rule_selector = SelectAttention(
            d_read=self.head_dim, d_write=select_dim, d_inter=select_dim, num_read=self.num_rules,
            num_write=self.num_entities, share_query=True
        )
        self.entity_selector = SelectAttention(
            d_read=self.head_dim, d_write=select_dim, d_inter=select_dim//2, num_read=self.top_k,
            num_write=self.num_entities, share_query=False
        )
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(self.head_dim, d_model, bias=True)

        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = self.dropout
        self.fc1 = GroupMLP(d_model, d_model * 4, 1, d_model // 2)
        self.fc2 = GroupMLP(d_model * 4, d_model, 1, d_model // 2)
        self.final_layer_norm = nn.LayerNorm(d_model)

        if torch.cuda.is_available():
            try:
                from apex.normalization import FusedLayerNorm
                self.self_attn_layer_norm = FusedLayerNorm(d_model)
                self.final_layer_norm = FusedLayerNorm(d_model)
            except ImportError:
                pass

        self.init_weights()

    def init_weights(self):
        xavier_normal_(self.rules_embed)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_rules, self.head_dim).transpose(1, 2).contiguous()

    def get_topk_masks(self, batch_size, rules_emb, entities_emb, dim=1):
        entities_emb = entities_emb.repeat(self.top_k, 1, 1)
        entity_rule_scores = self.entity_rule_selector(read_q=rules_emb, write_k=entities_emb)
        entity_rule_scores = entity_rule_scores.reshape(batch_size*self.top_k, -1)
        if self.training:
            mask = topk_gumbel_softmax(entity_rule_scores, tau=1.0, hard=True, dim=dim, k=1)
        else:
            mask = TopkArgMax().apply(entity_rule_scores, dim, 1)
        return mask

    def get_entity_mask(self, batch_size, selected_rules, entities_emb, dim=1):
        entity_scores = self.entity_selector(read_q=selected_rules, write_k=entities_emb)
        entity_scores = entity_scores.view(batch_size * self.top_k, self.num_entities)
        if self.training:
            mask = topk_gumbel_softmax(entity_scores, tau=0.5, hard=True, dim=dim, k=self.ctx_size)
        else:
            mask = TopkArgMax().apply(entity_scores, dim, self.ctx_size)
        return mask

    def get_masks(self, hidden_states, attention_mask):
        bsz, tgt_len, embed_dim = hidden_states.size()
        entities_emb, entity_seq_mask = self.entity_encoder(hidden_states, self.seq_entity_lengths, attention_mask)

        rules_emb = self.rules_embed.repeat(bsz, 1, 1)
        mask = self.get_topk_masks(bsz, rules_emb, entities_emb)

        rule_mask = (mask.view(bsz, self.top_k, self.num_rules, self.num_entities, 1)).sum(dim=3)
        rules_emb = rules_emb.unsqueeze(1).view(bsz, self.top_k, self.num_rules, -1)
        selected_rules_emb = (rules_emb * rule_mask).sum(dim=2)

        entity_mask = self.get_entity_mask(bsz, selected_rules_emb, entities_emb)
        return entity_mask, entity_seq_mask, rule_mask

    def topk_self_attn(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:

        bsz, tgt_len, embed_dim = hidden_states.size()
        entity_mask, entity_seq_mask, rule_mask = self.get_masks(hidden_states, attention_mask)

        proj_shape = (bsz * self.num_rules, -1, self.head_dim)
        hidden_states = hidden_states.view(bsz, 1, 1, tgt_len, embed_dim)
        entity_mask = entity_mask.view(bsz, self.top_k, self.num_entities, 1, 1)
        entity_seq_mask = entity_seq_mask.view(bsz, 1, self.num_entities, tgt_len, 1)
        hidden_states = (hidden_states * entity_mask * entity_seq_mask).sum(dim=(1, 2))

        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        query_states = self._shape(self.q_proj(hidden_states) * self.scaling, -1, bsz)
        query_states = query_states.view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = attn_weights.view(bsz, self.num_rules, tgt_len, src_len) + attention_mask.unsqueeze(1)
        attn_weights = attn_weights.view(bsz * self.num_rules, tgt_len, src_len)
        attn_probs = F.softmax(attn_weights, dim=-1)

        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.num_rules, tgt_len, self.head_dim)
        attn_output = attn_output.unsqueeze(1)
        attn_output = (attn_output * rule_mask.unsqueeze(-1)).sum(2)
        attn_output = attn_output.transpose(1, 2)
        attn_output = self.out_proj(attn_output).sum(2)

        return attn_output

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.topk_self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = hidden_states.view(-1, 1, hidden_states.size(-1))
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = hidden_states.view(*residual.size())
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states

    def rule_net(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """NOT USED"""
        bsz, tgt_len, embed_dim = hidden_states.size()
        entity_mask, entity_seq_mask, rule_mask = self.get_masks(hidden_states, attention_mask)

        value_states = self.rule_proj(hidden_states).view(bsz, 1, 1, tgt_len, embed_dim)
        entity_mask = entity_mask.view(bsz, self.top_k, self.num_entities, 1, 1)
        entity_seq_mask = entity_seq_mask.view(bsz, 1, self.num_entities, tgt_len, 1)
        rule_mlp_input = (value_states * entity_mask * entity_seq_mask).sum(dim=(1, 2))
        rule_mlp_input = rule_mlp_input.view(bsz * tgt_len, 1, embed_dim).expand(-1, self.num_rules, -1)
        attn_output = self.rule_prod(rule_mlp_input)
        attn_output = attn_output.view(bsz, tgt_len, self.num_rules, -1)
        attn_output = attn_output.unsqueeze(2).expand(-1, -1, self.top_k, -1, -1)
        attn_output = (attn_output * rule_mask.unsqueeze(1)).sum(3).mean(2)

        return attn_output
