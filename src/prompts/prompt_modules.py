import math
from typing import Tuple, Any

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_

from src.prompts.prompt_functional import attentive_max_pooling, linear


class EntityEncoder(nn.Module):
    """Encodes sequences into a fixed sized vector via attentive pooling."""
    def __init__(self, emb_dim, d_model, num_entities):
        super(EntityEncoder, self).__init__()
        self.entity_trsf_w = nn.Parameter(torch.empty(emb_dim, d_model))
        self.entity_trsf_b = nn.Parameter(torch.empty(1, emb_dim))
        self.num_entities = num_entities

        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.entity_trsf_w)
        constant_(self.entity_trsf_b, 0)

    def forward(self, hidden: torch.Tensor, lengths: dict, hidden_mask: torch.Tensor = None) -> Tuple[torch.Tensor, Any]:
        entity_reps = []
        cumm_len = 0
        ent_count = 0
        new_hidden_mask = torch.zeros_like(hidden_mask[:, :, 0]).unsqueeze(1).repeat(1, self.num_entities, 1)
        mask_val = torch.finfo(hidden.dtype).min

        for ent_name, len_ in lengths.items():
            entity_seq_mask = hidden_mask[:, cumm_len:cumm_len + len_, cumm_len:cumm_len + len_]
            if torch.any(torch.all((entity_seq_mask == mask_val), dim=-1).sum(-1)):
                # all masked produces nans in the entity_rep
                entity_seq_mask[:, 0] = 0.0
            entity_seq = hidden[:, cumm_len:cumm_len + len_, :]
            new_hidden_mask[:, ent_count, cumm_len:cumm_len + len_] = 1
            entity_rep, _ = attentive_max_pooling(
                entity_seq, entity_seq_mask, self.entity_trsf_w, self.entity_trsf_b
            )
            entity_reps.append(entity_rep)
            cumm_len += len_
            ent_count += 1
        return torch.stack(entity_reps, dim=1), new_hidden_mask


class GroupMLP(nn.Module):
    def __init__(self, in_dim, out_dim, num, rank):
        super().__init__()
        self.group_mlp1 = GroupLinearLayer(in_dim, rank, num)
        self.group_mlp2 = GroupLinearLayer(rank, out_dim, num)

    def forward(self, x):
        x = torch.relu(self.group_mlp1(x))
        x = self.group_mlp2(x)
        return x


class GroupLinearLayer(nn.Module):
    def __init__(self, din, dout, num_blocks, bias=True, a=None):
        super(GroupLinearLayer, self).__init__()
        self.nb = num_blocks
        self.dout = dout
        if a is None:
            a = 1. / math.sqrt(dout)
        self.weight = nn.Parameter(torch.FloatTensor(num_blocks, din, dout).uniform_(-a, a))
        self.bias = bias
        if bias is True:
            self.bias = nn.Parameter(torch.FloatTensor(num_blocks, dout).uniform_(-a, a))
        else:
            self.bias = None

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = torch.bmm(x, self.weight)
        x = x.permute(1, 0, 2)
        if self.bias is not None:
            x = x + self.bias
        return x


class SelectAttention(nn.Module):
    """docstring for SelectAttention"""

    def __init__(self, d_read, d_write, d_inter, num_read, num_write, share_query=False, share_key=False):
        super(SelectAttention, self).__init__()
        if not share_query:
            self.gll_read = GroupLinearLayer(d_read, d_inter, num_read)
        else:
            self.gll_read = nn.Linear(d_read, d_inter)
        if not share_key:
            self.gll_write = GroupLinearLayer(d_write, d_inter, num_write)
        else:
            self.gll_write = nn.Linear(d_write, d_inter)
        self.temperature = math.sqrt(d_inter)

    def forward(self, read_q, write_k):
        read = self.gll_read(read_q)
        write = self.gll_write(write_k)
        return torch.bmm(read, write.permute(0, 2, 1)) / self.temperature


class TopkArgMax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, dim=-1, k=2):
        idx = torch.topk(input, k=k, dim=dim)[1]
        ctx._input_shape = input.shape
        ctx._input_dtype = input.dtype
        ctx._input_device = input.device
        op = torch.zeros(input.size()).to(input.device)
        op.scatter_(1, idx, 1)
        ctx.save_for_backward(op)
        return op

    @staticmethod
    def backward(ctx, grad_output):
        op, = ctx.saved_tensors
        grad_input = grad_output * op
        return grad_input


