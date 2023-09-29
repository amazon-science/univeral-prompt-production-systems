import torch
from torch.overrides import has_torch_function_variadic, handle_torch_function

from typing import Optional



def linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Shape:

        - Input: :math:`(N, *, in\_features)` N is the batch size, `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    if has_torch_function_variadic(input, weight):
        return handle_torch_function(linear, (input, weight), input, weight, bias=bias)
    return torch._C._nn.linear(input, weight, bias)


def attentive_max_pooling(
        hidden: torch.Tensor,
        attn_mask: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None
):
    hidden = linear(hidden, weight, bias)
    k = hidden.permute(0, 2, 1)
    k = k/k.norm(dim=1).unsqueeze(1)  # normalized key
    q = hidden.max(dim=1)[0].unsqueeze(1)  # max pooled query
    attn = torch.bmm(q, k)
    if attn_mask is not None:
        attn += attn_mask[:, :1, :]
    attn = torch.softmax(attn, dim=2)
    pooled_rep = torch.matmul(attn, hidden).squeeze(1)
    return pooled_rep, attn


def get_gumbels(logits: torch.Tensor):
    gumbels_ = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )
    return gumbels_


def topk_gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1, k=2):
    # type: (Tensor, Tensor, float, bool, float, int) -> Tensor
    r"""
    Samples from the `Gumbel-Softmax distribution`_ and optionally discretizes.
    You can use this function to replace "F.gumbel_softmax".

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.
    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.
    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`
      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)
    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.topk_gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.topk_gumbel_softmax(logits, tau=1, hard=True)
    .. _Gumbel-Softmax distribution:
        https://arxiv.org/abs/1611.00712
        https://arxiv.org/abs/1611.01144
    """

    def _gen_gumbels():
        gumbels = get_gumbels(logits)
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum() or torch.isneginf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.topk(k=k, dim=dim)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = None
    return ret
