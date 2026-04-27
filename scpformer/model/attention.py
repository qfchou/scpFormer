from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class MultiheadAttention(Module):
    """Modified multi-head attention from torch.nn.MultiheadAttention.

    Uses identity projection weights (no learned in-projection or out-projection)
    to serve as a cross-attention module in the FlashScpFormerMHA layer.

    Reference: `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.
    """

    __constants__ = ["batch_first"]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        batch_first=True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self._qkv_same_embed_dim = True

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        # Identity projection weights (no learned parameters)
        self.in_proj_weight = torch.eye(embed_dim, **factory_kwargs)
        self.out_proj_weight = torch.eye(embed_dim, **factory_kwargs)

    def __setstate__(self, state):
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True
        super(MultiheadAttention, self).__setstate__(state)

    def _apply(self, fn):
        self.in_proj_weight = fn(self.in_proj_weight)
        self.out_proj_weight = fn(self.out_proj_weight)
        return super()._apply(fn)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            query: Query embeddings of shape (N, L, E_q) when batch_first=True.
            key: Key embeddings of shape (N, S, E_k) when batch_first=True.
            value: Value embeddings of shape (N, S, E_v) when batch_first=True.
            key_padding_mask: If specified, a mask of shape (N, S). True means ignore.
            need_weights: If True, return attention weights.
            attn_mask: 2D or 3D mask preventing attention to certain positions.
            average_attn_weights: If True, average weights across heads.

        Returns:
            attn_output: Attention output of shape (N, L, E).
            attn_output_weights: Attention weights (if need_weights=True).
        """
        is_batched = query.dim() == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != torch.bool and not torch.is_floating_point(
                key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            in_proj_weight=None,
            in_proj_bias=None,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.dropout,
            out_proj_weight=self.out_proj_weight,
            out_proj_bias=None,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=self.in_proj_weight,
            k_proj_weight=self.in_proj_weight,
            v_proj_weight=self.in_proj_weight,
            average_attn_weights=average_attn_weights,
        )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
