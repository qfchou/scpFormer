from functools import lru_cache
import math
from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.transformer import _get_clones

# Flash Attention 2.x imports
from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input
from .attention import MultiheadAttention


class FlashScpFormerMHA(nn.Module):
    """
    Custom MHA layer for ScpFormer. This takes two separate forward passes on the pect
    proteins, and on the gen proteins.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
    ) -> None:
        assert batch_first
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)

        # Flash Attention 2.x uses functional API directly
        self.attention_dropout = attention_dropout

        self.cross_attn = MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=batch_first,
            **factory_kwargs,
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(
        self,
        ctx_total_embs: Tensor,
        tgt_total_embs: Tensor,
        ctx_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        need_weights=False,
    ):
        """
        ctx_total_embs: (batch, ctx_len, hidden_dim) (where hidden_dim = num heads * head dim)
        tgt_total_embs: (batch, tgt_len, hidden_dim)
        ctx_key_padding_mask: bool tensor of shape (batch, ctx_len), 1 means valid and 0 means not valid.
        tgt_key_padding_mask: bool tensor of shape (batch, tgt_len), 1 means valid and 0 means not valid.
        """
        batch_size = ctx_total_embs.shape[0]
        ctx_seqlen = ctx_total_embs.shape[1]
        ctx_qkv = self.Wqkv(ctx_total_embs)

        ctx_qkv = rearrange(
            ctx_qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads
        )

        # full self attention on ctx proteins
        # Flash Attention 2.x functional API with proper padding mask support
        if ctx_key_padding_mask is not None and not ctx_key_padding_mask.all():
            # Has padding tokens - use varlen version
            # unpad_input expects 1=valid, 0=padding
            ctx_qkv_for_unpad = rearrange(ctx_qkv, "b s three h d -> b s (three h d)")
            ctx_qkv_unpad, indices, cu_seqlens, max_seqlen, seqused = unpad_input(
                ctx_qkv_for_unpad, ctx_key_padding_mask
            )
            ctx_qkv_unpad = rearrange(
                ctx_qkv_unpad, "nnz (three h d) -> nnz three h d",
                three=3, h=self.num_heads
            )

            # Use varlen flash attention
            ctx_context_unpad = flash_attn_varlen_qkvpacked_func(
                ctx_qkv_unpad,
                cu_seqlens,
                max_seqlen,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=self.causal,
            )

            # Pad back to original shape
            ctx_context = pad_input(
                rearrange(ctx_context_unpad, "nnz h d -> nnz (h d)"),
                indices,
                batch_size,
                ctx_seqlen
            )
            ctx_context = rearrange(ctx_context, "b s (h d) -> b s h d", h=self.num_heads)
        else:
            # No padding - use regular version
            ctx_context = flash_attn_qkvpacked_func(
                ctx_qkv,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=self.causal,
            )
        ctx_attn_weights = None if not need_weights else None  # Flash Attention 2 doesn't return weights

        ctx_context = self.out_proj(rearrange(ctx_context, "b s h d -> b s (h d)"))

        if tgt_total_embs is None:
            return (ctx_context, None), (ctx_attn_weights, None)

        tgt_qkv = self.Wqkv(tgt_total_embs)
        tgt_qkv = rearrange(
            tgt_qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads
        )

        # CROSS ATTENTION USING RAW PYTORCH IMPLEMENTATION
        cross_q = tgt_qkv[:, :, 0, :, :]  # (batch, tgt_len, nheads, head_dim)
        cross_q = rearrange(cross_q, "b tgt_s h d -> b tgt_s (h d)")
        cross_kv = torch.cat(
            [ctx_qkv[:, :, 1:, :, :], tgt_qkv[:, :, 1:, :, :]], dim=1
        )  # (batch, ctx_seq+tgt_seq, 2, nheads, head_dim)
        cross_kv = rearrange(cross_kv, "b ctx_tgt_s two h d -> b ctx_tgt_s two (h d)")

        # make the attention mask, for pytorch implementation, true means attention is not allowed
        @lru_cache(maxsize=1)
        def make_mask(q_len, k_len, device):
            attention_mask = torch.zeros(
                (q_len, k_len), device=device, dtype=torch.bool
            )  # (tgt_len, ctx_len+tgt_len)
            # make the last tgt_len by tgt_gen to be true, only the diagonal is allowed with false
            attention_mask[:, -q_len:] = ~torch.eye(
                q_len, device=device, dtype=torch.bool
            )
            return attention_mask

        attention_mask = make_mask(cross_q.shape[1], cross_kv.shape[1], cross_q.device)

        if ctx_key_padding_mask is None and tgt_key_padding_mask is None:
            key_padding_mask = None
        else:
            if ctx_key_padding_mask is None:
                ctx_key_padding_mask = torch.ones(
                    (ctx_qkv.shape[0], ctx_qkv.shape[1]),
                    device=ctx_qkv.device,
                    dtype=torch.bool,
                )
            elif tgt_key_padding_mask is None:
                tgt_key_padding_mask = torch.ones(
                    (tgt_qkv.shape[0], tgt_qkv.shape[1]),
                    device=tgt_qkv.device,
                    dtype=torch.bool,
                )
            key_padding_mask = ~torch.cat(
                [ctx_key_padding_mask, tgt_key_padding_mask], dim=1
            )
        cross_context, _ = self.cross_attn(
            cross_q,
            cross_kv[:, :, 0, :],
            cross_kv[:, :, 1, :],
            key_padding_mask=key_padding_mask,
            attn_mask=attention_mask,
        )
        tgt_context = cross_context  # (batch, tgt_len, hidden_dim)
        tgt_attn_weights = None

        return (ctx_context, tgt_context), (ctx_attn_weights, tgt_attn_weights)


class FlashScpFormerLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    The class is modified from torch.nn.TransformerEncoderLayer to support the
    FlashAttention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """

    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",  # "pre" or "post"
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = FlashScpFormerMHA(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=batch_first,
            attention_dropout=dropout,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if norm_scheme not in ["pre", "post"]:
            raise ValueError("norm_scheme must be either pre or post")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def _reverse_key_padding_mask(self, src_key_padding_mask):
        """
        Reverse the true false values of the key padding mask. This is because
        we follow pytorch rule that the mask is True for padded tokens, but
        in the inner flash MHA, it assumes the mask is False for padded tokens.
        """
        if src_key_padding_mask is None:
            return None

        if not src_key_padding_mask.any().item():
            # no padding tokens in src
            return None
        return ~src_key_padding_mask

    def forward(
        self,
        ctx_total_embs: Tensor,
        tgt_total_embs: Tensor,
        ctx_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        ctx_key_padding_mask_ = self._reverse_key_padding_mask(ctx_key_padding_mask)
        tgt_key_padding_mask_ = self._reverse_key_padding_mask(tgt_key_padding_mask)

        if self.norm_scheme == "pre":
            ctx_total_embs = self.norm1(ctx_total_embs)
            if tgt_total_embs is not None:
                tgt_total_embs = self.norm1(tgt_total_embs)
            ctx_total_embs2, tgt_total_embs2 = self.self_attn(
                ctx_total_embs,
                tgt_total_embs,
                ctx_key_padding_mask=ctx_key_padding_mask_,
                tgt_key_padding_mask=tgt_key_padding_mask_,
            )[0]
            ctx_total_embs = ctx_total_embs + self.dropout1(ctx_total_embs2)
            ctx_total_embs = self.norm2(ctx_total_embs)
            ctx_total_embs2 = self.linear2(
                self.dropout(self.activation(self.linear1(ctx_total_embs)))
            )
            ctx_total_embs = ctx_total_embs + self.dropout2(ctx_total_embs2)

            if tgt_total_embs is not None:
                tgt_total_embs = tgt_total_embs + self.dropout1(tgt_total_embs2)
                tgt_total_embs = self.norm2(tgt_total_embs)
                tgt_total_embs2 = self.linear2(
                    self.dropout(self.activation(self.linear1(tgt_total_embs)))
                )
                tgt_total_embs = tgt_total_embs + self.dropout2(tgt_total_embs2)
        else:
            ctx_total_embs2, tgt_total_embs2 = self.self_attn(
                ctx_total_embs,
                tgt_total_embs,
                ctx_key_padding_mask=ctx_key_padding_mask_,
                tgt_key_padding_mask=tgt_key_padding_mask_,
            )[0]
            ctx_total_embs = ctx_total_embs + self.dropout1(ctx_total_embs2)
            ctx_total_embs = self.norm1(ctx_total_embs)
            ctx_total_embs2 = self.linear2(
                self.dropout(self.activation(self.linear1(ctx_total_embs)))
            )
            ctx_total_embs = ctx_total_embs + self.dropout2(ctx_total_embs2)
            ctx_total_embs = self.norm2(ctx_total_embs)

            if tgt_total_embs is not None:
                tgt_total_embs = tgt_total_embs + self.dropout1(tgt_total_embs2)
                tgt_total_embs = self.norm1(tgt_total_embs)
                tgt_total_embs2 = self.linear2(
                    self.dropout(self.activation(self.linear1(tgt_total_embs)))
                )
                tgt_total_embs = tgt_total_embs + self.dropout2(tgt_total_embs2)
                tgt_total_embs = self.norm2(tgt_total_embs)

        return ctx_total_embs, tgt_total_embs


class FlashScpFormerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        mask_check=True,
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.mask_check = mask_check

    def forward(
        self,
        ctx_total_embs: Tensor,
        tgt_total_embs: Tensor,
        ctx_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if ctx_key_padding_mask is not None:
            _skpm_dtype = ctx_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(
                ctx_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )

        for mod in self.layers:
            ctx_total_embs, tgt_total_embs = mod(
                ctx_total_embs,
                tgt_total_embs,
                ctx_key_padding_mask,
                tgt_key_padding_mask,
            )

        if self.norm is not None:
            ctx_total_embs = self.norm(ctx_total_embs)
            tgt_total_embs = self.norm(tgt_total_embs)

        return ctx_total_embs, tgt_total_embs
