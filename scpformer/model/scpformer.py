"""
ScpFormer: a transformer model for single-cell proteomics.

Encoder-only transformer that encodes single-cell protein expression profiles
using pre-computed protein language model embeddings as the token embedding
layer.

Classes:
    ScpFormerOutput          -- dataclass wrapping all model outputs.
    ScpFormerModel           -- core encoder model (HuggingFace PreTrainedModel).
    ScpFormerForClassification -- thin wrapper adding a classification head.
"""

import os
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from .config import ScpFormerConfig
from .flash_attention import FlashScpFormerEncoder, FlashScpFormerLayer
from .modules import ClsDecoder, ContinuousValueEncoder, ExprDecoder


@dataclass
class ScpFormerOutput(ModelOutput):
    """Container for ScpFormer forward-pass outputs."""

    cell_emb: Optional[Tensor] = None
    cls_output: Optional[Tensor] = None
    ctx_preds: Optional[Tensor] = None
    tgt_preds: Optional[Tensor] = None
    prot_emb: Optional[Tensor] = None


class ScpFormerModel(PreTrainedModel):
    """Transformer encoder for single-cell proteomics.

    Args:
        config: Model hyper-parameters (see :class:`ScpFormerConfig`).
        protein_emb_dim: Dimensionality of the input protein embeddings
            (e.g., 960 for ESM-2 ``esm2_t30_150M``).
    """

    config_class = ScpFormerConfig

    def _init_weights(self, module):
        pass

    PAD_TOKEN_IDX = 0
    CLS_TOKEN_IDX = 1

    def __init__(self, config: ScpFormerConfig, protein_emb_dim: int = 960):
        super().__init__(config)
        d_model = config.n_embd

        self.d_model = d_model
        self.protein_emb_dim = protein_emb_dim
        self.use_generative_training = config.use_generative_training

        # ── Embedding layers ─────────────────────────────────────────
        self.special_token_embeddings = nn.Embedding(2, d_model)
        nn.init.normal_(self.special_token_embeddings.weight, mean=0.0, std=0.02)

        self.protein_embedding_projection = nn.Sequential(
            nn.Linear(protein_emb_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.value_encoder = ContinuousValueEncoder(d_model, config.dropout)
        self.flag_encoder = nn.Embedding(2, d_model)
        self.input_layernorm = nn.LayerNorm(d_model)

        # ── Transformer encoder ──────────────────────────────────────
        if config.use_generative_training:
            layer = FlashScpFormerLayer(
                d_model, config.n_head, config.d_hid, config.dropout,
                batch_first=True, norm_scheme="post",
            )
            self.transformer_encoder = FlashScpFormerEncoder(layer, config.n_layer)
        else:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
            layer = TransformerEncoderLayer(
                d_model, config.n_head, config.d_hid, config.dropout,
                batch_first=True,
            )
            self.transformer_encoder = TransformerEncoder(layer, config.n_layer)

        self.decoder = ExprDecoder(d_model)

    # ==================================================================
    # Embedding helpers (private)
    # ==================================================================

    def _embed_proteins(
        self,
        protein_embs: Tensor,
        expression_values: Tensor,
        padding_mask: Optional[Tensor],
    ) -> Tensor:
        """Project protein embeddings, fill special tokens, and add value encoding."""
        batch_size, seq_len, _ = protein_embs.shape
        device = protein_embs.device
        dtype = protein_embs.dtype

        token_embs = torch.zeros(batch_size, seq_len, self.d_model, dtype=dtype, device=device)

        real_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        real_mask[:, 0] = False  # position 0 is CLS
        if padding_mask is not None:
            real_mask = real_mask & (~padding_mask)

        if real_mask.any():
            token_embs[real_mask] = self.protein_embedding_projection(protein_embs[real_mask])

        token_embs[:, 0, :] = self.special_token_embeddings.weight[self.CLS_TOKEN_IDX]
        if padding_mask is not None:
            token_embs[padding_mask] = self.special_token_embeddings.weight[self.PAD_TOKEN_IDX]

        value_embs = torch.zeros_like(token_embs)
        if real_mask.any():
            value_embs[real_mask] = self.value_encoder(expression_values[real_mask])

        return self.input_layernorm(token_embs + value_embs)

    def _transformer_forward(
        self,
        ctx_embs: Tensor,
        ctx_padding_mask: Tensor,
        tgt_protein_embs: Optional[Tensor],
        tgt_padding_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the transformer encoder on perception (and optionally generation) inputs."""
        device = ctx_embs.device

        if tgt_protein_embs is not None:
            tgt_token_embs = self.protein_embedding_projection(tgt_protein_embs)
            tgt_flags = self.flag_encoder(
                torch.tensor(1, device=device),
            ).expand(tgt_protein_embs.shape[0], tgt_protein_embs.shape[1], -1)
            tgt_embs = self.input_layernorm(tgt_token_embs + tgt_flags)
        else:
            tgt_embs = None

        ctx_output, tgt_output = self.transformer_encoder(
            ctx_embs,
            tgt_embs,
            ctx_padding_mask.bool(),
            tgt_padding_mask.bool() if tgt_padding_mask is not None else None,
        )

        return ctx_output, tgt_output

    # ==================================================================
    # Public forward
    # ==================================================================

    def generative_forward_with_embeddings(
        self,
        ctx_prot_embs: Tensor,
        ctx_expr: Tensor,
        ctx_key_padding_mask: Tensor,
        tgt_prot_embs: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Mapping[str, Tensor]:
        """Main forward pass using pre-computed protein embeddings.

        Returns dict containing ``"cell_emb"``, ``"prot_emb"``, ``"ctx_preds"``,
        ``"tgt_preds"``.
        """
        ctx_embs = self._embed_proteins(
            ctx_prot_embs, ctx_expr, ctx_key_padding_mask,
        )

        ctx_output, tgt_output = self._transformer_forward(
            ctx_embs, ctx_key_padding_mask,
            tgt_protein_embs=tgt_prot_embs,
            tgt_padding_mask=tgt_key_padding_mask,
        )

        if tgt_output is None:
            transformer_output = ctx_output
        else:
            transformer_output = torch.cat([ctx_output, tgt_output], dim=1)

        full_preds = self.decoder(transformer_output)["pred"]

        ctx_len = ctx_prot_embs.shape[1]
        return {
            "cell_emb": transformer_output[:, 0, :],
            "prot_emb": transformer_output,
            "ctx_preds": full_preds[:, :ctx_len],
            "tgt_preds": full_preds[:, ctx_len:],
        }

    def forward(self, **kwargs) -> ScpFormerOutput:
        """HuggingFace-compatible forward that wraps :meth:`generative_forward_with_embeddings`."""
        return ScpFormerOutput(**self.generative_forward_with_embeddings(**kwargs))

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def load_pretrained_weights(model: "ScpFormerModel", checkpoint_path: str):
        """Load pretrained weights, matching by key name and shape."""
        bin_path = os.path.join(str(checkpoint_path), "pytorch_model.bin")
        if os.path.isfile(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            pretrained = ScpFormerModel.from_pretrained(str(checkpoint_path))
            state_dict = pretrained.state_dict()
            del pretrained

        model_dict = model.state_dict()
        matched = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        skipped = [k for k in state_dict if k not in matched]
        model_dict.update(matched)
        model.load_state_dict(model_dict)

        print(f"  Loaded {len(matched)} params, skipped {len(skipped)}")
        if skipped:
            print(f"  Skipped keys: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

        return model


class ScpFormerForClassification(ScpFormerModel):
    """ScpFormer with a classification head on the ``<cls>`` token.

    Args:
        config: Model configuration.
        num_classes: Number of output classes.
    """

    def __init__(self, config: ScpFormerConfig, num_classes: int):
        super().__init__(config)
        self.cls_decoder = ClsDecoder(
            d_model=config.n_embd,
            n_cls=num_classes,
            nlayers=3,
            activation=nn.LeakyReLU,
        )

    def forward(
        self,
        prot_embs: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        **kwargs,
    ) -> ScpFormerOutput:
        outputs = self.generative_forward_with_embeddings(
            ctx_prot_embs=prot_embs,
            ctx_expr=values,
            ctx_key_padding_mask=src_key_padding_mask,
            tgt_prot_embs=None,
            tgt_key_padding_mask=None,
        )
        cell_emb = outputs["cell_emb"]
        return ScpFormerOutput(
            cell_emb=cell_emb,
            cls_output=self.cls_decoder(cell_emb),
        )
