"""
Shared data loading and preprocessing utilities for protein-based cell data.

Provides:
- load_protein_embeddings: load pre-computed protein embeddings from .npz
- preprocess_expressions: log1p + min-max normalization
- prepend_special_tokens / pad_batch: sequence construction utilities
- ProteinCellDataset: unified dataset for all downstream tasks
- ProteinCollator: unified collator for perception-based tasks (annotation, embedding extraction)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def load_protein_embeddings(embeddings_path):
    """
    Load pre-computed protein embeddings from .npz file.

    Expected keys in npz: 'accessions', 'mean_embeddings'

    Args:
        embeddings_path: Path to .npz file

    Returns:
        accession_to_embedding: dict mapping accession string -> 1D numpy array
        emb_dim: int, dimension of protein embeddings
    """
    print(f"Loading protein embeddings from {embeddings_path}...")
    npz_data = np.load(embeddings_path, allow_pickle=True)
    accessions = npz_data['accessions']
    mean_embeddings = npz_data['mean_embeddings']

    accession_to_embedding = {}
    for acc, emb in zip(accessions, mean_embeddings):
        if emb.ndim == 2 and emb.shape[0] == 1:
            accession_to_embedding[acc] = emb[0]
        elif emb.ndim > 1:
            accession_to_embedding[acc] = emb.mean(0)
        else:
            accession_to_embedding[acc] = emb

    emb_dim = list(accession_to_embedding.values())[0].shape[-1]
    print(f"  Loaded {len(accession_to_embedding)} protein embeddings (dim={emb_dim})")
    return accession_to_embedding, emb_dim


def preprocess_expressions(exprs, do_log1p=False, do_arcsinh=False):
    """
    Shared expression preprocessing: optional log1p or arcsinh -> min-max normalize to [0, 10].

    Args:
        exprs: 1D torch tensor of expression values (only valid protein positions,
               no special tokens like CLS/PAD)
        do_log1p: whether to apply log1p transformation first
        do_arcsinh: whether to apply arcsinh transformation first

    Returns:
        Preprocessed 1D torch tensor
    """
    if len(exprs) == 0:
        return exprs

    if do_log1p:
        exprs = torch.log1p(exprs)
    elif do_arcsinh:
        exprs = torch.arcsinh(exprs/5)

    expr_min = exprs.min()
    expr_max = exprs.max()
    if expr_max > expr_min:
        exprs = (exprs - expr_min) / (expr_max - expr_min) * 10.0
    else:
        exprs = torch.zeros_like(exprs)

    return exprs.float()


def prepend_special_tokens(embs, exprs, pad_value=-2):
    """
    Prepend CLS token (and optionally batch token) to a single cell's data.

    Args:
        embs: [n_proteins, emb_dim] tensor of protein embeddings
        exprs: [n_proteins] tensor of expression values
        pad_value: value used for special token expression positions
        use_batch_token: if True, also prepend a batch token (mean of protein embeddings)

    Returns:
        embs: [n_proteins + num_special, emb_dim] with CLS (and batch token) prepended
        exprs: [n_proteins + num_special] with pad_value at special positions
    """
    emb_dim = embs.shape[1] if embs.dim() == 2 and embs.shape[0] > 0 else (embs.shape[1] if embs.dim() == 2 else 0)
    cls_emb = torch.zeros(1, emb_dim, dtype=torch.float32)
    cls_expr = torch.tensor([pad_value], dtype=torch.float32)

    embs = torch.cat([cls_emb, embs.float()], dim=0)
    exprs = torch.cat([cls_expr, exprs], dim=0)

    return embs, exprs


def pad_batch(embs_list, exprs_list, pad_value=-2, max_length=None):
    """
    Pad a batch of variable-length sequences to the same length.

    Args:
        embs_list: list of [seq_i, emb_dim] tensors
        exprs_list: list of [seq_i] tensors
        pad_value: value used for padding expression positions
        max_length: optional maximum sequence length cap

    Returns:
        prot_embs: [B, max_len, emb_dim]
        values: [B, max_len]
        src_key_padding_mask: [B, max_len] bool tensor (True = padded)
    """
    batch_max = max(e.shape[0] for e in embs_list)
    if max_length is not None:
        batch_max = min(batch_max, max_length)

    emb_dim = embs_list[0].shape[1]
    pad_emb = torch.zeros(emb_dim, dtype=torch.float32)

    padded_embs = []
    padded_vals = []
    padded_masks = []

    for embs, vals in zip(embs_list, exprs_list):
        cur_len = embs.shape[0]
        if cur_len < batch_max:
            pad_len = batch_max - cur_len
            embs = torch.cat([embs, pad_emb.unsqueeze(0).expand(pad_len, -1)], dim=0)
            vals = torch.cat([vals, torch.full((pad_len,), pad_value, dtype=torch.float32)], dim=0)
            mask = torch.cat([
                torch.zeros(cur_len, dtype=torch.bool),
                torch.ones(pad_len, dtype=torch.bool),
            ])
        elif cur_len > batch_max:
            embs = embs[:batch_max]
            vals = vals[:batch_max]
            mask = torch.zeros(batch_max, dtype=torch.bool)
        else:
            mask = torch.zeros(batch_max, dtype=torch.bool)

        padded_embs.append(embs)
        padded_vals.append(vals)
        padded_masks.append(mask)

    return (
        torch.stack(padded_embs),
        torch.stack(padded_vals),
        torch.stack(padded_masks),
    )


class ProteinCellDataset(Dataset):
    """
    Unified dataset for protein-based cell data.

    Pre-computes valid protein indices (those with embeddings) and returns
    per-cell: prot_embeddings, expressions, and optionally labels.

    Args:
        adata: AnnData object
        accession_to_embedding: dict mapping accession -> embedding vector
        protein_emb_dim: dimension of protein embeddings
        celltype_labels: optional numpy array of int celltype labels [N]
        batch_labels: optional numpy array of int batch labels [N]
        embedding_key: column in adata.var to match against accession_to_embedding
    """

    def __init__(
        self,
        adata,
        accession_to_embedding,
        protein_emb_dim=960,
        celltype_labels=None,
        batch_labels=None,
        embedding_key="accession",
        do_log1p=False,
    ):
        self.adata = adata
        self.celltype_labels = celltype_labels
        self.batch_labels = batch_labels
        self.do_log1p = do_log1p

        # Determine match keys from adata.var
        if embedding_key in adata.var.columns:
            match_keys = adata.var[embedding_key].tolist()
        elif "protein_name" in adata.var.columns:
            match_keys = adata.var["protein_name"].tolist()
        else:
            match_keys = adata.var.index.tolist()

        # Pre-compute valid protein indices and embeddings
        self.valid_indices = []
        valid_embs = []
        for i, key in enumerate(match_keys):
            if key in accession_to_embedding:
                self.valid_indices.append(i)
                valid_embs.append(accession_to_embedding[key])

        self.valid_indices = np.array(self.valid_indices)
        self.protein_embeddings = np.array(valid_embs, dtype=np.float32)

        print(
            f"  Dataset: {adata.shape[0]} cells, "
            f"{len(self.valid_indices)}/{adata.shape[1]} proteins with embeddings"
        )

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        # Get expression values for valid proteins
        if hasattr(self.adata.X, 'toarray'):
            expressions = self.adata.X[idx].toarray().flatten()
        else:
            expressions = self.adata.X[idx].flatten()

        valid_expressions = expressions[self.valid_indices].astype(np.float32)
        exprs_tensor = torch.from_numpy(valid_expressions)

        # Preprocess: log1p + per-cell min-max normalize to [0, 10]
        if len(exprs_tensor) > 0:
            exprs_tensor = preprocess_expressions(exprs_tensor, do_log1p=self.do_log1p)

        item = {
            'prot_embeddings': torch.from_numpy(self.protein_embeddings.copy()),
            'expressions': exprs_tensor,
        }

        if self.celltype_labels is not None:
            item['celltype_label'] = torch.tensor(
                self.celltype_labels[idx], dtype=torch.long
            )
        if self.batch_labels is not None:
            item['batch_labels'] = torch.tensor(
                self.batch_labels[idx], dtype=torch.long
            )

        return item


class ProteinCollator:
    """
    Unified data collator for perception-based tasks (annotation, integration).

    Handles: truncation → CLS prepend → padding → optional MLM masking.
    Automatically collects celltype_labels and batch_labels when present.

    Args:
        pad_value: value used for CLS and padding positions (default -2)
        max_length: maximum sequence length including CLS token
        do_mlm: if True, randomly mask expression values and return target_values
        mlm_probability: fraction of non-special tokens to mask (when do_mlm=True)
        mask_value: value to fill at masked positions (default -1)
    """

    def __init__(
        self,
        pad_value=-2,
        max_length=2000,
        do_mlm=False,
        mlm_probability=0.4,
        mask_value=-1,
    ):
        self.pad_value = pad_value
        self.max_length = max_length
        self.do_mlm = do_mlm
        self.mlm_probability = mlm_probability
        self.mask_value = mask_value
        self.num_special_tokens = 1  # CLS

    def __call__(self, features):
        prot_embs_list = []
        values_list = []
        celltype_labels = []
        batch_labels = []

        for f in features:
            embs = f["prot_embeddings"]   # [n_valid, emb_dim]
            exprs = f["expressions"]      # [n_valid]
            num_proteins = len(exprs)

            # Truncate if too many proteins
            max_proteins = self.max_length - self.num_special_tokens
            if num_proteins > max_proteins:
                indices = torch.randperm(num_proteins)[:max_proteins]
                embs = embs[indices]
                exprs = exprs[indices]

            # Prepend CLS token
            embs, exprs = prepend_special_tokens(
                embs, exprs, pad_value=self.pad_value,
            )

            prot_embs_list.append(embs)
            values_list.append(exprs)

            if "celltype_label" in f:
                celltype_labels.append(f["celltype_label"])
            if "batch_labels" in f:
                batch_labels.append(f["batch_labels"])

        # Pad to batch max length
        prot_embs, values, src_key_padding_mask = pad_batch(
            prot_embs_list, values_list,
            pad_value=self.pad_value,
            max_length=self.max_length,
        )

        result = {
            "prot_embs": prot_embs,                        # [B, seq, emb_dim]
            "values": values,                               # [B, seq]
            "src_key_padding_mask": src_key_padding_mask,   # [B, seq]
        }

        # MLM masking (for integration-style training)
        if self.do_mlm:
            result["target_values"] = values.clone()
            mask_positions = torch.rand(values.shape) < self.mlm_probability
            mask_positions = mask_positions & (~src_key_padding_mask)
            mask_positions[:, :self.num_special_tokens] = False
            result["values"] = torch.where(
                mask_positions,
                torch.tensor(self.mask_value, dtype=values.dtype),
                values,
            )

        if celltype_labels:
            result["celltype_labels"] = torch.stack(celltype_labels)
        if batch_labels:
            result["batch_labels"] = torch.stack(batch_labels)
        else:
            result["batch_labels"] = None

        return result
