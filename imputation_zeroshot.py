"""
Impute zero-valued expressions using a pretrained scpFormer model.

Follows the same ctx/tgt split as training (imputation.py):
- ctx: non-zero proteins  → [CLS] + observed expression values (visible to model)
- tgt: zero proteins      → protein embeddings only, model predicts expression

All output values are in the normalized space (log1p + min-max to [0, 10]):
- Non-zero positions: normalized input values
- Zero positions: model tgt_preds

Result is saved to adata.obsm['X_impu'].
"""

import argparse
import os
import json
import numpy as np
import torch
from tqdm import tqdm
import anndata as ad
import scipy.sparse as sp

from scpformer.model import ScpFormerModel
from scpformer.data import load_protein_embeddings, preprocess_expressions, prepend_special_tokens, pad_batch


class CellImputor:
    """Impute zero-valued expressions using a trained ScpFormerModel."""

    def __init__(self, model_path, protein_embeddings_path, device="cuda",
                 batch_size=64, do_log1p=False):
        self.device = device
        self.batch_size = batch_size
        self.do_log1p = do_log1p
        self.pad_value = -2

        # Load model config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)

        print(f"\n{'='*60}")
        print(f"Model Configuration:")
        print(f"  - Model path: {model_path}")
        print(f"  - n_embd: {config.get('n_embd')}")
        print(f"  - n_layer: {config.get('n_layer')}")
        print(f"{'='*60}\n")

        print(f"Loading model from {model_path}...")
        self.model = ScpFormerModel.from_pretrained(model_path, torch_dtype=torch.float16)
        self.model.to(device)
        self.model.eval()

        print(f"Loading protein embeddings from {protein_embeddings_path}...")
        self.accession_to_embedding, self.protein_emb_dim = load_protein_embeddings(
            protein_embeddings_path
        )

    def prepare_batch(self, cells_data):
        """
        Prepare a batch of cells with ctx/tgt split matching training.

        ctx = non-zero proteins (CLS + expression visible to model)
        tgt = zero proteins (embedding only, model predicts expression)

        Returns dict with GPU tensors for model inference plus CPU metadata
        for writing results back to the output matrix.
        """
        ctx_embs_list = []
        ctx_expr_list = []
        tgt_embs_list = []
        # Per-cell original indices for write-back
        cell_info = []  # list of (nonzero_orig_indices, zero_orig_indices)

        for accessions, expressions in cells_data:
            protein_embs = []
            valid_exprs = []
            valid_orig_indices = []

            for i, (acc, expr) in enumerate(zip(accessions, expressions)):
                if acc in self.accession_to_embedding:
                    protein_embs.append(self.accession_to_embedding[acc])
                    valid_exprs.append(expr)
                    valid_orig_indices.append(i)

            if len(protein_embs) > 0:
                prot_embs = torch.tensor(np.array(protein_embs), dtype=torch.float32)
                exprs_raw = torch.tensor(valid_exprs, dtype=torch.float32)
                orig_indices = torch.tensor(valid_orig_indices, dtype=torch.long)

                # Split: ctx = non-zero, tgt = zero
                nonzero_mask = (exprs_raw != 0)
                zero_mask = ~nonzero_mask

                ctx_embs = prot_embs[nonzero_mask]
                ctx_exprs_raw = exprs_raw[nonzero_mask]
                tgt_embs = prot_embs[zero_mask]

                nonzero_orig_indices = orig_indices[nonzero_mask]
                zero_orig_indices = orig_indices[zero_mask]

                # Normalize ctx expressions: log1p + per-cell min-max to [0, 10]
                if len(ctx_exprs_raw) > 0:
                    ctx_exprs = preprocess_expressions(ctx_exprs_raw, do_log1p=self.do_log1p)
                else:
                    ctx_exprs = ctx_exprs_raw
            else:
                ctx_embs = torch.empty(0, self.protein_emb_dim, dtype=torch.float32)
                ctx_exprs = torch.empty(0, dtype=torch.float32)
                tgt_embs = torch.empty(0, self.protein_emb_dim, dtype=torch.float32)
                nonzero_orig_indices = torch.empty(0, dtype=torch.long)
                zero_orig_indices = torch.empty(0, dtype=torch.long)

            # Prepend CLS to ctx only (same as training)
            ctx_embs, ctx_exprs = prepend_special_tokens(
                ctx_embs, ctx_exprs, pad_value=self.pad_value,
            )

            ctx_embs_list.append(ctx_embs)
            ctx_expr_list.append(ctx_exprs)
            tgt_embs_list.append(tgt_embs)
            cell_info.append((nonzero_orig_indices, zero_orig_indices))

        # Pad ctx and tgt separately
        ctx_prot_embs, ctx_expr, ctx_key_padding_mask = pad_batch(
            ctx_embs_list, ctx_expr_list, pad_value=self.pad_value,
        )

        # tgt needs dummy expr tensor for pad_batch (model doesn't use tgt expressions)
        tgt_dummy_exprs = [torch.zeros(e.shape[0], dtype=torch.float32) for e in tgt_embs_list]
        tgt_prot_embs, _, tgt_key_padding_mask = pad_batch(
            tgt_embs_list, tgt_dummy_exprs, pad_value=self.pad_value,
        )

        return {
            # GPU tensors for model
            'ctx_prot_embs': ctx_prot_embs.to(self.device, dtype=torch.float16),
            'ctx_expr': ctx_expr.to(self.device, dtype=torch.float16),
            'ctx_key_padding_mask': ctx_key_padding_mask.to(self.device),
            'tgt_prot_embs': tgt_prot_embs.to(self.device, dtype=torch.float16),
            'tgt_key_padding_mask': tgt_key_padding_mask.to(self.device),
            # CPU metadata for write-back
            'ctx_expr_cpu': ctx_expr,                       # normalized non-zero values
            'tgt_key_padding_mask_cpu': tgt_key_padding_mask,
            'cell_info': cell_info,                         # (nonzero_orig_indices, zero_orig_indices) per cell
        }

    @torch.no_grad()
    def impute(self, adata):
        """
        Impute zero-valued expressions for all cells.

        All output values are in the normalized space (log1p + min-max [0, 10]):
        - Non-zero positions: normalized input values (from ctx)
        - Zero positions: model tgt_preds

        Args:
            adata: AnnData with 'accession' in adata.var

        Returns:
            imputed_matrix: np.ndarray (n_cells, n_proteins), all in normalized space
        """
        n_cells, n_proteins = adata.shape
        protein_accessions = adata.var['accession'].values

        n_valid = sum(1 for acc in protein_accessions if acc in self.accession_to_embedding)
        print(f"Imputing {n_cells} cells, {n_proteins} proteins "
              f"({n_valid} with protein embeddings)...")

        # Output matrix: all values in normalized space
        imputed = np.zeros((n_cells, n_proteins), dtype=np.float32)
        n_imputed_total = 0

        for start_idx in tqdm(range(0, n_cells, self.batch_size), desc="Imputing"):
            end_idx = min(start_idx + self.batch_size, n_cells)
            actual_batch_size = end_idx - start_idx

            # Build batch: (accessions, raw_expressions) per cell
            batch_cells = []
            for i in range(start_idx, end_idx):
                if sp.issparse(adata.X):
                    expressions = adata.X[i].toarray().flatten()
                else:
                    expressions = adata.X[i].flatten()
                batch_cells.append((protein_accessions, expressions))

            inputs = self.prepare_batch(batch_cells)

            outputs = self.model.generative_forward_with_embeddings(
                ctx_prot_embs=inputs['ctx_prot_embs'],
                ctx_expr=inputs['ctx_expr'],
                ctx_key_padding_mask=inputs['ctx_key_padding_mask'],
                tgt_prot_embs=inputs['tgt_prot_embs'],
                tgt_key_padding_mask=inputs['tgt_key_padding_mask'],
            )

            # tgt_preds: (batch, tgt_len) predictions for zero positions
            tgt_preds = outputs['tgt_preds'].float().cpu()
            ctx_expr_cpu = inputs['ctx_expr_cpu']                # (batch, ctx_len)
            tgt_mask_cpu = inputs['tgt_key_padding_mask_cpu']    # (batch, tgt_len), True=pad

            for b in range(actual_batch_size):
                cell_idx = start_idx + b
                nonzero_idxs, zero_idxs = inputs['cell_info'][b]

                # Non-zero positions: write back normalized ctx expression values
                # ctx layout: [CLS, prot_0, prot_1, ...], skip CLS at pos 0
                n_nonzero = len(nonzero_idxs)
                if n_nonzero > 0:
                    normed_ctx = ctx_expr_cpu[b, 1:1 + n_nonzero].numpy()
                    imputed[cell_idx, nonzero_idxs.numpy()] = normed_ctx

                # Zero positions: write back model tgt predictions
                # skip padding positions using tgt_key_padding_mask
                n_zero = len(zero_idxs)
                if n_zero > 0:
                    valid_tgt = ~tgt_mask_cpu[b, :n_zero]    # True = real tgt (not padding)
                    preds = tgt_preds[b, :n_zero].numpy()
                    preds = np.maximum(preds, 0)             # clamp negative predictions
                    imputed[cell_idx, zero_idxs[valid_tgt].numpy()] = preds[valid_tgt.numpy()]
                    n_imputed_total += int(valid_tgt.sum().item())

        print(f"\nImputation complete (all values in normalized space):")
        print(f"  Zero entries imputed: {n_imputed_total}")
        print(f"  Non-zero entries (normalized): {n_valid * n_cells - n_imputed_total}")

        return imputed


def main():
    parser = argparse.ArgumentParser(
        description="Impute zero expressions using pretrained scpFormer",
    )

    parser.add_argument(
        "--model_path", type=str,
        default=None, required=True,
        help="Path to trained model checkpoint directory"
    )
    parser.add_argument(
        "--h5ad_path", type=str,
        default=None, required=True,
        help="Path to input h5ad file"
    )
    parser.add_argument(
        "--protein_embeddings_path", type=str,
        default=None, required=True,
        help="Path to protein embeddings npz"
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Output h5ad file"
    )
    parser.add_argument("--batch_size", type=int, default=320)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--do_log1p", action="store_true",
                        help="Apply log1p transformation (must match training)")

    args = parser.parse_args()

    # Determine output path
    if args.output_path is None:
        base, ext = os.path.splitext(args.h5ad_path)
        args.output_path = f"{base}_imputed{ext}"

    # Check paths
    for name, path in [("Model", args.model_path), ("H5AD", args.h5ad_path),
                        ("Protein embeddings", args.protein_embeddings_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    # Load data
    print(f"\nLoading h5ad data from {args.h5ad_path}...")
    adata = ad.read_h5ad(args.h5ad_path)
    print(f"Loaded {adata.shape[0]} cells with {adata.shape[1]} proteins")

    # Count sparsity
    if sp.issparse(adata.X):
        total = adata.shape[0] * adata.shape[1]
        nnz = adata.X.nnz
        print(f"Sparsity: {(total - nnz) / total * 100:.1f}% zeros")
    else:
        total = adata.X.size
        nnz = np.count_nonzero(adata.X)
        print(f"Sparsity: {(total - nnz) / total * 100:.1f}% zeros")

    # Create imputor
    imputor = CellImputor(
        model_path=args.model_path,
        protein_embeddings_path=args.protein_embeddings_path,
        device=args.device,
        batch_size=args.batch_size,
        do_log1p=args.do_log1p,
    )

    # Impute
    imputed_matrix = imputor.impute(adata)

    # Save to obsm
    adata.obsm['X_impu'] = imputed_matrix
    print(f"\nSaved imputed matrix to adata.obsm['X_impu'] "
          f"(shape: {imputed_matrix.shape})")

    # Write output
    print(f"Writing to {args.output_path}...")
    adata.write_h5ad(args.output_path)

    print(f"\nDone! Output: {args.output_path}")


if __name__ == "__main__":
    main()
