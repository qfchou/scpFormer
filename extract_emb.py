"""
Simple script to extract CLS embeddings from a trained ScpFormer model
and save them to adata.obsm['X_emb'].

Usage:
    python embed_h5ad.py \
        --model_path /path/to/model \
        --protein_embeddings_path /path/to/protein_embedding.npz \
        --h5ad_path /path/to/data.h5ad \
        --output_path /path/to/output.h5ad
"""

import argparse

import anndata as ad
import numpy as np
import torch
from tqdm import tqdm

from scpformer.data import (
    load_protein_embeddings,
    pad_batch,
    prepend_special_tokens,
    preprocess_expressions,
)
from scpformer.model import ScpFormerModel

PAD_VALUE = -2


def prepare_batch(cells_data, accession_to_embedding, protein_emb_dim, device, do_log1p=False):
    """Build model-ready tensors for a list of (accessions, expressions) tuples."""
    prot_embs_list, exprs_list = [], []

    for accessions, expressions in cells_data:
        valid_embs, valid_exprs = [], []
        for acc, expr in zip(accessions, expressions):
            if acc in accession_to_embedding:
                valid_embs.append(accession_to_embedding[acc])
                valid_exprs.append(expr)

        if valid_embs:
            prot_embs = torch.tensor(np.array(valid_embs), dtype=torch.float32)
            exprs = torch.tensor(valid_exprs, dtype=torch.float32)
            exprs = preprocess_expressions(exprs, do_log1p=do_log1p)
        else:
            prot_embs = torch.empty(0, protein_emb_dim, dtype=torch.float32)
            exprs = torch.empty(0, dtype=torch.float32)

        prot_embs, exprs = prepend_special_tokens(prot_embs, exprs, pad_value=PAD_VALUE)
        prot_embs_list.append(prot_embs)
        exprs_list.append(exprs)

    prot_embeddings, expressions, src_key_padding_mask = pad_batch(
        prot_embs_list, exprs_list, pad_value=PAD_VALUE
    )

    return {
        "prot_embeddings": prot_embeddings.to(device, dtype=torch.float16),
        "values": expressions.to(device, dtype=torch.float16),
        "src_key_padding_mask": src_key_padding_mask.to(device),
    }


@torch.no_grad()
def extract_cls_embeddings(model, adata, accession_to_embedding, protein_emb_dim,
                            device, batch_size=256, do_log1p=False):
    """Extract CLS embeddings for all cells in adata."""
    protein_accessions = adata.var["accession"].values
    n_cells = adata.n_obs
    all_embeddings = []

    for start in tqdm(range(0, n_cells, batch_size), desc="Extracting embeddings"):
        end = min(start + batch_size, n_cells)
        batch_cells = []
        for i in range(start, end):
            row = adata.X[i]
            expressions = row.toarray().flatten() if hasattr(row, "toarray") else row.flatten()
            batch_cells.append((protein_accessions, expressions))

        inputs = prepare_batch(batch_cells, accession_to_embedding, protein_emb_dim, device, do_log1p)
        outputs = model.generative_forward_with_embeddings(
            ctx_prot_embs=inputs["prot_embeddings"],
            ctx_expr=inputs["values"],
            ctx_key_padding_mask=inputs["src_key_padding_mask"],
        )
        all_embeddings.append(outputs["cell_emb"].float().cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
    return embeddings / norms


def main():
    parser = argparse.ArgumentParser(description="Extract ScpFormer CLS embeddings into obsm['X_emb']")
    parser.add_argument("--model_path", required=True, help="Path to trained model directory")
    parser.add_argument("--protein_embeddings_path", required=True, help="Path to protein_embedding.npz")
    parser.add_argument("--h5ad_path", required=True, help="Input h5ad file")
    parser.add_argument("--output_path", required=True, help="Output h5ad file")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--do_log1p", action="store_true")
    args = parser.parse_args()

    print(f"Loading data from {args.h5ad_path}...")
    adata = ad.read_h5ad(args.h5ad_path)
    print(f"  {adata.n_obs} cells x {adata.n_vars} proteins")

    accession_to_embedding, protein_emb_dim = load_protein_embeddings(args.protein_embeddings_path)

    print(f"Loading model from {args.model_path}...")
    model = ScpFormerModel.from_pretrained(args.model_path, torch_dtype=torch.float16)
    model.to(args.device).eval()

    embeddings = extract_cls_embeddings(
        model, adata, accession_to_embedding, protein_emb_dim,
        device=args.device, batch_size=args.batch_size, do_log1p=args.do_log1p,
    )
    print(f"Embeddings shape: {embeddings.shape}")

    adata.obsm["X_emb"] = embeddings
    adata.write_h5ad(args.output_path)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
