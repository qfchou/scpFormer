import argparse
import requests
import pandas as pd
import time
import sys
import logging
from pathlib import Path
from tqdm import tqdm

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Standard organism taxonomy IDs for UniProt
TAXONOMY_IDS = {
    "human": "9606"
}

def search_by_uniprot_id(uniprot_id):
    """
    Search UniProt directly by UniProt accession ID.

    Args:
        uniprot_id (str): UniProt accession ID (e.g., P12345, Q8N5F7)

    Returns:
        list: List containing a single UniProt entry dictionary, or empty if not found.
    """
    base_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}"

    params = {
        "format": "json",
        "fields": "accession,gene_names,protein_name,sequence,length,organism_name"
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        entry = response.json()

        result = {
            "accession": entry.get("primaryAccession", ""),
            "gene_names": entry.get("genes", [{}])[0].get("geneName", {}).get("value", "") if entry.get("genes") else "",
            "protein_name": entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
            "sequence": entry.get("sequence", {}).get("value", ""),
            "length": entry.get("sequence", {}).get("length", 0),
            "organism": entry.get("organism", {}).get("scientificName", "")
        }

        return [result]

    except requests.exceptions.RequestException as e:
        logging.warning(f"Error querying UniProt for ID {uniprot_id}: {e}")
        return []

def search_uniprot(protein_name, organism_id, reviewed_only=True):
    """
    Search UniProt for a protein by gene name or protein name.

    Args:
        protein_name (str): Protein/gene name to search
        organism_id (str): NCBI taxonomy ID (e.g., 9606 for human)
        reviewed_only (bool): If True, only search SwissProt (reviewed) entries

    Returns:
        list: List of top matching UniProt entries
    """
    base_url = "https://rest.uniprot.org/uniprotkb/search"

    review_filter = " AND reviewed:true" if reviewed_only else ""
    query = f'(gene:{protein_name} OR protein_name:{protein_name}) AND organism_id:{organism_id}{review_filter}'

    params = {
        "query": query,
        "format": "json",
        "fields": "accession,gene_names,protein_name,sequence,length,organism_name",
        "size": 5  
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        results = []
        for entry in data.get("results", []):
            result = {
                "accession": entry.get("primaryAccession", ""),
                "gene_names": entry.get("genes", [{}])[0].get("geneName", {}).get("value", "") if entry.get("genes") else "",
                "protein_name": entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                "sequence": entry.get("sequence", {}).get("value", ""),
                "length": entry.get("sequence", {}).get("length", 0),
                "organism": entry.get("organism", {}).get("scientificName", "")
            }
            results.append(result)

        return results

    except requests.exceptions.RequestException as e:
        logging.warning(f"Error querying UniProt for {protein_name}: {e}")
        return []

def query_protein_batch(protein_names, organism_id=None, delay=0.5, reviewed_only=True, use_id=False):
    """
    Query UniProt for a batch of proteins.

    Args:
        protein_names (list): List of protein names or UniProt IDs
        organism_id (str): NCBI taxonomy ID (ignored if use_id=True)
        delay (float): Delay between requests in seconds to respect rate limits
        reviewed_only (bool): If True, only search SwissProt entries (ignored if use_id=True)
        use_id (bool): Treat protein_names as UniProt IDs instead of gene/protein names

    Returns:
        tuple: (DataFrame of successful queries, List of not-found queries)
    """
    results = []
    not_found = []

    desc = "Querying by UniProt ID" if use_id else "Querying by Gene/Protein Name"

    for protein in tqdm(protein_names, desc=desc):
        if use_id:
            matches = search_by_uniprot_id(protein)
        else:
            if not organism_id:
                raise ValueError("Organism ID is required for name-based searches.")
            matches = search_uniprot(protein, organism_id, reviewed_only)

        if matches:
            best_match = matches[0]
            results.append({
                "query_name": protein,
                "accession": best_match["accession"],
                "gene_name": best_match["gene_names"],
                "protein_name": best_match["protein_name"],
                "sequence": best_match["sequence"],
                "length": best_match["length"],
                "organism": best_match["organism"],
                "num_matches": len(matches)
            })
        else:
            # Fallback to unreviewed if reviewed search failed
            if not use_id and reviewed_only:
                matches = search_uniprot(protein, organism_id, reviewed_only=False)
                if matches:
                    best_match = matches[0]
                    results.append({
                        "query_name": protein,
                        "accession": best_match["accession"],
                        "gene_name": best_match["gene_names"],
                        "protein_name": best_match["protein_name"],
                        "sequence": best_match["sequence"],
                        "length": best_match["length"],
                        "organism": best_match["organism"],
                        "num_matches": len(matches)
                    })
                else:
                    not_found.append(protein)
            else:
                not_found.append(protein)

        time.sleep(delay) 

    df = pd.DataFrame(results)
    return df, not_found

def save_results(df, not_found, output_dir):
    """Save results as CSV and a not-found text list."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not df.empty:        
        full_csv_path = out_path / "protein_found.csv"
        df.to_csv(full_csv_path, index=False)
        logging.info(f"Saved complete mapping to {full_csv_path}")
    
    if not_found:
        not_found_path = out_path / f"protein_not_found.txt"
        with open(not_found_path, 'w') as f:
            for protein in not_found:
                f.write(str(protein) + "\n")
        logging.info(f"Saved {len(not_found)} not-found proteins to {not_found_path}")

def main():
    parser = argparse.ArgumentParser(description="Query UniProt for protein sequences and metadata.")
    parser.add_argument("--input", required=True, help="Path to input CSV or TXT file containing protein names/IDs.")
    parser.add_argument("--outdir", required=True, help="Directory to save output files.")
    parser.add_argument("--column", default="query_name", help="Column name to read if input is a CSV.")
    parser.add_argument("--organism", default="human", help="Organism taxonomy to query.")
    parser.add_argument("--use-id", action="store_true", help="Flag to search directly by UniProt accession ID instead of gene names.")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between API calls in seconds.")
    
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Load input data
    if input_path.suffix.lower() == '.csv':
        df_input = pd.read_csv(input_path)
        if args.column not in df_input.columns:
            logging.error(f"Column '{args.column}' not found in CSV. Available columns: {list(df_input.columns)}")
            sys.exit(1)
        protein_list = df_input[args.column].dropna().astype(str).tolist()
    else:
        # Assume it's a raw text file
        with open(input_path, 'r') as f:
            protein_list = [line.strip() for line in f if line.strip()]

    logging.info(f"Loaded {len(protein_list)} queries from {args.input}")

    # Set organism ID based on selection (ignored if use_id is True)
    organism_id = TAXONOMY_IDS[args.organism] if not args.use_id else None

    # Run query
    df_results, not_found_list = query_protein_batch(
        protein_names=protein_list,
        organism_id=organism_id,
        delay=args.delay,
        use_id=args.use_id
    )

    # Save outputs
    save_results(df_results, not_found_list, args.outdir)
    logging.info(f"Run complete. Found: {len(df_results)}, Not found: {len(not_found_list)}")

if __name__ == "__main__":
    main()