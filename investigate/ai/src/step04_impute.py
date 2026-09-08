"""
step04_impute.py
----------------
Add the paper's pseudocount imputation and re-normalize.

What the paper did
------------------
From Materials and methods:

    "To recapture genes that were clearly present in the starting pool but
     completely absent after 10 generation growth, we performed a data
     imputation: genes with at least 20 normalized read counts (>5th percentile
     of normalized reads) in all three replicates of the starting pool but
     missing reads from the end-point analysis received a pseudocount of 1
     added to the barcode reads at 10 generations."

Three details matter for reproducing this:

  * The threshold of 20 is applied to *normalized* counts at generation 0.
  * It must hold in *all* replicates of the starting pool for that strain.
  * The pseudocount of 1 is added to the *raw* barcode reads at generation 10,
    which means the matrix has to be re-normalized afterwards.

This step therefore reads both the raw and the normalized matrices, decides
which (gene, strain) pairs qualify, writes an imputed raw matrix, and
re-normalizes it. It also writes an imputation mask so that later steps can
exclude imputed values from figures, which the paper does for Figures 2B and 3B
("Imputed ratios were not included").

Ambiguity worth flagging
------------------------
"missing reads from the end-point analysis" is read here as a raw count of zero
in *every* generation-10 replicate of that strain. Set
--endpoint-rule any_zero in the config if you want to impute whenever any single
replicate is zero. Check the count of imputed values against Tab 4 of
Supplementary file 4 to decide which reading the authors used.

Usage
-----
    python src/step04_impute.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from common import (ensure_dir, get_logger, load_config, load_sample_sheet,
                    read_table, sample_columns_for_strain, write_table)
from step03_normalize import library_size_normalize

log = get_logger("step04")


def find_imputable(raw: pd.DataFrame,
                   normed: pd.DataFrame,
                   sheet: pd.DataFrame,
                   threshold: float,
                   endpoint_rule: str) -> pd.DataFrame:
    """Return a boolean DataFrame, genes x samples, marking cells to impute.

    Only generation-10 columns are ever marked.
    """
    mask = pd.DataFrame(False, index=raw.index, columns=raw.columns)

    for strain in sorted(sheet["strain"].unique()):
        g0, g10 = sample_columns_for_strain(sheet, strain)
        if not g0 or not g10:
            log.warning("Strain %s is missing a generation arm; skipping", strain)
            continue

        # Condition 1: well measured in the starting pool, in every replicate.
        present_at_gen0 = (normed[g0] >= threshold).all(axis=1)

        # Condition 2: gone at the endpoint.
        if endpoint_rule == "all_zero":
            absent_at_gen10 = (raw[g10] == 0).all(axis=1)
        elif endpoint_rule == "any_zero":
            absent_at_gen10 = (raw[g10] == 0).any(axis=1)
        else:
            raise ValueError(f"Unknown endpoint_rule: {endpoint_rule}")

        qualifies = present_at_gen0 & absent_at_gen10
        n = int(qualifies.sum())
        log.info("Strain %-10s : %d genes qualify for imputation", strain, n)

        # Mark only the zero cells among that strain's generation-10 columns.
        for col in g10:
            mask.loc[qualifies & (raw[col] == 0), col] = True

    return mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--counts", default="counts_1mismatch.tsv")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Normalized-count threshold at generation 0 (paper: 20)")
    ap.add_argument("--pseudocount", type=float, default=None,
                    help="Raw reads added at generation 10 (paper: 1)")
    ap.add_argument("--endpoint-rule", choices=["all_zero", "any_zero"], default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    imp_cfg = cfg["imputation"]
    threshold = args.threshold if args.threshold is not None else imp_cfg["gen0_min_normalized"]
    pseudocount = args.pseudocount if args.pseudocount is not None else imp_cfg["pseudocount"]
    endpoint_rule = args.endpoint_rule or imp_cfg["endpoint_rule"]

    counts_dir = Path(cfg["paths"]["counts_dir"])
    norm_dir = Path(cfg["paths"]["normalized_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["imputed_dir"]))

    raw = read_table(counts_dir / args.counts)
    normed = read_table(norm_dir / "normalized_counts.tsv")
    sheet = load_sample_sheet(Path(cfg["paths"]["sample_sheet"]))

    # Keep the two matrices aligned in case a sample was dropped somewhere.
    shared = [c for c in raw.columns if c in normed.columns]
    raw, normed = raw[shared], normed[shared]

    log.info("Imputation rule: gen0 normalized >= %g in all replicates, "
             "gen10 raw %s, add %g raw read(s)",
             threshold, endpoint_rule.replace("_", " "), pseudocount)

    mask = find_imputable(raw, normed, sheet, threshold, endpoint_rule)
    n_imputed = int(mask.values.sum())
    log.info("Total imputed cells: %d", n_imputed)

    raw_imputed = raw.copy().astype(float)
    raw_imputed[mask] = raw_imputed[mask] + pseudocount

    normed_imputed = library_size_normalize(raw_imputed)

    write_table(raw_imputed.round(4), out_dir / "counts_imputed_raw.tsv")
    write_table(normed_imputed.round(6), out_dir / "normalized_counts_imputed.tsv")
    write_table(mask, out_dir / "imputation_mask.tsv")

    print("\n=== Imputation summary ===")
    print(f"Cells imputed: {n_imputed}")
    per_sample = mask.sum(axis=0)
    print("\nImputed cells per sample:")
    print(per_sample[per_sample > 0].to_string())
    genes_touched = int((mask.any(axis=1)).sum())
    print(f"\nDistinct genes with at least one imputed value: {genes_touched}")
    print("\nStep 05 will run the differential test twice: once on the "
          "un-imputed matrix (Supplementary file 4, Tab 3) and once on this "
          "imputed matrix (Tab 4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
