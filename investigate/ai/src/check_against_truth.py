"""
check_against_truth.py
----------------------
Compare the pipeline's recovered fitness scores with the ground truth written by
make_test_data.py. This is a self-test of the code, not a reproduction of the
paper. Use it after changing any of the steps to confirm nothing broke.

Expect Pearson r above 0.95 on log2FC, recall near 1.0 on deleterious genes, and
precision near 1.0.

One residual caveat: the simulator assigns each gene an absolute effect, whereas
the pipeline reports abundance relative to the rest of the library. When many
genes drop out the neutral genes rise slightly in relative terms, so a few
borderline calls are expected. The paper's fitness scores are relative in exactly
the same way: "Barcode abundance was normalized to the total number of reads per
sample, thus producing a fitness score relative to the total set of genes
expressed in each strain." That effect is small. If precision falls well below
0.9, suspect a bug in the statistics rather than the reference frame, and run
test_statistics.py.

Usage
-----
    python src/check_against_truth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import PROJECT_ROOT, get_logger, load_config, read_table

log = get_logger("selftest")


def main() -> int:
    cfg = load_config()
    truth_path = PROJECT_ROOT / "data" / "test" / "ground_truth.tsv"
    if not truth_path.exists():
        log.error("No ground truth found. Run make_test_data.py first.")
        return 1

    truth = read_table(truth_path)
    fitness = read_table(Path(cfg["paths"]["fitness_dir"]) / "fitness_scores_imputed.tsv")

    strains = [c for c in truth.columns if c != "forced_dropout"]
    rows = []
    for strain in strains:
        col = f"{strain}_log2FC"
        fdr_col = f"{strain}_FDR"
        if col not in fitness.columns:
            log.warning("No results for %s", strain)
            continue

        shared = truth.index.intersection(fitness.index)
        # Exclude forced dropouts: their true log2FC is negative infinity.
        keep = shared[~truth.loc[shared, "forced_dropout"].astype(bool)]
        a = truth.loc[keep, strain]
        b = fitness.loc[keep, col]
        ok = b.notna() & np.isfinite(b)

        true_del = set(keep[(a < -0.5)])
        called_del = set(keep[(fitness.loc[keep, fdr_col] < 0.05)
                              & (fitness.loc[keep, col] < 0)])
        tp = len(true_del & called_del)
        rows.append({
            "strain": strain,
            "n_genes": int(ok.sum()),
            "pearson_log2FC": round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 4),
            "median_abs_error": round(float(np.median(np.abs(a[ok] - b[ok]))), 4),
            "true_deleterious": len(true_del),
            "called_deleterious": len(called_del),
            "recall": round(tp / len(true_del), 3) if true_del else np.nan,
            "precision": round(tp / len(called_del), 3) if called_del else np.nan,
        })

    out = pd.DataFrame(rows)
    print("\n=== Self-test against synthetic ground truth ===")
    print(out.to_string(index=False))

    # The imputation step should have rescued the forced-dropout genes.
    mask_path = Path(cfg["paths"]["imputed_dir"]) / "imputation_mask.tsv"
    if mask_path.exists():
        mask = read_table(mask_path)
        dropouts = set(truth.index[truth["forced_dropout"].astype(bool)])
        imputed = set(mask.index[mask.any(axis=1)])
        print(f"\nForced dropout genes: {len(dropouts)}")
        print(f"Genes with at least one imputed value: {len(imputed)}")
        print(f"Dropouts that were imputed: {len(dropouts & imputed)}")

    good = (out["pearson_log2FC"] > 0.9).all()
    print("\nSelf-test", "PASSED" if good else "NEEDS REVIEW",
          "(expect Pearson r > 0.9 on log2FC for every strain)")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
