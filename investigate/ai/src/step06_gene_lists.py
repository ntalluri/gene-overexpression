"""
step06_gene_lists.py
--------------------
Fitness scores -> commonly deleterious, strain-specific deleterious, and
strain-specific beneficial gene lists, plus the background sets used later for
enrichment tests.

Definitions taken from the paper
--------------------------------
Commonly deleterious (Results, "Genes whose overexpression is deleterious
across many strains are functionally related"):

    "There were 431 OE genes that produced a significant defect in at least 66%
     of strains (FDR<0.05), and we refer to these as 'commonly deleterious' OE
     genes."

The Figure 2C legend states the same set as "a deleterious effect in >=10
strains", which is 10 of 15. Both readings give the same threshold, so the
default here is 10 strains.

Strain-specific (Results, "Strain-specific responses to specific genes"):

    "we identified genes whose OE produced a significant fitness effect in each
     strain and not more than two others, which we defined as 'strain-specific'
     gene lists."

Background sets (Supplementary file 8):
    Tab 1  genes measured in all three replicates at generation 0 in at least
           one strain, minus the 431 common genes
    Tab 2  genes with no effect (FDR>0.1) in each strain
    Tab 3  genes significant in at least one strain (FDR<0.05)
    Tab 4  Tab 3 minus the 431 common genes

One reading to check
--------------------
"significant fitness effect in each strain and not more than two others" does
not say whether the two others must show an effect in the same direction. This
script counts same-direction strains by default (--specificity-rule
same_direction). Pass --specificity-rule any_direction to count a strain as one
of the "others" whenever it shows any significant effect. Compare the resulting
list sizes against Tabs 6 and 7 of Supplementary file 4 to pick the right one;
the paper reports 41 genes for Y2209 and 1763 for Y12.

Note on YPS606
--------------
The paper excludes YPS606 from Figure 5 because only duplicates were analyzed.
This script reports it but flags it, and --exclude-strains YPS606 drops it from
the strain-specific tallies.

Usage
-----
    python src/step06_gene_lists.py
    python src/step06_gene_lists.py --specificity-rule any_direction
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (ensure_dir, get_logger, load_config, load_sample_sheet,
                    read_table, sample_columns_for_strain, write_table)

log = get_logger("step06")


def split_wide(wide: pd.DataFrame):
    """Split the wide fitness table into a log2FC frame and an FDR frame."""
    strains = sorted({c.rsplit("_", 1)[0] for c in wide.columns})
    logfc = pd.DataFrame(
        {s: wide[f"{s}_log2FC"] for s in strains if f"{s}_log2FC" in wide},
        index=wide.index,
    )
    fdr = pd.DataFrame(
        {s: wide[f"{s}_FDR"] for s in strains if f"{s}_FDR" in wide},
        index=wide.index,
    )
    return logfc, fdr


def significance_calls(logfc: pd.DataFrame, fdr: pd.DataFrame, alpha: float):
    """Return boolean frames for deleterious, beneficial, and any effect."""
    sig = fdr < alpha
    deleterious = sig & (logfc < 0)
    beneficial = sig & (logfc > 0)
    return deleterious, beneficial, sig


def commonly_deleterious(deleterious: pd.DataFrame, min_strains: int) -> pd.Index:
    """Genes deleterious in at least `min_strains` strains."""
    counts = deleterious.sum(axis=1)
    return deleterious.index[counts >= min_strains]


def strain_specific(effect: pd.DataFrame, other_effect: pd.DataFrame,
                    max_others: int) -> dict[str, pd.Index]:
    """Genes with an effect in one strain and in no more than `max_others`.

    `effect` decides membership for the focal strain. `other_effect` decides
    what counts as an "other" strain, which lets the caller choose between
    same-direction and any-direction counting.
    """
    out = {}
    for strain in effect.columns:
        others = [c for c in other_effect.columns if c != strain]
        n_other = other_effect[others].sum(axis=1)
        selected = effect[strain] & (n_other <= max_others)
        out[strain] = effect.index[selected]
    return out


def measured_at_gen0(normed: pd.DataFrame, sheet: pd.DataFrame,
                     min_normalized: float) -> pd.Index:
    """Genes measured in all generation-0 replicates of at least one strain.

    Used for Supplementary file 8, Tab 1. "Measured" is taken as a nonzero
    normalized count, with an optional higher threshold.
    """
    ok = pd.Series(False, index=normed.index)
    for strain in sorted(sheet["strain"].unique()):
        g0, _ = sample_columns_for_strain(sheet, strain)
        g0 = [c for c in g0 if c in normed.columns]
        if not g0:
            continue
        ok |= (normed[g0] > min_normalized).all(axis=1)
    return normed.index[ok]


def write_gene_list(genes, path: Path, extra: pd.DataFrame | None = None):
    """Write a one-column gene list, optionally with attached score columns.

    Gene names are forced to string dtype so that an empty list still writes a
    valid (header-only) file rather than failing on a float64 merge key.
    """
    df = pd.DataFrame({"gene": pd.Series(list(genes), dtype="string")})
    if extra is not None and len(df):
        right = extra.copy()
        right.index = right.index.astype("string")
        right = right.reset_index()
        right = right.rename(columns={right.columns[0]: "gene"})
        df = df.merge(right, on="gene", how="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--which", choices=["imputed", "no_imputation"], default="imputed",
                    help="Which fitness table to build lists from. The paper's "
                         "significance calls come from the imputed analysis.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--common-min-strains", type=int, default=10)
    ap.add_argument("--max-other-strains", type=int, default=2)
    ap.add_argument("--specificity-rule",
                    choices=["same_direction", "any_direction"],
                    default="same_direction")
    ap.add_argument("--exclude-strains", nargs="*", default=[],
                    help="Strains to drop from the strain-specific tallies, "
                         "e.g. YPS606")
    args = ap.parse_args()

    cfg = load_config(args.config)
    fitness_dir = Path(cfg["paths"]["fitness_dir"])
    norm_dir = Path(cfg["paths"]["normalized_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["gene_lists_dir"]))

    wide = read_table(fitness_dir / f"fitness_scores_{args.which}.tsv")
    sheet = load_sample_sheet(Path(cfg["paths"]["sample_sheet"]))

    logfc, fdr = split_wide(wide)
    log.info("Fitness table: %d genes x %d strains", *logfc.shape)

    deleterious, beneficial, sig_any = significance_calls(logfc, fdr, args.alpha)

    # --- commonly deleterious -----------------------------------------------
    common = commonly_deleterious(deleterious, args.common_min_strains)
    log.info("Commonly deleterious (>=%d strains): %d genes",
             args.common_min_strains, len(common))
    write_gene_list(common, out_dir / "commonly_deleterious.tsv",
                    extra=logfc.loc[common])

    # Figure 2C: how many genes are deleterious in exactly N strains.
    binned = deleterious.sum(axis=1)
    dist = binned[binned > 0].value_counts().sort_index()
    dist.index.name = "n_strains_deleterious"
    dist.name = "n_genes"
    write_table(dist.to_frame(), out_dir / "deleterious_strain_count_distribution.tsv")

    # --- strain-specific ----------------------------------------------------
    keep = [c for c in logfc.columns if c not in args.exclude_strains]
    del_keep = deleterious[keep]
    ben_keep = beneficial[keep]
    any_keep = sig_any[keep]

    if args.specificity_rule == "same_direction":
        ss_del = strain_specific(del_keep, del_keep, args.max_other_strains)
        ss_ben = strain_specific(ben_keep, ben_keep, args.max_other_strains)
    else:
        ss_del = strain_specific(del_keep, any_keep, args.max_other_strains)
        ss_ben = strain_specific(ben_keep, any_keep, args.max_other_strains)

    ss_dir = ensure_dir(out_dir / "strain_specific")
    rows = []
    for strain in keep:
        write_gene_list(ss_del[strain], ss_dir / f"{strain}_deleterious.tsv",
                        extra=logfc.loc[ss_del[strain], [strain]])
        write_gene_list(ss_ben[strain], ss_dir / f"{strain}_beneficial.tsv",
                        extra=logfc.loc[ss_ben[strain], [strain]])
        rows.append({
            "strain": strain,
            "strain_specific_deleterious": len(ss_del[strain]),
            "strain_specific_beneficial": len(ss_ben[strain]),
        })
    ss_summary = pd.DataFrame(rows).sort_values("strain_specific_deleterious")
    write_table(ss_summary, out_dir / "strain_specific_counts.tsv", index=False)

    # --- background sets (Supplementary file 8) ------------------------------
    bg_dir = ensure_dir(out_dir / "backgrounds")

    normed = read_table(norm_dir / "normalized_counts.tsv")
    measured = measured_at_gen0(normed, sheet,
                                cfg["gene_lists"]["gen0_measured_min_normalized"])
    write_gene_list(measured.difference(common), bg_dir / "tab1_measured_minus_common.tsv")

    # Tab 2: per strain, genes with no effect at FDR>0.1.
    no_effect_dir = ensure_dir(bg_dir / "tab2_no_effect_per_strain")
    for strain in logfc.columns:
        ne = fdr.index[fdr[strain] > 0.1]
        write_gene_list(ne, no_effect_dir / f"{strain}_no_effect.tsv")

    sig_at_least_one = sig_any.index[sig_any.any(axis=1)]
    write_gene_list(sig_at_least_one, bg_dir / "tab3_significant_any_strain.tsv")
    write_gene_list(sig_at_least_one.difference(common),
                    bg_dir / "tab4_significant_minus_common.tsv")

    # --- console summary -----------------------------------------------------
    print("\n=== Gene list summary ===")
    print(f"Analysis used            : {args.which}")
    print(f"Specificity rule         : {args.specificity_rule}")
    print(f"Genes significant in >=1 strain : {len(sig_at_least_one)}   (paper: 4064)")
    per_strain_sig = sig_any.sum(axis=0)
    print(f"Median significant genes per strain : {int(per_strain_sig.median())}   (paper: 1726)")
    print(f"Commonly deleterious     : {len(common)}   (paper: 431)")
    print("\nStrain-specific counts:")
    print(ss_summary.to_string(index=False))
    print("\nPaper reference points: 41 strain-specific genes in Y2209, "
          "1763 in Y12.")

    print("\nDeleterious-gene count per strain (Figure 2A):")
    print(deleterious.sum(axis=0).sort_values().to_string())
    print("\nPaper reference points: Y2209 635 deleterious genes, Y12 3060.")

    stats = {
        "significant_any_strain": int(len(sig_at_least_one)),
        "median_significant_per_strain": int(per_strain_sig.median()),
        "commonly_deleterious": int(len(common)),
        "deleterious_per_strain": deleterious.sum(axis=0).astype(int).to_dict(),
        "beneficial_per_strain": beneficial.sum(axis=0).astype(int).to_dict(),
    }
    (out_dir / "summary_stats.json").write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
