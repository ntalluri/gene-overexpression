"""
step07_core_analyses.py
-----------------------
The analyses in the Results section that need nothing beyond the pipeline's own
output. Each one is a separate function so you can run them individually.

Covered here
------------
  replicate_correlations   Reproduces "The mean correlation among replicates per
                           strain was generally high (0.74-0.89), aside of three
                           triplicated strains (Y2209, YJM1592, and YJM978)
                           [where] the correlation was lower (0.55-0.65)".
  figure2a                 Deleterious genes per strain.
  figure2b                 Distribution of log2 fitness scores among each
                           strain's deleterious genes, imputed values excluded.
  figure2c                 Genes binned by the number of strains in which they
                           are deleterious.
  figure3b                 Distribution of log2 fitness scores for the commonly
                           deleterious genes, per strain.
  cluster_heatmap_matrix   Hierarchically clustered matrix behind Figures 1B and
                           3A. The paper used Cluster 3.0 with Java TreeView;
                           this writes the ordered matrix plus the linkage so
                           you can render it however you like.
  beneficial_cluster       The 21 genes strongly beneficial in over 60% of
                           strains but not the lab strain (Figure 6 section).

Not covered here
----------------
Anything needing an outside annotation source. Those go in step08 once you have
the files; see the README section "External data you still need". The list is:
BioGRID interaction counts, the Pu et al. 2009 complex membership table, IUPred
disorder scores, mRNA and protein abundance tables, GO annotations, the
Makanae et al. 2013 deleterious gene list, per-strain SNP and amino acid
difference tables, protein amino acid composition, centromere coordinates, and
the RNA-seq counts from GSE171585.

Usage
-----
    python src/step07_core_analyses.py
    python src/step07_core_analyses.py --only replicate_correlations figure2c
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from common import (ensure_dir, get_logger, load_config, load_sample_sheet,
                    read_table, sample_columns_for_strain, write_table)

log = get_logger("step07")


# ---------------------------------------------------------------------------

def replicate_correlations(cfg, sheet, out_dir: Path) -> pd.DataFrame:
    """Pearson correlation among a strain's replicates, at each generation.

    Correlations are computed on log2 normalized counts so that a handful of
    very abundant barcodes do not dominate.
    """
    normed = read_table(Path(cfg["paths"]["normalized_dir"]) / "normalized_counts.tsv")
    logged = np.log2(normed + 1)

    rows = []
    for strain in sorted(sheet["strain"].unique()):
        g0, g10 = sample_columns_for_strain(sheet, strain)
        for gen, cols in (("gen0", g0), ("gen10", g10)):
            cols = [c for c in cols if c in logged.columns]
            if len(cols) < 2:
                continue
            corr = logged[cols].corr()
            # Mean of the off-diagonal entries.
            vals = corr.to_numpy()[np.triu_indices(len(cols), k=1)]
            rows.append({"strain": strain, "generation": gen,
                         "n_replicates": len(cols),
                         "mean_pearson": round(float(np.mean(vals)), 4),
                         "min_pairwise": round(float(np.min(vals)), 4)})

    df = pd.DataFrame(rows)
    write_table(df, out_dir / "replicate_correlations.tsv", index=False)

    print("\n=== Replicate correlations ===")
    print(df.to_string(index=False))
    print("\nPaper: mean correlation per strain generally 0.74-0.89, with")
    print("Y2209, YJM1592 and YJM978 lower at 0.55-0.65. The paper attributes")
    print("the lower values to noise in near-zero log2 changes rather than to")
    print("poor replicate agreement.")
    return df


def _load_calls(cfg, which: str):
    """Return (logfc, fdr, deleterious, beneficial) frames."""
    wide = read_table(Path(cfg["paths"]["fitness_dir"]) / f"fitness_scores_{which}.tsv")
    strains = sorted({c.rsplit("_", 1)[0] for c in wide.columns})
    logfc = pd.DataFrame({s: wide[f"{s}_log2FC"] for s in strains
                          if f"{s}_log2FC" in wide}, index=wide.index)
    fdr = pd.DataFrame({s: wide[f"{s}_FDR"] for s in strains
                        if f"{s}_FDR" in wide}, index=wide.index)
    alpha = cfg["statistics"]["fdr_alpha"]
    return logfc, fdr, (fdr < alpha) & (logfc < 0), (fdr < alpha) & (logfc > 0)


def figure2a(cfg, out_dir: Path) -> pd.DataFrame:
    """Number of deleterious and beneficial genes per strain."""
    _, _, deleterious, beneficial = _load_calls(cfg, "imputed")
    df = pd.DataFrame({
        "deleterious": deleterious.sum(axis=0),
        "beneficial": beneficial.sum(axis=0),
    }).astype(int).sort_values("deleterious")
    df.index.name = "strain"
    write_table(df, out_dir / "figure2a_significant_per_strain.tsv")
    print("\n=== Figure 2A / 2-supplement-1: genes per strain ===")
    print(df.to_string())
    print("\nPaper: Y2209 lowest at 635 deleterious, Y12 highest at 3060.")
    return df


def figure2b(cfg, out_dir: Path) -> pd.DataFrame:
    """Per-strain distribution of log2 fitness among deleterious genes.

    The paper excludes imputed ratios from this figure, so the imputation mask
    from step 04 is used to blank those values.
    """
    logfc, _, deleterious, _ = _load_calls(cfg, "imputed")
    mask_path = Path(cfg["paths"]["imputed_dir"]) / "imputation_mask.tsv"
    sheet = load_sample_sheet(Path(cfg["paths"]["sample_sheet"]))

    imputed_gene_by_strain = {}
    if mask_path.exists():
        mask = read_table(mask_path)
        for strain in logfc.columns:
            _, g10 = sample_columns_for_strain(sheet, strain)
            g10 = [c for c in g10 if c in mask.columns]
            if g10:
                imputed_gene_by_strain[strain] = set(mask.index[mask[g10].any(axis=1)])

    records, summary = [], []
    for strain in logfc.columns:
        genes = set(deleterious.index[deleterious[strain]])
        genes -= imputed_gene_by_strain.get(strain, set())
        values = logfc.loc[sorted(genes), strain].dropna()
        for gene, v in values.items():
            records.append({"strain": strain, "gene": gene, "log2FC": v})
        summary.append({
            "strain": strain,
            "n_deleterious_non_imputed": len(values),
            "median_log2FC": round(float(values.median()), 4) if len(values) else np.nan,
            "q1": round(float(values.quantile(0.25)), 4) if len(values) else np.nan,
            "q3": round(float(values.quantile(0.75)), 4) if len(values) else np.nan,
            "min_log2FC": round(float(values.min()), 4) if len(values) else np.nan,
        })

    long_df = pd.DataFrame(records)
    sum_df = pd.DataFrame(summary).sort_values("n_deleterious_non_imputed")
    write_table(long_df, out_dir / "figure2b_deleterious_log2fc_long.tsv", index=False)
    write_table(sum_df, out_dir / "figure2b_summary.tsv", index=False)
    print("\n=== Figure 2B: severity of deleterious effects ===")
    print(sum_df.to_string(index=False))
    return sum_df


def figure2c(cfg, out_dir: Path) -> pd.DataFrame:
    """Genes binned by how many strains they are deleterious in."""
    _, _, deleterious, _ = _load_calls(cfg, "imputed")
    counts = deleterious.sum(axis=1)
    dist = counts[counts > 0].value_counts().sort_index()
    dist.index.name = "n_strains_deleterious"
    dist.name = "n_genes"
    df = dist.to_frame()
    df["cumulative_from_top"] = df["n_genes"][::-1].cumsum()[::-1]
    write_table(df, out_dir / "figure2c_strain_count_bins.tsv")
    print("\n=== Figure 2C: deleterious in how many strains ===")
    print(df.to_string())
    print("\nPaper: over half of the 4064 significant genes produce a defect in")
    print("four or fewer strains; 431 genes are deleterious in 10 or more.")
    return df


def figure3b(cfg, out_dir: Path) -> pd.DataFrame:
    """Per-strain distribution of log2 fitness for the commonly deleterious set."""
    logfc, _, _, _ = _load_calls(cfg, "imputed")
    common_path = Path(cfg["paths"]["gene_lists_dir"]) / "commonly_deleterious.tsv"
    if not common_path.exists():
        log.warning("Run step 06 first; no commonly deleterious list found")
        return pd.DataFrame()

    common = pd.read_csv(common_path, sep="\t")["gene"].tolist()
    sub = logfc.loc[logfc.index.intersection(common)]

    df = pd.DataFrame({
        "n_genes": sub.notna().sum(),
        "median_log2FC": sub.median().round(4),
        "q1": sub.quantile(0.25).round(4),
        "q3": sub.quantile(0.75).round(4),
    })
    df.index.name = "strain"
    write_table(df, out_dir / "figure3b_common_gene_severity.tsv")
    write_table(sub, out_dir / "figure3a_common_gene_matrix.tsv")
    print("\n=== Figure 3B: severity among commonly deleterious genes ===")
    print(df.to_string())
    print("\nPaper: strains with more deleterious genes generally show more")
    print("severe median costs, except the North American oak-soil strains")
    print("(YPS128, YPS163, YPS606, Y389 lineage), which are unusually severe.")
    return df


def cluster_heatmap_matrix(cfg, out_dir: Path):
    """Hierarchically cluster the fitness matrix, as in Figures 1B and 3A.

    The paper used Cluster 3.0 with average linkage on the log2 scores. That is
    reproduced here with scipy. Genes not measured in a strain are mean-filled
    for the purpose of clustering only; the written matrix keeps the NaNs.
    """
    logfc, fdr, _, _ = _load_calls(cfg, "imputed")
    alpha = cfg["statistics"]["fdr_alpha"]

    # Figure 1B is restricted to genes significant in at least one strain.
    keep = (fdr < alpha).any(axis=1)
    sub = logfc.loc[keep]
    log.info("Clustering %d genes x %d strains", *sub.shape)

    filled = sub.apply(lambda r: r.fillna(r.mean()), axis=1).fillna(0.0)

    # Correlation distance is undefined for a row (or column) with no variance,
    # which happens for genes that score identically in every strain. Those rows
    # are set aside and appended at the end of the ordering.
    row_var = filled.var(axis=1)
    constant_rows = filled.index[row_var == 0]
    if len(constant_rows):
        log.info("%d genes have identical scores across strains; "
                 "appended after the clustered block", len(constant_rows))
    variable = filled.drop(index=constant_rows)

    if len(variable) < 3 or variable.shape[1] < 3:
        log.warning("Too few variable genes or strains for correlation "
                    "clustering; falling back to euclidean distance")
        metric = "euclidean"
        variable = filled
        constant_rows = filled.index[[]]
    else:
        metric = "correlation"

    row_link = linkage(pdist(variable.to_numpy(), metric=metric), method="average")
    col_link = linkage(pdist(variable.T.to_numpy(), metric=metric), method="average")

    row_order = leaves_list(row_link)
    col_order = leaves_list(col_link)

    ordered_index = list(variable.index[row_order]) + list(constant_rows)
    ordered = sub.loc[ordered_index, variable.columns[col_order]]
    write_table(ordered, out_dir / "figure1b_clustered_matrix.tsv")
    np.save(out_dir / "figure1b_row_linkage.npy", row_link)
    np.save(out_dir / "figure1b_col_linkage.npy", col_link)

    print("\n=== Figure 1B: clustered matrix ===")
    print(f"Wrote {ordered.shape[0]} genes x {ordered.shape[1]} strains")
    print("Strain order after clustering:")
    print("  " + ", ".join(ordered.columns))
    print("\nCompare with elife-70564-fig1-data1-v2.txt, the published")
    print("hierarchically clustered fitness scores.")
    return ordered


def beneficial_cluster(cfg, out_dir: Path, lab_strain: str = "BY4743",
                       min_fraction: float = 0.6) -> pd.DataFrame:
    """Genes strongly beneficial in most strains but not the lab strain.

    The paper describes a cluster of 21 such genes, over half of which sit next
    to a centromere. Identifying which of them cloned a CEN needs the centromere
    coordinates, which is a step-08 job; this function just finds the cluster.
    """
    logfc, fdr, _, beneficial = _load_calls(cfg, "imputed")
    n_strains = beneficial.shape[1]
    threshold = int(np.ceil(min_fraction * n_strains))

    n_beneficial = beneficial.sum(axis=1)
    selected = n_beneficial >= threshold
    if lab_strain in beneficial.columns:
        selected &= ~beneficial[lab_strain]

    genes = logfc.index[selected]
    df = logfc.loc[genes].copy()
    df.insert(0, "n_strains_beneficial", n_beneficial.loc[genes])
    df = df.sort_values("n_strains_beneficial", ascending=False)
    write_table(df, out_dir / "beneficial_cluster.tsv")

    print(f"\n=== Beneficial cluster (>= {threshold} of {n_strains} strains, "
          f"not {lab_strain}) ===")
    print(f"Genes found: {len(df)}   (paper: 21)")
    if len(df):
        print(df.head(25).to_string())
    print("\nPaper notes these include RRP6, RRP7, LOC1, RPL35B and DOM34, and")
    print("that over half sit next to a centromere that was cloned into the")
    print("plasmid's upstream region.")
    return df


ANALYSES = {
    "replicate_correlations": lambda cfg, sheet, d: replicate_correlations(cfg, sheet, d),
    "figure2a": lambda cfg, sheet, d: figure2a(cfg, d),
    "figure2b": lambda cfg, sheet, d: figure2b(cfg, d),
    "figure2c": lambda cfg, sheet, d: figure2c(cfg, d),
    "figure3b": lambda cfg, sheet, d: figure3b(cfg, d),
    "cluster_heatmap_matrix": lambda cfg, sheet, d: cluster_heatmap_matrix(cfg, d),
    "beneficial_cluster": lambda cfg, sheet, d: beneficial_cluster(cfg, d),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", nargs="*", choices=sorted(ANALYSES),
                    help="Run only these analyses")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sheet = load_sample_sheet(Path(cfg["paths"]["sample_sheet"]))
    out_dir = ensure_dir(Path(cfg["paths"]["analysis_dir"]))

    names = args.only or list(ANALYSES)
    for name in names:
        log.info("Running %s", name)
        try:
            ANALYSES[name](cfg, sheet, out_dir)
        except Exception as exc:
            log.error("%s failed: %s", name, exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
