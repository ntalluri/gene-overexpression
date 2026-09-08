"""
plots.py
--------
Render the paper's figures from pipeline output. Everything here reads files
written by steps 06, 07 and 09; nothing is recomputed, so if a number in a plot
looks wrong the problem is upstream.

Figures produced
----------------
  fig1b_heatmap            Clustered log2 fitness scores, all genes significant
                           in at least one strain. Blue for depleted plasmids
                           (fitness defects), yellow for enriched, matching the
                           paper's colour scheme.
  fig2a_bars               Deleterious genes per strain, coloured by lineage.
  fig2_supp1_bars          Deleterious and beneficial genes per strain.
  fig2b_boxplots           Distribution of log2 scores among deleterious genes.
  fig2c_bars               Genes binned by number of strains affected.
  fig3a_heatmap            The same heatmap restricted to commonly deleterious.
  fig3b_boxplots           Severity among commonly deleterious genes.
  replicate_correlations   Per-strain replicate agreement.

Lineage colours come from Supplementary file 1 as described in the Figure 1B
legend. Strains not in the table are drawn grey.

Usage
-----
    python src/plots.py
    python src/plots.py --only fig2a_bars fig2c_bars --format pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from common import ensure_dir, get_logger, load_config, read_table

log = get_logger("plots")

# Lineage assignments from Supplementary file 1, used for the bar colours.
LINEAGE = {
    "BC187": "European Wine", "DBVPG1373": "European Wine",
    "NCYC3290": "European Wine",
    "YPS128": "North American Oak", "YPS163": "North American Oak",
    "YPS606": "North American Oak", "Y389": "North American Oak",
    "Y12": "Asian", "Y2209": "Asian",
    "YJM978": "West African",
    "BY4743": "Lab",
    "Y7568": "Mosaic", "YJM1273": "Mosaic", "YJM1389": "Mosaic",
    "YJM1592": "Mosaic",
}
LINEAGE_COLORS = {
    "European Wine": "#8c1b1b",
    "North American Oak": "#7b52a1",
    "Asian": "#1f6fb4",
    "West African": "#2e8b57",
    "Lab": "#333333",
    "Mosaic": "#d98b00",
    "Unknown": "#9a9a9a",
}

# The paper's heatmap: blue for depleted plasmids, yellow for enriched.
FITNESS_CMAP = LinearSegmentedColormap.from_list(
    "fitness", ["#1a4f8a", "#4f86c6", "#f2f2f2", "#e8c34a", "#c08a00"])


def strain_color(strain: str) -> str:
    return LINEAGE_COLORS[LINEAGE.get(strain, "Unknown")]


def save(fig, out_dir: Path, name: str, fmt: str):
    path = out_dir / f"{name}.{fmt}"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", path)


def lineage_legend(ax):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=k)
               for k, c in LINEAGE_COLORS.items()]
    ax.legend(handles=handles, fontsize=7, frameon=False,
              loc="upper left", title="Lineage", title_fontsize=7)


# ---------------------------------------------------------------------------

def _heatmap(matrix: pd.DataFrame, title: str, out_dir: Path, name: str, fmt: str):
    """Shared heatmap renderer for Figures 1B and 3A."""
    data = matrix.to_numpy(dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        log.warning("%s: nothing to plot", name)
        return
    limit = float(np.nanpercentile(np.abs(finite), 98)) or 1.0
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    height = max(4.0, min(14.0, matrix.shape[0] / 260))
    fig, ax = plt.subplots(figsize=(1.1 + 0.45 * matrix.shape[1], height))
    im = ax.imshow(data, aspect="auto", cmap=FITNESS_CMAP, norm=norm,
                   interpolation="nearest")

    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
    for label in ax.get_xticklabels():
        label.set_color(strain_color(label.get_text()))
    ax.set_yticks([])
    ax.set_ylabel(f"{matrix.shape[0]:,} genes", fontsize=8)
    ax.set_title(title, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log2 relative fitness score", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    save(fig, out_dir, name, fmt)


def fig1b_heatmap(cfg, ana: Path, out_dir: Path, fmt: str):
    path = ana / "figure1b_clustered_matrix.tsv"
    if not path.exists():
        log.warning("Run step07 cluster_heatmap_matrix first")
        return
    m = read_table(path)
    _heatmap(m, "Figure 1B: clustered log2 relative fitness scores",
             out_dir, "fig1b_heatmap", fmt)


def fig3a_heatmap(cfg, ana: Path, out_dir: Path, fmt: str):
    path = ana / "figure3a_common_gene_matrix.tsv"
    if not path.exists():
        log.warning("Run step07 figure3b first")
        return
    m = read_table(path)
    # Order genes by mean severity so the block structure is visible.
    m = m.loc[m.mean(axis=1).sort_values().index]
    _heatmap(m, f"Figure 3A: {m.shape[0]} commonly deleterious genes",
             out_dir, "fig3a_heatmap", fmt)


def fig2a_bars(cfg, ana: Path, out_dir: Path, fmt: str):
    path = ana / "figure2a_significant_per_strain.tsv"
    if not path.exists():
        log.warning("Run step07 figure2a first")
        return
    df = read_table(path).sort_values("deleterious")

    fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(df)), 3.6))
    ax.bar(df.index, df["deleterious"],
           color=[strain_color(s) for s in df.index])
    ax.set_ylabel("Deleterious OE genes (FDR < 0.05)", fontsize=8)
    ax.set_title("Figure 2A: deleterious genes per strain", fontsize=9)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    lineage_legend(ax)
    save(fig, out_dir, "fig2a_bars", fmt)


def fig2_supp1_bars(cfg, ana: Path, out_dir: Path, fmt: str):
    path = ana / "figure2a_significant_per_strain.tsv"
    if not path.exists():
        return
    df = read_table(path).sort_values("deleterious")
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(df)), 3.6))
    ax.bar(x - 0.2, df["deleterious"], width=0.4,
           color="#3a3a3a", label="Deleterious")
    ax.bar(x + 0.2, df["beneficial"], width=0.4,
           color="#c0c0c0", label="Beneficial")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=90, fontsize=7)
    ax.set_ylabel("Significant OE genes (FDR < 0.05)", fontsize=8)
    ax.set_title("Figure 2, supplement 1: significant genes per strain", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out_dir, "fig2_supp1_bars", fmt)


def _boxplot(long_df: pd.DataFrame, order, title: str, ylabel: str,
             out_dir: Path, name: str, fmt: str):
    groups = [long_df.loc[long_df["strain"] == s, "log2FC"].dropna().to_numpy()
              for s in order]
    groups = [g for g in groups if len(g)]
    order = [s for s, g in zip(order, groups) if len(g)]
    if not groups:
        log.warning("%s: nothing to plot", name)
        return

    fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(order)), 3.8))
    bp = ax.boxplot(groups, patch_artist=True, showfliers=False, widths=0.65)
    for patch, strain in zip(bp["boxes"], order):
        patch.set_facecolor(strain_color(strain))
        patch.set_alpha(0.75)
        patch.set_edgecolor("#333333")
    for element in ("whiskers", "caps", "medians"):
        for item in bp[element]:
            item.set_color("#333333")

    ax.axhline(0, color="#888888", linewidth=0.7, linestyle="--")
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out_dir, name, fmt)


def fig2b_boxplots(cfg, ana: Path, out_dir: Path, fmt: str):
    long_path = ana / "figure2b_deleterious_log2fc_long.tsv"
    sum_path = ana / "figure2b_summary.tsv"
    if not long_path.exists():
        log.warning("Run step07 figure2b first")
        return
    long_df = pd.read_csv(long_path, sep="\t")
    order = list(pd.read_csv(sum_path, sep="\t")
                 .sort_values("n_deleterious_non_imputed")["strain"])
    _boxplot(long_df, order,
             "Figure 2B: severity of deleterious effects (imputed excluded)",
             "log2 relative fitness score", out_dir, "fig2b_boxplots", fmt)


def fig3b_boxplots(cfg, ana: Path, out_dir: Path, fmt: str):
    path = ana / "figure3a_common_gene_matrix.tsv"
    if not path.exists():
        log.warning("Run step07 figure3b first")
        return
    m = read_table(path)
    long_df = m.reset_index().melt(id_vars=m.index.name or "index",
                                   var_name="strain", value_name="log2FC")

    order_path = ana / "figure2a_significant_per_strain.tsv"
    if order_path.exists():
        order = [s for s in read_table(order_path)
                 .sort_values("deleterious").index if s in m.columns]
    else:
        order = list(m.columns)
    _boxplot(long_df, order,
             "Figure 3B: severity among commonly deleterious genes",
             "log2 relative fitness score", out_dir, "fig3b_boxplots", fmt)


def fig2c_bars(cfg, ana: Path, out_dir: Path, fmt: str):
    path = ana / "figure2c_strain_count_bins.tsv"
    if not path.exists():
        log.warning("Run step07 figure2c first")
        return
    df = read_table(path)
    threshold = cfg["gene_lists"]["common_min_strains"]
    colors = ["#c0392b" if int(i) >= threshold else "#7f8c8d" for i in df.index]

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar(df.index.astype(int), df["n_genes"], color=colors)
    ax.set_xlabel("Number of strains in which the gene is deleterious", fontsize=8)
    ax.set_ylabel("Genes", fontsize=8)
    n_common = int(df.loc[df.index.astype(int) >= threshold, "n_genes"].sum())
    ax.set_title(f"Figure 2C: {n_common} genes deleterious in "
                 f"{threshold} or more strains", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out_dir, "fig2c_bars", fmt)


def replicate_correlations(cfg, ana: Path, out_dir: Path, fmt: str):
    path = ana / "replicate_correlations.tsv"
    if not path.exists():
        log.warning("Run step07 replicate_correlations first")
        return
    df = pd.read_csv(path, sep="\t")
    pivot = df.pivot(index="strain", columns="generation", values="mean_pearson")
    pivot = pivot.sort_values(pivot.columns[0])
    x = np.arange(len(pivot))

    fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(pivot)), 3.4))
    width = 0.8 / max(1, len(pivot.columns))
    for i, col in enumerate(pivot.columns):
        ax.bar(x + i * width - 0.4 + width / 2, pivot[col], width=width, label=col)

    # The paper's reported bands for most strains and for the three noisy ones.
    ax.axhspan(0.74, 0.89, color="#2e8b57", alpha=0.12,
               label="paper: most strains")
    ax.axhspan(0.55, 0.65, color="#c0392b", alpha=0.12,
               label="paper: Y2209/YJM1592/YJM978")

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=90, fontsize=7)
    ax.set_ylabel("Mean pairwise Pearson r (log2 counts)", fontsize=8)
    ax.set_title("Replicate agreement per strain", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out_dir, "replicate_correlations", fmt)


FIGURES = {
    "fig1b_heatmap": fig1b_heatmap,
    "fig2a_bars": fig2a_bars,
    "fig2_supp1_bars": fig2_supp1_bars,
    "fig2b_boxplots": fig2b_boxplots,
    "fig2c_bars": fig2c_bars,
    "fig3a_heatmap": fig3a_heatmap,
    "fig3b_boxplots": fig3b_boxplots,
    "replicate_correlations": replicate_correlations,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES))
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    ana = Path(cfg["paths"]["analysis_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "10_figures")

    for name in (args.only or list(FIGURES)):
        log.info("Rendering %s", name)
        try:
            FIGURES[name](cfg, ana, out_dir, args.format)
        except Exception as exc:
            log.error("%s failed: %s", name, exc)

    print(f"\nFigures written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
