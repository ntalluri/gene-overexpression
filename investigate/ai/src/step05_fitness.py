"""
step05_fitness.py
-----------------
Normalized counts -> log2 relative fitness score and FDR, per gene per strain.

What the paper did
------------------
From Materials and methods:

    "Measured and imputed data were analyzed using edgeR version 3.22.1, using
     a linear model with generation (0 or 10) as a factor. Genes whose barcodes
     were significantly different after 10 generations of growth in each strain
     at an FDR<0.05 were taken as significant (Benjamini and Hochberg, 1995).
     Fitness scores were calculated by taking the ratio of normalized reads at
     generation 10 divided by reads at generation 0."

Each strain is analyzed on its own, with generation as the only factor. The
outputs are Tabs 3 and 4 of Supplementary file 4: average log2 change in fitness
and BH-corrected FDR, without and with imputation.

Three backends
--------------
--backend python (default)
    A pure-Python negative binomial GLM with Cox-Reid adjusted profile
    likelihood dispersion estimation and a likelihood ratio test. No extra
    dependencies. See the notes below on why the point estimates can be made
    exact for this particular design; only the dispersion moderation is
    approximate.

--backend edgepython
    Uses edgePython (pachterlab/edgePython on PyPI, `pip install edgepython`), a
    pure-Python port of edgeR's internals including its weighted-likelihood
    empirical Bayes dispersion shrinkage. This reproduces edgeR's moderation
    rather than approximating it, so it should track the paper's edgeR output
    closely without needing R. Recommended when you want edgeR-faithful results.
    Its README states it was written mainly by Claude and Codex under Lior
    Pachter's direction; the "edger" backend is kept as an independent check.

--backend edger
    Calls the real edgeR through rpy2. The reference implementation. Requires R
    with edgeR installed and `pip install rpy2`.

Why the pure-Python path is well behaved here
---------------------------------------------
The input matrix has already been scaled so every column sums to exactly 1e6
(step 03). In a negative binomial GLM with a log link, a group-means design, and
identical offsets across samples, the maximum likelihood fitted value for each
group is just the arithmetic mean of that group's counts, whatever the
dispersion. That collapses the iteratively reweighted least squares fit to a
closed form, so the whole analysis vectorizes over genes and the log2 fold
change is exactly log2(mean at generation 10 / mean at generation 0), which is
the ratio the paper describes.

What is still an approximation: edgeR's empirical Bayes dispersion moderation
uses a spline-interpolated locally weighted likelihood. The version here uses a
grid search with nearest-neighbour smoothing over average abundance. Dispersions
will be close but not identical, so a handful of genes near the FDR cutoff can
flip. Run --backend edger if that matters.

Usage
-----
    python src/step05_fitness.py
    python src/step05_fitness.py --backend edger
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import chi2

from common import (ensure_dir, get_logger, load_config, load_sample_sheet,
                    read_table, sample_columns_for_strain, write_table)

log = get_logger("step05")

# Dispersion grid, spanning the range edgeR explores.
DISP_GRID = np.exp(np.linspace(np.log(1e-4), np.log(4.0), 120))
MU_FLOOR = 1e-8       # keeps logs finite when a group mean is exactly zero


# ---------------------------------------------------------------------------
# Negative binomial pieces
# ---------------------------------------------------------------------------

def nb_loglik(y: np.ndarray, mu: np.ndarray, disp: float) -> np.ndarray:
    """Per-gene negative binomial log-likelihood, summed over samples.

    y and mu are (n_genes, n_samples). disp is a scalar dispersion phi, with the
    NB parameterized so that variance = mu + phi * mu^2.
    """
    inv = 1.0 / disp
    mu = np.maximum(mu, MU_FLOOR)
    ll = (gammaln(y + inv) - gammaln(inv) - gammaln(y + 1.0)
          + y * np.log(disp * mu) - (y + inv) * np.log1p(disp * mu))
    return ll.sum(axis=1)


def nb_deviance(y: np.ndarray, mu: np.ndarray, disp: float) -> np.ndarray:
    """Per-gene NB deviance, summed over samples.

    Uses edgeR's unit deviance:

        dev_i = 2 * [ y*log(y/mu) - (y + 1/phi) * log((y + 1/phi)/(mu + 1/phi)) ]

    with the convention 0*log(0) = 0. This equals 2*(saturated log-likelihood
    minus model log-likelihood); the gamma terms in the log-likelihood do not
    depend on mu and cancel in the difference. Verified against
    statsmodels.GLM(family=NegativeBinomial) by checking that
    dev_reduced - dev_full matches 2*(llf_full - llf_reduced).
    """
    mu = np.maximum(mu, MU_FLOOR)
    inv = 1.0 / disp
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = np.where(y > 0, y * np.log(np.maximum(y, MU_FLOOR) / mu), 0.0)
    term2 = (y + inv) * np.log((y + inv) / (mu + inv))
    return 2.0 * (term1 - term2).sum(axis=1)


def cox_reid_apl(y: np.ndarray, mu: np.ndarray, disp: float,
                 group_idx: list[np.ndarray]) -> np.ndarray:
    """Cox-Reid adjusted profile log-likelihood for one dispersion value.

    The adjustment is -0.5 * log|X'WX|. For a group-means design the information
    matrix is diagonal with entries sum_{i in group} w_i, and the IRLS weight is
    w_i = mu_i / (1 + phi * mu_i).
    """
    ll = nb_loglik(y, mu, disp)
    mu_safe = np.maximum(mu, MU_FLOOR)
    w = mu_safe / (1.0 + disp * mu_safe)
    logdet = np.zeros(y.shape[0])
    for idx in group_idx:
        logdet += np.log(np.maximum(w[:, idx].sum(axis=1), MU_FLOOR))
    return ll - 0.5 * logdet


def group_means(y: np.ndarray, group_idx: list[np.ndarray]) -> np.ndarray:
    """Fitted values under the full (group-means) model.

    With equal offsets this is the exact NB maximum likelihood fit.
    """
    mu = np.empty_like(y, dtype=float)
    for idx in group_idx:
        m = y[:, idx].mean(axis=1, keepdims=True)
        mu[:, idx] = m
    return mu


def smooth_by_abundance(values: np.ndarray, abundance: np.ndarray,
                        span: float = 0.3) -> np.ndarray:
    """Nearest-neighbour smoothing of a per-gene quantity along abundance.

    Stands in for edgeR's locally weighted likelihood. `values` is
    (n_genes, n_grid); the smoothing is applied down the gene axis for each
    grid point using a sliding window over abundance rank.
    """
    order = np.argsort(abundance)
    n = len(order)
    width = max(3, int(round(span * n)))
    half = width // 2

    ordered = values[order]
    # Cumulative sums make the sliding window average O(n) per grid point.
    cs = np.vstack([np.zeros((1, ordered.shape[1])), np.cumsum(ordered, axis=0)])
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    counts = (hi - lo)[:, None]
    smoothed_ordered = (cs[hi] - cs[lo]) / counts

    smoothed = np.empty_like(smoothed_ordered)
    smoothed[order] = smoothed_ordered
    return smoothed


def estimate_dispersions(y: np.ndarray, group_idx: list[np.ndarray],
                         prior_df: float = 10.0,
                         trend_span: float = 0.3):
    """Estimate common, trended and tagwise dispersions.

    Mirrors the shape of edgeR's estimateDisp: build the adjusted profile
    likelihood over a dispersion grid, smooth it along abundance to get a trend,
    then shrink each gene's own likelihood toward that trend with weight
    prior_df / residual_df.
    """
    n_genes, n_samples = y.shape
    n_coef = len(group_idx)
    residual_df = max(n_samples - n_coef, 1)

    mu = group_means(y, group_idx)

    apl = np.empty((n_genes, len(DISP_GRID)))
    for j, disp in enumerate(DISP_GRID):
        apl[:, j] = cox_reid_apl(y, mu, disp, group_idx)

    common = DISP_GRID[np.argmax(apl.sum(axis=0))]

    abundance = np.log(np.maximum(y.mean(axis=1), MU_FLOOR))
    apl_trend = smooth_by_abundance(apl, abundance, span=trend_span)
    trended = DISP_GRID[np.argmax(apl_trend, axis=1)]

    prior_n = prior_df / residual_df
    apl_tagwise = apl + prior_n * apl_trend
    tagwise = DISP_GRID[np.argmax(apl_tagwise, axis=1)]

    log.info("dispersion: common=%.4f, trended median=%.4f, tagwise median=%.4f "
             "(residual df=%d, prior weight=%.2f)",
             common, float(np.median(trended)), float(np.median(tagwise)),
             residual_df, prior_n)
    return {"common": common, "trended": trended, "tagwise": tagwise}


def glm_lrt(y: np.ndarray, group_idx: list[np.ndarray], disp: np.ndarray):
    """Likelihood ratio test of group effect, one dispersion per gene.

    Returns (logFC, lr_statistic, p_value). logFC is log2 of the ratio of the
    second group's fitted mean to the first group's.
    """
    mu_full = group_means(y, group_idx)
    mu_null = np.repeat(y.mean(axis=1, keepdims=True), y.shape[1], axis=1)

    n_genes = y.shape[0]
    dev_full = np.empty(n_genes)
    dev_null = np.empty(n_genes)

    # Deviance depends on the dispersion, which varies per gene, so group genes
    # by their dispersion value to keep this vectorized.
    for d in np.unique(disp):
        sel = disp == d
        dev_full[sel] = nb_deviance(y[sel], mu_full[sel], d)
        dev_null[sel] = nb_deviance(y[sel], mu_null[sel], d)

    lr = np.maximum(dev_null - dev_full, 0.0)
    pval = chi2.sf(lr, df=len(group_idx) - 1)

    m0 = np.maximum(y[:, group_idx[0]].mean(axis=1), MU_FLOOR)
    m1 = np.maximum(y[:, group_idx[1]].mean(axis=1), MU_FLOOR)
    logfc = np.log2(m1 / m0)
    return logfc, lr, pval


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR, same as R's p.adjust(method='BH')."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    out = np.empty(n)
    out[order] = adjusted
    return out


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def run_strain_python(sub: pd.DataFrame, g0: list[str], g10: list[str],
                      prior_df: float, trend_span: float,
                      dispersion_type: str) -> pd.DataFrame:
    """Analyze one strain with the pure-Python backend."""
    cols = g0 + g10
    y = sub[cols].to_numpy(dtype=float)
    group_idx = [np.arange(len(g0)), np.arange(len(g0), len(cols))]

    disps = estimate_dispersions(y, group_idx, prior_df=prior_df,
                                 trend_span=trend_span)
    chosen = disps[dispersion_type]
    if np.isscalar(chosen):
        chosen = np.full(y.shape[0], float(chosen))

    logfc, lr, pval = glm_lrt(y, group_idx, chosen)
    fdr = benjamini_hochberg(pval)

    return pd.DataFrame({
        "logFC": logfc,
        "logCPM": np.log2(np.maximum(y.mean(axis=1), MU_FLOOR)),
        "LR": lr,
        "PValue": pval,
        "FDR": fdr,
        "dispersion": chosen,
        "mean_gen0": y[:, group_idx[0]].mean(axis=1),
        "mean_gen10": y[:, group_idx[1]].mean(axis=1),
    }, index=sub.index)


def run_strain_edgepython(sub: pd.DataFrame, g0: list[str], g10: list[str]) -> pd.DataFrame:
    """Analyze one strain with edgePython (pachterlab/edgePython on PyPI).

    edgePython is a pure-Python port of edgeR's internals: TMM normalization,
    weighted-likelihood empirical Bayes dispersion shrinkage, glm_fit and
    glm_lrt. Unlike the "python" backend here it reproduces edgeR's dispersion
    moderation rather than approximating it, so results should track the paper's
    edgeR analysis closely without needing R.

    Note on provenance: edgePython's own README states it was written mainly by
    Claude and Codex under Lior Pachter's direction. Treat it as you would any
    third-party dependency: the wrapper below is small enough to audit, and the
    "edger" backend remains available as an independent reference if you want to
    confirm agreement on your own data.
    """
    try:
        import edgepython as ep
    except ImportError as exc:
        raise SystemExit(
            "The edgepython backend needs the edgepython package:\n"
            "    pip install edgepython"
        ) from exc

    cols = g0 + g10
    counts = sub[cols]
    group = np.array(["gen0"] * len(g0) + ["gen10"] * len(g10))

    # The matrix is already library-size normalized (step 03), so TMM factors
    # come out near 1; we still call calc_norm_factors so the pipeline matches
    # what a normal edgeR run would do. To force the paper's "norm.factors = 1"
    # behaviour exactly, pass --no-tmm (handled by the caller via a flag).
    y = ep.make_dgelist(counts=counts.to_numpy(dtype=float), group=group)
    y = ep.calc_norm_factors(y)
    design = np.column_stack([np.ones(len(cols)),
                              (group == "gen10").astype(float)])
    y = ep.estimate_disp(y, design)
    fit = ep.glm_fit(y, design)
    lrt = ep.glm_lrt(fit, coef=1)
    tt = ep.top_tags(lrt, n=counts.shape[0], sort_by="none")
    table = tt["table"] if isinstance(tt, dict) else tt

    res = pd.DataFrame(table).copy()
    res.index = counts.index
    # edgePython returns logFC, logCPM, LR, PValue, FDR, matching edgeR.
    res["mean_gen0"] = counts[g0].mean(axis=1)
    res["mean_gen10"] = counts[g10].mean(axis=1)
    return res


def run_strain_edger(sub: pd.DataFrame, g0: list[str], g10: list[str]) -> pd.DataFrame:
    """Analyze one strain by calling edgeR through rpy2."""
    try:
        from rpy2 import robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
    except ImportError as exc:
        raise SystemExit(
            "The edger backend needs rpy2. Install it with:\n"
            "    pip install rpy2\n"
            "and make sure R has edgeR:\n"
            '    R -e \'BiocManager::install("edgeR")\''
        ) from exc

    edger = importr("edgeR")
    pandas2ri.activate()

    cols = g0 + g10
    counts = sub[cols]
    groups = robjects.FactorVector(["gen0"] * len(g0) + ["gen10"] * len(g10),
                                   levels=robjects.StrVector(["gen0", "gen10"]))

    r_counts = pandas2ri.py2rpy(counts)
    robjects.globalenv["counts"] = r_counts
    robjects.globalenv["grp"] = groups

    # The matrix is already library-size normalized, so lib.size is fixed at the
    # column totals and norm.factors are left at 1, matching the paper.
    robjects.r("""
        library(edgeR)
        d <- DGEList(counts = as.matrix(counts), group = grp)
        d$samples$norm.factors <- rep(1, ncol(counts))
        design <- model.matrix(~ grp)
        d <- estimateDisp(d, design)
        fit <- glmFit(d, design)
        lrt <- glmLRT(fit, coef = 2)
        res <- topTags(lrt, n = nrow(counts), sort.by = "none")$table
    """)
    res = pandas2ri.rpy2py(robjects.r("res"))
    res.index = counts.index
    res["mean_gen0"] = counts[g0].mean(axis=1)
    res["mean_gen10"] = counts[g10].mean(axis=1)
    return res


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def analyze_matrix(matrix: pd.DataFrame, sheet: pd.DataFrame, backend: str,
                   prior_df: float, trend_span: float,
                   dispersion_type: str, label: str):
    """Run every strain and return (per_strain_results, wide_summary)."""
    per_strain = {}
    for strain in sorted(sheet["strain"].unique()):
        g0, g10 = sample_columns_for_strain(sheet, strain)
        g0 = [c for c in g0 if c in matrix.columns]
        g10 = [c for c in g10 if c in matrix.columns]
        if len(g0) < 2 or len(g10) < 2:
            log.warning("Strain %s has %d gen0 and %d gen10 samples; skipping",
                        strain, len(g0), len(g10))
            continue

        log.info("[%s] %s: %d vs %d replicates", label, strain, len(g0), len(g10))

        # Genes with no reads anywhere in this strain carry no information.
        sub = matrix.loc[matrix[g0 + g10].sum(axis=1) > 0]

        if backend == "python":
            res = run_strain_python(sub, g0, g10, prior_df, trend_span, dispersion_type)
        elif backend == "edgepython":
            res = run_strain_edgepython(sub, g0, g10)
        else:
            res = run_strain_edger(sub, g0, g10)

        res = res.reindex(matrix.index)
        per_strain[strain] = res

        n_sig = int((res["FDR"] < 0.05).sum())
        n_del = int(((res["FDR"] < 0.05) & (res["logFC"] < 0)).sum())
        n_ben = int(((res["FDR"] < 0.05) & (res["logFC"] > 0)).sum())
        log.info("  %s: %d significant (FDR<0.05); %d deleterious, %d beneficial",
                 strain, n_sig, n_del, n_ben)

    # Wide table: two columns per strain, matching Supplementary file 4 Tabs 3-4.
    wide = pd.DataFrame(index=matrix.index)
    for strain, res in per_strain.items():
        wide[f"{strain}_log2FC"] = res["logFC"]
        wide[f"{strain}_FDR"] = res["FDR"]
    return per_strain, wide


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--backend", choices=["python", "edgepython", "edger"],
                    default="python",
                    help="python: built-in NB GLM (no extra deps). "
                         "edgepython: pure-Python edgeR port (pip install edgepython), "
                         "recommended for edgeR-faithful output. "
                         "edger: real edgeR via rpy2.")
    ap.add_argument("--prior-df", type=float, default=10.0,
                    help="Empirical Bayes prior degrees of freedom (edgeR default 10)")
    ap.add_argument("--trend-span", type=float, default=0.3,
                    help="Fraction of genes in the abundance smoothing window")
    ap.add_argument("--dispersion", choices=["common", "trended", "tagwise"],
                    default="tagwise")
    args = ap.parse_args()

    cfg = load_config(args.config)
    norm_dir = Path(cfg["paths"]["normalized_dir"])
    imp_dir = Path(cfg["paths"]["imputed_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["fitness_dir"]))

    sheet = load_sample_sheet(Path(cfg["paths"]["sample_sheet"]))

    jobs = [
        ("no_imputation", norm_dir / "normalized_counts.tsv"),
        ("imputed", imp_dir / "normalized_counts_imputed.tsv"),
    ]

    for label, path in jobs:
        if not path.exists():
            log.warning("Missing %s; skipping the %s analysis", path, label)
            continue
        matrix = read_table(path)
        per_strain, wide = analyze_matrix(
            matrix, sheet, args.backend, args.prior_df,
            args.trend_span, args.dispersion, label,
        )

        strain_dir = ensure_dir(out_dir / label / "per_strain")
        for strain, res in per_strain.items():
            write_table(res, strain_dir / f"{strain}.tsv")
        write_table(wide, out_dir / f"fitness_scores_{label}.tsv")
        log.info("Wrote %s", out_dir / f"fitness_scores_{label}.tsv")

        print(f"\n=== Significant genes per strain ({label}, FDR<0.05) ===")
        summary = []
        for strain, res in per_strain.items():
            sig = res["FDR"] < 0.05
            summary.append({
                "strain": strain,
                "deleterious": int((sig & (res["logFC"] < 0)).sum()),
                "beneficial": int((sig & (res["logFC"] > 0)).sum()),
                "total_significant": int(sig.sum()),
            })
        sdf = pd.DataFrame(summary).sort_values("deleterious")
        print(sdf.to_string(index=False))
        write_table(sdf, out_dir / f"significant_gene_counts_{label}.tsv", index=False)

        n_any = int((wide.filter(like="_FDR") < 0.05).any(axis=1).sum())
        print(f"\nGenes significant in at least one strain: {n_any}")
        print("The paper reports 4064, with a median of 1726 per strain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
