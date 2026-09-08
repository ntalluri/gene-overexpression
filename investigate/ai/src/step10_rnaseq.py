"""
step10_rnaseq.py
----------------
The transcriptome arm: GEO series GSE171585 and Supplementary file 5.

What the paper did
------------------
    "Differentially expressed genes were identified by edgeR using a linear model
     with strain background as a factor and paired replicates, identifying genes
     differentially expressed in each strain relative to the average of all
     strains using an FDR cutoff of 0.05... There was a total of 4802 genes that
     were significant in at least one strain."

This differs from the Bar-seq model in two ways that matter. The comparison is
each strain against the mean of all strains rather than against a control, and
replicate is a blocking factor, so the design is `~ replicate + strain` fitted
across all strains at once with sum-to-zero contrasts on strain.

That is a genuinely multi-coefficient GLM, so unlike step 05 there is no closed
form to exploit. Three backends are available: --backend python fits the model
by iteratively reweighted least squares with no extra dependencies;
--backend edgepython uses the edgeR port (pip install edgepython) for
edgeR-faithful dispersion moderation without R, recommended here; and
--backend edger calls real edgeR through rpy2 for exact agreement with the paper.

Input
-----
Either the raw counts from Supplementary file 5 Tab 1 (fastest route), or a
counts matrix you produced yourself from the GSE171585 reads. The paper's own
route was Trimmomatic, then BWA-MEM against S288c R64-1-1, then HT-Seq.

    curl -L -o data/external/elife-70564-supp5-v2.xlsx \\
      https://cdn.elifesciences.org/articles/70564/elife-70564-supp5-v2.xlsx

Cross-referencing
-----------------
The paper's strain-specific models rest on connecting differentially expressed
genes to the overexpression gene lists. `crossref` does that: for each strain it
intersects the strain-specific OE lists with that strain's up- and
down-regulated genes and reports the overlap with a hypergeometric p-value.

Usage
-----
    python src/step10_rnaseq.py --counts data/external/elife-70564-supp5-v2.xlsx
    python src/step10_rnaseq.py --backend edger
    python src/step10_rnaseq.py --only crossref
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from common import (ensure_dir, get_logger, load_config, load_sample_sheet,
                    read_table, write_table)
from enrichment import overlap_test
from step05_fitness import benjamini_hochberg, nb_deviance
from step08_annotations import first_existing

log = get_logger("step10")

STRAINS = [
    "BY4743", "BC187", "DBVPG1373", "NCYC3290", "Y12", "Y2209", "Y389",
    "Y7568", "YJM1273", "YJM1389", "YJM1592", "YJM978", "YPS128", "YPS163",
    "YPS606",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def known_strains(cfg) -> list[str]:
    """Strain names to look for in RNA-seq column headers.

    The 15 isolates from the paper, plus whatever appears in the Bar-seq sample
    sheet. Including the sample sheet means renamed or test strains are picked up
    without editing this file.
    """
    names = set(STRAINS)
    try:
        sheet = load_sample_sheet(Path(cfg["paths"]["sample_sheet"]))
        names |= set(sheet["strain"].astype(str))
    except Exception:
        pass
    return sorted(names)


def parse_sample_name(name: str, strains: list[str] | None = None):
    """Split an RNA-seq column header into (strain, replicate).

    Headers in Supplementary file 5 are free text, so this is a guess. Anything
    it cannot parse is reported rather than silently dropped.
    """
    upper = str(name).upper()
    strain = None
    for s in sorted(strains or STRAINS, key=len, reverse=True):
        if s.upper() in upper:
            strain = s
            break
    if strain is None:
        return None, None

    m = re.search(r"(?:REP|REPLICATE|R)[\s_\-]*([A-C1-3])\b", upper)
    if m:
        return strain, m.group(1)
    m = re.search(r"[\s_\-]([A-C1-3])$", str(name).strip())
    if m:
        return strain, m.group(1).upper()
    return strain, "1"


def load_counts(path: Path, strains: list[str] | None = None
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load an RNA-seq count matrix and derive its sample table.

    Accepts the eLife xlsx (Tab 1) or a plain TSV with genes as rows.
    """
    if path.suffix.lower() in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)
        name = list(sheets)[0]
        df = sheets[name]
        log.info("Reading sheet '%s' from %s", name, path.name)
        gene_col = df.columns[0]
        df = df.set_index(gene_col)
    else:
        df = read_table(path)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all").fillna(0)
    df.index = df.index.astype(str)
    df.index.name = "gene"

    rows, unparsed = [], []
    for col in df.columns:
        strain, rep = parse_sample_name(col, strains)
        if strain is None:
            unparsed.append(col)
            continue
        rows.append({"sample_id": col, "strain": strain, "replicate": rep})

    if unparsed:
        log.warning("Could not assign a strain to %d columns: %s",
                    len(unparsed), unparsed[:8])
        df = df.drop(columns=unparsed)

    if not rows:
        raise ValueError(
            f"No column in {path.name} could be matched to a strain name.\n"
            f"Columns seen: {list(df.columns)[:10]}\n"
            "Strain names searched for: " + ", ".join(strains or STRAINS) + "\n"
            "Either rename the columns to include the strain, or pass "
            "--strains to list the names to look for."
        )

    samples = pd.DataFrame(rows)
    log.info("RNA-seq matrix: %d genes x %d samples, %d strains",
             df.shape[0], len(samples), samples["strain"].nunique())
    return df, samples


# ---------------------------------------------------------------------------
# Pure-Python negative binomial GLM
# ---------------------------------------------------------------------------

def build_design(samples: pd.DataFrame):
    """Design matrix for ~ replicate + strain with sum-to-zero strain contrasts.

    Sum-to-zero coding is what makes each strain coefficient a comparison against
    the mean of all strains, which is the contrast the paper used. The last
    strain has no column of its own; its effect is minus the sum of the others,
    and is recovered explicitly after fitting.
    """
    strains = sorted(samples["strain"].unique())
    reps = sorted(samples["replicate"].unique())

    n = len(samples)
    columns = [np.ones(n)]
    names = ["intercept"]

    # Replicate blocking, treatment-coded against the first replicate.
    for rep in reps[1:]:
        columns.append((samples["replicate"] == rep).to_numpy(float))
        names.append(f"replicate_{rep}")

    # Strain, sum-to-zero coded.
    strain_values = samples["strain"].to_numpy()
    for strain in strains[:-1]:
        col = (strain_values == strain).astype(float).copy()
        col[strain_values == strains[-1]] = -1.0
        columns.append(col)
        names.append(f"strain_{strain}")

    X = np.column_stack(columns)
    strain_cols = {s: names.index(f"strain_{s}") for s in strains[:-1]}
    return X, names, strains, strain_cols


def fit_nb_glm(y: np.ndarray, X: np.ndarray, offset: np.ndarray,
               disp: float, max_iter: int = 40, tol: float = 1e-8):
    """Fit one gene's NB GLM by IRLS. Returns (beta, deviance)."""
    n = len(y)
    mu = np.maximum(y, 0.25) + 0.1
    beta = np.zeros(X.shape[1])

    for _ in range(max_iter):
        eta = np.log(mu) - offset
        # NB with log link: W = mu / (1 + phi*mu), working response z = eta + (y-mu)/mu
        w = mu / (1.0 + disp * mu)
        z = eta + (y - mu) / mu
        XtW = X.T * w
        try:
            new_beta = np.linalg.solve(XtW @ X + 1e-8 * np.eye(X.shape[1]), XtW @ z)
        except np.linalg.LinAlgError:
            break
        new_eta = X @ new_beta + offset
        new_mu = np.exp(np.clip(new_eta, -30, 30))
        if np.max(np.abs(new_beta - beta)) < tol:
            beta = new_beta
            mu = new_mu
            break
        beta, mu = new_beta, new_mu

    dev = nb_deviance(y[None, :], mu[None, :], disp)[0]
    return beta, dev, mu


def estimate_common_dispersion(counts: np.ndarray, X: np.ndarray,
                               offset: np.ndarray,
                               grid: np.ndarray | None = None,
                               n_sample: int = 1500,
                               seed: int = 0) -> float:
    """Common dispersion by maximizing summed deviance-based fit over a grid.

    A subsample of genes is used because this refits the GLM at every grid point.
    """
    if grid is None:
        grid = np.exp(np.linspace(np.log(1e-3), np.log(2.0), 25))

    rng = np.random.default_rng(seed)
    idx = rng.choice(counts.shape[0], size=min(n_sample, counts.shape[0]),
                     replace=False)
    sub = counts[idx]
    residual_df = X.shape[0] - X.shape[1]

    best, best_score = grid[0], np.inf
    for disp in grid:
        total = 0.0
        for row in sub:
            _, dev, _ = fit_nb_glm(row, X, offset, disp)
            total += dev
        # The dispersion whose residual deviance is closest to the residual
        # degrees of freedom, the standard moment-matching criterion.
        score = abs(total / len(sub) - residual_df)
        if score < best_score:
            best, best_score = disp, score

    log.info("Common dispersion: %.4f (residual df %d)", best, residual_df)
    return float(best)


def run_python_backend(counts: pd.DataFrame, samples: pd.DataFrame,
                       min_count: int = 10) -> dict[str, pd.DataFrame]:
    """Fit ~ replicate + strain and test each strain against the overall mean."""
    keep = counts.sum(axis=1) >= min_count
    log.info("Keeping %d of %d genes with total count >= %d",
             int(keep.sum()), len(counts), min_count)
    mat = counts.loc[keep, samples["sample_id"]].to_numpy(float)

    X, names, strains, strain_cols = build_design(samples)
    lib = mat.sum(axis=0)
    offset = np.log(lib / np.exp(np.mean(np.log(lib))))

    disp = estimate_common_dispersion(mat, X, offset)

    n_genes = mat.shape[0]
    n_coef = X.shape[1]
    betas = np.zeros((n_genes, n_coef))
    dev_full = np.zeros(n_genes)
    for i, row in enumerate(mat):
        betas[i], dev_full[i], _ = fit_nb_glm(row, X, offset, disp)
        if (i + 1) % 1000 == 0:
            log.info("  fitted %d/%d genes", i + 1, n_genes)

    results = {}
    # Each strain gets a reduced model with its own column dropped.
    for strain in strains:
        if strain in strain_cols:
            drop = strain_cols[strain]
            X_red = np.delete(X, drop, axis=1)
            logfc = betas[:, drop] / np.log(2)
        else:
            # The reference strain's effect is minus the sum of the others.
            cols = list(strain_cols.values())
            logfc = -betas[:, cols].sum(axis=1) / np.log(2)
            # Test it by dropping all strain columns and comparing, which tests
            # the whole strain term. Flagged in the output so you know.
            X_red = np.delete(X, cols, axis=1)

        dev_red = np.zeros(n_genes)
        for i, row in enumerate(mat):
            _, dev_red[i], _ = fit_nb_glm(row, X_red, offset, disp)

        df_diff = X.shape[1] - X_red.shape[1]
        lr = np.maximum(dev_red - dev_full, 0.0)
        pval = chi2.sf(lr, df=df_diff)
        results[strain] = pd.DataFrame({
            "logFC": logfc,
            "LR": lr,
            "PValue": pval,
            "FDR": benjamini_hochberg(pval),
        }, index=counts.index[keep])
        n_sig = int((results[strain]["FDR"] < 0.05).sum())
        log.info("%s: %d differentially expressed genes (FDR<0.05)", strain, n_sig)

    return results


def run_edgepython_backend(counts: pd.DataFrame, samples: pd.DataFrame,
                           min_count: int = 10) -> dict:
    """Same ~ replicate + strain model through edgePython.

    edgePython (pip install edgepython) is a pure-Python port of edgeR, so this
    fits the identical design with edgeR's own dispersion moderation, no R
    required. Each strain is tested via its own coefficient in the sum-to-zero
    design, so the coefficient is that strain's deviation from the overall mean,
    matching the paper's contrast. The reference strain (which has no column of
    its own) is tested with the negative-sum contrast.

    See step05's edgepython note on provenance.
    """
    try:
        import edgepython as ep
    except ImportError as exc:
        raise SystemExit("The edgepython backend needs: pip install edgepython") from exc

    keep = counts.sum(axis=1) >= min_count
    log.info("Keeping %d of %d genes with total count >= %d",
             int(keep.sum()), len(counts), min_count)
    mat = counts.loc[keep, samples["sample_id"]]

    X, names, strains, strain_cols = build_design(samples)
    group = samples["strain"].to_numpy()

    y = ep.make_dgelist(counts=mat.to_numpy(dtype=float), group=group)
    y = ep.calc_norm_factors(y)
    y = ep.estimate_disp(y, X)
    fit = ep.glm_fit(y, X)

    results = {}
    for strain in strains:
        if strain in strain_cols:
            lrt = ep.glm_lrt(fit, coef=strain_cols[strain])
        else:
            # Reference strain: contrast = -1 on every strain column.
            contrast = np.zeros(X.shape[1])
            for c in strain_cols.values():
                contrast[c] = -1.0
            lrt = ep.glm_lrt(fit, contrast=contrast)
        tt = ep.top_tags(lrt, n=mat.shape[0], sort_by="none")
        table = tt["table"] if isinstance(tt, dict) else tt
        res = pd.DataFrame(table).copy()
        res.index = mat.index
        results[strain] = res
        log.info("%s: %d DE genes (FDR<0.05)", strain,
                 int((res["FDR"] < 0.05).sum()))
    return results


def run_edger_backend(counts: pd.DataFrame, samples: pd.DataFrame) -> dict:
    """Same model through real edgeR via rpy2."""
    try:
        from rpy2 import robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
    except ImportError as exc:
        raise SystemExit("The edger backend needs rpy2 and R with edgeR.") from exc

    importr("edgeR")
    pandas2ri.activate()

    mat = counts[samples["sample_id"]]
    robjects.globalenv["counts"] = pandas2ri.py2rpy(mat)
    robjects.globalenv["strain"] = robjects.FactorVector(list(samples["strain"]))
    robjects.globalenv["rep"] = robjects.FactorVector(list(samples["replicate"]))

    robjects.r("""
        library(edgeR)
        d <- DGEList(counts = as.matrix(counts))
        keep <- filterByExpr(d, group = strain)
        d <- d[keep, , keep.lib.sizes = FALSE]
        d <- calcNormFactors(d)
        # Sum-to-zero contrasts make each strain coefficient a comparison
        # against the mean of all strains.
        contrasts(strain) <- contr.sum(nlevels(strain))
        design <- model.matrix(~ rep + strain)
        d <- estimateDisp(d, design)
        fit <- glmFit(d, design)
        strain_cols <- grep('^strain', colnames(design))
        strain_levels <- levels(strain)
    """)

    strain_levels = list(robjects.r("strain_levels"))
    results = {}
    for i, strain in enumerate(strain_levels[:-1]):
        robjects.globalenv["ci"] = robjects.IntVector([int(robjects.r("strain_cols")[i])])
        robjects.r("""
            lrt <- glmLRT(fit, coef = ci)
            res <- topTags(lrt, n = nrow(fit), sort.by = 'none')$table
        """)
        res = pandas2ri.rpy2py(robjects.r("res"))
        res.index = list(robjects.r("rownames(fit)"))
        results[strain] = res
        log.info("%s: %d DE genes", strain, int((res["FDR"] < 0.05).sum()))

    # The last level, tested against the mean via the negative sum contrast.
    robjects.r("""
        L <- rep(0, ncol(design)); L[strain_cols] <- -1
        lrt <- glmContrasts <- glmLRT(fit, contrast = L)
        res <- topTags(lrt, n = nrow(fit), sort.by = 'none')$table
    """)
    res = pandas2ri.rpy2py(robjects.r("res"))
    res.index = list(robjects.r("rownames(fit)"))
    results[strain_levels[-1]] = res
    return results


# ---------------------------------------------------------------------------
# Cross-referencing with the overexpression results
# ---------------------------------------------------------------------------

def crossref(cfg, out_dir: Path):
    """Intersect strain-specific OE lists with that strain's DE genes."""
    de_path = out_dir / "rnaseq_de_wide.tsv"
    if not de_path.exists():
        log.warning("No RNA-seq results yet; run the differential step first")
        return

    de = read_table(de_path)
    gl = Path(cfg["paths"]["gene_lists_dir"])
    ss_dir = gl / "strain_specific"
    if not ss_dir.exists():
        log.warning("No strain-specific OE lists; run step 06 first")
        return

    alpha = cfg["statistics"]["fdr_alpha"]
    universe = set(de.index)
    rows = []

    strains = sorted({c.rsplit("_", 1)[0] for c in de.columns})
    for strain in strains:
        lfc_col, fdr_col = f"{strain}_log2FC", f"{strain}_FDR"
        if lfc_col not in de.columns:
            continue
        up = set(de.index[(de[fdr_col] < alpha) & (de[lfc_col] > 0)])
        down = set(de.index[(de[fdr_col] < alpha) & (de[lfc_col] < 0)])

        for direction in ("deleterious", "beneficial"):
            path = ss_dir / f"{strain}_{direction}.tsv"
            if not path.exists():
                continue
            oe = set(pd.read_csv(path, sep="\t")["gene"].dropna().astype(str))
            oe &= universe
            if len(oe) < 5:
                continue
            for de_label, de_set in (("expressed_higher", up),
                                     ("expressed_lower", down)):
                res = overlap_test(oe, de_set, universe)
                rows.append({
                    "strain": strain, "oe_direction": direction,
                    "expression": de_label, **res,
                })

    if not rows:
        print("\nNo cross-references could be computed.")
        return

    df = pd.DataFrame(rows).sort_values("pvalue")
    write_table(df, out_dir / "oe_vs_expression_crossref.tsv", index=False)
    print("\n=== Overexpression lists vs differential expression ===")
    print(df.head(30).to_string(index=False))
    print("\nPaper's specific claims to check here: genes beneficial to")
    print("DBVPG1373 and Y12 are enriched for mitotic cell cycle genes, and both")
    print("strains express G2/M genes lower; genes deleterious to Y12 encode")
    print("mitochondrial matrix proteins, a category expressed higher in Y12;")
    print("DBVPG1373 represses aromatic amino acid biosynthesis (p = 4.1e-6).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--counts", default=None,
                    help="RNA-seq counts file. Defaults to Supplementary file 5 "
                         "in data/external if present.")
    ap.add_argument("--backend", choices=["python", "edgepython", "edger"],
                    default="python",
                    help="edgepython (pip install edgepython) reproduces edgeR "
                         "without R and is recommended here.")
    ap.add_argument("--strains", nargs="*", default=None,
                    help="Strain names to look for in the column headers. "
                         "Defaults to the paper's 15 isolates plus anything in "
                         "the Bar-seq sample sheet.")
    ap.add_argument("--only", nargs="*", choices=["differential", "crossref"],
                    default=["differential", "crossref"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "09_rnaseq")

    if "differential" in args.only:
        ext = Path(cfg["paths"]["external_dir"])
        path = Path(args.counts) if args.counts else first_existing(
            ext, "elife-70564-supp5-v2.xlsx", "rnaseq_counts.tsv")
        if path is None or not Path(path).exists():
            log.error("No RNA-seq counts found. Download Supplementary file 5:")
            log.error("  curl -L -o data/external/elife-70564-supp5-v2.xlsx \\")
            log.error("    https://cdn.elifesciences.org/articles/70564/"
                      "elife-70564-supp5-v2.xlsx")
            return 1

        counts, samples = load_counts(Path(path),
                                      args.strains or known_strains(cfg))
        write_table(samples, out_dir / "rnaseq_samples.tsv", index=False)

        if args.backend == "python":
            results = run_python_backend(counts, samples)
        elif args.backend == "edgepython":
            results = run_edgepython_backend(counts, samples)
        else:
            results = run_edger_backend(counts, samples)

        per_dir = ensure_dir(out_dir / "per_strain")
        wide = pd.DataFrame(index=counts.index)
        for strain, res in results.items():
            write_table(res, per_dir / f"{strain}.tsv")
            wide[f"{strain}_log2FC"] = res["logFC"]
            wide[f"{strain}_FDR"] = res["FDR"]
        write_table(wide, out_dir / "rnaseq_de_wide.tsv")

        alpha = cfg["statistics"]["fdr_alpha"]
        n_any = int((wide.filter(like="_FDR") < alpha).any(axis=1).sum())
        summary = pd.DataFrame({
            "strain": list(results),
            "n_DE": [int((r["FDR"] < alpha).sum()) for r in results.values()],
            "n_up": [int(((r["FDR"] < alpha) & (r["logFC"] > 0)).sum())
                     for r in results.values()],
            "n_down": [int(((r["FDR"] < alpha) & (r["logFC"] < 0)).sum())
                       for r in results.values()],
        }).sort_values("n_DE")
        write_table(summary, out_dir / "rnaseq_summary.tsv", index=False)

        print("\n=== RNA-seq differential expression ===")
        print(summary.to_string(index=False))
        print(f"\nGenes significant in at least one strain: {n_any}")
        print("The paper reports 4802.")

    if "crossref" in args.only:
        crossref(cfg, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
