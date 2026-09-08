"""
step09_enrichments.py
---------------------
Run the enrichment analyses reported in the Results section, using the gene
lists from step 06, the annotations from step 08, and the test engine in
enrichment.py.

Background sets, as specified in the Methods
--------------------------------------------
    "taking as the background data set the total number of measured genes
     (except for strain-specific gene lists, in which the background data set
     was a list of insignificant genes in that strain with FDR>0.1 and measured
     in at least two of the biological replicates)"

So the commonly deleterious analysis uses the measured-gene background from
Supplementary file 8 Tab 1 or Tab 3, and each strain-specific list uses that
strain's own FDR>0.1 set from Tab 2. Both are produced by step 06.

Significance threshold
----------------------
    "Because gene lists are heavily overlapping, standard FDR calculations
     over-correct p-values. We therefore took a stringent p-value of 5x10-4 as
     significant, but also cite FDR significance in data files."

enrichment.py applies p < 5e-4 as the call and reports BH FDR alongside.

Analyses
--------
  common_genes           Categorical and continuous enrichment of the 431
                         commonly deleterious genes. Run twice, once on the
                         whole set and once with translation-related genes
                         removed, because the paper checks that its enrichments
                         survive that removal.
  strain_specific        Each strain's deleterious and beneficial lists against
                         that strain's own no-effect background.
  makanae_overlap        Hypergeometric overlap of BY4743 deleterious genes with
                         the Makanae et al. 2013 set. Paper: 851 genes,
                         p = 8e-45.
  balance_hypothesis     The two specific claims: commonly deleterious genes have
                         more physical interactions (paper p = 4.0e-69, Wilcoxon)
                         and include more complex members (p = 6.6e-12,
                         hypergeometric), both before and after removing
                         translation genes.

Usage
-----
    python src/step09_enrichments.py
    python src/step09_enrichments.py --only common_genes balance_hypothesis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import ensure_dir, get_logger, load_config, read_table, write_table
from enrichment import (P_THRESHOLD, hypergeometric_enrichment, load_categorical,
                        overlap_test, wilcoxon_enrichment)

log = get_logger("step09")


def load_annotations(cfg):
    """Load the categorical and continuous tables written by step 08."""
    ann_dir = Path(cfg["paths"]["results_dir"]) / "07_annotations"
    cat_path = ann_dir / "categories.tsv"
    feat_path = ann_dir / "features_continuous.tsv"

    categories = load_categorical(cat_path) if cat_path.exists() else {}
    features = read_table(feat_path) if feat_path.exists() else pd.DataFrame()

    if not categories:
        log.warning("No categorical annotations found; hypergeometric tests skipped")
    if features.empty:
        log.warning("No continuous features found; Wilcoxon tests skipped")
    return categories, features, ann_dir


def read_list(path: Path) -> set[str]:
    """Read a gene list written by step 06."""
    if not path.exists():
        return set()
    df = pd.read_csv(path, sep="\t")
    col = "gene" if "gene" in df.columns else df.columns[0]
    return set(df[col].dropna().astype(str))


def run_both_tests(query: set[str], background: set[str],
                   categories, features, out_prefix: Path, label: str):
    """Run the hypergeometric and Wilcoxon tests and write both tables."""
    results = {}

    if categories:
        hyper = hypergeometric_enrichment(query, background, categories)
        if not hyper.empty:
            hyper.to_csv(f"{out_prefix}_hypergeometric.tsv", sep="\t", index=False)
            results["hypergeometric"] = hyper

    if not features.empty:
        wilcox = wilcoxon_enrichment(query, background, features)
        if not wilcox.empty:
            wilcox.to_csv(f"{out_prefix}_wilcoxon.tsv", sep="\t", index=False)
            results["wilcoxon"] = wilcox

    n_hyper = int(results.get("hypergeometric", pd.DataFrame()).get(
        "passes_p_threshold", pd.Series(dtype=bool)).sum())
    n_wilcox = int(results.get("wilcoxon", pd.DataFrame()).get(
        "passes_p_threshold", pd.Series(dtype=bool)).sum())
    log.info("%s: query=%d background=%d, %d categories and %d features "
             "pass p<%.0e", label, len(query), len(background),
             n_hyper, n_wilcox, P_THRESHOLD)
    return results


def analysis_common_genes(cfg, categories, features, out_dir: Path):
    """Enrichment of the commonly deleterious genes, with and without translation."""
    gl = Path(cfg["paths"]["gene_lists_dir"])
    ann_dir = Path(cfg["paths"]["results_dir"]) / "07_annotations"

    common = read_list(gl / "commonly_deleterious.tsv")
    if not common:
        log.warning("No commonly deleterious list; run step 06 first")
        return

    # Tab 1 background: measured genes minus the common set.
    background = read_list(gl / "backgrounds" / "tab1_measured_minus_common.tsv")
    if not background:
        background = read_list(gl / "backgrounds" / "tab3_significant_any_strain.tsv")
    universe = common | background

    sub = ensure_dir(out_dir / "common_genes")
    run_both_tests(common, universe, categories, features,
                   sub / "all", "commonly deleterious, all")

    # The paper repeats several tests after removing translation-related genes.
    trans_path = ann_dir / "translation_genes.txt"
    if trans_path.exists():
        translation = set(pd.read_csv(trans_path, header=None)[0].astype(str))
        common_nt = common - translation
        universe_nt = universe - translation
        log.info("Removed %d translation-related genes from the common set "
                 "(%d remain of %d)", len(common & translation),
                 len(common_nt), len(common))
        run_both_tests(common_nt, universe_nt, categories, features,
                       sub / "no_translation",
                       "commonly deleterious, translation removed")
    else:
        log.warning("No translation gene list; skipping the without-translation "
                    "repeat. See fetch_external_data.py --list for how to get "
                    "Supplementary file 2.")

    print("\n=== Commonly deleterious enrichments ===")
    for name in ["all_hypergeometric", "all_wilcoxon",
                 "no_translation_hypergeometric", "no_translation_wilcoxon"]:
        p = sub / f"{name}.tsv"
        if not p.exists():
            continue
        df = pd.read_csv(p, sep="\t")
        top = df[df["passes_p_threshold"]].head(12)
        print(f"\n--- {name} ({len(df)} tested, "
              f"{int(df['passes_p_threshold'].sum())} pass p<5e-4) ---")
        if len(top):
            keep = [c for c in ["category", "feature", "observed", "expected",
                                "fold_enrichment", "median_query",
                                "median_background", "direction", "pvalue"]
                    if c in top.columns]
            print(top[keep].to_string(index=False))
    print("\nPaper: this set is enriched for translation, ribosomal proteins,")
    print("ribosome biogenesis, ESR-repressed genes, helicases and ATP binding,")
    print("mitosis regulators, nuclear localization and essential genes, all at")
    print("p < 1e-4, and all still significant after translation genes are removed.")


def analysis_strain_specific(cfg, categories, features, out_dir: Path):
    """Enrichment of each strain's specific lists against its own no-effect set."""
    gl = Path(cfg["paths"]["gene_lists_dir"])
    ss_dir = gl / "strain_specific"
    bg_dir = gl / "backgrounds" / "tab2_no_effect_per_strain"

    if not ss_dir.exists():
        log.warning("No strain-specific lists; run step 06 first")
        return

    sub = ensure_dir(out_dir / "strain_specific")
    summary = []

    strains = sorted({p.name.rsplit("_", 1)[0] for p in ss_dir.glob("*.tsv")})
    for strain in strains:
        background = read_list(bg_dir / f"{strain}_no_effect.tsv")
        if not background:
            log.warning("No no-effect background for %s; skipping", strain)
            continue

        for direction in ("deleterious", "beneficial"):
            query = read_list(ss_dir / f"{strain}_{direction}.tsv")
            if len(query) < 5:
                continue
            res = run_both_tests(query, query | background, categories, features,
                                 sub / f"{strain}_{direction}",
                                 f"{strain} {direction}")
            for kind, df in res.items():
                hits = df[df["passes_p_threshold"]]
                for row in hits.head(5).itertuples():
                    summary.append({
                        "strain": strain,
                        "direction": direction,
                        "test": kind,
                        "term": getattr(row, "category", None) or getattr(row, "feature", ""),
                        "pvalue": row.pvalue,
                    })

    if summary:
        sdf = pd.DataFrame(summary).sort_values(["strain", "direction", "pvalue"])
        write_table(sdf, out_dir / "strain_specific_top_enrichments.tsv", index=False)
        print("\n=== Top strain-specific enrichments (p < 5e-4) ===")
        print(sdf.to_string(index=False))
    else:
        print("\nNo strain-specific enrichments passed p < 5e-4.")

    print("\nPaper highlights to look for: mitotic cell cycle among genes")
    print("beneficial to DBVPG1373 and Y12 (p < 0.0004); mitochondrial matrix")
    print("among genes deleterious to Y12; tryptophan content among genes")
    print("deleterious to DBVPG1373 (p = 8.1e-5, Wilcoxon); DNA binding among")
    print("genes deleterious to Y7568 (p = 1.8e-4); nonsynonymous SNP counts in")
    print("BC187 and Y7568 beneficial genes (p < 0.0009, Wilcoxon).")


def analysis_makanae_overlap(cfg, out_dir: Path):
    """Hypergeometric overlap of BY4743 deleterious genes with Makanae et al. 2013."""
    ext = Path(cfg["paths"]["external_dir"])
    path = None
    for name in ("makanae2013_deleterious.txt", "makanae2013.tsv"):
        if (ext / name).exists():
            path = ext / name
            break
    if path is None:
        log.warning("No Makanae 2013 list found. See fetch_external_data.py --list.")
        return

    makanae = {g for g in pd.read_csv(path, sep="\t", header=None)[0].astype(str)
               if g.startswith("Y")}

    wide = read_table(Path(cfg["paths"]["fitness_dir"]) / "fitness_scores_imputed.tsv")
    if "BY4743_FDR" not in wide.columns:
        log.warning("No BY4743 results in the fitness table")
        return

    alpha = cfg["statistics"]["fdr_alpha"]
    ours = set(wide.index[(wide["BY4743_FDR"] < alpha) & (wide["BY4743_log2FC"] < 0)])
    universe = set(wide.index)

    res = overlap_test(ours, makanae, universe)
    df = pd.DataFrame([res])
    write_table(df, out_dir / "makanae2013_overlap.tsv", index=False)

    print("\n=== Overlap with Makanae et al. 2013 ===")
    print(df.to_string(index=False))
    print(f"\nOur BY4743 deleterious genes: {len(ours)}   (paper: 851)")
    print(f"Overlap p-value: {res['pvalue']:.2e}   (paper: 8e-45)")


def analysis_balance_hypothesis(cfg, categories, features, out_dir: Path):
    """The two Balance Hypothesis claims, before and after removing translation genes."""
    gl = Path(cfg["paths"]["gene_lists_dir"])
    ann_dir = Path(cfg["paths"]["results_dir"]) / "07_annotations"

    common = read_list(gl / "commonly_deleterious.tsv")
    background = read_list(gl / "backgrounds" / "tab1_measured_minus_common.tsv")
    if not common or not background:
        log.warning("Need step 06 output for this analysis")
        return

    translation = set()
    if (ann_dir / "translation_genes.txt").exists():
        translation = set(pd.read_csv(ann_dir / "translation_genes.txt",
                                      header=None)[0].astype(str))

    rows = []
    for label, drop in [("all genes", set()), ("translation removed", translation)]:
        if label == "translation removed" and not translation:
            continue
        query = common - drop
        universe = (common | background) - drop

        # Claim 1: more physical interactions and more disorder (Wilcoxon).
        interest = [c for c in ["n_physical_interactions", "n_total_interactions",
                                "median_iupred", "mrna_abundance",
                                "protein_abundance", "protein_length"]
                    if c in features.columns]
        if interest:
            w = wilcoxon_enrichment(query, universe, features[interest])
            for row in w.itertuples():
                rows.append({
                    "subset": label, "test": "Wilcoxon", "term": row.feature,
                    "median_query": row.median_query,
                    "median_background": row.median_background,
                    "direction": row.direction, "pvalue": row.pvalue,
                    "passes_p_threshold": row.passes_p_threshold,
                })

        # Claim 2: more complex members (hypergeometric).
        complex_terms = {k: v for k, v in categories.items()
                         if k == "in_any_complex" or k.startswith("complex:")}
        if complex_terms:
            h = hypergeometric_enrichment(query, universe,
                                          {"in_any_complex": complex_terms.get(
                                              "in_any_complex", set())})
            for row in h.itertuples():
                rows.append({
                    "subset": label, "test": "Hypergeometric", "term": row.category,
                    "median_query": row.observed,
                    "median_background": row.expected,
                    "direction": "higher" if row.observed > row.expected else "lower",
                    "pvalue": row.pvalue,
                    "passes_p_threshold": row.passes_p_threshold,
                })

    if not rows:
        log.warning("No interaction, complex or disorder annotations available")
        return

    df = pd.DataFrame(rows)
    write_table(df, out_dir / "balance_hypothesis.tsv", index=False)
    print("\n=== Balance Hypothesis tests ===")
    print(df.to_string(index=False))
    print("\nPaper: commonly deleterious genes have more protein interactions")
    print("(p = 4.0e-69, Wilcoxon), include more complex members (p = 6.6e-12,")
    print("hypergeometric), and have higher median IUPred disorder (p < 4.7e-12,")
    print("Wilcoxon). All three survive removing translation genes. Higher mRNA")
    print("and protein abundance does NOT survive that removal, which is the")
    print("paper's evidence that expression alone does not predict toxicity.")


ANALYSES = {
    "common_genes": lambda cfg, cat, feat, d: analysis_common_genes(cfg, cat, feat, d),
    "strain_specific": lambda cfg, cat, feat, d: analysis_strain_specific(cfg, cat, feat, d),
    "makanae_overlap": lambda cfg, cat, feat, d: analysis_makanae_overlap(cfg, d),
    "balance_hypothesis": lambda cfg, cat, feat, d: analysis_balance_hypothesis(cfg, cat, feat, d),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", nargs="*", choices=sorted(ANALYSES))
    args = ap.parse_args()

    cfg = load_config(args.config)
    categories, features, _ = load_annotations(cfg)
    out_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "08_enrichments")

    for name in (args.only or list(ANALYSES)):
        log.info("Running %s", name)
        try:
            ANALYSES[name](cfg, categories, features, out_dir)
        except Exception as exc:
            log.error("%s failed: %s", name, exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
