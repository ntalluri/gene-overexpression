"""
enrichment.py
-------------
The statistical engine behind the paper's functional and biophysical
enrichments. Imported by step07; also usable on its own.

What the paper did
------------------
From Materials and methods:

    "Functional and biophysical enrichments were assessed using Wilcoxon
     rank-sum tests for continuous data (e.g., gene length, # of SNPs, and %
     amino acid content) and Hypergeometric tests for categorical terms, taking
     as the background data set the total number of measured genes (except for
     strain-specific gene lists, in which the background data set was a list of
     insignificant genes in that strain with FDR>0.1 and measured in at least
     two of the biological replicates). Because gene lists are heavily
     overlapping, standard FDR calculations over-correct p-values. We therefore
     took a stringent p-value of 5x10-4 as significant, but also cite FDR
     significance in data files."

So: two test types, an explicit background set per query list, a fixed
p-value threshold of 5e-4, and BH FDR reported alongside rather than used as
the cutoff. All four choices are implemented here.

Input format
------------
Categorical annotations: a two-column TSV, gene<TAB>category. A gene may appear
on several rows. Example, a GO slim file.

Continuous annotations: a TSV with a gene column and one column per feature.
Example, a table of IUPred disorder scores, protein lengths, and amino acid
percentages.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, mannwhitneyu

# The paper's fixed significance threshold, chosen because overlapping gene
# lists make ordinary FDR correction too conservative.
P_THRESHOLD = 5e-4


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH step-up FDR, same as R's p.adjust(method='BH')."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    adjusted = p[order] * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def load_categorical(path: str | Path) -> dict[str, set[str]]:
    """Read a gene<TAB>category file into category -> set of genes."""
    df = pd.read_csv(path, sep="\t", header=None, names=["gene", "category"],
                     comment="#")
    mapping = defaultdict(set)
    for gene, category in zip(df["gene"], df["category"]):
        mapping[str(category)].add(str(gene))
    return dict(mapping)


def hypergeometric_enrichment(query: set[str], background: set[str],
                              categories: dict[str, set[str]],
                              min_category_size: int = 3) -> pd.DataFrame:
    """One-sided hypergeometric test for over-representation.

    The universe is `background`. `query` should be a subset of it; genes in the
    query but not the background are dropped, with the count reported.
    """
    query = set(query) & set(background)
    N = len(background)

    rows = []
    for name, members in categories.items():
        members_in_bg = members & background
        K = len(members_in_bg)
        if K < min_category_size:
            continue
        k = len(query & members_in_bg)
        n = len(query)
        if k == 0:
            continue
        # P(X >= k) under the hypergeometric null.
        pval = hypergeom.sf(k - 1, N, K, n)
        expected = n * K / N if N else np.nan
        rows.append({
            "category": name,
            "background_size": N,
            "query_size": n,
            "category_size": K,
            "observed": k,
            "expected": round(expected, 3),
            "fold_enrichment": round(k / expected, 3) if expected else np.nan,
            "pvalue": pval,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["FDR"] = benjamini_hochberg(out["pvalue"].to_numpy())
    out["passes_p_threshold"] = out["pvalue"] < P_THRESHOLD
    out["passes_fdr_05"] = out["FDR"] < 0.05
    return out.sort_values("pvalue").reset_index(drop=True)


def wilcoxon_enrichment(query: set[str], background: set[str],
                        features: pd.DataFrame,
                        alternative: str = "two-sided") -> pd.DataFrame:
    """Wilcoxon rank-sum test of each continuous feature, query vs background.

    `features` is indexed by gene, one column per feature. The comparison set is
    background minus query, so the two groups are disjoint.
    """
    query = set(query) & set(features.index)
    compare = (set(background) & set(features.index)) - query

    rows = []
    for col in features.columns:
        a = pd.to_numeric(features.loc[list(query), col], errors="coerce").dropna()
        b = pd.to_numeric(features.loc[list(compare), col], errors="coerce").dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        stat, pval = mannwhitneyu(a, b, alternative=alternative)
        rows.append({
            "feature": col,
            "n_query": len(a),
            "n_background": len(b),
            "median_query": float(np.median(a)),
            "median_background": float(np.median(b)),
            "direction": "higher" if np.median(a) > np.median(b) else "lower",
            "statistic": float(stat),
            "pvalue": float(pval),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["FDR"] = benjamini_hochberg(out["pvalue"].to_numpy())
    out["passes_p_threshold"] = out["pvalue"] < P_THRESHOLD
    out["passes_fdr_05"] = out["FDR"] < 0.05
    return out.sort_values("pvalue").reset_index(drop=True)


def overlap_test(set_a: set[str], set_b: set[str],
                 universe: set[str]) -> dict:
    """Hypergeometric test of overlap between two gene lists.

    This is the test used for the Makanae et al. comparison, where the paper
    reports p = 8e-45 for the overlap with its 851 deleterious genes in BY4743.
    """
    a = set(set_a) & universe
    b = set(set_b) & universe
    N, K, n = len(universe), len(b), len(a)
    k = len(a & b)
    pval = hypergeom.sf(k - 1, N, K, n) if k else 1.0
    expected = n * K / N if N else np.nan
    return {
        "universe": N,
        "list_a": n,
        "list_b": K,
        "overlap": k,
        "expected_overlap": round(expected, 2),
        "fold_enrichment": round(k / expected, 3) if expected else np.nan,
        "pvalue": pval,
    }
