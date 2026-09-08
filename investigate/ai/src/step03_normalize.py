"""
step03_normalize.py
-------------------
Unnormalized read counts -> normalized (library-size scaled) read counts.

What the paper did
------------------
From Materials and methods, "Moby normalization and analyses":

    "We experimented with several normalization strategies, including TMM in
     the edgeR package and simple library size normalization, in which barcode
     reads were divided by the total barcode read count in the sample,
     multiplied by 1 million to rescale for edgeR analysis. The latter provided
     the most robust procedure with the fewest assumptions."

So the normalization is counts-per-million computed per sample, with no TMM and
no other scaling factor. That is what this step does. The result matches Tab 2
of Supplementary file 4 ("Library-size normalized and scaled read counts").

TMM is implemented here as well, behind --method tmm, only so you can reproduce
the comparison the authors say they made. Do not use it for the main pipeline.

Usage
-----
    python src/step03_normalize.py
    python src/step03_normalize.py --input counts_exact.tsv --method tmm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import ensure_dir, get_logger, load_config, read_table, write_table

log = get_logger("step03")

SCALE = 1_000_000     # "multiplied by 1 million to rescale for edgeR analysis"


def library_size_normalize(counts: pd.DataFrame, scale: float = SCALE) -> pd.DataFrame:
    """Divide each column by its own total, then multiply by 1e6.

    This is plain counts-per-million. Samples with a zero total are left as
    zeros rather than producing NaN.
    """
    totals = counts.sum(axis=0)
    zero_samples = totals[totals == 0].index.tolist()
    if zero_samples:
        log.warning("Samples with zero total counts, left as zeros: %s", zero_samples)
    safe_totals = totals.replace(0, np.nan)
    normed = counts.divide(safe_totals, axis=1) * scale
    return normed.fillna(0.0)


def tmm_factors(counts: pd.DataFrame,
                log_ratio_trim: float = 0.3,
                sum_trim: float = 0.05) -> pd.Series:
    """Compute edgeR-style TMM normalization factors.

    Provided only for the comparison the authors describe. The reference sample
    is the one whose upper-quartile-scaled count profile is closest to the mean,
    as in edgeR's calcNormFactors.
    """
    lib_sizes = counts.sum(axis=0)
    # Pick the reference column the way edgeR does.
    f75 = counts.divide(lib_sizes, axis=1).quantile(0.75, axis=0)
    ref_col = (f75 - f75.mean()).abs().idxmin()

    ref = counts[ref_col].astype(float)
    ref_size = lib_sizes[ref_col]

    factors = {}
    for col in counts.columns:
        obs = counts[col].astype(float)
        obs_size = lib_sizes[col]

        keep = (obs > 0) & (ref > 0)
        if keep.sum() == 0:
            factors[col] = 1.0
            continue

        o = obs[keep].values
        r = ref[keep].values

        # M = log ratio, A = mean expression, v = approximate asymptotic variance
        logR = np.log2((o / obs_size) / (r / ref_size))
        absE = (np.log2(o / obs_size) + np.log2(r / ref_size)) / 2
        v = (obs_size - o) / obs_size / o + (ref_size - r) / ref_size / r

        finite = np.isfinite(logR) & np.isfinite(absE) & (absE > -1e10)
        logR, absE, v = logR[finite], absE[finite], v[finite]
        if len(logR) == 0:
            factors[col] = 1.0
            continue

        n = len(logR)
        lo_r, hi_r = np.floor(n * log_ratio_trim) + 1, n + 1 - np.floor(n * log_ratio_trim)
        lo_s, hi_s = np.floor(n * sum_trim) + 1, n + 1 - np.floor(n * sum_trim)

        rank_r = pd.Series(logR).rank().values
        rank_s = pd.Series(absE).rank().values
        keep2 = (rank_r >= lo_r) & (rank_r <= hi_r) & (rank_s >= lo_s) & (rank_s <= hi_s)

        if keep2.sum() == 0:
            factors[col] = 1.0
            continue
        w = 1.0 / v[keep2]
        factors[col] = 2 ** (np.sum(w * logR[keep2]) / np.sum(w))

    f = pd.Series(factors)
    # edgeR centres the factors so their geometric mean is 1.
    return f / np.exp(np.mean(np.log(f)))


def tmm_normalize(counts: pd.DataFrame, scale: float = SCALE) -> pd.DataFrame:
    """CPM using effective library sizes (lib size x TMM factor)."""
    factors = tmm_factors(counts)
    effective = counts.sum(axis=0) * factors
    return counts.divide(effective, axis=1) * scale


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--input", default="counts_1mismatch.tsv",
                    help="Which count table in the counts directory to use")
    ap.add_argument("--method", choices=["library_size", "tmm"],
                    default="library_size",
                    help="library_size reproduces the paper; tmm is for comparison only")
    args = ap.parse_args()

    cfg = load_config(args.config)
    counts_dir = Path(cfg["paths"]["counts_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["normalized_dir"]))

    counts = read_table(counts_dir / args.input)
    log.info("Loaded counts: %d genes x %d samples", *counts.shape)

    if args.method == "library_size":
        normed = library_size_normalize(counts)
        out_name = "normalized_counts.tsv"
    else:
        normed = tmm_normalize(counts)
        out_name = "normalized_counts_tmm.tsv"

    write_table(normed.round(6), out_dir / out_name)
    log.info("Wrote %s", out_dir / out_name)

    print("\n=== Normalization summary ===")
    print(f"Method: {args.method}")
    print(f"Column sums after scaling (should all be ~{SCALE:,} for library_size):")
    print(normed.sum(axis=0).round(1).to_string())
    print("\n5th percentile of nonzero normalized counts per sample:")
    print("(the imputation threshold of 20 in step 04 is described in the paper")
    print(" as sitting above this percentile, so it is worth eyeballing)")
    pcts = normed.apply(lambda c: np.percentile(c[c > 0], 5) if (c > 0).any() else np.nan)
    print(pcts.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
