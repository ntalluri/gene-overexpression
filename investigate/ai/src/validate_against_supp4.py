"""
validate_against_supp4.py
-------------------------
Compare each stage of the pipeline against the published values in
Supplementary file 4 of Robinson et al. 2021 (eLife 70564).

The supplementary workbook has seven tabs, described by eLife as:

    Tab 1: Unnormalized read counts for each strain.
    Tab 2: Library-size normalized and scaled read counts for each strain.
    Tab 3: Average (log2) change in fitness and BH-corrected FDR from edgeR,
           without data imputation.
    Tab 4: Same, using data in which some ratios had been imputed.
    Tab 5: List of commonly deleterious genes.
    Tab 6: Strain-specific deleterious genes for each strain.
    Tab 7: Strain-specific beneficial genes for each strain.

Download it first:

    curl -L -o data/external/elife-70564-supp4-v2.xlsx \\
      https://cdn.elifesciences.org/articles/70564/elife-70564-supp4-v2.xlsx

The exact column headers inside those tabs are not documented, so this script
inspects them and matches columns to strains by name rather than by position.
It prints what it found before comparing, so you can see whether the matching
was sensible. If a tab is laid out differently than assumed, the script reports
that rather than producing a misleading correlation.

Usage
-----
    python src/validate_against_supp4.py
    python src/validate_against_supp4.py --stage fitness
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import PROJECT_ROOT, get_logger, load_config, read_table

log = get_logger("validate")

STRAINS = [
    "BY4743", "BC187", "DBVPG1373", "NCYC3290", "Y12", "Y2209", "Y389",
    "Y7568", "YJM1273", "YJM1389", "YJM1592", "YJM978", "YPS128", "YPS163",
    "YPS606",
]


def find_strain(text: str) -> str | None:
    """Return the strain named in a column header, longest match first."""
    upper = str(text).upper()
    for strain in sorted(STRAINS, key=len, reverse=True):
        if strain.upper() in upper:
            return strain
    return None


def describe_workbook(path: Path) -> dict[str, pd.DataFrame]:
    """Load every sheet and print its shape and headers."""
    sheets = pd.read_excel(path, sheet_name=None)
    print(f"\n=== {path.name} ===")
    for name, df in sheets.items():
        print(f"\n  Sheet '{name}': {df.shape[0]} rows x {df.shape[1]} cols")
        cols = list(df.columns)
        preview = cols[:10]
        print(f"    columns: {preview}{' ...' if len(cols) > 10 else ''}")
    return sheets


def pick_gene_column(df: pd.DataFrame) -> str:
    """Guess which column holds the systematic ORF name.

    Chooses the column with the highest fraction of values matching the yeast
    systematic-name pattern, e.g. YAL001C or YHR188C_1.
    """
    pattern = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?(_\d+)?$", re.IGNORECASE)
    best, best_frac = None, 0.0
    for col in df.columns:
        values = df[col].astype(str).head(500)
        frac = values.str.match(pattern).mean()
        if frac > best_frac:
            best, best_frac = col, frac
    if best_frac < 0.5:
        raise ValueError("Could not identify a gene-name column in this sheet")
    return best


def compare_numeric(ours: pd.Series, theirs: pd.Series, label: str,
                    log_transform: bool = False) -> dict:
    """Report overlap, correlation and disagreement between two gene-indexed series."""
    shared = ours.index.intersection(theirs.index)
    a = pd.to_numeric(ours.loc[shared], errors="coerce")
    b = pd.to_numeric(theirs.loc[shared], errors="coerce")
    ok = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]

    if log_transform:
        a, b = np.log1p(a), np.log1p(b)

    if len(a) < 10:
        return {"label": label, "n": len(a), "pearson": np.nan,
                "spearman": np.nan, "median_abs_diff": np.nan}

    return {
        "label": label,
        "n": int(len(a)),
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "spearman": float(pd.Series(a).corr(pd.Series(b), method="spearman")),
        "median_abs_diff": float(np.median(np.abs(a.values - b.values))),
        "max_abs_diff": float(np.max(np.abs(a.values - b.values))),
    }


def validate_counts(sheets, cfg, tab_index: int, our_path: Path, label: str):
    """Compare a count-like tab (Tab 1 or Tab 2) with one of our matrices."""
    names = list(sheets.keys())
    if tab_index >= len(names):
        log.warning("Workbook has no tab %d", tab_index + 1)
        return []
    df = sheets[names[tab_index]]

    try:
        gene_col = pick_gene_column(df)
    except ValueError as exc:
        log.warning("Tab %d (%s): %s", tab_index + 1, names[tab_index], exc)
        return []

    df = df.set_index(gene_col)
    ours = read_table(our_path)

    results = []
    for col in df.columns:
        strain = find_strain(col)
        if strain is None:
            continue
        # Match published sample columns to ours by shared strain plus any
        # generation/replicate hints in the header text.
        candidates = [c for c in ours.columns if strain.upper() in c.upper()]
        if not candidates:
            continue
        hint = str(col).upper()
        gen = "10" if re.search(r"(T\s*10|GEN\w*\s*10|G10)", hint) else (
              "0" if re.search(r"(T\s*0|GEN\w*\s*0|G0)", hint) else None)
        if gen is not None:
            narrowed = [c for c in candidates
                        if re.search(rf"(T\s*{gen}\b|GEN\w*\s*{gen}\b|G{gen}\b)",
                                     c.upper())]
            if narrowed:
                candidates = narrowed
        results.append(compare_numeric(ours[candidates[0]], df[col],
                                       f"{label}: {col} vs {candidates[0]}",
                                       log_transform=True))
    return results


def validate_fitness(sheets, cfg, tab_index: int, our_path: Path, label: str):
    """Compare a fitness tab (Tab 3 or Tab 4) with our wide fitness table."""
    names = list(sheets.keys())
    if tab_index >= len(names):
        log.warning("Workbook has no tab %d", tab_index + 1)
        return []
    df = sheets[names[tab_index]]

    try:
        gene_col = pick_gene_column(df)
    except ValueError as exc:
        log.warning("Tab %d (%s): %s", tab_index + 1, names[tab_index], exc)
        return []

    df = df.set_index(gene_col)
    ours = read_table(our_path)

    results = []
    for col in df.columns:
        strain = find_strain(col)
        if strain is None:
            continue
        header = str(col).upper()
        if "FDR" in header or "PADJ" in header or "Q" == header.strip():
            our_col = f"{strain}_FDR"
        else:
            our_col = f"{strain}_log2FC"
        if our_col not in ours.columns:
            continue
        results.append(compare_numeric(ours[our_col], df[col],
                                       f"{label}: {col} vs {our_col}"))

    # Overlap of the significance calls, which is what downstream lists depend on.
    fdr_cols = {find_strain(c): c for c in df.columns
                if find_strain(c) and "FDR" in str(c).upper()}
    for strain, col in fdr_cols.items():
        our_col = f"{strain}_FDR"
        if our_col not in ours.columns:
            continue
        shared = ours.index.intersection(df.index)
        theirs_sig = set(shared[(pd.to_numeric(df.loc[shared, col],
                                               errors="coerce") < 0.05).fillna(False)])
        ours_sig = set(shared[(ours.loc[shared, our_col] < 0.05).fillna(False)])
        inter = len(theirs_sig & ours_sig)
        union = len(theirs_sig | ours_sig)
        results.append({
            "label": f"{label}: {strain} FDR<0.05 call agreement",
            "n": union,
            "jaccard": round(inter / union, 4) if union else np.nan,
            "published_significant": len(theirs_sig),
            "our_significant": len(ours_sig),
        })
    return results


def validate_gene_lists(sheets, cfg):
    """Compare Tabs 5-7 with the lists produced by step 06."""
    gl_dir = Path(cfg["paths"]["gene_lists_dir"])
    names = list(sheets.keys())
    results = []

    # Tab 5: commonly deleterious, a single list.
    if len(names) > 4:
        df = sheets[names[4]]
        published = set()
        for col in df.columns:
            published |= set(df[col].dropna().astype(str))
        published = {g for g in published
                     if re.match(r"^Y[A-P][LR]\d{3}[WC]", g, re.IGNORECASE)}

        our_path = gl_dir / "commonly_deleterious.tsv"
        if our_path.exists() and published:
            ours = set(pd.read_csv(our_path, sep="\t")["gene"])
            inter = len(ours & published)
            results.append({
                "label": "Tab 5 commonly deleterious",
                "published": len(published),
                "ours": len(ours),
                "shared": inter,
                "jaccard": round(inter / len(ours | published), 4),
            })

    # Tabs 6 and 7: one column per strain.
    for tab_idx, kind, suffix in [(5, "deleterious", "deleterious"),
                                  (6, "beneficial", "beneficial")]:
        if tab_idx >= len(names):
            continue
        df = sheets[names[tab_idx]]
        for col in df.columns:
            strain = find_strain(col)
            if strain is None:
                continue
            published = {g for g in df[col].dropna().astype(str)
                         if re.match(r"^Y[A-P][LR]\d{3}[WC]", g, re.IGNORECASE)}
            our_path = gl_dir / "strain_specific" / f"{strain}_{suffix}.tsv"
            if not our_path.exists() or not published:
                continue
            ours = set(pd.read_csv(our_path, sep="\t")["gene"])
            inter = len(ours & published)
            results.append({
                "label": f"Tab {tab_idx + 1} {strain} strain-specific {kind}",
                "published": len(published),
                "ours": len(ours),
                "shared": inter,
                "jaccard": round(inter / len(ours | published), 4) if (ours | published) else np.nan,
            })
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--supp4",
                    default="data/external/elife-70564-supp4-v2.xlsx")
    ap.add_argument("--stage",
                    choices=["all", "describe", "counts", "normalized",
                             "fitness", "gene_lists"],
                    default="all")
    args = ap.parse_args()

    cfg = load_config(args.config)
    supp_path = Path(args.supp4)
    if not supp_path.is_absolute():
        supp_path = PROJECT_ROOT / supp_path

    if not supp_path.exists():
        log.error("Supplementary file 4 not found at %s", supp_path)
        log.error("Download it with:")
        log.error("  curl -L -o %s \\", supp_path)
        log.error("    https://cdn.elifesciences.org/articles/70564/"
                  "elife-70564-supp4-v2.xlsx")
        return 1

    sheets = describe_workbook(supp_path)
    if args.stage == "describe":
        return 0

    all_results = []

    if args.stage in ("all", "counts"):
        all_results += validate_counts(
            sheets, cfg, 0,
            Path(cfg["paths"]["counts_dir"]) / "counts_1mismatch.tsv",
            "Tab 1 raw counts")

    if args.stage in ("all", "normalized"):
        all_results += validate_counts(
            sheets, cfg, 1,
            Path(cfg["paths"]["normalized_dir"]) / "normalized_counts.tsv",
            "Tab 2 normalized")

    if args.stage in ("all", "fitness"):
        all_results += validate_fitness(
            sheets, cfg, 2,
            Path(cfg["paths"]["fitness_dir"]) / "fitness_scores_no_imputation.tsv",
            "Tab 3 no imputation")
        all_results += validate_fitness(
            sheets, cfg, 3,
            Path(cfg["paths"]["fitness_dir"]) / "fitness_scores_imputed.tsv",
            "Tab 4 imputed")

    if args.stage in ("all", "gene_lists"):
        all_results += validate_gene_lists(sheets, cfg)

    if not all_results:
        print("\nNo comparisons were made. Check that the earlier steps have run "
              "and that the workbook layout matches what this script expects.")
        return 1

    out = pd.DataFrame(all_results)
    print("\n=== Validation results ===")
    print(out.to_string(index=False))

    report_path = Path(cfg["paths"]["results_dir"]) / "validation_report.tsv"
    out.to_csv(report_path, sep="\t", index=False)
    print(f"\nSaved to {report_path}")

    print("\nHow to read this:")
    print("  Counts and normalized counts should correlate above ~0.99 if the")
    print("  barcode offset and demultiplexing are right.")
    print("  log2FC should correlate above ~0.95. FDR values will agree less")
    print("  closely with the pure-Python backend because of dispersion")
    print("  moderation; look at the call-agreement Jaccard instead.")
    print("  Gene-list Jaccard below ~0.9 usually means a definitional")
    print("  difference, not a numerical one. Re-run step 06 with the other")
    print("  --specificity-rule and compare.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
