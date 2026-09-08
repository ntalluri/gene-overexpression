"""
step08_annotations.py
---------------------
Turn whatever is in data/external/ into two tables the enrichment tests can use:

    results/07_annotations/categories.tsv          gene <TAB> category
    results/07_annotations/features_continuous.tsv gene x numeric features

Every source is optional. Missing files produce a warning and a line in the
coverage report, not a failure, so you can start with GO annotations alone and
add BioGRID or IUPred later.

Categorical sources
-------------------
  GO annotations       GAF 2.1 from SGD, split into the three aspects and
                       prefixed GO_BP / GO_MF / GO_CC
  essential genes      inferred from SGD_features.tab feature qualifiers, or
                       supplied directly
  complexes            Pu et al. 2009 / CYC2008, as complex membership plus a
                       single "in_any_complex" term
  ESR                  Gasch et al. 2000 induced and repressed sets
  translation genes    Supplementary file 2 of the paper, so they can be removed
                       from the analysis where the text says they were
  feature type         from SGD_features.tab, e.g. ORF, tRNA gene

Continuous sources
------------------
  protein_length       from orf_trans_all.fasta
  pct_<AA>             percent composition for each of the 20 amino acids;
                       pct_W is the tryptophan measure behind the DBVPG1373
                       result
  n_physical_interactions, n_genetic_interactions, n_total_interactions
                       from BioGRID TAB3, deduplicated by interaction partner
  median_iupred        supplied disorder scores
  mrna_abundance, protein_abundance
                       supplied expression tables
  distance_to_centromere
                       from SGD_features.tab, for the beneficial cluster result

Usage
-----
    python src/step08_annotations.py
    python src/step08_annotations.py --report-only
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from common import (PROJECT_ROOT, ensure_dir, get_logger, load_config,
                    open_maybe_gzip, write_table)

log = get_logger("step08")

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
GO_ASPECTS = {"P": "GO_BP", "F": "GO_MF", "C": "GO_CC"}


def first_existing(directory: Path, *names: str) -> Path | None:
    """Return the first of `names` that exists in `directory`."""
    for name in names:
        p = directory / name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# SGD chromosomal features
# ---------------------------------------------------------------------------

SGD_FEATURE_COLUMNS = [
    "sgdid", "feature_type", "feature_qualifier", "feature_name",
    "standard_name", "alias", "parent_feature", "secondary_sgdid",
    "chromosome", "start", "stop", "strand", "genetic_position",
    "coordinate_version", "sequence_version", "description",
]


def load_sgd_features(path: Path) -> pd.DataFrame:
    """Read SGD_features.tab, which has no header row."""
    df = pd.read_csv(path, sep="\t", header=None, names=SGD_FEATURE_COLUMNS,
                     dtype=str, na_filter=False, on_bad_lines="skip")
    log.info("SGD_features.tab: %d rows", len(df))
    return df


def sgdid_to_systematic(features: pd.DataFrame) -> dict[str, str]:
    """Map primary SGDID to systematic feature name."""
    return {row.sgdid: row.feature_name
            for row in features.itertuples()
            if row.sgdid and row.feature_name}


def centromere_distances(features: pd.DataFrame) -> pd.Series:
    """Distance in base pairs from each ORF to the nearest centromere on its chromosome.

    The paper notes that over half the strongly beneficial cluster sit next to a
    centromere, which the MoBY plasmid would have cloned into the upstream region.
    """
    cens = features[features["feature_type"].str.lower() == "centromere"]
    by_chrom = defaultdict(list)
    for row in cens.itertuples():
        try:
            start, stop = int(row.start), int(row.stop)
        except ValueError:
            continue
        by_chrom[row.chromosome].append((min(start, stop) + max(start, stop)) / 2)

    if not by_chrom:
        log.warning("No centromere features found in SGD_features.tab")
        return pd.Series(dtype=float)

    distances = {}
    orfs = features[features["feature_type"].str.upper().str.contains("ORF", na=False)]
    for row in orfs.itertuples():
        centers = by_chrom.get(row.chromosome)
        if not centers or not row.feature_name:
            continue
        try:
            start, stop = int(row.start), int(row.stop)
        except ValueError:
            continue
        mid = (min(start, stop) + max(start, stop)) / 2
        distances[row.feature_name] = min(abs(mid - c) for c in centers)

    log.info("Centromere distances computed for %d ORFs", len(distances))
    return pd.Series(distances, name="distance_to_centromere")


def essential_from_features(features: pd.DataFrame) -> set[str]:
    """Genes whose SGD feature qualifier or description marks them inviable.

    This is a rough proxy. If you have the curated essential-gene list, put it in
    data/external/essential_genes.txt instead; that takes priority.
    """
    hits = features[
        features["description"].str.contains("essential", case=False, na=False)
        | features["feature_qualifier"].str.contains("essential", case=False, na=False)
    ]
    genes = {n for n in hits["feature_name"] if n}
    log.info("Essential-gene proxy from descriptions: %d genes", len(genes))
    return genes


# ---------------------------------------------------------------------------
# GO
# ---------------------------------------------------------------------------

def load_go_term_names(obo_path: Path | None) -> dict[str, str]:
    """Parse go-basic.obo for GO ID -> term name."""
    if obo_path is None or not obo_path.exists():
        return {}
    names, current = {}, None
    with open(obo_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line == "[Term]":
                current = None
            elif line.startswith("id: GO:"):
                current = line[4:]
            elif line.startswith("name: ") and current:
                names[current] = line[6:]
                current = None
    log.info("GO term names: %d", len(names))
    return names


def load_go_annotations(gaf_path: Path, sgdid_map: dict[str, str],
                        term_names: dict[str, str],
                        exclude_evidence=("ND",)) -> list[tuple[str, str]]:
    """Parse a GAF 2.1 file into (gene, category) pairs.

    Columns used, 0-based: 1 DB Object ID, 4 GO ID, 6 Evidence, 8 Aspect,
    10 DB Object Synonym. The systematic name is taken from the SGDID map when
    possible, and from the synonym field otherwise.
    """
    pairs = []
    skipped = 0
    with open_maybe_gzip(gaf_path, "rt") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 11:
                continue
            db_object_id, go_id, evidence, aspect = f[1], f[4], f[6], f[8]
            if evidence in exclude_evidence:
                continue

            gene = sgdid_map.get(db_object_id)
            if gene is None:
                # Fall back to a systematic-looking entry in the synonym field.
                for syn in f[10].split("|"):
                    if syn.startswith("Y") and len(syn) >= 7:
                        gene = syn
                        break
            if gene is None:
                skipped += 1
                continue

            prefix = GO_ASPECTS.get(aspect, "GO")
            label = term_names.get(go_id, go_id)
            pairs.append((gene, f"{prefix}:{go_id}:{label}"))

    log.info("GO annotations: %d gene-term pairs (%d rows without a gene mapping)",
             len(pairs), skipped)
    return pairs


# ---------------------------------------------------------------------------
# Protein sequences
# ---------------------------------------------------------------------------

def protein_features(fasta_path: Path) -> pd.DataFrame:
    """Protein length and per-amino-acid percentage composition."""
    rows = []
    name, seq = None, []

    def flush():
        if name is None:
            return
        s = "".join(seq).upper().rstrip("*")
        if not s:
            return
        counts = {aa: s.count(aa) for aa in AMINO_ACIDS}
        n = len(s)
        row = {"gene": name, "protein_length": n}
        row.update({f"pct_{aa}": 100.0 * counts[aa] / n for aa in AMINO_ACIDS})
        rows.append(row)

    with open_maybe_gzip(fasta_path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                flush()
                # SGD headers look like ">YAL001C TFC3 SGDID:S000000001, ..."
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line.strip())
    flush()

    df = pd.DataFrame(rows).set_index("gene")
    log.info("Protein features for %d proteins", len(df))
    return df


# ---------------------------------------------------------------------------
# BioGRID
# ---------------------------------------------------------------------------

def biogrid_interaction_counts(path: Path,
                               organism_taxid: str = "559292") -> pd.DataFrame:
    """Count distinct interaction partners per gene from a BioGRID TAB3 file.

    Columns are located by name rather than position, since BioGRID has renamed
    them between format versions. Self-interactions are excluded and partners are
    deduplicated, so a pair seen in twenty publications counts once.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False, na_filter=False)
    cols = {c.lower(): c for c in df.columns}

    def find(*fragments):
        for lower, original in cols.items():
            if all(f in lower for f in fragments):
                return original
        return None

    col_a = find("systematic", "interactor a")
    col_b = find("systematic", "interactor b")
    col_type = find("experimental system type")
    col_org_a = find("organism", "interactor a") or find("organism id a")
    col_org_b = find("organism", "interactor b") or find("organism id b")

    if not (col_a and col_b):
        raise ValueError(
            f"Could not find systematic-name columns in {path.name}. "
            f"Columns present: {list(df.columns)[:12]}"
        )
    log.info("BioGRID columns: A=%s B=%s type=%s", col_a, col_b, col_type)

    # Keep same-organism interactions only.
    if col_org_a and col_org_b:
        before = len(df)
        df = df[(df[col_org_a] == organism_taxid) & (df[col_org_b] == organism_taxid)]
        log.info("Kept %d of %d rows for taxid %s", len(df), before, organism_taxid)

    partners = {"physical": defaultdict(set), "genetic": defaultdict(set)}
    for a, b, kind in zip(df[col_a], df[col_b],
                          df[col_type] if col_type else ["physical"] * len(df)):
        if a in ("-", "") or b in ("-", "") or a == b:
            continue
        bucket = partners.get(str(kind).lower())
        if bucket is None:
            continue
        bucket[a].add(b)
        bucket[b].add(a)

    genes = set(partners["physical"]) | set(partners["genetic"])
    out = pd.DataFrame({
        "n_physical_interactions": {g: len(partners["physical"].get(g, ())) for g in genes},
        "n_genetic_interactions": {g: len(partners["genetic"].get(g, ())) for g in genes},
    })
    out["n_total_interactions"] = (out["n_physical_interactions"]
                                   + out["n_genetic_interactions"])
    out.index.name = "gene"
    log.info("Interaction counts for %d genes (median physical: %.0f)",
             len(out), out["n_physical_interactions"].median())
    return out


# ---------------------------------------------------------------------------
# Simple two-column files
# ---------------------------------------------------------------------------

def load_pairs(path: Path, prefix: str | None = None) -> list[tuple[str, str]]:
    """Read a gene<TAB>category file, optionally prefixing the category."""
    df = pd.read_csv(path, sep="\t", header=None, names=["gene", "category"],
                     dtype=str, comment="#").dropna()
    if prefix:
        df["category"] = prefix + ":" + df["category"].astype(str)
    return list(zip(df["gene"], df["category"]))


def load_gene_column(path: Path) -> set[str]:
    """Read a one-column gene list, or the first column of a wider file."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)
        genes = set()
        for df in sheets.values():
            for col in df.columns:
                vals = df[col].dropna().astype(str)
                genes |= {v for v in vals if v.startswith("Y") and len(v) >= 7}
        return genes
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, comment="#")
    return {v for v in df[0].dropna().astype(str) if v}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--report-only", action="store_true",
                    help="Report which sources are present, build nothing")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ext = Path(cfg["paths"]["external_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "07_annotations")

    categorical: list[tuple[str, str]] = []
    continuous: list[pd.DataFrame] = []
    coverage: list[dict] = []

    def note(source: str, path, detail: str, ok: bool):
        coverage.append({
            "source": source,
            "file": Path(path).name if path else "(missing)",
            "status": "loaded" if ok else "missing",
            "detail": detail,
        })

    # --- SGD features --------------------------------------------------------
    features = None
    p = first_existing(ext, "SGD_features.tab")
    if p:
        features = load_sgd_features(p)
        for row in features.itertuples():
            if row.feature_name and row.feature_type:
                categorical.append((row.feature_name, f"feature_type:{row.feature_type}"))
        dist = centromere_distances(features)
        if len(dist):
            continuous.append(dist.to_frame())
        note("SGD features", p, f"{len(features)} rows, "
             f"{len(dist)} centromere distances", True)
    else:
        note("SGD features", None,
             "no systematic-name mapping, no centromere distances", False)

    # --- essential genes -----------------------------------------------------
    p = first_existing(ext, "essential_genes.txt", "essential_genes.tsv")
    if p:
        genes = load_gene_column(p)
        categorical += [(g, "essential") for g in genes]
        note("essential genes", p, f"{len(genes)} genes (curated list)", True)
    elif features is not None:
        genes = essential_from_features(features)
        categorical += [(g, "essential") for g in genes]
        note("essential genes", "SGD_features.tab",
             f"{len(genes)} genes (description proxy, not curated)", True)
    else:
        note("essential genes", None, "essential-gene enrichment unavailable", False)

    # --- GO ------------------------------------------------------------------
    p = first_existing(ext, "gene_association.sgd.gaf", "gene_association.sgd.gaf.gz",
                       "sgd.gaf", "sgd.gaf.gz")
    if p:
        sgdid_map = sgdid_to_systematic(features) if features is not None else {}
        term_names = load_go_term_names(first_existing(ext, "go-basic.obo"))
        pairs = load_go_annotations(p, sgdid_map, term_names)
        categorical += pairs
        note("GO annotations", p,
             f"{len(pairs)} pairs, "
             f"{'with' if term_names else 'without'} term names", True)
    else:
        note("GO annotations", None, "GO enrichment unavailable", False)

    # --- complexes -----------------------------------------------------------
    p = first_existing(ext, "cyc2008_complexes.tsv", "pu2009_complexes.tsv")
    if p:
        pairs = load_pairs(p, prefix="complex")
        categorical += pairs
        categorical += [(g, "in_any_complex") for g, _ in pairs]
        note("protein complexes", p, f"{len(pairs)} memberships", True)
    else:
        note("protein complexes", None,
             "Balance Hypothesis complex test unavailable", False)

    # --- ESR -----------------------------------------------------------------
    p = first_existing(ext, "esr_genes.tsv")
    if p:
        pairs = load_pairs(p)
        categorical += pairs
        note("ESR gene sets", p, f"{len(pairs)} assignments", True)
    else:
        note("ESR gene sets", None, "ESR enrichment unavailable", False)

    # --- translation genes (Supplementary file 2) ----------------------------
    p = first_existing(ext, "elife-70564-supp2-v2.xlsx", "translation_genes.txt")
    if p:
        genes = load_gene_column(p)
        categorical += [(g, "translation_related") for g in genes]
        pd.Series(sorted(genes), name="gene").to_csv(
            out_dir / "translation_genes.txt", index=False, header=False)
        note("translation genes", p,
             f"{len(genes)} genes; step09 can exclude these", True)
    else:
        note("translation genes", None,
             "cannot repeat the 'excluding translation factors' analyses", False)

    # --- proteins ------------------------------------------------------------
    p = first_existing(ext, "orf_trans_all.fasta", "orf_trans_all.fasta.gz")
    if p:
        prot = protein_features(p)
        continuous.append(prot)
        note("protein sequences", p,
             f"{len(prot)} proteins, length and 20 aa percentages", True)
    else:
        note("protein sequences", None,
             "tryptophan-content and length tests unavailable", False)

    # --- BioGRID -------------------------------------------------------------
    p = first_existing(ext, "BIOGRID-yeast.tab3.txt")
    if p:
        try:
            inter = biogrid_interaction_counts(p)
            continuous.append(inter)
            note("BioGRID", p, f"{len(inter)} genes with interaction counts", True)
        except ValueError as exc:
            note("BioGRID", p, f"parse failed: {exc}", False)
    else:
        note("BioGRID", None, "interaction-count test unavailable", False)

    # --- optional numeric tables --------------------------------------------
    for filename, label in [("iupred_median.tsv", "IUPred disorder"),
                            ("abundance.tsv", "mRNA/protein abundance")]:
        p = first_existing(ext, filename)
        if p:
            df = pd.read_csv(p, sep="\t", index_col=0)
            df.index.name = "gene"
            continuous.append(df.apply(pd.to_numeric, errors="coerce"))
            note(label, p, f"{df.shape[1]} columns, {len(df)} genes", True)
        else:
            note(label, None, "unavailable", False)

    # --- report --------------------------------------------------------------
    cov = pd.DataFrame(coverage)
    write_table(cov, out_dir / "annotation_coverage.tsv", index=False)
    print("\n=== Annotation sources ===")
    print(cov.to_string(index=False))

    if args.report_only:
        return 0

    # --- assemble ------------------------------------------------------------
    if categorical:
        cat_df = pd.DataFrame(categorical, columns=["gene", "category"]).drop_duplicates()
        cat_df = cat_df[cat_df["gene"].astype(str).str.len() > 0]
        cat_df.to_csv(out_dir / "categories.tsv", sep="\t", index=False, header=False)
        n_cat = cat_df["category"].nunique()
        print(f"\nWrote {len(cat_df):,} gene-category pairs across {n_cat:,} categories")
    else:
        log.warning("No categorical annotations were built")

    if continuous:
        feat = pd.concat(continuous, axis=1)
        feat = feat.loc[:, ~feat.columns.duplicated()]
        feat.index.name = "gene"
        write_table(feat, out_dir / "features_continuous.tsv")
        print(f"Wrote {feat.shape[0]:,} genes x {feat.shape[1]} continuous features")
        print("  " + ", ".join(feat.columns[:12]) +
              (" ..." if feat.shape[1] > 12 else ""))
    else:
        log.warning("No continuous features were built")

    missing = cov[cov["status"] == "missing"]
    if len(missing):
        print("\nAnalyses currently unavailable because of missing files:")
        for row in missing.itertuples():
            print(f"  {row.source}: {row.detail}")
        print("\nRun `python src/fetch_external_data.py --list` for how to get them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
