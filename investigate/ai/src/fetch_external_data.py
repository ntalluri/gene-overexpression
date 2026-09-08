"""
fetch_external_data.py
----------------------
Download the annotation files that steps 08 through 10 need, and print manual
instructions for the ones that cannot be fetched programmatically.

Everything lands in data/external/.

Caveat on the URLs
------------------
I could not test these downloads from the environment this was written in, and
SGD in particular reorganizes its archive paths from time to time. Each resource
below is therefore tried against several candidate URLs, and anything that fails
is reported with the page you should visit to find the current link. Nothing is
silently skipped.

Usage
-----
    python src/fetch_external_data.py                # everything automatable
    python src/fetch_external_data.py --only sgd_features biogrid
    python src/fetch_external_data.py --list         # show status, download nothing
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from common import PROJECT_ROOT, ensure_dir, get_logger

log = get_logger("fetch")

# Each resource lists candidate URLs tried in order, the local filename, and a
# human-readable note about what it is used for.
RESOURCES = {
    "sgd_features": {
        "urls": [
            "http://sgd-archive.yeastgenome.org/curation/chromosomal_feature/SGD_features.tab",
            "https://sgd-archive.yeastgenome.org/curation/chromosomal_feature/SGD_features.tab",
        ],
        "filename": "SGD_features.tab",
        "used_for": "systematic-name mapping, feature types, centromere coordinates",
        "fallback_page": "https://www.yeastgenome.org/search?category=download",
    },
    "sgd_go": {
        "urls": [
            "http://sgd-archive.yeastgenome.org/curation/literature/gene_association.sgd.gaf.gz",
            "https://sgd-archive.yeastgenome.org/curation/literature/gene_association.sgd.gaf.gz",
        ],
        "filename": "gene_association.sgd.gaf.gz",
        "used_for": "GO annotations for hypergeometric enrichment",
        "fallback_page": "http://current.geneontology.org/products/pages/downloads.shtml",
    },
    "sgd_proteins": {
        "urls": [
            "http://sgd-archive.yeastgenome.org/sequence/S288C_reference/orf_protein/orf_trans_all.fasta.gz",
            "https://sgd-archive.yeastgenome.org/sequence/S288C_reference/orf_protein/orf_trans_all.fasta.gz",
        ],
        "filename": "orf_trans_all.fasta.gz",
        "used_for": "protein length and amino acid composition, including the "
                    "tryptophan content behind the DBVPG1373 result",
        "fallback_page": "https://www.yeastgenome.org/search?category=download",
    },
    "go_obo": {
        "urls": [
            "http://current.geneontology.org/ontology/go-basic.obo",
        ],
        "filename": "go-basic.obo",
        "used_for": "human-readable names for GO term IDs",
        "fallback_page": "http://geneontology.org/docs/download-ontology/",
    },
    "biogrid": {
        "urls": [
            "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-ORGANISM-LATEST.tab3.zip",
        ],
        "filename": "BIOGRID-ORGANISM-LATEST.tab3.zip",
        "used_for": "physical interaction counts per protein, for the Balance "
                    "Hypothesis tests",
        "fallback_page": "https://downloads.thebiogrid.org/BioGRID",
        "note": "Large (hundreds of MB). Only the S. cerevisiae S288c member of "
                "the archive is extracted.",
    },
}

# Things that have no stable programmatic download.
MANUAL = {
    "elife_supp2_translation_genes": (
        "Translation-related genes removed from several analyses.\n"
        "  curl -L -o data/external/elife-70564-supp2-v2.xlsx \\\n"
        "    https://cdn.elifesciences.org/articles/70564/elife-70564-supp2-v2.xlsx"
    ),
    "elife_supp5_rnaseq": (
        "RNA-seq read counts, which save you re-running HT-Seq if you only want\n"
        "the counts rather than the raw reads.\n"
        "  curl -L -o data/external/elife-70564-supp5-v2.xlsx \\\n"
        "    https://cdn.elifesciences.org/articles/70564/elife-70564-supp5-v2.xlsx"
    ),
    "makanae_2013": (
        "Makanae et al. 2013, Genome Research 23:300-311, doi:10.1101/gr.146662.112.\n"
        "The paper reports 851 of our BY4743 deleterious genes overlapping their\n"
        "dosage-sensitive set at p = 8e-45. Download their supplementary gene list\n"
        "and save it as data/external/makanae2013_deleterious.txt, one systematic\n"
        "ORF name per line."
    ),
    "pu_2009_complexes": (
        "Pu et al. 2009, Nucleic Acids Research 37:825-831 (CYC2008 complex\n"
        "catalogue). Save as data/external/cyc2008_complexes.tsv with two columns,\n"
        "gene<TAB>complex_name."
    ),
    "esr_gene_lists": (
        "Environmental Stress Response gene lists from Gasch et al. 2000,\n"
        "Mol Biol Cell 11:4241-4257. Save as data/external/esr_genes.tsv with two\n"
        "columns, gene<TAB>ESR_induced or ESR_repressed."
    ),
    "iupred_disorder": (
        "IUPred2A median disorder score per protein (Meszaros et al. 2018).\n"
        "Run IUPred2A over orf_trans_all.fasta yourself, or use the web service,\n"
        "and save as data/external/iupred_median.tsv with two columns,\n"
        "gene<TAB>median_iupred."
    ),
    "expression_abundance": (
        "mRNA and protein abundance per gene, for the 'high expression alone is\n"
        "not a predictor of toxicity' analysis. Save as\n"
        "data/external/abundance.tsv with a gene column plus any of\n"
        "mrna_abundance and protein_abundance."
    ),
    "yeastract_regulons": (
        "Transcription factor targets (Monteiro et al. 2020, YEASTRACT+), for the\n"
        "Y7568 regulator analysis. Save as data/external/tf_targets.tsv with two\n"
        "columns, tf_gene<TAB>target_gene."
    ),
    "strain_variants": (
        "Per-strain nonsynonymous SNP and amino acid difference counts against the\n"
        "S288c allele. Derive these from the 1002 Yeast Genomes Project, Strope\n"
        "et al. 2015, or Bergstrom et al. 2014, and save as\n"
        "data/external/strain_snps.tsv with columns gene, strain, n_nonsyn_snps,\n"
        "pct_aa_different."
    ),
}


def download(url: str, dest: Path, timeout: int = 600) -> bool:
    """Try one URL. Returns True on success."""
    try:
        log.info("GET %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "moby-repro/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as out:
            shutil.copyfileobj(resp, out)
        size_mb = dest.stat().st_size / 1e6
        log.info("  saved %s (%.1f MB)", dest.name, size_mb)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        log.warning("  failed: %s", exc)
        if dest.exists():
            dest.unlink()
        return False


def extract_yeast_from_biogrid(zip_path: Path, out_dir: Path) -> Path | None:
    """Pull just the S. cerevisiae member out of the BioGRID organism archive."""
    with zipfile.ZipFile(zip_path) as zf:
        matches = [n for n in zf.namelist()
                   if "Saccharomyces_cerevisiae_S288c" in n and n.endswith(".txt")]
        if not matches:
            log.warning("No S288c member found inside %s", zip_path.name)
            return None
        member = matches[0]
        target = out_dir / "BIOGRID-yeast.tab3.txt"
        with zf.open(member) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    log.info("Extracted %s -> %s", member, target.name)
    return target


def gunzip(path: Path) -> Path:
    """Decompress a .gz file next to itself and return the new path."""
    target = path.with_suffix("")
    with gzip.open(path, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    log.info("Decompressed %s -> %s", path.name, target.name)
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=sorted(RESOURCES),
                    help="Fetch only these resources")
    ap.add_argument("--list", action="store_true",
                    help="Report what is present and what is missing, then exit")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if the file already exists")
    args = ap.parse_args()

    out_dir = ensure_dir(PROJECT_ROOT / "data" / "external")

    if args.list:
        print("\n=== Automatable downloads ===")
        for key, spec in RESOURCES.items():
            path = out_dir / spec["filename"]
            status = "present" if path.exists() else "MISSING"
            print(f"  [{status:>7}] {key:<16} {spec['filename']}")
            print(f"              {spec['used_for']}")
        print("\n=== Manual downloads ===")
        for key, note in MANUAL.items():
            print(f"  {key}:")
            for line in note.splitlines():
                print(f"    {line}")
            print()
        return 0

    names = args.only or list(RESOURCES)
    failures = []

    for key in names:
        spec = RESOURCES[key]
        dest = out_dir / spec["filename"]
        print(f"\n--- {key}: {spec['used_for']} ---")
        if "note" in spec:
            print(f"    {spec['note']}")

        if dest.exists() and not args.force:
            log.info("Already present, skipping (use --force to re-download)")
        else:
            if not any(download(url, dest) for url in spec["urls"]):
                failures.append((key, spec))
                continue

        # Post-processing.
        if key == "biogrid" and dest.exists():
            extract_yeast_from_biogrid(dest, out_dir)
        elif dest.suffix == ".gz" and not dest.with_suffix("").exists():
            gunzip(dest)

    print("\n" + "=" * 70)
    if failures:
        print("Could not download the following. Fetch them by hand:\n")
        for key, spec in failures:
            print(f"  {key}")
            print(f"    needed for : {spec['used_for']}")
            print(f"    save as    : data/external/{spec['filename']}")
            print(f"    look here  : {spec['fallback_page']}\n")
    else:
        print("All requested downloads succeeded.")

    print("These still need manual work. Run --list to see the instructions:")
    print("  " + ", ".join(MANUAL))
    print("\nNothing downstream requires all of them. step08 builds whatever")
    print("annotation table it can from the files that are present and reports")
    print("which analyses are unavailable as a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
