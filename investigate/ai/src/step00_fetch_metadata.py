"""
step00_fetch_metadata.py
------------------------
Pull the sample metadata for GEO series GSE171586 (the Bar-seq lanes from
Robinson et al. 2021, eLife 70564) and write a draft sample sheet.

Why this step exists
--------------------
The rest of the pipeline is driven by data/reference/samples.tsv, which maps
each FASTQ (and, if the FASTQs are still multiplexed, each inline index) to a
strain / generation / replicate. GEO holds that mapping in its sample titles
and characteristics fields. This script downloads the series SOFT record,
parses it, and writes a first draft of samples.tsv that you then check by hand.

Do not trust the draft blindly. The parser guesses strain, generation, and
replicate from the GEO sample title using regular expressions. GEO titles are
free text, so verify every row against the GEO page before running step 02.

Usage
-----
    python src/step00_fetch_metadata.py
    python src/step00_fetch_metadata.py --gse GSE171586 --out data/reference/samples_draft.tsv

Requires network access to ncbi.nlm.nih.gov.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from common import PROJECT_ROOT, get_logger, ensure_dir

log = get_logger("step00")

# GEO serves a plain-text SOFT record here. This avoids scraping the HTML page.
SOFT_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    "?acc={gse}&targ=gsm&form=text&view=brief"
)

# SRA Run Selector metadata for the linked BioProject, used to map GSM -> SRR.
SRA_RUNINFO_URL = (
    "https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc={acc}"
)


def fetch_text(url: str, timeout: int = 120) -> str:
    """Download a URL and return its body as text."""
    log.info("GET %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "moby-repro/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_soft(text: str) -> pd.DataFrame:
    """Parse a GEO SOFT 'brief' record into one row per GSM.

    The SOFT format is a sequence of `^SAMPLE = GSMxxxxxxx` blocks, each
    followed by `!Sample_<field> = <value>` lines. Fields can repeat, so we
    collect repeated values into a semicolon-joined string.
    """
    records = []
    current = None

    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("^SAMPLE"):
            if current:
                records.append(current)
            gsm = line.split("=", 1)[1].strip()
            current = {"gsm": gsm}
        elif line.startswith("!Sample_") and current is not None:
            key, _, value = line[len("!Sample_"):].partition("=")
            key = key.strip()
            value = value.strip()
            if key in current:
                current[key] = current[key] + "; " + value
            else:
                current[key] = value

    if current:
        records.append(current)

    if not records:
        raise RuntimeError(
            "No samples parsed from the SOFT record. GEO may have returned a "
            "CAPTCHA page instead of data. Open the GEO URL in a browser, or "
            "retry from a different network."
        )
    return pd.DataFrame(records)


# --- Heuristics for reading strain / generation / replicate out of free text ---

# The 15 isolates used in the paper (Supplementary file 1). Listing them
# explicitly is more reliable than trying to pattern-match arbitrary names.
KNOWN_STRAINS = [
    "BY4743", "BC187", "DBVPG1373", "NCYC3290", "Y12", "Y2209", "Y389",
    "Y7568", "YJM1273", "YJM1389", "YJM1592", "YJM978", "YPS128", "YPS163",
    "YPS606",
]


def guess_strain(text: str) -> str:
    """Return the first known strain name found in a free-text field."""
    upper = text.upper()
    # Sort longest-first so that YJM1389 is not shadowed by a shorter prefix.
    for strain in sorted(KNOWN_STRAINS, key=len, reverse=True):
        if strain.upper() in upper:
            return strain
    return ""


def guess_generation(text: str) -> str:
    """Return '0' or '10' if the text names a timepoint, else ''.

    Handles the common spellings: T0/T10, gen0/gen10, 'generation 10',
    'g0'/'g10', and the 0/10-generation phrasing used in the paper.
    """
    t = text.lower()
    patterns_10 = [r"\bt\s*10\b", r"\bgen\w*\s*10\b", r"\bg10\b",
                   r"10\s*generation", r"\b10\s*gen\b"]
    patterns_0 = [r"\bt\s*0\b", r"\bgen\w*\s*0\b", r"\bg0\b",
                  r"0\s*generation", r"\bstarting\s*pool\b", r"\binitial\b"]
    for pat in patterns_10:
        if re.search(pat, t):
            return "10"
    for pat in patterns_0:
        if re.search(pat, t):
            return "0"
    return ""


def guess_replicate(text: str) -> str:
    """Return a replicate label such as A, B, C, 1, 2, 3 if one is present."""
    m = re.search(r"\b(?:rep|replicate|biorep)[\s_\-]*([A-Ca-c1-3])\b", text)
    if m:
        return m.group(1).upper()
    # Fall back to a trailing single letter or digit, e.g. "BY4743 T0 A".
    m = re.search(r"[\s_\-]([A-Ca-c1-3])$", text.strip())
    if m:
        return m.group(1).upper()
    return ""


def build_draft_sheet(soft_df: pd.DataFrame) -> pd.DataFrame:
    """Turn parsed GEO records into a draft samples.tsv."""
    rows = []
    for _, rec in soft_df.iterrows():
        # Concatenate every text field so the heuristics see all available hints.
        blob = " ; ".join(
            str(rec.get(k, "")) for k in
            ["title", "source_name_ch1", "characteristics_ch1",
             "description", "supplementary_file_1"]
        )
        rows.append({
            "sample_id": rec.get("title", rec["gsm"]).strip().replace(" ", "_"),
            "gsm": rec["gsm"],
            "strain": guess_strain(blob),
            "generation": guess_generation(blob),
            "replicate": guess_replicate(str(rec.get("title", ""))),
            "index_seq": "",          # fill in only if FASTQs are multiplexed
            "fastq": "",              # fill in after downloading the reads
            "geo_title": rec.get("title", ""),
            "geo_characteristics": rec.get("characteristics_ch1", ""),
            "geo_supplementary": rec.get("supplementary_file_1", ""),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gse", default="GSE171586",
                    help="GEO series accession (default: GSE171586)")
    ap.add_argument("--out", default="data/reference/samples_draft.tsv",
                    help="Where to write the draft sample sheet")
    ap.add_argument("--soft-out", default="data/reference/GSE_soft.txt",
                    help="Where to save the raw SOFT record for inspection")
    args = ap.parse_args()

    out_path = PROJECT_ROOT / args.out
    soft_path = PROJECT_ROOT / args.soft_out
    ensure_dir(out_path.parent)

    text = fetch_text(SOFT_URL.format(gse=args.gse))
    soft_path.write_text(text)
    log.info("Saved raw SOFT record to %s", soft_path)

    soft_df = parse_soft(text)
    log.info("Parsed %d GEO samples", len(soft_df))

    draft = build_draft_sheet(soft_df)
    draft.to_csv(out_path, sep="\t", index=False)
    log.info("Wrote draft sample sheet to %s", out_path)

    # Report anything the heuristics could not resolve, so you know where to look.
    for col in ["strain", "generation", "replicate"]:
        blank = draft[draft[col] == ""]
        if len(blank):
            log.warning("%d rows have no %s guessed; fill these in by hand",
                        len(blank), col)

    print("\nDraft sample sheet preview:")
    print(draft[["sample_id", "gsm", "strain", "generation",
                 "replicate", "geo_title"]].to_string(index=False))
    print(
        "\nNext steps:\n"
        "  1. Check every row against the GEO page for {gse}.\n"
        "  2. Download the FASTQs (see README for the fasterq-dump commands)\n"
        "     and fill in the 'fastq' column.\n"
        "  3. If the FASTQs are still multiplexed lanes, run step01 to work out\n"
        "     the inline index sequences and fill in 'index_seq'.\n"
        "  4. Save the checked file as data/reference/samples.tsv.\n"
        .format(gse=args.gse)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
