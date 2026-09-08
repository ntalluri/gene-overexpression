"""
step02_count_barcodes.py
------------------------
FASTQ lanes -> unnormalized read counts per gene per sample.

This reimplements the logic of the Gasch lab's BarSeq.py
(github.com/wiscplace/BarSeq, Mike Place, 2016), which is the script the paper
used. The matching rules below are copied from that script deliberately, so
that the counts are comparable. The speed-ups are described under
"Optimizations" and none of them change which gene a read is assigned to.

Matching rules, taken from BarSeq.py
------------------------------------
1. Demultiplex. A read belongs to a sample if the read *starts with* that
   sample's inline index sequence, exact match. Reads matching no index are
   discarded.

2. Exact gene match. Take the read window starting at a fixed offset
   (28 in the original, i.e. the uptag begins at base 29). Try window lengths
   10, 11, ... 22 in ascending order and return the first window that is a
   known MoBY uptag.

3. One-mismatch fallback. If no exact match, slide the start offset over a
   small window (offset-3 to offset+2, i.e. 25-30 in the original) and again
   try lengths 10-22, looking each window up in a dictionary containing every
   single-base substitution of every uptag. Bases substituted in are
   G, C, A, T and N, so reads with one N in the tag still match.

4. Two output tables. "exact" holds only rule-2 hits. "1mismatch" holds
   rule-2 plus rule-3 hits summed. The paper's read counts correspond to the
   1mismatch table, which is what downstream steps use by default.

A caveat inherited from the original
------------------------------------
The one-mismatch dictionary is built with first-writer-wins: if two different
genes' uptags produce the same one-mismatch sequence, that sequence is assigned
to whichever gene appears first in the decode file. This script reproduces that
behaviour and additionally writes a collision report so you can see how many
sequences are affected. Use --strict-fuzzy to drop ambiguous sequences instead.

Optimizations
-------------
The original does up to 13 dictionary lookups per read per offset. Here we first
index every tag by its first 10 bases. Since every uptag is at least 10 bases
long, a read window whose first 10 bases are not a known prefix cannot match any
tag at any length, so we can reject it with one lookup. When the prefix does
match we try only the tag lengths that actually exist for that prefix, still in
ascending order. The assignment is identical; the work is roughly 10x less.

Usage
-----
    python src/step02_count_barcodes.py
    python src/step02_count_barcodes.py --offset 28 --threads 4
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from common import (PROJECT_ROOT, ensure_dir, get_logger, load_config,
                    load_sample_sheet, read_fastq, resolve_fastq, write_table)

log = get_logger("step02")

MIN_TAG_LEN = 10
MAX_TAG_LEN = 22
PREFIX_LEN = 10          # every uptag is at least this long, so a safe index key
SUBSTITUTION_BASES = ["G", "C", "A", "T", "N"]   # same set as tag_dictionary.py


# ---------------------------------------------------------------------------
# Reference dictionaries
# ---------------------------------------------------------------------------

def load_decode(decode_path: Path):
    """Read GENE_UPTAG_Decode.txt.

    Returns (exact_tag_to_gene, gene_order) where gene_order preserves the file
    order, because the one-mismatch collision rule depends on it.
    """
    exact = {}
    gene_order = []
    rows = []
    with open(decode_path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line.startswith("Y"):     # skips the header line
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            gene, tag = parts[0], parts[1]
            rows.append((gene, tag))
            exact[tag] = gene
            if gene not in gene_order:
                gene_order.append(gene)
    log.info("Decode file: %d tag entries, %d unique genes", len(rows), len(gene_order))
    return exact, gene_order, rows


def build_fuzzy_dict(rows, strict: bool = False):
    """Build the one-mismatch lookup table.

    For each uptag, every position is replaced by each of G, C, A, T, N. The
    original sequence is regenerated in the process, so exact tags are members
    of this dictionary too.

    Returns (fuzzy_dict, collisions) where collisions maps an ambiguous sequence
    to the set of genes that could produce it.
    """
    fuzzy = {}
    collisions = defaultdict(set)

    for gene, tag in rows:
        chars = list(tag)
        for i in range(len(chars)):
            original = chars[i]
            for base in SUBSTITUTION_BASES:
                chars[i] = base
                variant = "".join(chars)
                if variant in fuzzy:
                    if fuzzy[variant] != gene:
                        # Record the clash. First writer keeps the key, which is
                        # what the original tag_dictionary.py does.
                        collisions[variant].add(fuzzy[variant])
                        collisions[variant].add(gene)
                else:
                    fuzzy[variant] = gene
            chars[i] = original

    if strict:
        for variant in collisions:
            fuzzy.pop(variant, None)
        log.info("strict-fuzzy: dropped %d ambiguous sequences", len(collisions))

    log.info("One-mismatch dictionary: %d sequences, %d ambiguous",
             len(fuzzy), len(collisions))
    return fuzzy, collisions


def build_prefix_index(tag_dict: dict):
    """Map a 10-base prefix to the ascending list of tag lengths using it.

    This is the lookup that lets us reject a non-matching read window in one
    dictionary probe instead of thirteen.
    """
    index = defaultdict(set)
    for seq in tag_dict:
        index[seq[:PREFIX_LEN]].add(len(seq))
    return {prefix: sorted(lengths) for prefix, lengths in index.items()}


# ---------------------------------------------------------------------------
# Read matching
# ---------------------------------------------------------------------------

def match_at_offset(read: str, offset: int, tag_dict: dict, prefix_index: dict):
    """Try to match a gene uptag starting at `offset`. Returns a gene or None.

    Lengths are tried in ascending order, matching the original loop.
    """
    prefix = read[offset:offset + PREFIX_LEN]
    lengths = prefix_index.get(prefix)
    if not lengths:
        return None
    for length in lengths:
        gene = tag_dict.get(read[offset:offset + length])
        if gene is not None:
            return gene
    return None


def count_one_fastq(fastq_path: Path,
                    index_to_sample: dict,
                    exact_dict: dict, exact_prefix: dict,
                    fuzzy_dict: dict, fuzzy_prefix: dict,
                    offset: int,
                    fuzzy_window: tuple[int, int],
                    max_reads: int | None = None):
    """Count barcodes in one FASTQ.

    `index_to_sample` maps an inline index sequence to a sample_id. Pass the
    special key "" mapped to a single sample_id when the FASTQ is already
    demultiplexed, in which case every read is attributed to that sample.

    Returns (exact_counts, fuzzy_only_counts, qc) where the count objects are
    nested dicts sample_id -> gene -> count.
    """
    exact_counts = defaultdict(Counter)
    fuzzy_counts = defaultdict(Counter)

    qc = Counter()
    demultiplexed = list(index_to_sample.keys()) == [""]
    # Sort indexes longest-first so a longer index is preferred when one index
    # is a prefix of another.
    index_items = sorted(
        ((idx, sid) for idx, sid in index_to_sample.items() if idx),
        key=lambda kv: -len(kv[0]),
    )
    single_sample = index_to_sample.get("") if demultiplexed else None

    fuzzy_start, fuzzy_end = fuzzy_window

    for n, (_, seq, _, _) in enumerate(read_fastq(fastq_path)):
        if max_reads is not None and n >= max_reads:
            break
        qc["total_reads"] += 1

        # --- demultiplex -----------------------------------------------------
        if demultiplexed:
            sample_id = single_sample
        else:
            sample_id = None
            for idx, sid in index_items:
                if seq.startswith(idx):
                    sample_id = sid
                    break
            if sample_id is None:
                qc["no_index_match"] += 1
                continue
        qc["index_matched"] += 1

        # --- exact gene match at the fixed offset -----------------------------
        gene = match_at_offset(seq, offset, exact_dict, exact_prefix)
        if gene is not None:
            exact_counts[sample_id][gene] += 1
            qc["exact_hits"] += 1
            continue

        # --- one-mismatch fallback across a small offset window ---------------
        hit = None
        for off in range(fuzzy_start, fuzzy_end + 1):
            hit = match_at_offset(seq, off, fuzzy_dict, fuzzy_prefix)
            if hit is not None:
                break
        if hit is not None:
            fuzzy_counts[sample_id][hit] += 1
            qc["fuzzy_hits"] += 1
        else:
            qc["unmatched"] += 1

    return exact_counts, fuzzy_counts, qc


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--offset", type=int, default=None,
                    help="Start offset of the gene uptag. Defaults to the "
                         "config value (28, as in the original BarSeq.py). "
                         "Confirm with step01 before trusting it.")
    ap.add_argument("--fuzzy-window", type=int, default=None,
                    help="How far either side of --offset the one-mismatch "
                         "search slides. Original used offset-3 to offset+2, "
                         "so pass 3 to reproduce it (the +2 side is derived).")
    ap.add_argument("--strict-fuzzy", action="store_true",
                    help="Drop ambiguous one-mismatch sequences instead of "
                         "assigning them to the first gene in the decode file.")
    ap.add_argument("--max-reads", type=int, default=None,
                    help="Stop after this many reads per FASTQ. For testing.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    offset = args.offset if args.offset is not None else cfg["counting"]["uptag_offset"]
    slide = (args.fuzzy_window if args.fuzzy_window is not None
             else cfg["counting"]["fuzzy_slide_back"])
    fuzzy_window = (max(0, offset - slide), offset + cfg["counting"]["fuzzy_slide_forward"])

    decode_path = Path(cfg["paths"]["decode_file"])
    sheet_path = Path(cfg["paths"]["sample_sheet"])
    raw_dir = Path(cfg["paths"]["raw_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["counts_dir"]))

    log.info("uptag offset=%d, one-mismatch offsets %d-%d",
             offset, fuzzy_window[0], fuzzy_window[1])

    exact_dict, gene_order, rows = load_decode(decode_path)
    fuzzy_dict, collisions = build_fuzzy_dict(rows, strict=args.strict_fuzzy)
    exact_prefix = build_prefix_index(exact_dict)
    fuzzy_prefix = build_prefix_index(fuzzy_dict)

    # Save the collision report once; it is a property of the library, not the data.
    coll_rows = [{"sequence": s, "genes": ",".join(sorted(g))}
                 for s, g in collisions.items()]
    pd.DataFrame(coll_rows).to_csv(out_dir / "fuzzy_tag_collisions.tsv",
                                   sep="\t", index=False)

    sheet = load_sample_sheet(sheet_path)
    log.info("Sample sheet: %d samples across %d strains",
             len(sheet), sheet["strain"].nunique())

    # Group samples by the FASTQ they come from, so each file is read once.
    by_fastq = defaultdict(dict)
    for _, row in sheet.iterrows():
        path = resolve_fastq(row["fastq"], raw_dir)
        by_fastq[path][row["index_seq"]] = row["sample_id"]

    all_exact = defaultdict(Counter)
    all_fuzzy = defaultdict(Counter)
    qc_rows = []

    for fastq_path, index_map in by_fastq.items():
        if not fastq_path.exists():
            log.error("Missing FASTQ: %s", fastq_path)
            return 1
        log.info("Counting %s (%d samples multiplexed in it)",
                 fastq_path.name, len(index_map))
        ex, fz, qc = count_one_fastq(
            fastq_path, index_map, exact_dict, exact_prefix,
            fuzzy_dict, fuzzy_prefix, offset, fuzzy_window,
            max_reads=args.max_reads,
        )
        for sid, counter in ex.items():
            all_exact[sid].update(counter)
        for sid, counter in fz.items():
            all_fuzzy[sid].update(counter)

        qc_rows.append({"fastq": fastq_path.name, **qc})

    # --- assemble matrices ---------------------------------------------------
    sample_ids = list(sheet["sample_id"])

    exact_df = pd.DataFrame(
        {sid: pd.Series(all_exact.get(sid, {})) for sid in sample_ids}
    ).reindex(index=gene_order).fillna(0).astype(int)
    exact_df.index.name = "gene"

    # The 1-mismatch table is exact + fuzzy-only, matching BarSeq.py mergeCounts().
    combined = {}
    for sid in sample_ids:
        merged = Counter(all_exact.get(sid, {}))
        merged.update(all_fuzzy.get(sid, {}))
        combined[sid] = pd.Series(merged)
    combined_df = pd.DataFrame(combined).reindex(index=gene_order).fillna(0).astype(int)
    combined_df.index.name = "gene"

    write_table(exact_df, out_dir / "counts_exact.tsv")
    write_table(combined_df, out_dir / "counts_1mismatch.tsv")
    pd.DataFrame(qc_rows).to_csv(out_dir / "counting_qc.tsv", sep="\t", index=False)

    log.info("Wrote %s and %s", out_dir / "counts_exact.tsv",
             out_dir / "counts_1mismatch.tsv")

    # --- quick sanity numbers ------------------------------------------------
    print("\n=== Counting summary ===")
    print(pd.DataFrame(qc_rows).to_string(index=False))
    print(f"\nGenes in matrix: {combined_df.shape[0]}")
    print(f"Samples in matrix: {combined_df.shape[1]}")
    print("\nTotal assigned reads per sample (1-mismatch table):")
    print(combined_df.sum(axis=0).to_string())
    zero_genes = (combined_df.sum(axis=1) == 0).sum()
    print(f"\nGenes with zero reads in every sample: {zero_genes}")
    print("\nCompare these totals with the paper's reported median of")
    print("7,570,975 reads per barcode sample before deciding the run is good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
