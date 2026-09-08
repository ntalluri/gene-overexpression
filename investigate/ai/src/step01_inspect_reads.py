"""
step01_inspect_reads.py
-----------------------
Look at a small sample of reads from one FASTQ and report its structure.

Why this step exists
--------------------
The original Gasch lab counting script (wiscplace/BarSeq, BarSeq.py) hard-codes
two facts about the read layout:

  * the inline multiplex index is at the very start of the read, and
  * the MoBY gene uptag begins at base 29, i.e. offset 28 (0-based).

Those numbers describe the library prep used in 2016. If you have re-downloaded
the lanes from SRA, the reads may have been trimmed, the index may live in a
separate index read, or the FASTQ may already be demultiplexed per sample. This
step measures the layout instead of assuming it, so that step 02 counts with the
right offset.

What it reports
---------------
  1. Read length distribution.
  2. For each candidate offset, the fraction of reads whose next 10-22 bases
     match a known MoBY uptag exactly. The correct offset shows a sharp peak.
  3. The most common read prefixes of length 6-12, which are your candidate
     inline multiplex indexes if the lane is still multiplexed.

Usage
-----
    python src/step01_inspect_reads.py data/raw/lane1.fastq.gz
    python src/step01_inspect_reads.py data/raw/lane1.fastq.gz --n-reads 500000
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from common import get_logger, read_fastq, resolve_path

log = get_logger("step01")

# MoBY uptags in the decode file range from 10 to 22 bases.
MIN_TAG_LEN = 10
MAX_TAG_LEN = 22


def load_uptags(decode_path: Path) -> set[str]:
    """Return the set of exact uptag sequences from GENE_UPTAG_Decode.txt."""
    tags = set()
    with open(decode_path) as fh:
        for line in fh:
            line = line.rstrip()
            # Data lines start with a systematic ORF name, which begins with Y.
            if not line.startswith("Y"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                tags.add(parts[1])
    return tags


def scan_offsets(reads: list[str], tags: set[str],
                 max_offset: int = 45) -> list[tuple[int, int]]:
    """Count exact uptag hits at each candidate start offset.

    For a given offset we try every tag length from 10 to 22 and take the first
    window that is a known uptag. This mirrors the length loop in the original
    BarSeq.py matchGene().
    """
    hits = []
    for offset in range(0, max_offset + 1):
        n_hit = 0
        for read in reads:
            for length in range(MIN_TAG_LEN, MAX_TAG_LEN + 1):
                if read[offset:offset + length] in tags:
                    n_hit += 1
                    break
        hits.append((offset, n_hit))
    return hits


def common_prefixes(reads: list[str], length: int, top: int = 15):
    """Return the most frequent prefixes of a given length."""
    counter = Counter(r[:length] for r in reads if len(r) >= length)
    return counter.most_common(top)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fastq", help="FASTQ file to inspect (may be .gz)")
    ap.add_argument("--n-reads", type=int, default=200_000,
                    help="How many reads to sample from the top of the file")
    ap.add_argument("--decode",
                    default="data/reference/GENE_UPTAG_Decode.txt",
                    help="MoBY gene uptag decode file")
    ap.add_argument("--max-offset", type=int, default=45,
                    help="Highest start offset to test")
    args = ap.parse_args()

    decode_path = resolve_path(args.decode)
    tags = load_uptags(decode_path)
    log.info("Loaded %d exact uptags from %s", len(tags), decode_path.name)

    fastq_path = resolve_path(args.fastq)

    reads = []
    for i, (_, seq, _, _) in enumerate(read_fastq(fastq_path)):
        if i >= args.n_reads:
            break
        reads.append(seq)
    log.info("Sampled %d reads from %s", len(reads), fastq_path.name)

    if not reads:
        log.error("No reads found. Is the file empty or misformatted?")
        return 1

    # --- 1. Read lengths -----------------------------------------------------
    len_counter = Counter(len(r) for r in reads)
    print("\n=== Read length distribution (top 5) ===")
    for length, count in len_counter.most_common(5):
        print(f"  {length} bp : {count:>8,} reads ({100 * count / len(reads):.1f}%)")

    # --- 2. Uptag offset scan ------------------------------------------------
    print("\n=== Exact uptag hit rate by start offset ===")
    hits = scan_offsets(reads, tags, args.max_offset)
    best_offset, best_hits = max(hits, key=lambda x: x[1])
    for offset, n in hits:
        if n == 0:
            continue
        pct = 100 * n / len(reads)
        marker = "  <-- best" if offset == best_offset else ""
        print(f"  offset {offset:>2} : {n:>8,} hits ({pct:5.1f}%){marker}")

    if best_hits == 0:
        print("\n  No uptags matched at any offset in 0-{}.".format(args.max_offset))
        print("  Check that this really is MoBY Bar-seq data, that the reads are")
        print("  in the expected orientation, and that they have not been")
        print("  reverse-complemented. Try running again on the reverse")
        print("  complement of the reads before assuming the data are wrong.")
    else:
        print(f"\n  Recommended --offset for step 02: {best_offset}")
        print(f"  ({100 * best_hits / len(reads):.1f}% of sampled reads carry an "
              f"exact uptag at that offset)")
        print("  For reference, the original BarSeq.py assumed offset 28.")

    # --- 3. Candidate inline indexes ----------------------------------------
    print("\n=== Most common read prefixes (candidate inline indexes) ===")
    print("If this lane is already demultiplexed, no single prefix will dominate")
    print("and you should leave index_seq empty in samples.tsv.\n")
    for plen in (6, 8, 10, 12):
        top = common_prefixes(reads, plen, top=8)
        total_top = sum(c for _, c in top)
        print(f"  --- prefix length {plen} "
              f"(top 8 cover {100 * total_top / len(reads):.1f}% of reads) ---")
        for prefix, count in top:
            print(f"      {prefix}  {count:>8,}  ({100 * count / len(reads):.2f}%)")
        print()

    print("Reading the prefix table: in a multiplexed lane you expect a small")
    print("number of prefixes at roughly equal, high frequency, one per sample.")
    print("Those are your index_seq values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
