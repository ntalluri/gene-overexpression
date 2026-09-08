"""
make_test_data.py
-----------------
Generate a small synthetic Bar-seq dataset so you can run the whole pipeline in
a couple of minutes before pointing it at the real 7-million-read lanes.

The synthetic reads are built to the same layout the real ones are assumed to
have: a 10-base inline multiplex index, an 18-base constant region, then the
MoBY uptag at offset 28, then filler. A configurable fraction of reads carry a
single substitution inside the uptag so that the one-mismatch path gets
exercised, and a fraction are junk so that the unmatched counter is nonzero.

Ground truth is written to data/test/ground_truth.tsv: the true log2 fold change
that was used to simulate each gene in each strain. Compare it with
results/04_fitness/fitness_scores_imputed.tsv to confirm the statistics work.

Usage
-----
    python src/make_test_data.py
    python src/make_test_data.py --n-strains 4 --reads-per-sample 300000
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import PROJECT_ROOT, ensure_dir, get_logger

log = get_logger("testdata")

BASES = "ACGT"
CONSTANT_REGION_LEN = 18     # gap between the 10 bp index and the uptag
UPTAG_OFFSET = 28
FILLER_LEN = 40


def random_seq(n: int, rng: random.Random) -> str:
    return "".join(rng.choice(BASES) for _ in range(n))


def make_indexes(n: int, rng: random.Random) -> list[str]:
    """Generate n distinct 10-base index sequences."""
    seen = set()
    while len(seen) < n:
        seen.add(random_seq(10, rng))
    return sorted(seen)


def mutate(seq: str, rng: random.Random) -> str:
    """Substitute one random base."""
    i = rng.randrange(len(seq))
    alt = rng.choice([b for b in BASES if b != seq[i]])
    return seq[:i] + alt + seq[i + 1:]


def load_uptags(decode_path: Path, limit: int | None):
    """Return a list of (gene, uptag) pairs from the decode file."""
    pairs = []
    with open(decode_path) as fh:
        for line in fh:
            if not line.startswith("Y"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
            if limit and len(pairs) >= limit:
                break
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-strains", type=int, default=3)
    ap.add_argument("--n-genes", type=int, default=500)
    ap.add_argument("--reads-per-sample", type=int, default=120_000)
    ap.add_argument("--n-lanes", type=int, default=2)
    ap.add_argument("--mismatch-rate", type=float, default=0.06)
    ap.add_argument("--junk-rate", type=float, default=0.04)
    ap.add_argument("--dropout-genes", type=int, default=12,
                    help="Genes forced to zero at generation 10, to exercise "
                         "the imputation step")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    nprng = np.random.default_rng(args.seed)

    decode = PROJECT_ROOT / "data" / "reference" / "GENE_UPTAG_Decode.txt"
    pairs = load_uptags(decode, args.n_genes)
    genes = [g for g, _ in pairs]
    log.info("Simulating %d genes", len(genes))

    strains = [f"TESTSTR{i+1}" for i in range(args.n_strains)]
    replicates = ["A", "B", "C"]
    generations = [0, 10]

    samples = []
    for strain in strains:
        for gen in generations:
            for rep in replicates:
                samples.append({
                    "sample_id": f"{strain}_T{gen}_{rep}",
                    "strain": strain,
                    "generation": gen,
                    "replicate": rep,
                })

    indexes = make_indexes(len(samples), rng)
    for s, idx in zip(samples, indexes):
        s["index_seq"] = idx

    # Spread samples across lanes, as in the real experiment where replicates
    # were pooled and split across three lanes.
    for i, s in enumerate(samples):
        s["fastq"] = f"test_lane{(i % args.n_lanes) + 1}.fastq"

    # --- ground truth --------------------------------------------------------
    # Baseline abundance is shared across strains; the effect is strain-specific.
    baseline = nprng.lognormal(mean=0.0, sigma=0.8, size=len(genes))
    baseline = baseline / baseline.sum()

    truth = pd.DataFrame(index=genes)
    effects = {}
    for strain in strains:
        lfc = np.zeros(len(genes))
        # 15% of genes deleterious, 5% beneficial, rest neutral.
        n = len(genes)
        idx_del = nprng.choice(n, size=int(0.15 * n), replace=False)
        remaining = np.setdiff1d(np.arange(n), idx_del)
        idx_ben = nprng.choice(remaining, size=int(0.05 * n), replace=False)
        lfc[idx_del] = -nprng.uniform(0.8, 3.0, size=len(idx_del))
        lfc[idx_ben] = nprng.uniform(0.8, 2.0, size=len(idx_ben))
        effects[strain] = lfc
        truth[strain] = lfc

    # Force a few genes to vanish at generation 10 so imputation has work to do.
    dropouts = list(nprng.choice(len(genes), size=args.dropout_genes, replace=False))
    truth["forced_dropout"] = [i in dropouts for i in range(len(genes))]

    # --- write FASTQs --------------------------------------------------------
    raw_dir = ensure_dir(PROJECT_ROOT / "data" / "raw")
    lane_handles = {}
    for lane in range(1, args.n_lanes + 1):
        path = raw_dir / f"test_lane{lane}.fastq"
        lane_handles[f"test_lane{lane}.fastq"] = open(path, "w")

    read_id = 0
    for s in samples:
        fh = lane_handles[s["fastq"]]
        strain = s["strain"]
        lfc = effects[strain]

        if s["generation"] == 0:
            weights = baseline.copy()
        else:
            weights = baseline * (2.0 ** lfc)
            for d in dropouts:
                weights[d] = 0.0
        weights = weights / weights.sum()

        # Multinomial sampling of reads across genes, then noise from the
        # per-replicate library prep.
        noisy = weights * nprng.lognormal(0.0, 0.12, size=len(weights))
        noisy = noisy / noisy.sum()
        counts = nprng.multinomial(
            int(args.reads_per_sample * (1 - args.junk_rate)), noisy)

        for gene_i, count in enumerate(counts):
            if count == 0:
                continue
            uptag = pairs[gene_i][1]
            for _ in range(count):
                tag = mutate(uptag, rng) if rng.random() < args.mismatch_rate else uptag
                pad = "N" * max(0, UPTAG_OFFSET - 10 - CONSTANT_REGION_LEN)
                seq = (s["index_seq"]
                       + random_seq(CONSTANT_REGION_LEN, rng)
                       + pad + tag
                       + random_seq(FILLER_LEN, rng))
                read_id += 1
                fh.write(f"@read{read_id}\n{seq}\n+\n{'I' * len(seq)}\n")

        # Junk reads that carry the index but no recognizable uptag.
        for _ in range(int(args.reads_per_sample * args.junk_rate)):
            seq = s["index_seq"] + random_seq(80, rng)
            read_id += 1
            fh.write(f"@read{read_id}\n{seq}\n+\n{'I' * len(seq)}\n")

    for fh in lane_handles.values():
        fh.close()

    # --- write sample sheet and ground truth ---------------------------------
    sheet = pd.DataFrame(samples)[
        ["sample_id", "strain", "generation", "replicate", "index_seq", "fastq"]]
    sheet_path = PROJECT_ROOT / "data" / "reference" / "samples.tsv"
    sheet.to_csv(sheet_path, sep="\t", index=False)

    test_dir = ensure_dir(PROJECT_ROOT / "data" / "test")
    truth.index.name = "gene"
    truth.to_csv(test_dir / "ground_truth.tsv", sep="\t")

    log.info("Wrote %d reads across %d lanes", read_id, args.n_lanes)
    log.info("Sample sheet: %s", sheet_path)
    log.info("Ground truth: %s", test_dir / "ground_truth.tsv")
    print("\nNow run:")
    print("  python src/step02_count_barcodes.py")
    print("  python src/step03_normalize.py")
    print("  python src/step04_impute.py")
    print("  python src/step05_fitness.py")
    print("  python src/step06_gene_lists.py")
    print("  python src/check_against_truth.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
