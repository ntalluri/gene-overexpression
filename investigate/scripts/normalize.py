"""
Own version of Reproducing the "Moby normalization and analyses" section of
Robinson, Hose, Jochem, Place & Gasch, eLife 2021;10:e70564.

Pipeline:
  1. Sum the three sequencing lanes -> raw counts per sample
  2. CPM: divide each sample by its own total, x 1e6
  3. Imputation: genes with >=20 normalized counts in all three Gen0 replicates
     but no reads at Gen10 get a pseudocount of 1 added to their Gen10 counts
  4. Fitness score = log2(normalized Gen10 / normalized Gen0) and then taking the average between replicates
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path("raw_data")
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LANE_FILES = [f"GSE171586_MobyBarseq_Lane{i}_Exact-Match.txt.gz" for i in (1, 2, 3)]

SAMPLE_RE = re.compile(r"^(?P<strain>.+)_Rep(?P<rep>\d+)_(?P<gen>Gen0|Gen10)$") # TODO: add a comment on how this works

# YPS606 Rep3 is present in GEO but was excluded from the published supplement 4 data
# "only duplicates of that strain were analyzed" and in the Normalized Read Counts, they don't include these.
EXCLUDE = {"YPS606_YPD_Rep3_Gen0", "YPS606_YPD_Rep3_Gen10"}


def read_lane(path):
    """
    Reads a lane file as a DF
    drops the well position and index tag from column
    names: 'BC187_YPD_Rep1_Gen0A1-AATAGGCGCT' -> 'BC187_YPD_Rep1_Gen0'.
    """
    df = pd.read_csv(path, sep="\t", index_col="gene")
    df.columns = df.columns.str.replace(r"[A-H]\d{1,2}-[ACGT]+$", "", regex=True)
    return df


def sum_lanes(paths):
    """
    Sum the three lanes. 
    They are technical splits of one pooled library
    (Methods, "Library construction"), not independent samples.
    """
    lanes = [read_lane(p) for p in paths]
    return lanes[0] + lanes[1] + lanes[2]


def cpm(counts):
    """
    Library-size normalization: reads / total reads in sample * 1e6.
    """
    return counts / counts.sum(axis=0) * 1e6


def sample_groups(columns):
    """
    {strain: {'Gen0': [cols], 'Gen10': [cols]}}
    """
    groups = {}
    for col in columns:
        m = SAMPLE_RE.match(col)
        groups.setdefault(m["strain"], {}).setdefault(m["gen"], []).append(col)
    return groups


def impute(counts, normed):
    """
    Add a pseudocount of 1 to Gen10 raw counts for genes clearly present in
    the starting pool but absent at the endpoint.

    Evaluated per strain: >= `20` normalized counts in all
    three Gen0 replicates AND zero raw reads in all Gen10 replicates.

    "To recapture genes that were clearly present in the starting pool
    but completely absent after 10 generation growth, we performed a data imputation:
    genes with at least 20 normalized read counts (>5th percentile of normalized reads) 
    in all three replicates of the starting pool but missing reads from the end-point analysis 
    received a pseudocount of 1 added to the barcode reads at 10 generations."
    """


    out = counts.copy()
    for strain, gens in sample_groups(counts.columns).items():
        gen0, gen10 = gens.get("Gen0"), gens.get("Gen10")
        mask = (normed[gen0] >= 20).all(axis=1) & (counts[gen10] == 0).all(axis=1)
        out.loc[mask, gen10] += 1
    return out


def fitness_scores(normed):
    """
    log2(Gen10 / Gen0) per strain-replicate.

    Fitness scores were calculated by taking the ratio of
    normalized reads at generation 10 divided by reads at generation 0.
    
    Zeros give +/-inf, which becomes NaN for the strain average
    """

    scores = {}
    with np.errstate(divide="ignore", invalid="ignore"):

        for strain, gens in sample_groups(normed.columns).items():
            for c0, c10 in zip(gens["Gen0"], gens["Gen10"]):
                rep = SAMPLE_RE.match(c0)["rep"]
                scores[f"{strain}_Rep{rep}"] = np.log2(normed[c10] / normed[c0])

    return pd.DataFrame(scores).replace([np.inf, -np.inf], np.nan)

def avg_fitness_scores(normed):
    """
    The average fiteness across strains of the per-replicate log2 ratio fitness scores.
    """
    per_rep = fitness_scores(normed)

    out = {}
    for strain in sorted(sample_groups(normed.columns)):
        cols = [c for c in per_rep.columns if c.startswith(strain + "_Rep")]
        yeast_strain = strain.split("_")[0]
        out[f"{yeast_strain}_Avg_log2_Fitness_Score"] = per_rep[cols].mean(axis=1)
    
    return pd.DataFrame(out)

def main():

    # to use the raw barseq data (lanes) from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171586
    counts = sum_lanes([RAW_DIR / f for f in LANE_FILES])
    counts = counts.drop(columns=[c for c in EXCLUDE if c in counts.columns])

    # to use the raw counts directly from supplemnt 4 
    # counts = pd.read_csv(RAW_DIR / "Unnormalized-Read-Counts.csv", sep=",", header=0, index_col="gene")
  
    normed = cpm(counts)

    counts.to_csv(OUT_DIR / "raw_counts.tsv", sep="\t")
    normed.round(0).astype("Int64").to_csv(OUT_DIR / "normalized_counts.tsv", sep="\t")
    fitness_scores(normed).to_csv(OUT_DIR / "fitness_scores_no_imputed.tsv", sep="\t")
    avg_fitness_scores(normed).round(4).to_csv(OUT_DIR / "avg_log2_fitness_scores_no_imputation.tsv", sep="\t")
    
    counts_imp = impute(counts, normed)
    normed_imp = cpm(counts_imp)

    counts_imp.to_csv(OUT_DIR / "raw_counts_imputed.tsv", sep="\t")
    normed_imp.round(0).astype("Int64").to_csv(OUT_DIR / "normalized_counts_imputed.tsv", sep="\t")
    # TODO can we remove all the rows that are all empty from the avg_log2_fitness_scores_imputed.tsv

    fitness_scores(normed_imp).to_csv(OUT_DIR / "fitness_scores_imputed.tsv", sep="\t")
    avg_fitness_scores(normed_imp).round(4).to_csv(OUT_DIR / "avg_log2_fitness_scores_imputed.tsv", sep="\t")


if __name__ == "__main__":
    main()
    