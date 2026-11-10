"""
Random 70/15/15 split of the long fitness table, as a baseline to compare
against the DataSAIL split.

Genes are the unit, so all strain rows for a gene land in the same set. This
matches the cold-protein setting of the DataSAIL split, with the similarity
constraint removed.
"""

import numpy as np
import pandas as pd
import os

SPLIT_OUT_DIR = "data/split/prepped"
SEED = 15
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

os.makedirs(SPLIT_OUT_DIR, exist_ok=True)

fitness_scores = pd.read_csv("data/oe/prepped/long_fitness_scores.csv", sep="\t")  # comes from running prep_data.py

genes = np.sort(fitness_scores["UID"].unique())
rng = np.random.default_rng(SEED)
genes = rng.permutation(genes)

n = len(genes)
n_train = round(FRACTIONS["train"] * n)
n_val = round(FRACTIONS["val"] * n)

assignment = {}
assignment.update({g: "train" for g in genes[:n_train]})
assignment.update({g: "val" for g in genes[n_train:n_train + n_val]})
assignment.update({g: "test" for g in genes[n_train + n_val:]})

df = fitness_scores.copy()
df["Split"] = df["UID"].map(assignment)

# a gene split across sets is leakage no similarity check would catch
assert df["Split"].notna().all()
assert df.groupby("UID").Split.nunique().max() == 1
assert not df.duplicated(subset=["UID", "strain"]).any()

df[["UID", "Split"]].drop_duplicates().to_csv(
    f"{SPLIT_OUT_DIR}/gene_splits_random.tsv", sep="\t", index=False)

for s in ("train", "val", "test"):
    part = df[df.Split == s]
    part[["UID", "strain", "fitness_score", "Split"]].to_csv(f"{SPLIT_OUT_DIR}/{s}_random.tsv", sep="\t", index=False)
    
    print(f"{s:6} {len(part):6} rows  {part.UID.nunique():5} genes  "
          f"{part.UID.nunique() / n:.3f} of genes")