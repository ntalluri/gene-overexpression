# Requires XXX files ran with DataSAIL ran with the run_datasail_split.sh and .sub files on CHTC

import pandas as pd


#!/usr/bin/env python3
"""Split the long fitness table using the DataSAIL split (run C1e_10, corrected).

Fitness data is keyed by ORF name, the split table by UniProt accession, so the
mapping file bridges them. Every row of a gene follows that gene's split.
"""

fitness_scores = pd.read_csv("data/oe/prepped/long_fitness_scores.csv", sep="\t")
mapping = pd.read_csv("data/genes/prepped/mapping_from_yorf_filtered_to_uniprot.tsv", sep="\t")
split = pd.read_csv("data/split/C1e_10/Protein_clean_splits_move_to_train.tsv", sep="\t")

df = (fitness_scores
      .merge(mapping[["From", "Entry"]], left_on="UID", right_on="From", how="left")
      .merge(split, left_on="Entry", right_on="ID", how="left"))

lost = df[df.Split.isna()]
print(lost)
print(f"dropped {len(lost)} rows / {lost.UID.nunique()} genes with no split")
df = df.dropna(subset=["Split"])

# a gene split across sets is leakage no similarity check would catch
assert df.groupby("UID").Split.nunique().max() == 1

# TODO: add in that these will go into split? or this is the code I will use when I start to train the models
for s in ("train", "val", "test"):
    part = df[df.Split == s]
    part[["UID", "strain", "fitness_score", "Split"]].to_csv(
        f"{s}.tsv", sep="\t", index=False)
    print(f"{s:6} {len(part):6} rows  {part.UID.nunique():5} genes")