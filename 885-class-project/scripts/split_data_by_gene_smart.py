# Requires XXX files ran with DataSAIL ran with the run_datasail_split.sh and .sub files on CHTC


"""
Split the long fitness table using the DataSAIL split (run C1e_10).

Fitness data is keyed by ORF name, the split table by UniProt accession, so the
mapping file bridges them.
"""

import pandas as pd
import os

SPLIT_OUT_DIR = "data/split/prepped"
os.makedirs(SPLIT_OUT_DIR, exist_ok=True)

# Weird Cases
# UniProt entry (Protein)      Gene(s)
# One protein, two genes       P02309            YNL030W, YBR009C
# One protein, two genes       P10081            YKR059W, YJL138C
# One protein, two genes       P61830            YNL031C, YBR010W
# One gene, two entries        P0CT04, P0CT05    YNL015W
# One gene, two entries        P0C5Q8, Q3E767    YNL067W-B

# One protein, two genes: only one of the two genes is in mapping["From"], the
# other only in mapping["Gene Names"], so it would be lost in the merge below.
# Adding it as its own row ties both genes to the same Entry, hence same split.
PROTEIN_GROUPS = {
    "P02309": ["YNL030W", "YBR009C"],
    "P10081": ["YKR059W", "YJL138C"],
    "P61830": ["YNL031C", "YBR010W"],
}

# One gene, two entries: the gene merges to both entries and the fitness row is
# duplicated. Both entries must be in the same split, then one copy is dropped.
ONE_GENE_TWO_ENTRIES = ["YNL015W", "YNL067W-B"]

# data
fitness_scores = pd.read_csv("data/oe/prepped/long_fitness_scores.csv", sep="\t") # comes from running prep_data.py
mapping = pd.read_csv("data/genes/prepped/mapping_from_yorf_filtered_to_uniprot.tsv", sep="\t") # comes from running get_yorf_to_accession_mapping.py
split = pd.read_csv("data/split/C1e_10/Protein_clean_splits_move_to_train.tsv", sep="\t") # comes from running run_datasail_split.sh (except there is no seed so you must use what I defined)

# add the missing gene -> entry rows for the one-protein-two-genes cases into the mapping
extra = pd.DataFrame([{"From": gene, "Entry": entry} for entry, genes in PROTEIN_GROUPS.items() for gene in genes])
mapping = pd.concat([mapping[["From", "Entry"]], extra]).drop_duplicates().reset_index(drop=True)

# map from protein to gene name for the fitness scores
df = fitness_scores.merge(mapping[["From", "Entry"]], left_on="UID", right_on="From", how="left")

# split the df into train validation test
df = df.merge(split, left_on="Entry", right_on="ID", how="left")

# check that we didn't lose any genes
lost = df[df.Split.isna()]
# print(lost)
print(f"dropped {len(lost)} rows / {lost.UID.nunique()} genes with no split")
df = df.dropna(subset=["Split"])

# one gene, two entries: check that both entries are in the same split and have the same score per strain
for gene in ONE_GENE_TWO_ENTRIES:
    sub = df[df.UID == gene]
    print(f"{gene}: entries {sorted(sub.Entry.unique())} in splits {sorted(sub.Split.unique())}")
    assert sub.Split.nunique() == 1, f"{gene} entries land in different splits"
    assert sub.groupby("strain").fitness_score.nunique().max() == 1, \
        f"{gene} has different scores per strain"
    print(f"we are in the clear to drop one of the entries for {gene}")

before = len(df)
df = df.drop_duplicates(subset=["UID", "strain"], keep="first")
print(f"dropped {before - len(df)} duplicate gene-strain rows")

# a gene split across sets is leakage no similarity check would catch
assert df.groupby("UID").Split.nunique().max() == 1
assert not df.duplicated(subset=["UID", "strain"]).any()

# save the splits into seperate files
for s in ("train", "val", "test"):
    part = df[df.Split == s]
    part[["UID", "strain", "fitness_score", "Split"]].to_csv(f"{SPLIT_OUT_DIR}/{s}_smart.tsv", sep="\t", index=False)
    print(f"{s:6} {len(part):6} rows  {part.UID.nunique():5} genes")