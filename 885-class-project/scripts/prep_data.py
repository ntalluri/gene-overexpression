import itertools
import os
import re
import numpy as np
import pandas as pd

OE_OUT_DIR = "data/oe/prepped"
GENE_OUT_DIR = "data/genes/prepped"
os.makedirs(OE_OUT_DIR, exist_ok=True)
os.makedirs(GENE_OUT_DIR, exist_ok=True)


### PREP OVEREXPRESSION DATA ###

fs = pd.read_csv("data/oe/Fitness_Scores_Imputed.tsv", sep="\t", index_col="UID")
yorf_ids = pd.Series(fs.index.astype(str).str.strip(), name="Feature name")
yorf_ids.to_csv(f"{GENE_OUT_DIR}/yorf_ids.csv", sep="\t", index=False)

# Keep only the fitness score columns
fs = fs.filter(like="_Avg_log2_Fitness_Score")
fs.columns = fs.columns.str.replace("_Avg_log2_Fitness_Score", "")
print(fs.shape)  # expect 15 columns
fs.to_csv(f"{OE_OUT_DIR}/fitness_scores.csv", sep="\t")

# Long format: one row per gene-strain pair
long_fs = fs.reset_index().melt(id_vars="UID", var_name="strain", value_name="fitness_score")

# Drop gene-strain pairs with no score, recording what was dropped
dropped = long_fs[long_fs["fitness_score"].isna()].drop(columns="fitness_score")
dropped.to_csv(f"{GENE_OUT_DIR}/dropped_nan_pairs.csv", sep="\t", index=False)
print(dropped.groupby("strain").size())

long_fs.dropna().to_csv(f"{OE_OUT_DIR}/long_fitness_scores.csv", sep="\t", index=False)


### PREP GENES ###

sgd_columns = [
    "Primary SGDID", "Feature type", "Feature qualifier", "Feature name",
    "Standard gene name", "Alias", "Parent feature name", "Secondary SGDID",
    "Chromosome", "Start_coordinate", "Stop_coordinate", "Strand",
    "Genetic position", "Coordinate version", "Sequence version", "Description",
]
sgd = pd.read_csv(
    "data/genes/SGD_features.tab", sep="\t", header=None, names=sgd_columns, dtype=str
)

# Duplicate YORF ids
dups = yorf_ids[yorf_ids.duplicated(keep=False)].sort_values()
print(f"{yorf_ids.duplicated().sum()} exact duplicates: {dups.tolist()}")

# Are all YORF ids in SGD?
orfs = sgd[sgd["Feature type"] == "ORF"]  # "CDS" is the other option
orf_values = set(orfs.astype(str).stack().str.strip())
missing = yorf_ids[~yorf_ids.isin(orf_values)].tolist()
print(f"{len(yorf_ids) - len(missing)}/{len(yorf_ids)} IDs found; missing: {missing}")
# expect with CDS: 4107/4112 found
#   ['YBR098W-1', 'YDR315C-1', 'YDR315C-2', 'YER007W-1', 'YGL145W-1'] -- technical replicates
# expect with ORF: 4104/4112 found
#   adds 'YER109C', 'YOR031W' (blocked reading frames) and 'YFL056C' (pseudogene); none are dubious
# pseudogene: resembles a functional gene but cannot produce a functional protein
# blocked reading frame: ORF-like region that cannot produce the full-length protein in S288C
#   (premature stop or frameshift), while the intact gene exists in other strains or species


#### Investigating the clone ids (e.g. YBR098W-1)

# Group each "-N" clone with its base ORF
clone_groups = {}
for uid in fs.index.astype(str):
    if re.search(r"-\d+$", uid):
        clone_groups.setdefault(re.sub(r"-\d+$", "", uid), []).append(uid)

rows = []
clone_ids = set()
for base, clones in sorted(clone_groups.items()):
    members = ([base] if base in fs.index else []) + sorted(clones)
    clone_ids.update(members)
    print(f"{base}: {len(members)} rows -> {members}")
    for a, b in itertools.combinations(members, 2):
        pair = fs.loc[[a, b]].T.dropna()
        rows.append({
            "base_ORF": base,
            "id_1": a,
            "id_2": b,
            "n_strains_shared": len(pair),
            "pearson_r": pair[a].corr(pair[b]) if len(pair) > 1 else pd.NA,
            "spearman_r": pair[a].corr(pair[b], method="spearman") if len(pair) > 1 else pd.NA,
        })

clone_corr = pd.DataFrame(rows)
clone_corr.to_csv(f"{GENE_OUT_DIR}/clone_replicate_correlations.csv", sep="\t", index=False)
print(clone_corr.to_string(index=False))

# Null: how well do two unrelated genes correlate in this matrix?
rng = np.random.default_rng(0)
mat = fs.dropna()
idx = rng.choice(len(mat), size=(2000, 2))
for method in ["pearson", "spearman"]:
    null = [mat.iloc[i].corr(mat.iloc[j], method=method) for i, j in idx if i != j]
    print(f"{method} null percentiles (50, 90, 95, 99): {np.percentile(null, [50, 90, 95, 99]).round(3)}")

# Every clone pair falls at or below the null median, so the two copies of an ORF agree
# no better than two unrelated genes do. There is no justified way to pick one or to
# average them, so both copies of each ORF are removed.
pd.Series(sorted(clone_ids), name="Feature name").to_csv(
    f"{GENE_OUT_DIR}/dropped_clone_ids.csv", sep="\t", index=False
)
print(f"{len(clone_ids)} clone IDs dropped: {sorted(clone_ids)}")


#### Remove dubious genes and clone IDs from the yorf id list

dubious_ids = set(
    sgd.loc[sgd["Feature qualifier"].fillna("").str.contains("Dubious"), "Feature name"]
    .astype(str).str.strip()
)
is_dubious = yorf_ids.isin(dubious_ids)
is_clone = yorf_ids.isin(clone_ids)

yorf_ids[is_dubious].to_csv(f"{GENE_OUT_DIR}/dropped_dubious_yorf_ids.csv", sep="\t", index=False)
yorf_ids[~is_dubious & ~is_clone].to_csv(
    f"{GENE_OUT_DIR}/yorf_ids_filtered.csv", sep="\t", index=False
)
print(f"{is_dubious.sum()} dubious removed, {is_clone.sum()} clones removed, "
      f"{(~is_dubious & ~is_clone).sum()} remain")