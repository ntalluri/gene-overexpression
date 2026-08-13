from pathlib import Path
import pandas as pd

RAW_DIR = Path("raw_data")

OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOI_DIR = OUT_DIR / "genes_of_interest"
GOI_DIR.mkdir(parents=True, exist_ok=True)

FITNESS_FILE = OUT_DIR / "fitness_and_fdr.tsv" 

# uncomment to use the data from supplement 4 instead of my analysis
# FITNESS_FILE = RAW_DIR / "Fitness_Scores_Imputed.tsv" 

SUPP_CD_GENES = RAW_DIR / "Commonly-Deleterious-Genes.csv"
SUPP_SSBEN_GENES = RAW_DIR / "Strain-Specific-Beneficial.csv"
SUPP_SSDEL_GENES = RAW_DIR / "Strain-Specific-Deleterious.csv"

def read_fitness(path=FITNESS_FILE):
    """
    Split the wide table into aligned score and FDR frames, one column per strain.
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    strains = [c.replace("_Avg_log2_Fitness_Score", "") for c in df.columns if c.endswith("_Avg_log2_Fitness_Score")]
    score = pd.DataFrame({s: df[f"{s}_Avg_log2_Fitness_Score"] for s in strains})
    fdr = pd.DataFrame({s: df[f"{s}_FDR"] for s in strains})
    return score, fdr


def deleterious_or_beneficial(path=FITNESS_FILE, fdr_cutoff=0.05):
    """
    To be deleterious, the gene must have an FDR < 0.05 and a log2 avg fitness score < 0
    To be beneficial, the gene must have an FDR < 0.05 and a log2 avg fitness score > 0
    """
    score, fdr = read_fitness(path)
    significant = fdr < fdr_cutoff
    return significant & (score < 0), significant & (score > 0)


def commonly_deleterious(path=FITNESS_FILE, min_strains=10, fdr_cutoff=0.05):
    """
    To be commonly deleterious, the gene needs to be in at least 10 of the 15 strains (66%).
    """
    deleterious, _ = deleterious_or_beneficial(path, fdr_cutoff)
    n = deleterious.sum(axis=1)
    return n[n >= min_strains].sort_values(ascending=False).rename_axis("Commonly Deleterious Genes").rename("# strains where deleterious")


def strain_specific(path=FITNESS_FILE, max_strains=3, fdr_cutoff=0.05):
    """
    To be strain-specific deleterious or strain-specific beneficial, the gene can only be in max 3 strains

    """
    deleterious, beneficial = deleterious_or_beneficial(path, fdr_cutoff)
    rare_del = deleterious.sum(axis=1) <= max_strains
    rare_ben = beneficial.sum(axis=1) <= max_strains
    return deleterious.mul(rare_del, axis=0), beneficial.mul(rare_ben, axis=0)


def as_gene_columns(mask, suffix):
    """
    Boolean frame -> one column of gene names per strain
    """
    return pd.DataFrame({f"{s}_{suffix}_Genes": pd.Series(mask.index[mask[s]].tolist()) for s in mask.columns})


# find the commonly deleterious genes between the strains
cd = commonly_deleterious()
cd = cd.to_frame()
cd.to_csv(GOI_DIR/"commonly_deleterious.csv", sep="\t")

# compare with the supplement
supp_cd = pd.read_csv(SUPP_CD_GENES, header=0, sep=",", index_col=0)
merged = pd.merge(cd, supp_cd, how='outer', indicator=True, left_index=True, right_index=True)
label_map = {
    'left_only': 'cd',
    'right_only': 'supp_cd',
    'both': 'both'
}
merged['_merge'] = merged['_merge'].map(label_map)
merged.columns=["# strains where deleterious_cd", "# strains where deleterious_supp_cd", "merge"]
merged["difference"] = merged["# strains where deleterious_cd"].fillna(0) - merged["# strains where deleterious_supp_cd"].fillna(0)
merged["# of strains differ"] = merged["difference"].ne(0)
merged.drop(columns=["difference"], inplace=True)
merged.sort_values(by=["# of strains differ"], inplace=True, ascending=False)
merged.to_csv(GOI_DIR/"compare_commonly_deleterious_to_supplement.csv", sep="\t")
pd.crosstab(merged["merge"], merged["# of strains differ"]).to_csv(GOI_DIR/"compare_commonly_deleterious_to_supplement_stats.csv", sep="\t")

# find the strain specific genes
ss_del, ss_ben = strain_specific()
as_gene_columns(ss_del, "Deleterious").to_csv(GOI_DIR/"strain_specific_deleterious.csv", sep="\t", index=False)
as_gene_columns(ss_ben, "Beneficial").to_csv(GOI_DIR/"strain_specific_beneficial.csv", sep="\t", index=False)

# compare with the supplements
# want to see how many genes are overlapping and how many are unique in 
# my work vs the supplemnt per strain
def compare_lists(mine, supp):
    """
    Per-strain overlap between my gene columns and the supplement's.
    """
    COUNT_COLS = ["mine", "supp", "overlap"]
    rows = []
    for col in mine.columns:
        strain = col.split("_")[0]
        m = set(mine[col].dropna())
        
        if col not in supp.columns:
            rows.append({"strain": strain, "mine": len(m), "supp": None, "overlap": None, "jaccard": None})
            continue
        
        s = set(supp[col].dropna())
        ov = m & s
        rows.append({"strain": strain, "mine": len(m), "supp": len(s), "overlap": len(ov), "jaccard": round(len(ov) / len(m | s), 3)})
    
    df = pd.DataFrame(rows).set_index("strain")
    df[COUNT_COLS] = df[COUNT_COLS].astype("Int64")
    df["jaccard"] = df["jaccard"].round(3)
    return df
    

supp_del = pd.read_csv(SUPP_SSDEL_GENES, header=0, sep=",")
supp_ben = pd.read_csv(SUPP_SSBEN_GENES, header=0, sep=",")
ss_del = as_gene_columns(ss_del, "Deleterious")
ss_ben = as_gene_columns(ss_ben, "Beneficial")

stats_ben = compare_lists(ss_ben, supp_ben) 
stats_ben.to_csv(GOI_DIR/"compare_strain_specific_beneficial_to_supplement_stats.csv", sep = "\t")

stats_del = compare_lists(ss_del, supp_del) 
stats_del.to_csv(GOI_DIR/"compare_strain_specific_deleterious_to_supplement_stats.csv", sep = "\t")
