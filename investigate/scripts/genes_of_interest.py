from pathlib import Path
import pandas as pd

OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOI_DIR = OUT_DIR / "genes_of_interest"
GOI_DIR.mkdir(parents=True, exist_ok=True)

FITNESS_FILE = OUT_DIR / "fitness_and_fdr.tsv"

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
    return n[n >= min_strains].sort_values(ascending=False)

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

cd = commonly_deleterious()
# TODO: rename the columns Commonly Deleterious Genes, # strains where deleterious
cd.to_csv(GOI_DIR/"commonly_deleterious.csv", sep="\t")

ss_del, ss_ben = strain_specific()
as_gene_columns(ss_del, "Deleterious").to_csv(GOI_DIR/"strain_specific_deleterious.csv", sep="\t", index=False)
as_gene_columns(ss_ben, "Beneficial").to_csv(GOI_DIR/"strain_specific_beneficial.csv", sep="\t", index=False)

# TODO: then compare with the supplement and see how much overlap