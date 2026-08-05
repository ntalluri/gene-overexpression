from pathlib import Path
import pandas as pd

OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FITNESS_FILE = OUT_DIR / "moby_fitness_and_fdr.tsv"

def read_fitness(path=FITNESS_FILE):
    """Split the wide table into aligned score and FDR frames, one column per strain."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    strains = [c.replace("_Avg_log2_Fitness_Score", "")
               for c in df.columns if c.endswith("_Avg_log2_Fitness_Score")]
    score = pd.DataFrame({s: df[f"{s}_Avg_log2_Fitness_Score"] for s in strains})
    fdr = pd.DataFrame({s: df[f"{s}_FDR"] for s in strains})
    return score, fdr


def deleterious_or_beneficial(path=FITNESS_FILE, fdr_cutoff=0.05):
    """Two boolean frames, genes x strains."""
    score, fdr = read_fitness(path)
    significant = fdr < fdr_cutoff
    return significant & (score < 0), significant & (score > 0)


# commonly deleterious: deleterious in at least 10 of the 15 strains (66%).
def commonly_deleterious(path=FITNESS_FILE, min_strains=10, fdr_cutoff=0.05):
    """
    Series of gene -> number of strains where deleterious, filtered to >= min_strains.
    """
    deleterious, _ = deleterious_or_beneficial(path, fdr_cutoff)
    n = deleterious.sum(axis=1)
    return n[n >= min_strains].sort_values(ascending=False)

def strain_specific(path=FITNESS_FILE, max_strains=3, fdr_cutoff=0.05):
    """
    Two boolean frames: strain-specific deleterious and strain-specific beneficial.
    """
    deleterious, beneficial = deleterious_or_beneficial(path, fdr_cutoff)
    rare_del = deleterious.sum(axis=1) <= max_strains
    rare_ben = beneficial.sum(axis=1) <= max_strains
    return deleterious.mul(rare_del, axis=0), beneficial.mul(rare_ben, axis=0)

def as_gene_columns(mask, suffix):
    """Boolean frame -> one column of gene names per strain, ragged and blank-padded."""
    return pd.DataFrame({f"{s}_{suffix}_Genes": pd.Series(mask.index[mask[s]].tolist())
                         for s in mask.columns})

cd = commonly_deleterious()
cd.to_csv(OUT_DIR/"commonly_deleterious.csv", sep="\t")

ss_del, ss_ben = strain_specific()
as_gene_columns(ss_del, "Deleterious").to_csv(OUT_DIR/"strain_specific_deleterious.csv", sep="\t", index=False)
as_gene_columns(ss_ben, "Beneficial").to_csv(OUT_DIR/"strain_specific_beneficial.csv", sep="\t", index=False)

