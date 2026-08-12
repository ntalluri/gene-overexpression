"""
edgeR step of the analysis, run with Python via edgePython.

From paper:
Measured and imputed data were analyzed using edgeR version
3.22.1, using a linear model with generation (0 or 10) as a factor. Genes whose
barcodes were significantly different after 10 generations of growth in each
strain at an FDR<0.05 were taken as significant (Benjamini and Hochberg, 1995).
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import edgepython as ep
from normalize import sample_groups

OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FDR_CUTOFF = 0.05

COUNTS_FILE = OUT_DIR / "normalized_counts_imputed.tsv" # I think this is what their using (not the raw counts)
SCORES_FILE = OUT_DIR / "avg_log2_fitness_scores_imputed.tsv"


def fdr_strain(counts, gen0_cols, gen10_cols):
    """
    Fit per strain. 
    Returns FDR by gene (and logFC/logCPM/LR/PValue).
    """
    cols = list(gen0_cols) + list(gen10_cols)
    mat = counts[cols].values.astype(float) # integer count matrix (genes x samples)

    y = ep.make_dgelist(counts=mat) # the raw counts
    y = ep.calc_norm_factors(y) # calculate normalization factors

    # defining the design of our comparisonl comparing gen0 vs gen10 samples
    # b0 is the Gen0 baseline and b1 is the shift from Gen0 to Gen10
    # Gen0 rows: 1·b0 + 0·b1 = b0
    # Gen10 rows: 1·b0 + 1·b1 = b0 + b1
    # column 0 is the intercept Gen0, column 1 marks the Gen10 samples
    is_gen10 = np.array([c.endswith("_Gen10") for c in cols], dtype=float)
    design = np.column_stack([np.ones(len(cols)), is_gen10])

    y = ep.estimate_disp(y, design) # estimation of the negative binomial dispersions
    fit = ep.glm_fit(y, design) # fitting of the negative binomial model to the count data
    lrt = ep.glm_lrt(fit, coef=1) # hypothesis testing 

    table = ep.top_tags(lrt, n=mat.shape[0], adjust_method='BH')["table"] # turns the raw p-values into a usable results table using the Benjamini & Hochberg (BH) FDR method
    table.index = counts.index[table.index] # pull out those gene names in that order
    return table

def load_saved():
    """
    Read the two tables normalize.py already wrote to data/.
    """
    counts_imp = pd.read_csv(COUNTS_FILE, sep="\t", index_col="gene")

    avg = pd.read_csv(SCORES_FILE, sep="\t", index_col="gene")
    if not counts_imp.index.equals(avg.index):
        raise ValueError(f"{COUNTS_FILE.name} and {SCORES_FILE.name} have different genes")
    return counts_imp, avg


def main():
    counts_imp, avg = load_saved()

    out = {}
    for strain, gens in sorted(sample_groups(counts_imp.columns).items()):
        
        res = fdr_strain(counts_imp, gens["Gen0"], gens["Gen10"])
        
        yeast_strain = strain.split("_")[0]

        out[f"{yeast_strain}_Avg_log2_Fitness_Score"] = avg[f"{yeast_strain}_Avg_log2_Fitness_Score"]
        out[f"{yeast_strain}_FDR"] = res["FDR"]
        # print(f"{yeast_strain:14s} {int((res['FDR'] < FDR_CUTOFF).sum()):5d} genes at FDR < {FDR_CUTOFF}")

    table = pd.DataFrame(out)
    table.index.name = "gene"
    table.to_csv(OUT_DIR / "fitness_and_fdr.tsv", sep="\t")

if __name__ == "__main__":
    main()