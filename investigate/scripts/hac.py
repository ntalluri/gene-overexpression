from pathlib import Path
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import pdist
import matplotlib
matplotlib.use("Agg")  # matplotlib-base has no GUI backend; write files instead
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = Path("data")
 
METRIC = "cosine"   # uncentered Pearson correlation, the Eisen default
METHOD = "average"

FITNESS_PATH = OUT_DIR / "fitness_scores_imputed.tsv"


def read_fitness(path=FITNESS_PATH):
    """Per-replicate log2 fitness scores, one column per strain-replicate."""
    return pd.read_csv(path, sep="\t", index_col="gene").dropna(how="all")

def cluster(matrix, metric=METRIC, method=METHOD):
    """Cluster rows and columns. Returns (row_linkage, col_linkage).
 
    NaN cells are filled with 0, meaning no fitness change. scipy's pdist has
    no missing-data handling, unlike Cluster 3.0's pairwise-complete
    correlation. At 4.7% missing this is a minor approximation, but it does
    pull genes with dropouts toward the no-effect cluster.
    """
    X = matrix.fillna(0.0).values
    row_linkage = linkage(pdist(X, metric=metric), method=method)
    col_linkage = linkage(pdist(X.T, metric=metric), method=method)
    return row_linkage, col_linkage
 
def reorder(matrix, row_linkage, col_linkage):
    """The clustered matrix, rows and columns in dendrogram leaf order.
    This is what Java TreeView displays."""
    return matrix.iloc[leaves_list(row_linkage), leaves_list(col_linkage)]
 
def plot_column_dendrogram(matrix, col_linkage, path, color_threshold=None):
    """Dendrogram over the 44 strain-replicate columns.
 
    This is the readable one. A dendrogram over 4216 gene rows is a solid
    smear; use the heatmap for those.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(col_linkage, labels=list(matrix.columns), leaf_rotation=90,
               color_threshold=color_threshold, ax=ax)
    ax.set_ylabel(f"{METRIC} distance ({METHOD} linkage)")
    ax.set_title("Strain-replicate clustering of log2 fitness scores")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
 
 
def plot_heatmap(matrix, row_linkage, col_linkage, path, vmax=2.0):
    """Figure 1B: clustered heatmap with both dendrograms attached."""
 
    grid = sns.clustermap(
        matrix.fillna(0.0), row_linkage=row_linkage, col_linkage=col_linkage,
        cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        yticklabels=False, xticklabels=True, figsize=(11, 13),
        cbar_kws={"label": "log2 fitness score"},
    )
    grid.ax_heatmap.set_xlabel("")
    grid.ax_heatmap.set_ylabel(f"{matrix.shape[0]} genes")
    plt.setp(grid.ax_heatmap.get_xticklabels(), rotation=90, fontsize=7)
    grid.savefig(path, dpi=200)
    plt.close(grid.figure)



M = read_fitness()

row_linkage, col_linkage = cluster(M)
clustered = reorder(M, row_linkage, col_linkage)
clustered.to_csv(OUT_DIR / "clustered_fitness_scores.tsv", sep="\t", na_rep="")

plot_column_dendrogram(M, col_linkage, OUT_DIR / "dendrogram_strains.png")
plot_heatmap(M, row_linkage, col_linkage, OUT_DIR / "heatmap_figure.png")