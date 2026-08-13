from pathlib import Path
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import pdist
import matplotlib
matplotlib.use("Agg")  # matplotlib-base has no GUI backend; write files instead
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


RAW_DIR = Path("raw_data")
OUT_DIR = Path("data")
HAC_DIR = OUT_DIR / "hac"
HAC_DIR.mkdir(parents=True, exist_ok=True)

FITNESS_PATH = OUT_DIR / "fitness_scores_imputed.tsv"
COMMONLY_DELETERIOUS_PATH = OUT_DIR / "genes_of_interest" / "commonly_deleterious.csv"

METRIC = "cosine"
METHOD = "average"
BLUE_YELLOW = LinearSegmentedColormap.from_list(
    "blue_yellow", ["#00A6E0", "#000000", "#FFFF00"]
)

def cluster(matrix, metric=METRIC, method=METHOD):
    """
    Cluster rows and columns. Returns (row_linkage, col_linkage).
 
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
    """
    The clustered matrix, rows and columns in dendrogram leaf order.
    This is what Java TreeView displays.
    """
    return matrix.iloc[leaves_list(row_linkage), leaves_list(col_linkage)]
 
def plot_column_dendrogram(matrix, col_linkage, path, color_threshold=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(col_linkage, labels=list(matrix.columns), leaf_rotation=90,
               color_threshold=color_threshold, ax=ax)
    ax.set_ylabel(f"{METRIC} distance ({METHOD} linkage)")
    ax.set_title("Strain-replicate clustering of log2 fitness scores")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
 
def plot_heatmap(matrix, row_linkage, col_linkage, path, vmax=3.0):
    grid = sns.clustermap(
        matrix.fillna(0.0), row_linkage=row_linkage, col_linkage=col_linkage,
        cmap=BLUE_YELLOW, vmin=-vmax, vmax=vmax,
        yticklabels=False, xticklabels=True, figsize=(11, 13),
        cbar_kws={
            "label": "log2 fitness score",
            "extend": "both",          # arrowheads flag that the ends are clipped
            "ticks": [-3, -2, -1, 0, 1, 2, 3],
        },
    )
    grid.ax_cbar.set_yticklabels(
        ["≤ 8x depleted", "4x", "2x", "0", "2x", "4x", "≥ 8x enriched"]
    )
    grid.ax_heatmap.collections[0].set_rasterized(True)
    grid.ax_heatmap.set_xlabel("")
    grid.ax_heatmap.set_ylabel(f"{matrix.shape[0]} genes")
    plt.setp(grid.ax_heatmap.get_xticklabels(), rotation=90, fontsize=7)
    grid.savefig(path, dpi=300)
    plt.close(grid.figure)



df = pd.read_csv(FITNESS_PATH, sep="\t", index_col="gene").dropna(how="all")
row_linkage, col_linkage = cluster(df)
clustered = reorder(df, row_linkage, col_linkage)
# clustered.to_csv(OUT_DIR / "clustered_fitness_scores.tsv", sep="\t", na_rep="")
# plot_column_dendrogram(df, col_linkage, OUT_DIR / "hac_strains.png")
plot_heatmap(df, row_linkage, col_linkage, HAC_DIR / "heatmap_hac_figure.png")

df_cd = pd.read_csv(COMMONLY_DELETERIOUS_PATH, sep="\t").drop("# strains where deleterious", axis=1)
genes = df_cd["Commonly Deleterious Genes"].astype(str).str.strip().drop_duplicates()
deleterious = df.loc[genes[genes.isin(df.index)]]
row_linkage, col_linkage = cluster(deleterious)
clustered = reorder(deleterious, row_linkage, col_linkage)
plot_heatmap(deleterious, row_linkage, col_linkage, HAC_DIR / "heatmap_hac_figure_commonly_deleterious.png")


# TODO: do it for their data, for 1) use the elife-70564-fig1-data1-v2.txt file, for 2) use the Commonly-Deleterious-Genes.csv file

