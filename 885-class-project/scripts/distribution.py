import pandas as pd
import matplotlib.pyplot as plt

# TODO: add the make folder if dne part

BINS = 100
long = pd.read_csv("data/oe/prepped/long_fitness_scores.csv", sep="\t", header=0,)

plt.hist(long["fitness_score"], bins=BINS)
plt.xlabel("log2 fitness score")
plt.ylabel("Gene-strain pairs") # The pooled histogram counts each gene once per strain. A gene appears up to 15 times, so each count is a gene-strain pair
plt.axvline(0, color="black", lw=0.8)
plt.title(f"All strains pooled (n = {len(long)})") # Pooled plot: n ≈ number of genes × 15 strains
plt.savefig("figures/all_strains_distribution.png")

fs_df = pd.read_csv("data/oe/prepped/fitness_scores.csv", sep="\t", header=0, index_col="UID")

fig, axes = plt.subplots(3, 5, figsize=(18, 10), sharex=True)
for ax, strain in zip(axes.flat, fs_df.columns):
    vals = fs_df[strain].dropna()
    ax.hist(vals, bins=BINS)
    ax.axvline(0, color="black", lw=0.8)  # reference line at no effect
    ax.set_title(f"{strain} (n = {len(vals)})") # n is genes with a non-missing score in that strain

for ax in axes[-1]:
    ax.set_xlabel("log2 fitness score")
for ax in axes[:, 0]:
    ax.set_ylabel("Genes") # a gene appears once per strain plot

fig.tight_layout()
plt.savefig("figures/per_strain_distribution.png")