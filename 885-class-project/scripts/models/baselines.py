# Basic setup: one-hot features and 4 baseline models, fit on train and
# scored on train/val/test.
#
# Usage:
#   python -m scripts.models.baselines --split random
#   python -m scripts.models.baselines --split smart

import argparse
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import os 
from scripts.features.one_hot_encoded_features import OneHotDesign

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["random", "smart"], required=True,
                    help="random = random 70/15/15 by gene, smart = DataSAIL C1e")
args = parser.parse_args()

SPLIT_DIR = f"data/split/prepped"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------- Load the data

# load the train/val/test splits based on the split choice
train = pd.read_csv(f"{SPLIT_DIR}/train_{args.split}.tsv", sep="\t")
val = pd.read_csv(f"{SPLIT_DIR}/val_{args.split}.tsv", sep="\t")
test = pd.read_csv(f"{SPLIT_DIR}/test_{args.split}.tsv", sep="\t")
splits = {"train": train, "val": val, "test": test}

# set what y is
y_train = train["fitness_score"].to_numpy()

# one hot encode gene and strain names from training data
onehot = OneHotDesign().fit(train) 
n_gene = onehot.n_gene_cols()

# # Encoding a val/test gene as all zeros is the intended behaviour here, so the
# # warning about it is noise. predict() handles what the zeros mean.
# warnings.filterwarnings("ignore", message="Found unknown categories")

def design(df, block):
    """
    Build the design matrix X for one baseline. One row per row of df, in order.

      "ones"    a single column of 1s. The bias column for the total-mean model.
      "strain"  strain one hot encoded, one column per train strain minus the dropped
                reference level.
      "gene"    gene one hot encoded, same construction, one column per train gene
                minus the reference.
      "both"    the gene block followed by the strain block, in that order.

    "ones" is handled here rather than in OneHotDesign because a column of 1s is
    not an encoding of anything, so it does not belong in the encoder.
    """
    if block == "ones":
        return np.ones((len(df), 1))
    return onehot.transform(df, block)


# ---------------------------------------------- what each baseline should learn from train only
total_mean = y_train.mean()
train_strain_mean = train.groupby("strain")["fitness_score"].mean()
train_gene_mean = train.groupby("UID")["fitness_score"].mean()


def expected(df, block):
    """
    The group means the fitted model should reproduce on train.
    """
    s = df["strain"].map(train_strain_mean).to_numpy()
    g = df["UID"].map(train_gene_mean).to_numpy()
    if block == "ones":
        return np.full(len(df), total_mean)
    if block == "strain":
        return s
    if block == "gene":
        return g
    return g + s - total_mean


# ------------------------------------------------------------------------- fit the models
# Total-mean model: y = w0 * 1. Every row gets the same input, so the model can
# only learn one constant. The bias column is the intercept, hence no intercept.
m_total = LinearRegression(fit_intercept=False).fit(design(train, "ones"), y_train)
print("\ntotal mean model")
print(f"w0 = {m_total.coef_[0]:.6f}; train mean = {total_mean:.6f}; |diff| = {abs(m_total.coef_[0] - total_mean):.1e}")

m_str = LinearRegression().fit(design(train, "strain"), y_train)  # y = w0 + ws*s
m_gen = LinearRegression().fit(design(train, "gene"), y_train)    # y = w0 + wg*g
m_add = LinearRegression().fit(design(train, "both"), y_train)    # y = w0 + wg*g + ws*s


# ------------------------------------------------------------- predict
# A val/test gene was never seen, so its one-hot row is all zeros, which the
# model reads as the reference gene. The reference gene is an arbitrary choice,
# so instead give an unseen gene the average gene effect over the training rows:
# the honest statement that the model knows nothing about it beyond the training
# distribution. Averaging over rows rather than over genes makes the gene model's
# fallback land exactly on the train mean, since OLS residuals sum to zero.

mean_wg = {
    "gene": float((design(train, "gene") @ m_gen.coef_).mean()),
    "both": float((design(train, "gene") @ m_add.coef_[:n_gene]).mean()),
}

def predict(model, df, block):
    yhat = model.predict(design(df, block))
    if block in ("gene", "both"):
        yhat = yhat + onehot.unseen_genes(df) * mean_wg[block]
    return yhat


MODELS = [
    ("total mean", m_total, "ones"),
    ("strain one-hot", m_str, "strain"),
    ("gene one-hot", m_gen, "gene"),
    ("additive", m_add, "both"),
]

# ------------------------------------------------------- recovery check
# Does each model reproduce the group means it is supposed to learn on train?
# The additive identity gene_mean + strain_mean - total_mean only holds for a
# complete design, so it is also reported over genes measured in all strains.

complete = (train.groupby("UID")["strain"].transform("size").to_numpy()
            == train["strain"].nunique())

print("\nrecovery of the training means (max |yhat - expected|)")
for name, model, block in MODELS:
    resid = np.abs(predict(model, train, block) - expected(train, block))
    extra = "" if block != "both" else f"   complete cells only = {resid[complete].max():.1e}"
    print(f"  {name:16s} {resid.max():.1e}{extra}")


# ---------------------------------------------------------------- score
# R2 is computed against each split's own variance, so a negative value means
# the model does worse than predicting that split's mean.
rows = [
    {
        "model": name,
        "split": split_name,
        "R2": r2_score(df["fitness_score"], predict(model, df, block)),
        "MAE": mean_absolute_error(df["fitness_score"], predict(model, df, block)),
        "RMSE": root_mean_squared_error(df["fitness_score"], predict(model, df, block)),
    }
    for name, model, block in MODELS
    for split_name, df in splits.items()
]
scores = pd.DataFrame(rows)
print("\nscores")
print(scores.pivot(index="model", columns="split",
                   values=["R2", "MAE", "RMSE"]).round(4).to_string())
scores.to_csv(f"{OUTPUT_DIR}/baseline_scores_{args.split}.tsv", sep="\t", index=False)

# -------------------------------------------------------------- weights
strain_levels = onehot.strain_levels
gene_levels = onehot.gene_levels
s_means = train_strain_mean.reindex(strain_levels).to_numpy()
g_means = train_gene_mean.reindex(gene_levels).to_numpy()


def weight_table(levels, coef, intercept, level_col, compare_name, compare):
    """One row per level, with the dropped reference level's 0 put back in."""
    w = np.r_[0.0, coef]
    return pd.DataFrame({
        level_col: levels,
        "w": w,
        "w0 + w": intercept + w,
        compare_name: compare,
    })


# Reference level: prediction = w0. Every other level: w0 + w.
print(f"\nstrain one-hot model   w0 = {m_str.intercept_:.4f} "
      f"(train mean of reference strain {strain_levels[0]})")
print(weight_table(strain_levels, m_str.coef_, m_str.intercept_,
                   "strain", "train strain mean", s_means).to_string(index=False))

print(f"\ngene one-hot model   w0 = {m_gen.intercept_:.4f} "
      f"(train mean of reference gene {gene_levels[0]})")
print(weight_table(gene_levels, m_gen.coef_, m_gen.intercept_,
                   "UID", "train gene mean", g_means)
      .head(10).to_string(index=False))  # one row per train gene; first 10 shown

# coef_ holds gene weights first, then strain weights, same order as the "both"
# block. Prediction for gene g in strain s: w0 + wg + ws. The mean-difference
# columns equal the weights only if no gene-strain cells are missing.
print(f"\nadditive model   w0 = {m_add.intercept_:.4f} "
      f"(prediction for {gene_levels[0]} in {strain_levels[0]})")
print(weight_table(strain_levels, m_add.coef_[n_gene:], m_add.intercept_,
                   "strain", "strain mean diff", s_means - s_means[0]).to_string(index=False))
print()
print(weight_table(gene_levels, m_add.coef_[:n_gene], m_add.intercept_,
                   "UID", "gene mean diff", g_means - g_means[0])
      .head(10).to_string(index=False))

