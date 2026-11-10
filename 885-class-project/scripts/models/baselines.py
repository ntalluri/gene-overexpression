# Basic setup: one-hot features and 4 baseline models, fit on train and
# scored on train/val/test.
#
# Usage:
#   python baselines.py --split random
#   python baselines.py --split smart

import argparse
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import os 

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["random", "smart"], required=True,
                    help="random = random 70/15/15 by gene, smart = DataSAIL C1e")
args = parser.parse_args()

SPLIT_DIR = f"data/split/prepped"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the split long tables (one row per gene-strain pair)
train = pd.read_csv(f"{SPLIT_DIR}/train_{args.split}.tsv", sep="\t")
val = pd.read_csv(f"{SPLIT_DIR}/val_{args.split}.tsv", sep="\t")
test = pd.read_csv(f"{SPLIT_DIR}/test_{args.split}.tsv", sep="\t")
splits = {"train": train, "val": val, "test": test}

y_train = train["fitness_score"].to_numpy()

print(f"split = {args.split}")
for name, df in splits.items():
    print(f"  {name:5s} {len(df):6d} rows  {df.UID.nunique():5d} genes  "
          f"{df.strain.nunique():3d} strains")

# Both splits are cold-gene: no gene appears in more than one set.
train_genes = set(train["UID"])
assert not (set(val["UID"]) & train_genes)
assert not (set(test["UID"]) & train_genes)
print("  no gene appears in two sets (cold-gene split)")

# Build one hot encoded features (using sparse matrices)
# claude said that a dense gene one-hot would be ~73,000 x 4,900 floats (~3 GB).

# One-hot encoding all levels plus an intercept causes perfect multicollinearity.
# Each row has exactly one level set to 1, so the one-hot columns sum to 1.
# That sum equals the intercept column (all 1s).
# So any one-hot column is a linear combination of the intercept and the others.
# The weights are not unique: infinitely many sets give identical predictions.
# The solver returns one set
# Predictions are fine; individual weights are not interpretable.
# https://medium.com/@ramdhanhdy/understanding-the-dummy-variable-trap-78d00f8bf20a

# the fix is to drop one level per categorical feature (the reference level).
# The intercept absorbs the reference level.
# Other weights become differences from the reference level.
# With one categorical feature, w0 = the reference level's mean.
# Two one-hot blocks (gene + strain) are collinear even without an intercept.

# This matters for unpenalized linear models (OLS; logistic with no penalty).
# Ridge and sklearn's default logistic regression (L2) have unique weights, so all levels can be kept
# Pure lasso (L1) can still have non-unique weights with collinear columns.
# Tree models do not need a dropped level.


def check_collinearity(X, name, intercept=True):
    """Count redundant columns in X (plus an intercept column, if the model has one)."""
    if intercept:
        X = sp.hstack([np.ones((X.shape[0], 1)), X], format="csr")
    XtX = (X.T @ X).toarray()  # same rank as X, but only n_cols x n_cols
    rank = np.linalg.matrix_rank(XtX)
    empty = int((np.asarray(X.sum(axis=0)).ravel() == 0).sum())
    print(f"{name:28s} columns = {X.shape[1]:5d}  rank = {rank:5d}  "
          f"redundant = {X.shape[1] - rank}  empty columns = {empty}")


# Encoders are fit on train only. Fitting on the full table would leak the
# identity of the val/test genes into the feature space.
# handle_unknown="ignore" so val/test genes do not raise; see predict() for why
# an all-zero row still needs fixing up.
strain_encoder = OneHotEncoder(drop="first", handle_unknown="ignore")
strain_encoder.fit(train[["strain"]])

gene_encoder = OneHotEncoder(drop="first", handle_unknown="ignore")
gene_encoder.fit(train[["UID"]])

levels = strain_encoder.categories_[0]
gene_levels = gene_encoder.categories_[0]
n_gene = len(gene_levels) - 1  # drop="first" removes one column
# check_collinearity(gene_encoder.transform(train[["UID"]]), "uid")


# Encoding a val/test gene as all zeros is the intended behaviour here, so the
# warning about it is noise. predict() handles what the zeros mean.
warnings.filterwarnings("ignore", message="Found unknown categories")


def design(df, block):
    """Build the design matrix for one of the four baselines."""
    if block == "ones":
        return np.ones((len(df), 1))  # bias column: all 1s
    Xs = strain_encoder.transform(df[["strain"]])
    Xg = gene_encoder.transform(df[["UID"]])
    if block == "strain":
        return Xs
    if block == "gene":
        return Xg
    return sp.hstack([Xg, Xs], format="csr")


# What each baseline should learn, from train only
total_mean = y_train.mean()
train_strain_mean = train.groupby("strain")["fitness_score"].mean()
train_gene_mean = train.groupby("UID")["fitness_score"].mean()


def expected(df, block):
    """The group means the fitted model should reproduce on train."""
    s = df["strain"].map(train_strain_mean).to_numpy()
    g = df["UID"].map(train_gene_mean).to_numpy()
    if block == "ones":
        return np.full(len(df), total_mean)
    if block == "strain":
        return s
    if block == "gene":
        return g
    return g + s - total_mean


# Total-mean model: y = w0 * 1
# Every row gets the same input (1), so the model can only learn one constant.
m_total = LinearRegression(fit_intercept=False)  # the bias column is the intercept
m_total.fit(design(train, "ones"), y_train)
w0 = m_total.coef_[0]
print("\ntotal mean model")
print(f"w0 = {w0:.6f}   train mean = {total_mean:.6f}   |diff| = {abs(w0 - total_mean):.1e}")

# Fit three other baselines on train
m_str = LinearRegression().fit(design(train, "strain"), y_train)  # y = w0 + ws*s
m_gen = LinearRegression().fit(design(train, "gene"), y_train)    # y = w0 + wg*g
m_add = LinearRegression().fit(design(train, "both"), y_train)    # y = w0 + wg*g + ws*s

# Gene weights, with the reference gene's 0 put back in
wg = np.r_[0.0, m_gen.coef_]
wg_add = np.r_[0.0, m_add.coef_[:n_gene]]
ws_add = np.r_[0.0, m_add.coef_[n_gene:]]

# A val/test gene was never seen, so its one-hot row is all zeros, which the
# model reads as the reference gene. The reference gene is an arbitrary choice,
# so instead give an unseen gene the average gene effect over the training rows:
# the honest statement that the model knows nothing about it beyond the
# training distribution. Averaging over rows rather than over genes makes the
# gene model's fallback land exactly on the train mean, since OLS residuals
# sum to zero.
mean_wg = {
    "gene": float((design(train, "gene") @ m_gen.coef_).mean()),
    "both": float((design(train, "gene") @ m_add.coef_[:n_gene]).mean()),
}


def predict(model, df, block):
    yhat = model.predict(design(df, block))
    if block in ("gene", "both"):
        unseen = ~df["UID"].isin(train_genes).to_numpy()
        yhat = yhat + unseen * mean_wg[block]
    return yhat


MODELS = [
    ("total mean", m_total, "ones"),
    ("strain one-hot", m_str, "strain"),
    ("gene one-hot", m_gen, "gene"),
    ("additive", m_add, "both"),
]

# Does each model reproduce the group means it is supposed to learn on train?
# The additive identity gene_mean + strain_mean - total_mean only holds for a
# complete design, so it is also reported over the genes measured in all strains.
n_strain = train["strain"].nunique()
complete = train.groupby("UID")["strain"].transform("size").to_numpy() == n_strain
print("\nrecovery of the training means (max |yhat - expected|)")
for name, model, block in MODELS:
    resid = np.abs(predict(model, train, block) - expected(train, block))
    extra = "" if block != "both" else \
        f"   complete cells only = {resid[complete].max():.1e}"
    print(f"  {name:16s} {resid.max():.1e}{extra}")
# print(f"  {int((~complete).sum())} of {len(train)} train rows are in a gene "
#       f"missing at least one strain")

# Score every model on every split.
# R2 is computed against each split's own variance, so a negative value means
# the model does worse than predicting that split's mean.
print("\nscores")
rows = []
for name, model, block in MODELS:
    for split_name, df in splits.items():
        yt = df["fitness_score"].to_numpy()
        yhat = predict(model, df, block)
        rows.append({
            "model": name,
            "split": split_name,
            "R2": r2_score(yt, yhat),
            "MAE": mean_absolute_error(yt, yhat),
            "RMSE": root_mean_squared_error(yt, yhat),
        })
scores = pd.DataFrame(rows)
print(scores.pivot(index="model", columns="split",
                   values=["R2", "MAE", "RMSE"]).round(4).to_string())
scores.to_csv(f"{OUTPUT_DIR}/baseline_scores_{args.split}.tsv", sep="\t", index=False)

# Under a cold-gene split the gene block carries no information at test time,
# so the gene model collapses onto the total mean and the additive model onto
# the strain model. Only the strain effect transfers.
for split_name in ("val", "test"):
    df = splits[split_name]
    g = predict(m_gen, df, "gene")
    t = predict(m_total, df, "ones")
    a = predict(m_add, df, "both")
    s = predict(m_str, df, "strain")
    print(f"  {split_name}: max |gene - total mean| = {np.max(np.abs(g - t)):.1e}   "
          f"max |additive - strain| = {np.max(np.abs(a - s)):.1e}")

# Read the strain model's weights
# Reference strain: prediction = w0. Every other strain: w0 + ws.
w = np.r_[0.0, m_str.coef_]  # reference strain has no column, so w = 0
weights = pd.DataFrame({
    "strain": levels,
    "w_s": w,
    "w0 + w_s": m_str.intercept_ + w,
    "train strain mean": train_strain_mean.reindex(levels).to_numpy(),
    "test strain mean": test.groupby("strain")["fitness_score"].mean().reindex(levels).to_numpy(),
})

print(f"\nstrain linear regression model")
print(f"w0 = {m_str.intercept_:.4f} (train mean of reference strain {levels[0]})")
print(weights.to_string(index=False))

# Read the gene model's weights
# Reference gene: prediction = w0. Every other gene: w0 + wg.
gene_weights = pd.DataFrame({
    "UID": gene_levels,
    "w_g": wg,
    "w0 + w_g": m_gen.intercept_ + wg,
    "train gene mean": train_gene_mean.reindex(gene_levels).to_numpy(),
})
print(f"\ngene linear regression model")
print(f"w0 = {m_gen.intercept_:.4f} (train mean of reference gene {gene_levels[0]})")
print(gene_weights.head(10).to_string(index=False))  # one row per train gene; first 10 shown

# Read the additive model's weights
# coef_ holds gene weights first, then strain weights (same order as X_both).
# Prediction for gene g in strain s: w0 + wg + ws.
print(f"\nadditive linear regression model")
s_means = train_strain_mean.reindex(levels).to_numpy()
g_means = train_gene_mean.reindex(gene_levels).to_numpy()

add_strain_weights = pd.DataFrame({
    "strain": levels,
    "w_s": ws_add,
    "strain mean diff (strain - dropped)": s_means - s_means[0],  # equals w_s only if no cells are missing
})

add_gene_weights = pd.DataFrame({
    "UID": gene_levels,
    "w_g": wg_add,
    "gene mean diff (gene - dropped)": g_means - g_means[0],  # equals w_g only if no cells are missing
})

print(f"w0 = {m_add.intercept_:.4f} (prediction for {gene_levels[0]} in {levels[0]})")
print(add_strain_weights.to_string(index=False))
print()
print(add_gene_weights.head(10).to_string(index=False))

# TODO: next step try to look at other ways to split the data
# TODO: try out regulizered models
# - bayesion prior on 0's for regression models (interesting ideas)