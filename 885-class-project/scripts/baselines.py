# Basic setup: one-hot features and 4 baseline models.

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

# Load data and convert to long format (one row per gene-strain pair)

long = pd.read_csv("data/oe/prepped/long_fitness_scores.csv", sep="\t", header=0,)
# print(long)
y = long["fitness_score"].to_numpy()
# print(long.shape, long["UID"].nunique(), "genes", long["strain"].nunique(), "strains")
# #TODO need to check why is this 4112 genes and not the 4064 by looking in the paper 

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


strain_encoder = OneHotEncoder(drop="first")
X_strain = strain_encoder.fit_transform(long[["strain"]])
# print(strain_encoder.categories_)
# print(strain_encoder.categories_[0][0])
# print(X_strain)
# check_collinearity(X_strain, "strain")

gene_encoder = OneHotEncoder(drop="first")
X_gene = gene_encoder.fit_transform(long[["UID"]])
# print(gene_encoder.categories_)
# print(gene_encoder.categories_[0][0])
# print(X_gene)
# print("strain one-hot:", X_strain.shape, " gene one-hot:", X_gene.shape)
# check_collinearity(X_gene, "uid")

X_both = sp.hstack([X_gene, X_strain])
# print(X_both)
# check_collinearity(X_both, "additive")

# What each baseline should learn
total_mean = y.mean()
strain_mean = long.groupby("strain")["fitness_score"].transform("mean").to_numpy()
gene_mean = long.groupby("UID")["fitness_score"].transform("mean").to_numpy()
additive = gene_mean + strain_mean - total_mean 


# Total-mean model: y = w0 * 1
# Every row gets the same input (1), so the model can only learn one constant.
X_ones = np.ones((len(long), 1))  # bias column: all 1s
m_total = LinearRegression(fit_intercept=False)  # the bias column is the intercept
m_total.fit(X_ones, y)
w0 = m_total.coef_[0]
print("\ntotal mean model")
print(f"w0 = {w0:.6f}   total mean = {total_mean:.6f}   |diff| = {abs(w0 - total_mean):.1e}")
print(f"R2 = {r2_score(y, m_total.predict(X_ones)):.4f}\n")  # 0 by definition

def fit_and_check(X, expected, name):
    """
    Fit with an intercept (w0), then compare predictions to expected means.
    """
    model = LinearRegression(tol=1e-10)  # fit_intercept=True is the default
    model.fit(X, y)
    yhat = model.predict(X)
    diff = np.max(np.abs(yhat - expected))
    print(f"{name:16s} R2 = {r2_score(y, yhat):.4f}   max |yhat - expected| = {diff:.1e}") 
    # prints the r^2 score (coefficient of determination) measures the proportion of variance 
    # in the dependent variable (Y) that is predictable from the independent variable(s) (X)
    # prints the largest gap between the model's predictions and the means it should have learned.
    return model


# Fit three other baselines
m_str = fit_and_check(X_strain, strain_mean, "strain one-hot")  # y = w0 + ws*s
m_gen = fit_and_check(X_gene, gene_mean, "gene one-hot")  # y = w0 + wg*g
m_add = fit_and_check(X_both, additive, "additive")  # y = w0 + wg*g + ws*s

# Read the strain model's weights
# Reference strain: prediction = w0. Every other strain: w0 + ws.
levels = strain_encoder.categories_[0]
w = np.r_[0.0, m_str.coef_]  # reference strain has no column, so w = 0
weights = pd.DataFrame({
    "strain": levels,
    "w_g": w,
    "w0 + w_g": m_str.intercept_ + w,
    "strain specific mean": long.groupby("strain")["fitness_score"].mean().reindex(levels).to_numpy(),
})

print(f"\nstrain linear regression model")
print(f"w0 = {m_str.intercept_:.4f} (mean of reference strain {levels[0]})")
print(weights.to_string(index=False))

# Read the gene model's weights
# Reference gene: prediction = w0. Every other gene: w0 + wg.
gene_levels = gene_encoder.categories_[0]
wg = np.r_[0.0, m_gen.coef_]  # reference gene has no column, so w = 0
gene_weights = pd.DataFrame({
    "UID": gene_levels,
    "w_s": wg,
    "w0 + w_s": m_gen.intercept_ + wg,
    "gene specific mean": long.groupby("UID")["fitness_score"].mean().reindex(gene_levels).to_numpy(),
})
print(f"\ngene linear regression model")
print(f"w0 = {m_gen.intercept_:.4f} (mean of reference gene {gene_levels[0]})")
print(gene_weights.head(10).to_string(index=False))  # 4,900 rows; first 10 shown

# Read the additive model's weights
# coef_ holds gene weights first, then strain weights (same order as X_both).
# Prediction for gene g in strain s: w0 + wg + ws.
print(f"\nadditive linear regression model")
n_gene = X_gene.shape[1]
wg_add = np.r_[0.0, m_add.coef_[:n_gene]]  # reference gene: w = 0
ws_add = np.r_[0.0, m_add.coef_[n_gene:]]  # reference strain: w = 0

## comparing to the gene and strain only models
# add_strain_weights = pd.DataFrame({
#     "strain": levels,
#     "w_s": ws_add,
#     "strain-only w": np.r_[0.0, m_str.coef_],  # equal to w if no cells are missing
# })
# add_gene_weights = pd.DataFrame({
#     "UID": gene_levels,
#     "w_g": wg_add,
#     "gene-only w": wg,  # equal to w if no cells are missing
# })

s_means = long.groupby("strain")["fitness_score"].mean().reindex(levels).to_numpy()
g_means = long.groupby("UID")["fitness_score"].mean().reindex(gene_levels).to_numpy()

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