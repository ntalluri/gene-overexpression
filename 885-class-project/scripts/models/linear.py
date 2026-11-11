"""
Per-strain linear models on the categorical gene features.

Fifteen models, one per strain, all sharing the same gene feature matrix. Each
model learns how gene properties combine into a predicted OE fitness effect in
that strain, so comparing the fifteen weight vectors shows whether the same
properties matter in every background.

The gene split is shared across all fifteen, so a gene held out of training is
held out in every strain. Within a strain, a split is just that strain's rows
of the long split frame.

Usage:
  python -m scripts.models.linear --split smart
  python -m scripts.models.linear --split random
"""

import argparse
import os

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error

from scripts.features.categorical_features import CategoricalFeatures

FILTERED_GENES = "data/genes/prepped/yorf_ids_filtered.csv"
SPLIT_DIR = "data/split/prepped"
OUTPUT_DIR = "output"

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["random", "smart"], required=True,
                    help="random = random 70/15/15 by gene, smart = DataSAIL C1e")
args = parser.parse_args()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- load
# Long tables, one row per gene-strain pair. NaN pairs were dropped in
# prep_data.py, so every row here is a real measurement.
splits = {
    name: pd.read_csv(f"{SPLIT_DIR}/{name}_{args.split}.tsv", sep="\t")
    for name in ("train", "val", "test")
}
train = splits["train"]
strains = sorted(train["strain"].unique())

# ------------------------------------------------------------ features
# Fit on the training rows only
cat_feats = CategoricalFeatures().fit(train)

filtered = pd.DataFrame({
    "UID": pd.read_csv(FILTERED_GENES, sep="\t")["Feature name"].astype(str).str.strip()
})

X = cat_feats.transform(filtered) # df(4095, 1995)
feature_names = cat_feats.feature_names()

# ---------------------------------------------- per-strain train/val/test
def xy(df, strain):
    """
    X and y for one strain's rows of one split, rows aligned.
    """
    sub = df[df["strain"] == strain]
    return X.loc[sub["UID"]], sub["fitness_score"].to_numpy()

data = {strain: {name: xy(df, strain) for name, df in splits.items()} for strain in strains}

# ---------------------------------------------------------------- models

# Every strain reads rows out of the same X, so only y changes between models.
ALPHAS = list(np.logspace(1, 5, 9))
WEIGHT_ALPHA = min(ALPHAS, key=lambda a: abs(a - 316.0))   # shared alpha, so the weights compare

def score(y, yhat):
    flat = np.ptp(yhat) == 0  # correlation is undefined for a constant
    return {
        "n": len(y),
        "r2": r2_score(y, yhat),
        "pearson": np.nan if flat else pearsonr(yhat, y)[0],
        "spearman": np.nan if flat else spearmanr(yhat, y)[0],
        "rmse": root_mean_squared_error(y, yhat),
    }

score_rows, weight_rows = [], []

for strain in strains:
    X_train, y_train = data[strain]["train"]
    X_val, y_val = data[strain]["val"]

    A_train = X_train.to_numpy(dtype=np.float64)   # convert once, not per alpha
    A_val = X_val.to_numpy(dtype=np.float64)
    ref = y_train.mean()                           # the per-strain mean baseline

    score_rows.append({
        "strain": strain, "model": "strain mean", "alpha": np.nan, "split": "val",
        **score(y_val, np.full(len(y_val), ref)),
    })

    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(A_train, y_train)

        for split_name, A, y in (("train", A_train, y_train), ("val", A_val, y_val)):
            score_rows.append({
                "strain": strain, "model": "ridge", "alpha": alpha, "split": split_name,
                **score(y, model.predict(A)),
            })

        if alpha == WEIGHT_ALPHA:
            weight_rows.append(pd.DataFrame({
                "strain": strain,
                "feature": feature_names + ["intercept"],
                "weight": np.r_[model.coef_, model.intercept_],
            }))

    print(f"fit {strain}")


# ---------------------------------------------------------------- output
scores = pd.DataFrame(score_rows)
weights = pd.concat(weight_rows, ignore_index=True)

scores.to_csv(f"{OUTPUT_DIR}/linear_scores_{args.split}.tsv", sep="\t", index=False)
weights.to_csv(f"{OUTPUT_DIR}/linear_weights_{args.split}.tsv", sep="\t", index=False)

ridge = scores[scores.model == "ridge"]

print("\nr2, mean across the fifteen strains (range)")
for split_name in ("train", "val"):
    print(f"  {split_name}")
    for alpha, g in ridge[ridge.split == split_name].groupby("alpha"):
        print(f"    alpha {alpha:9.1f}  {g.r2.mean():7.4f}  "
              f"[{g.r2.min():7.4f}, {g.r2.max():7.4f}]")

# Per strain: the baseline (strain mean) and the best ridge on val.
best = (ridge[ridge.split == "val"]
        .loc[lambda d: d.groupby("strain").r2.idxmax()]
        .set_index("strain")[["alpha", "r2", "spearman"]]
        .rename(columns={"r2": "ridge_r2", "spearman": "ridge_rho"}))

val_r2 = scores[scores.split == "val"].set_index(["model", "strain"])["r2"]
summary = (best.join(val_r2.loc["strain mean"].rename("baseline_r2"))
               .sort_values("ridge_r2", ascending=False))

print("\nval per strain")
print(summary.round(4).to_string())
print(f"\nmean ridge r2 {summary.ridge_r2.mean():.4f}")

# Weights, at the one shared alpha.
top = (weights[weights.feature != "intercept"]
       .groupby("feature")["weight"]
       .agg(mean_weight="mean", sd_across_strains="std"))
print(f"\ntop 15 features by |mean weight| across strains, alpha = {WEIGHT_ALPHA:.1f}")
print(top.reindex(top.mean_weight.abs().sort_values(ascending=False).index)
         .head(15).round(4).to_string())