"""
One-hot design matrices for gene and strain identity.

Encoders are fit on training rows only. Fitting on the full table would put the
identity of the val/test genes into the feature space, which is what the
cold-gene split exists to prevent.

drop="first" removes one level per block to avoid the dummy variable trap, so an
unseen gene arrives as an all-zero row. What to do about that is the caller's
problem, not the encoder's. unseen_genes() tells you which rows those are.

Blocks come back as CSR (Compressed Sparse Row matrix). "both" is the gene block followed by the strain block.
"""

import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder

GENE_COL = "UID"
STRAIN_COL = "strain"

# An all-zero row for an unseen gene is intended, so the warning about it is noise.
warnings.filterwarnings("ignore", message="Found unknown categories")


class OneHotDesign:
    def fit(self, train):
        self.gene_encoder = OneHotEncoder(drop="first", handle_unknown="ignore")
        self.gene_encoder.fit(train[[GENE_COL]])

        self.strain_encoder = OneHotEncoder(drop="first", handle_unknown="ignore")
        self.strain_encoder.fit(train[[STRAIN_COL]])

        self.train_genes = frozenset(train[GENE_COL])
        return self

    def transform(self, df, block):
        
        Xg = sp.csr_matrix(self.gene_encoder.transform(df[[GENE_COL]]))
        Xs = sp.csr_matrix(self.strain_encoder.transform(df[[STRAIN_COL]]))
        
        if block == "gene":
            return Xg
        if block == "strain":
            return Xs
        if block == "both":
            return sp.hstack([Xg, Xs], format="csr")
        
        raise ValueError(f"block must be gene, strain, or both. got {block!r}")

    def feature_names(self, block):
        """
        Column names aligned to transform(df, block).
        """
        
        gene = [f"gene={n.removeprefix(GENE_COL + '_')}"
                for n in self.gene_encoder.get_feature_names_out([GENE_COL])]
        strain = [f"strain={n.removeprefix(STRAIN_COL + '_')}"
                  for n in self.strain_encoder.get_feature_names_out([STRAIN_COL])]
        return {"gene": gene, "strain": strain, "both": gene + strain}[block]

    def n_gene_cols(self):
        """
        Width of the gene block, for slicing the additive model's coefficients.
        """
        
        return len(self.gene_encoder.categories_[0]) - 1

    def unseen_genes(self, df):
        """
        True where the row's gene was not in the training set.
        """
        
        return ~df[GENE_COL].isin(self.train_genes).to_numpy()
    @property
    def gene_levels(self):
        return self.gene_encoder.categories_[0]

    @property
    def strain_levels(self):
        return self.strain_encoder.categories_[0]



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
