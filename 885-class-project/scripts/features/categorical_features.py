"""
Gene annotation features from the curated categorical table.

Source: Features_categorical_Table_v3.1.xlsx, one row per gene and 1993
annotation columns, with Feature_Table_data_info_v3.1.xlsx giving the citation
for each. Every column is already a 0/1 indicator (GO terms, RBP targets,
regulator targets, and so on), so nothing needs encoding. The work is pruning
and deciding what a missing value means.

A NaN means the gene was not scored by that study, not that the gene is a
negative. Filling with 0 merges the two, which is the usual compromise for
sparse membership data, so every gene also carries coverage: the fraction
of the table that was actually measured for it. That lets a model tell an
annotation-poor gene from one that is genuinely negative everywhere.

Pruning is fit on train genes only. Dropping a feature because it is rare
across the whole table lets held-out genes shape the feature space, which is
what the cold-gene split exists to prevent.

The annotation table is per gene. transform() reindexes it onto a long frame's
UID column, so the result has one row per row of df, matching OneHotDesign. A
gene's annotations repeat across its 15 strain rows.
"""

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

FEATURES = "data/genes/features/Features_categorical_Table.csv"
TABLE_GENE_COL = "gene"  # gene column in the annotation table
GENE_COL = "UID"         # gene column in the split frames


def load_table(path=FEATURES):
    """
    Categorial feature table indexed by gene.
    """
    table = pd.read_csv(path)
    table = table.set_index(TABLE_GENE_COL)

    assert table.index.is_unique, "annotation table has duplicate gene rows"

    # Every cell should be 0, 1, or NaN. A continuous column would be silently
    # truncated by the int8 cast in give_me_feature_rows_for_these_genes, so fail loudly here instead.
    bad = ~table.isin([0, 1]) & table.notna()
    if bad.to_numpy().any():
        cols = bad.any(axis=0)
        print(f"{int(bad.to_numpy().sum())} cells are not 0 or 1, "
              f"across {int(cols.sum())} of {table.shape[1]} columns")
        for c in table.columns[cols][:10]:
            print(f"  {c}: {pd.unique(table.loc[bad[c], c])[:5]}")
        raise AssertionError("table is not all 0/1")
    
    return table


class CategoricalFeatures:
    """
    Pruned 0/1 annotation features, plus two summary columns.

      cat_n_positive   how many of the kept annotations this gene has
      cat_coverage     fraction of the full table measured for this gene,
                       counted over every column, not just the kept ones

    Those two are on a different scale from the 0/1 block, so standardize
    before any penalized model.
    """

    def __init__(self):
        self.table = load_table()

    # ---------------------------------------------------------------- fit

    def fit(self, train):
        """
        Record the feature columns and their train prevalence.

        Nothing is pruned. Every annotation column is kept.
        """
        genes = train[GENE_COL].drop_duplicates()
        annotations, _ = self.give_me_feature_rows_for_these_genes(genes)

        self.keep = list(annotations.columns)
        self.train_positives = annotations.sum(axis=0)
        self.n_train_genes = len(genes)

        # How many train genes each study actually scored, before the fill.
        # The per-column counterpart of cat_coverage, which is per gene.
        self.train_scored = self.table.reindex(genes).notna().sum(axis=0)

        # Columns with the same train pattern are indistinguishable to any
        # model. Not pruned, but recorded so the report can show them.
        groups = {}
        arr = annotations.to_numpy()
        for i, c in enumerate(self.keep):
            groups.setdefault(arr[:, i].tobytes(), []).append(c)
        self.duplicate_group = {c: i for i, g in enumerate(groups.values()) for c in g}
        self.duplicate_group_size = {c: len(g) for g in groups.values() for c in g}

        return self

    # ---------------------------------------------------------- transform
    
    def transform(self, df):
        """
        Feature frame, one row per row of df, in order, indexed by gene.

        Every column is 0/1, so nothing here needs scaling before a penalized
        model and every weight comes out on the same scale.
        """
        annotations, _ = self.give_me_feature_rows_for_these_genes(df[GENE_COL])

        out = annotations[self.keep].astype(np.float32)
        out.index = df[GENE_COL].to_numpy()
        return out

    def space_efficient_transform(self, df):
        """Same matrix as transform(), stored as CSR.
        called like: X = cat_feats.space_efficient_transform(filtered) """
        return sp.csr_matrix(self.transform(df).to_numpy())

    def feature_names(self):
        """Column names aligned to transform(df)."""
        return list(self.keep)
    
    # ------------------------------------------------------------ helper functions
    def coverage(self, df):
        """
        Fraction of the table actually measured per gene. A diagnostic, not a
        feature: it describes how much attention the literature paid to a gene,
        which is not a property of the gene.

        called like: 
        """
        _, cov = self.give_me_feature_rows_for_these_genes(df[GENE_COL])
        return pd.Series(cov, index=df[GENE_COL].to_numpy())
    
    def missing_genes(self, df):
        """
        Genes in df with no row in the annotation table.
        """
        genes = df[GENE_COL].drop_duplicates()
        return genes[~genes.isin(self.table.index)].to_numpy()


    def feature_report(self):
        """
        One row per feature, with what is known about it from the train genes.

        train_scored        train genes this study actually scored
        train_unscored      the rest, whose 0 in the matrix is a fill, not a result
        prevalence_scored   positives among the genes that were scored, which is
                            the honest prevalence when a study covered only part
                            of the genome
        duplicate_group     features with the same train pattern share an id
        constant            flags the columns that carry no information at all 
                            in train but still get a weight.
        """
        pos = self.train_positives.to_numpy()
        scored = self.train_scored.reindex(self.keep).to_numpy()

        report = pd.DataFrame({
            "feature": self.keep,
            "train_positives": pos,
            "train_prevalence": pos / self.n_train_genes,
            "train_scored": scored,
            "train_unscored": self.n_train_genes - scored,
            "prevalence_scored": np.divide(pos, scored, out=np.full(len(pos), np.nan),
                                           where=scored > 0),
            "duplicate_group": [self.duplicate_group[c] for c in self.keep],
            "duplicate_group_size": [self.duplicate_group_size[c] for c in self.keep],
        })
        report["constant"] = (report.train_positives == 0) | \
                             (report.train_positives == self.n_train_genes)

        # Secondary sort on name so tied counts come out in a stable order.
        return report.sort_values(["train_positives", "feature"],
                                  ascending=[False, True]).reset_index(drop=True)
    
    def give_me_feature_rows_for_these_genes(self, gene_ids):
        """
        Reindex self.table onto gene_ids, one row per id, in the order given.

        gene_ids is any column of gene names: a long frame's UID column (with a
        gene repeated once per strain) or a per-gene frame. Repeats are kept.

        Turns every NaN into a 0. A NaN cell means that study never scored this
        gene, which is not the same as scoring it negative, and a gene with no
        row in the table at all comes back as a row of NaNs. Both end up as 0.

        Returns:
          filled    the annotation rows, NaNs replaced by 0
          coverage  fraction of each row that was not NaN, measured before the
                    fill. The only surviving record of which zeros are real.
        """
        rows = self.table.reindex(gene_ids)
        
        coverage = rows.notna().mean(axis=1).to_numpy(dtype=np.float32)
        filled = rows.fillna(0).astype(np.int8)

        return filled, coverage