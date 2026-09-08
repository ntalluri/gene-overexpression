"""
test_statistics.py
------------------
Regression tests for the statistical core. These need no data and no network;
run them after touching step05, step10 or enrichment.py.

    python src/test_statistics.py

What is checked
---------------
1. The negative binomial deviance equals 2*(saturated log-likelihood minus model
   log-likelihood). An earlier version of this code had a sign error inside the
   log that these tests would have caught.
2. The two-group closed form used by step 05 gives the same fitted values,
   coefficients and likelihood ratio statistic as a full GLM fit.
3. The IRLS fit used by step 10 matches statsmodels on a multi-coefficient
   design, and its likelihood ratio statistic matches too.
4. The sum-to-zero design really does compare each strain against the mean.
5. Benjamini-Hochberg matches a direct implementation of the step-up procedure.
6. The enrichment engine finds planted signal and rejects noise.

statsmodels is used as the reference. It is in requirements.txt.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

TOL = 1e-6
results = []


def check(name: str, passed: bool, detail: str = ""):
    results.append((name, passed, detail))
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))


def test_deviance_identity():
    from step05_fitness import nb_deviance, nb_loglik

    rng = np.random.default_rng(2)
    y = rng.negative_binomial(10, 10 / 310, (8, 6)).astype(float)
    worst = 0.0
    for disp in (0.01, 0.05, 0.2, 0.8, 2.0):
        mu = np.full_like(y, 300.0)
        analytic = nb_deviance(y, mu, disp)
        from_loglik = 2 * (nb_loglik(y, np.maximum(y, 1e-8), disp)
                           - nb_loglik(y, mu, disp))
        worst = max(worst, float(np.max(np.abs(analytic - from_loglik))))
    check("NB deviance equals 2*(saturated - model) log-likelihood",
          worst < 1e-8, f"max diff {worst:.2e}")


def test_two_group_closed_form():
    import statsmodels.api as sm
    from step05_fitness import group_means, nb_deviance

    rng = np.random.default_rng(7)
    X = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
    gidx = [np.arange(3), np.arange(3, 6)]

    worst_mu, worst_lfc, worst_lr = 0.0, 0.0, 0.0
    for _ in range(12):
        disp = float(rng.choice([0.01, 0.1, 0.5]))
        m0, m1 = rng.uniform(50, 5000), rng.uniform(50, 5000)
        y = np.concatenate([
            rng.negative_binomial(1 / disp, (1 / disp) / ((1 / disp) + m0), 3),
            rng.negative_binomial(1 / disp, (1 / disp) / ((1 / disp) + m1), 3),
        ]).astype(float)[None, :]

        mu_full = group_means(y, gidx)
        mu_null = np.repeat(y.mean(axis=1, keepdims=True), 6, axis=1)
        our_lr = float((nb_deviance(y, mu_null, disp)
                        - nb_deviance(y, mu_full, disp))[0])
        our_lfc = float(np.log2(y[0, 3:].mean() / y[0, :3].mean()))

        fam = sm.families.NegativeBinomial(alpha=disp)
        full = sm.GLM(y[0], X, family=fam).fit()
        null = sm.GLM(y[0], np.ones((6, 1)), family=fam).fit()

        worst_mu = max(worst_mu, float(np.max(np.abs(mu_full[0] - full.fittedvalues))))
        worst_lfc = max(worst_lfc, abs(our_lfc - full.params[1] / np.log(2)))
        worst_lr = max(worst_lr, abs(our_lr - 2 * (full.llf - null.llf)))

    check("step05 closed-form fitted values match a full GLM fit",
          worst_mu < 1e-6, f"max diff {worst_mu:.2e}")
    check("step05 log2FC matches the GLM coefficient",
          worst_lfc < TOL, f"max diff {worst_lfc:.2e}")
    check("step05 likelihood ratio matches 2*delta log-likelihood",
          worst_lr < 1e-6, f"max diff {worst_lr:.2e}")


def test_irls_multicoef():
    import statsmodels.api as sm
    from step10_rnaseq import build_design, fit_nb_glm

    rng = np.random.default_rng(4)
    samples = pd.DataFrame([{"sample_id": f"{s}_{r}", "strain": s, "replicate": r}
                            for s in "ABCD" for r in "123"])
    X, names, strains, cols = build_design(samples)

    check("sum-to-zero strain columns each sum to zero",
          bool(np.allclose(X[:, 3:].sum(axis=0), 0)))

    offset = np.log(rng.uniform(0.8, 1.25, len(samples)))
    disp = 0.15
    drop = cols["A"]
    X_red = np.delete(X, drop, axis=1)

    worst_beta, worst_lr = 0.0, 0.0
    for _ in range(8):
        mu = np.exp(X @ rng.normal(0, 0.5, X.shape[1]) + offset) * 300
        y = rng.negative_binomial(1 / disp, (1 / disp) / ((1 / disp) + mu)).astype(float)

        beta, dev_full, _ = fit_nb_glm(y, X, offset, disp)
        _, dev_red, _ = fit_nb_glm(y, X_red, offset, disp)

        fam = sm.families.NegativeBinomial(alpha=disp)
        full = sm.GLM(y, X, family=fam, offset=offset).fit()
        red = sm.GLM(y, X_red, family=fam, offset=offset).fit()

        worst_beta = max(worst_beta, float(np.max(np.abs(beta - full.params))))
        worst_lr = max(worst_lr, abs((dev_red - dev_full) - 2 * (full.llf - red.llf)))

    check("step10 IRLS coefficients match statsmodels",
          worst_beta < 1e-4, f"max diff {worst_beta:.2e}")
    check("step10 likelihood ratio matches 2*delta log-likelihood",
          worst_lr < 1e-5, f"max diff {worst_lr:.2e}")


def test_sum_to_zero_meaning():
    """A strain coefficient should be that strain's deviation from the mean."""
    from step10_rnaseq import build_design

    samples = pd.DataFrame([{"sample_id": f"{s}_{r}", "strain": s, "replicate": r}
                            for s in "ABCD" for r in "123"])
    X, names, strains, cols = build_design(samples)

    # Build a response with known per-strain offsets summing to zero.
    true = {"A": 0.4, "B": -0.1, "C": -0.5, "D": 0.2}
    eta = np.array([true[s] for s in samples["strain"]]) + 1.0
    beta, *_ = np.linalg.lstsq(X, eta, rcond=None)

    recovered = {s: beta[cols[s]] for s in strains[:-1]}
    recovered[strains[-1]] = -sum(recovered.values())
    worst = max(abs(recovered[s] - true[s]) for s in true)
    check("sum-to-zero coefficients recover each strain's deviation from the mean",
          worst < 1e-8, f"max diff {worst:.2e}")


def test_bh():
    from step05_fitness import benjamini_hochberg

    rng = np.random.default_rng(1)
    p = np.concatenate([rng.uniform(0, 1e-4, 20), rng.uniform(0, 1, 200)])

    def reference(pv):
        n = len(pv)
        order = np.argsort(pv)
        out = np.empty(n)
        running = 1.0
        for rank in range(n - 1, -1, -1):
            running = min(running, pv[order[rank]] * n / (rank + 1))
            out[order[rank]] = running
        return np.minimum(out, 1.0)

    diff = float(np.max(np.abs(benjamini_hochberg(p) - reference(p))))
    check("Benjamini-Hochberg matches a direct step-up implementation",
          diff < 1e-12, f"max diff {diff:.2e}")
    check("BH output is monotone in the p-value ranking",
          bool(np.all(np.diff(np.sort(benjamini_hochberg(p))) >= -1e-12)))


def test_enrichment_engine():
    from enrichment import (hypergeometric_enrichment, overlap_test,
                            wilcoxon_enrichment)

    rng = np.random.default_rng(11)
    universe = {f"G{i:04d}" for i in range(4000)}
    query = {f"G{i:04d}" for i in range(400)}

    planted = ({f"G{i:04d}" for i in range(200)}
               | {f"G{i:04d}" for i in range(3000, 3100)})
    noise = set(rng.choice(sorted(universe), 300, replace=False))

    h = hypergeometric_enrichment(query, universe,
                                  {"planted": planted, "noise": noise})
    h = h.set_index("category")
    check("hypergeometric detects planted categorical enrichment",
          bool(h.loc["planted", "passes_p_threshold"]),
          f"p={h.loc['planted', 'pvalue']:.1e}")
    check("hypergeometric rejects a random category",
          not bool(h.loc["noise", "passes_p_threshold"]),
          f"p={h.loc['noise', 'pvalue']:.2f}")

    feat = pd.DataFrame(index=sorted(universe))
    feat["noise"] = rng.normal(0, 1, len(feat))
    feat["planted"] = rng.normal(0, 1, len(feat))
    feat.loc[sorted(query), "planted"] += 1.0

    w = wilcoxon_enrichment(query, universe, feat).set_index("feature")
    check("Wilcoxon detects a planted continuous shift",
          bool(w.loc["planted", "passes_p_threshold"])
          and w.loc["planted", "direction"] == "higher",
          f"p={w.loc['planted', 'pvalue']:.1e}")
    check("Wilcoxon rejects a random feature",
          not bool(w.loc["noise", "passes_p_threshold"]),
          f"p={w.loc['noise', 'pvalue']:.2f}")

    o = overlap_test(query, planted, universe)
    check("overlap_test reports the correct observed and expected counts",
          o["overlap"] == 200 and abs(o["expected_overlap"] - 30.0) < 1e-6)


def test_edgepython_backend_agreement():
    """If edgepython is installed, its glm_lrt should track the built-in backend.

    They will not be identical: edgePython adds edgeR's empirical Bayes
    dispersion moderation and a prior_count that shrinks fold changes for
    near-zero genes, while the built-in backend uses the raw ratio. What must
    hold is that they agree on the ranking (Spearman near 1) and on the bulk of
    the fold changes once the handful of extreme imputed-gene estimates are set
    aside. This test would flag a wrapper that passed the wrong coefficient,
    transposed the matrix, or mismatched the design.
    """
    try:
        import edgepython as ep
    except ImportError:
        check("edgepython backend agreement", True,
              "edgepython not installed, skipped")
        return

    from scipy.stats import spearmanr
    from step05_fitness import (benjamini_hochberg, glm_lrt, group_means,
                                nb_deviance, run_strain_edgepython)

    rng = np.random.default_rng(3)
    n_genes = 500
    base = rng.lognormal(0, 1, n_genes)
    base /= base.sum()
    lfc = np.zeros(n_genes)
    lfc[:80] = -rng.uniform(1, 3, 80)
    lfc[80:110] = rng.uniform(1, 2, 30)

    def draw(weights, tot=1e6):
        w = weights / weights.sum()
        return rng.multinomial(int(tot), w)

    cols = [f"g0_{i}" for i in range(3)] + [f"g10_{i}" for i in range(3)]
    data = np.array([draw(base) for _ in range(3)]
                    + [draw(base * 2 ** lfc) for _ in range(3)]).T.astype(float)
    data = data / data.sum(axis=0) * 1e6
    genes = [f"Y{i:05d}" for i in range(n_genes)]
    sub = pd.DataFrame(data, index=genes, columns=cols)
    g0, g10 = cols[:3], cols[3:]

    # Built-in backend (closed-form point estimate + grid-search dispersion).
    y = sub[cols].to_numpy(dtype=float)
    gidx = [np.arange(3), np.arange(3, 6)]
    m0 = np.maximum(y[:, :3].mean(axis=1), 1e-8)
    m1 = np.maximum(y[:, 3:].mean(axis=1), 1e-8)
    ours_logfc = np.log2(m1 / m0)

    ep_res = run_strain_edgepython(sub, g0, g10)
    their_logfc = ep_res["logFC"].to_numpy()

    finite = np.isfinite(ours_logfc) & np.isfinite(their_logfc)
    rho = spearmanr(ours_logfc[finite], their_logfc[finite]).statistic
    check("edgepython logFC ranking matches the built-in backend",
          rho > 0.98, f"Spearman={rho:.4f}")

    # Bulk Pearson after dropping the few most-extreme disagreements.
    diff = np.abs(ours_logfc[finite] - their_logfc[finite])
    order = np.argsort(diff)
    keep = order[:-5] if len(order) > 20 else order
    r = np.corrcoef(ours_logfc[finite][keep], their_logfc[finite][keep])[0, 1]
    check("edgepython logFC matches the built-in backend on the bulk of genes",
          r > 0.97, f"Pearson(trim 5)={r:.4f}")


def main() -> int:
    print("Running statistical regression tests\n")
    for fn in (test_deviance_identity, test_two_group_closed_form,
               test_irls_multicoef, test_sum_to_zero_meaning,
               test_bh, test_enrichment_engine,
               test_edgepython_backend_agreement):
        print(f"{fn.__name__}:")
        try:
            fn()
        except Exception as exc:
            check(fn.__name__, False, f"raised {type(exc).__name__}: {exc}")
        print()

    failed = [name for name, ok, _ in results if not ok]
    print("=" * 60)
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("\nFailed:")
        for name in failed:
            print(f"  {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
