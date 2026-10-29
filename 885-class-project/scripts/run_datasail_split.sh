#!/bin/bash
# Leakage-controlled dataset splitter (replaces run_protein_clustering.sh).
#
# The previous script clustered proteins and stopped at cluster memberships.
# That was the wrong end product: the real goal is train/val/test splits with no
# similar target leaking across splits. DataSAIL does clustering + an integer
# program that assigns whole clusters to splits, so we drive DataSAIL directly
# and let it emit the splits.
#
# Method/technique tokens below are taken from DataSAIL's own source
# (datasail/settings.py: SIM_ALGOS, TEC_*), not from memory:
#   --e-sim  : mmseqs | mmseqs2 | mmseqspp | cdhit | foldseek | diamond | <matrix.tsv>
#   -t       : C1e (similarity cold-split, protein entity) | I1e (identity) | R (random)
#
# NOTE (HTCondor): runs inside the datasail docker image, which has datasail +
# the backend binaries + SCIP on PATH (/opt/conda/bin via the image's ENV PATH).

set -euo pipefail

# ---------------------------------------------------------------------------
# Scratch root. In HTCondor DOCKER universe the job starts in the Condor scratch
# directory, and transfer_output_files reads ONLY from there. _CONDOR_SCRATCH_DIR
# is NOT reliably set inside the container, so we capture $PWD here, at the top,
# BEFORE anything can cd. Everything we want transferred back must live under this
# path. This is the single source of truth for "where Condor looks."
# ---------------------------------------------------------------------------
SCRATCH_ROOT="$PWD"

# Keep matplotlib's cache off $HOME (not writable in the container); silences the
# MPLCONFIGDIR warnings DataSAIL emits when it renders the cluster plots.
export MPLCONFIGDIR="${SCRATCH_ROOT}/mpl"; mkdir -p "${MPLCONFIGDIR}"

# ---------------------------------------------------------------------------
# Arguments (positional, to stay easy to wire from the .sub 'arguments =' line)
# ---------------------------------------------------------------------------
TECH="${1:-C1e}"                 # C1e = cold-target split (what you want). See above.
SIM="${2:-mmseqs2}"               # similarity source: a method token OR a matrix .tsv path.
SPLIT_SPEC="${3:-0.7:0.15:0.15}" # colon-separated fractions; names auto-derived below.

# --- Tunables via env (all have explained defaults) ---
INPUT_FASTA="${INPUT_FASTA:-clean.fasta}"
RUNS="${RUNS:-5}"                # -r: independent splits per technique, for variance
                                 #     reporting. 5 mirrors how you already run multi-seed.
EPSILON="${EPSILON:-0.1}"       # -e: allowed deviation from target split sizes. Cold
                                 #     splits move WHOLE clusters at once, so exact ratios
                                 #     are usually infeasible; 0 would frequently make the
                                 #     ILP unsolvable. 5% slack is a sane starting point.
SOLVER="${SOLVER:-SCIP}"         # open-source ILP solver that ships with the conda install.
                                 #     Change to GUROBI/CPLEX only if you have a license.
TIMEOUT="${TIMEOUT:-1800}"       # --to-sec: cap optimizer time (excludes clustering) so a
                                 #     hard instance can't burn the whole slot. 30 min.
THREADS="${THREADS:-$(nproc)}"   # forwarded to the backend clustering tool. 0 = all CPUs.

DELTA="${DELTA:-0.05}"           # -d: A multiplicative factor by how much the limits
                                 # (as defined in the -s / –splits argument defined)
                                 # of the stratification can be exceeded. Default is 0.05


echo "===== DataSAIL split job started ====="
date; hostname; pwd
echo "HTCondor ClusterId: ${CLUSTER:-unknown}"
echo "HTCondor ProcId:    ${PROCESS:-unknown}"
echo "Technique:          ${TECH}"
echo "Similarity source:  ${SIM}"
echo "Split spec:         ${SPLIT_SPEC}"
echo "Runs / epsilon:     ${RUNS} / ${EPSILON}"
echo "Solver / timeout:   ${SOLVER} / ${TIMEOUT}s"

command -v datasail >/dev/null 2>&1 || { echo "ERROR: datasail not on PATH." >&2; exit 127; }
if [[ ! -s "${INPUT_FASTA}" ]]; then
    echo "ERROR: input FASTA '${INPUT_FASTA}' is missing or empty." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Validate technique and resolve split fractions -> names
# ---------------------------------------------------------------------------
case "${TECH}" in
  C1e|I1e|R) : ;;   # supported single-entity protein techniques
  *) echo "ERROR: technique '${TECH}' unsupported here. Use C1e, I1e, or R." >&2; exit 1 ;;
esac

IFS=':' read -ra FRACS <<< "${SPLIT_SPEC}"
case "${#FRACS[@]}" in
  2) NAMES=(train test) ;;
  3) NAMES=(train val test) ;;
  *) echo "ERROR: split spec must have 2 or 3 fractions (got ${#FRACS[@]})." >&2; exit 1 ;;
esac

# Validate the similarity source: either a known method token or an existing file.
# (For R, similarity is irrelevant and omitted entirely below.)
KNOWN_SIM="mmseqs mmseqs2 mmseqspp cdhit foldseek diamond"
if [[ "${TECH}" != "R" ]]; then
    if [[ -f "${SIM}" ]]; then
        echo "Using precomputed similarity matrix: ${SIM}"
    elif [[ " ${KNOWN_SIM} " == *" ${SIM} "* ]]; then
        echo "Using built-in similarity method: ${SIM}"
    else
        echo "ERROR: '${SIM}' is neither a known method (${KNOWN_SIM}) nor a file." >&2
        exit 1
    fi
fi

# OUTDIR lives UNDER the scratch root so its contents are transferable. Note it is
# output/<tag>/ ; the whole output/ tree is what we archive below.
OUTDIR="${SCRATCH_ROOT}/output/datasail_${TECH}_${SIM##*/}_${CLUSTER:-X}.${PROCESS:-Y}"
mkdir -p "${OUTDIR}"

echo; echo "===== DataSAIL version ====="; datasail --version || true

# ---------------------------------------------------------------------------
# Build and run the DataSAIL command.
# Command assembled as an array so flags can be included conditionally without
# fragile string interpolation.
# ---------------------------------------------------------------------------
CMD=(datasail
    --output    "${OUTDIR}"
    --techniques "${TECH}"
    --splits    "${FRACS[@]}"
    --names     "${NAMES[@]}"
    --e-type    P # this makes the proteins the e-entity type
    --e-data    "${INPUT_FASTA}"
    --runs      "${RUNS}"
    --epsilon   "${EPSILON}"
    --delta     "${DELTA}"
    --overflow  assign            # keep each cluster WHOLE in one split. 'break' would
                                  # slice an oversized cluster across splits and reintroduce
                                  # exactly the leakage we're removing, so we forbid it.
    --solver    "${SOLVER}"
    --to-sec    "${TIMEOUT}"
    --threads   "${THREADS}"
    --verbose   I
)
# Similarity source only matters for identity/cluster techniques, not random.
if [[ "${TECH}" != "R" ]]; then
    CMD+=(--e-sim "${SIM}")
fi

echo; echo "===== Running DataSAIL ====="
printf '%q ' "${CMD[@]}"; echo
"${CMD[@]}"

[ -d mmseqspp_results ] && cp mmseqspp_results/alis.tsv "${OUTDIR}/" || true


# ---------------------------------------------------------------------------
# Leakage evaluation with DataSAIL's eval_split (needs DataSAIL >= 1.3.0).
# Uses 'mmseqspp' (pairwise MMseqs2 similarities). 'mmseqs2' only gives
# cluster-level similarity and scores a good split the same as a random one.
#
# Writes to ${OUTDIR}:
#   leakage.tsv            scaled_dl = cross-set similarity / total similarity
#                          (lower = less leakage), split by set pair; the
#                          set-pair columns add up to scaled_dl.
#   leakage_threshold.tsv  per-protein check: how many val/test proteins have a
#                          close relative (identity >= THRESHOLDS) in another set.
#                          Paralogs from yeast's genome duplication can share
#                          sequence and OE effects; with one in train and one in
#                          test, a model could predict the test gene from its copy.
#   mmseqspp_alignments.tsv.gz  the pairwise alignment table (query, target,
#                          fident) that both checks use.
#   C1e_*/cross_set_pairs.tsv  every protein pair in different sets with
#                          identity >= the lowest threshold, highest first.
# Each check is also run on a random split of the same sizes, for comparison.
# ---------------------------------------------------------------------------
THRESHOLDS="${THRESHOLDS:-0.5}"   # identity cutoff(s), e.g. "0.3 0.5 0.7"

echo; echo "===== Evaluating leakage ====="
echo "Leakage evaluation: starting"
if python3 - "${OUTDIR}" "${INPUT_FASTA}" "${NAMES[*]}" "${THRESHOLDS}" << 'PYEOF'
import glob, itertools, os, sys
import numpy as np
import pandas as pd
import datasail.eval as ds_eval

# Wrap DataSAIL's clustering step to (1) make the matrix writable (DataSAIL 1.3.0
# + pandas >= 3 gives a read-only one) and (2) keep a copy for our own checks.
captured = {}
_cluster = ds_eval.cluster
def _wrapped(ds, **kw):
    ds = _cluster(ds, **kw)
    ds.cluster_similarity = np.array(ds.cluster_similarity)
    captured["names"] = np.array(ds.cluster_names)
    captured["sim"] = ds.cluster_similarity.copy()
    return ds
ds_eval.cluster = _wrapped
eval_split = ds_eval.eval_split

outdir, fasta, split_names = sys.argv[1], sys.argv[2], sys.argv[3].split()
thresholds = sorted(float(t) for t in sys.argv[4].split())
pairs = list(itertools.combinations(split_names, 2))   # train-val, train-test, val-test

# mmseqspp sorts its similarity matrix by ID but labels it in FASTA order,
# so evaluate on a copy of the FASTA sorted by ID.
seqs, name = {}, None
for line in open(fasta):
    line = line.strip()
    if line.startswith(">"):
        name = line[1:]
        seqs[name] = ""
    elif line:
        seqs[name] += line
fasta = "eval_sorted.fasta"
with open(fasta, "w") as out:
    for n in sorted(seqs):
        out.write(f">{n}\n{seqs[n]}\n")

threshold_rows = []

def evaluate(label, assign, run_dir=None):
    scaled, total, max_dl = eval_split("P", fasta, None, "mmseqspp", None, None, assign)[:3]
    names, S = captured["names"], captured["sim"]

    # first call only: save the alignment table and build a symmetric copy
    # (A vs B and B vs A can differ slightly; keep the higher one, drop self-pairs)
    if "sym" not in captured:
        r, c = np.nonzero(S)
        pd.DataFrame({"query": names[r], "target": names[c], "fident": S[r, c]}).to_csv(
            os.path.join(outdir, "mmseqspp_alignments.tsv.gz"), sep="\t", index=False)
        sym = np.maximum(S, S.T).astype(np.float32)
        np.fill_diagonal(sym, 0)
        captured["sym"] = sym
    sym = captured["sym"]

    lab = np.array([assign.get(n, "") for n in names])
    row = {"run": label, "scaled_dl": scaled}
    for a, b in pairs:
        ia, ib = lab == a, lab == b
        row[f"{a}-{b}"] = (S[np.ix_(ia, ib)].sum() + S[np.ix_(ib, ia)].sum()) / max_dl
        # per-protein check: closest relative in set a for each protein in set b
        closest = sym[np.ix_(ib, ia)].max(axis=1) if ia.any() and ib.any() else np.zeros(ib.sum())
        for t in thresholds:
            n_with = int((closest >= t).sum())
            threshold_rows.append({
                "run": label, "set_pair": f"{a}-{b}", "threshold": t,
                "checked_set": b, "n_checked": int(ib.sum()),
                "n_with_relative": n_with,
                "n_without_relative": int(ib.sum()) - n_with,
                "pct_with_relative": round(100 * n_with / max(int(ib.sum()), 1), 2)})
    if abs(sum(row[f"{a}-{b}"] for a, b in pairs) - scaled) > 1e-6:
        print(f"WARNING: set-pair breakdown for {label} does not add up to scaled_dl", flush=True)
    row.update(total_dl=total, max_dl=max_dl)

    # list every cross-set pair at or above the lowest threshold
    if run_dir is not None:
        i, j = np.nonzero(np.triu(sym >= thresholds[0], 1))
        keep = np.isin(lab[i], split_names) & np.isin(lab[j], split_names) & (lab[i] != lab[j])
        i, j = i[keep], j[keep]
        pd.DataFrame({"protein_1": names[i], "set_1": lab[i],
                      "protein_2": names[j], "set_2": lab[j],
                      "identity": np.round(sym[i, j], 3)}).sort_values(
            "identity", ascending=False).to_csv(
            os.path.join(run_dir, "cross_set_pairs.tsv"), sep="\t", index=False)
    return row

rows = []
files = sorted(glob.glob(os.path.join(outdir, "*", "*_splits.tsv")))
print(f"Leakage evaluation: found {len(files)} split file(s)", flush=True)
if not files:
    sys.exit("ERROR: no *_splits.tsv files found")
for f in files:
    print(f"Leakage evaluation: scoring {f}", flush=True)
    sp = pd.read_csv(f, sep="\t", dtype=str)
    assign = dict(zip(sp.ID, sp.Split))   # must be a dict; a file path fails in v1.4.0
    rows.append(evaluate(os.path.basename(os.path.dirname(f)), assign, os.path.dirname(f)))

# baseline: shuffle the last run's labels (same split sizes, no similarity control)
print("Leakage evaluation: scoring random baseline", flush=True)
shuffled = dict(zip(sp.ID, np.random.default_rng(0).permutation(sp.Split.to_numpy())))
rows.append(evaluate("random_baseline", shuffled))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(outdir, "leakage.tsv"), sep="\t", index=False)
print(df.to_string(index=False, float_format="%.4f"))

th = pd.DataFrame(threshold_rows)
th.to_csv(os.path.join(outdir, "leakage_threshold.tsv"), sep="\t", index=False)
print("\nProteins in the second set with a relative in the first set at identity >= threshold:")
print(th.to_string(index=False))
PYEOF
then
    echo "Leakage evaluation: DONE -> ${OUTDIR}/leakage.tsv, leakage_threshold.tsv"
else
    echo "WARNING: leakage evaluation FAILED (splits are still packaged)." >&2
fi


# ---------------------------------------------------------------------------
# Package results for transfer.
# Destination = SCRATCH_ROOT (where Condor looks). Member = the relative path
# 'output', archived from SCRATCH_ROOT so no absolute paths are embedded and the
# archive does NOT try to include itself (it sits beside output/, not inside it).
# transfer_output_files in the .sub must name this exact file:
#     transfer_output_files = datasail_results_$(ClusterId).tar.gz
# CLUSTER defaults to 'local' so a bare `docker run` (no CLUSTER env) still works
# under set -u instead of aborting.
# ---------------------------------------------------------------------------
RESULTS_TARBALL="datasail_results_${CLUSTER:-local}.tar.gz"
tar -czf "${SCRATCH_ROOT}/${RESULTS_TARBALL}" -C "${SCRATCH_ROOT}" output
echo "Packaged results -> ${SCRATCH_ROOT}/${RESULTS_TARBALL}"

echo; echo "===== Done ====="
echo "Splits, reports, and logs written under: ${OUTDIR}"
echo "(DataSAIL writes one assignment table per technique/run plus a leakage report;"
echo " inspect the report before trusting the split — watch for a percolated mega-cluster"
echo " that forces an imbalanced split.)"
date
