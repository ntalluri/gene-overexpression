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
# After the split, the script runs ONE leakage check: a pairwise identity
# threshold. DataSAIL's own scaled_dl score is deliberately not computed, since
# a summed-similarity ratio does not tell you which genes to move.
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

# ===========================================================================
# ARGUMENTS
# ===========================================================================

# --- Positional (easy to wire from the .sub 'arguments =' line) ------------
TECH="${1:-C1e}"                 # C1e = cold-target split (what you want). See above.
SIM="${2:-mmseqs2}"              # similarity source: a method token OR a matrix .tsv path.
SPLIT_SPEC="${3:-0.7:0.15:0.15}" # colon-separated fractions; names auto-derived below.

# --- Split construction, via env -------------------------------------------
INPUT_FASTA="${INPUT_FASTA:-clean.fasta}"
RUNS="${RUNS:-15}"                # -r: independent splits per technique, for variance
                                 #     reporting. 5 mirrors how you already run multi-seed.
EPSILON="${EPSILON:-0.1}"        # -e: allowed deviation from target split sizes. Cold
                                 #     splits move WHOLE clusters at once, so exact ratios
                                 #     are usually infeasible; 0 would frequently make the
                                 #     ILP unsolvable. 10% slack is a sane starting point.
DELTA="${DELTA:-0.05}"           # -d: multiplicative factor by how much the stratification
                                 #     limits (set by --splits) may be exceeded. DataSAIL
                                 #     default is 0.05.
SOLVER="${SOLVER:-SCIP}"         # open-source ILP solver that ships with the conda install.
                                 #     Change to GUROBI/CPLEX only if you have a license.
TIMEOUT="${TIMEOUT:-1800}"       # --to-sec: cap optimizer time (excludes clustering) so a
                                 #     hard instance can't burn the whole slot. 30 min.
THREADS="${THREADS:-$(nproc)}"   # forwarded to the backend clustering tool. 0 = all CPUs.
NUM_CLASSES="${NUM_CLASSES:-200}" # --e-num-classes: number of cluster groups DataSAIL
                                 #     condenses the entity clusters into before the ILP.
                                 #     DataSAIL default is 50; more groups = finer-grained
                                 #     units to move between splits, so sizes can land
                                 #     closer to the requested 0.7/0.15/0.15. Costs ILP
                                 #     time. Irrelevant for TECH=R, so only added below.

# --- Leakage check, via env ------------------------------------------------
THRESHOLDS="${THRESHOLDS:-0.3}"  # identity cutoff(s), space-separated for several.
                                 #     0.3 is the usual cutoff in protein ML.
EVAL_SIM="${EVAL_SIM:-mmseqspp}" # similarity used by the CHECK (not by the split).
                                 #     Must be pairwise: 'mmseqs2' only gives cluster-level
                                 #     similarity and would score a good split like a
                                 #     random one.
TRAIN_SET="${TRAIN_SET:-}"       # which split absorbs the leaking genes. Empty = the
                                 #     first name in --names, i.e. 'train'.
BASELINE_REPS="${BASELINE_REPS:-15}"  # random splits to score for comparison. Each one
                                 #     reshuffles the first DataSAIL run's labels, so the
                                 #     set sizes are identical and only the similarity
                                 #     control is gone. 0 = skip the baseline.
BASELINE_SEED="${BASELINE_SEED:-}"   # seed for those shuffles. Empty = fresh
                                 #     randomness every run. The seed actually used
                                 #     is printed in the log either way, so a run
                                 #     can be reproduced after the fact by passing
                                 #     that number back in here.
SAVE_ALIGNMENTS="${SAVE_ALIGNMENTS:-1}"  # 1 = also write the full pairwise identity table
                                 #     (pairwise_identity.tsv.gz). 0 = skip it.

# ===========================================================================

echo "===== DataSAIL split job started ====="
date; hostname; pwd
echo "HTCondor ClusterId: ${CLUSTER:-unknown}"
echo "HTCondor ProcId:    ${PROCESS:-unknown}"
echo "Technique:          ${TECH}"
echo "Similarity source:  ${SIM}"
echo "Split spec:         ${SPLIT_SPEC}"
echo "Runs / epsilon:     ${RUNS} / ${EPSILON}"
echo "Solver / timeout:   ${SOLVER} / ${TIMEOUT}s"
echo "Cluster groups:     ${NUM_CLASSES} (--e-num-classes; DataSAIL default is 50)"
echo "Leakage check:      ${EVAL_SIM} at identity >= ${THRESHOLDS}"
echo "Random baselines:   ${BASELINE_REPS} (seed ${BASELINE_SEED:-none, drawn at run time})"

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
TRAIN_SET="${TRAIN_SET:-${NAMES[0]}}"

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
    --e-type    P                 # proteins are the e-entity type
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
    CMD+=(--e-num-classes "${NUM_CLASSES}")
fi

echo; echo "===== Running DataSAIL ====="
printf '%q ' "${CMD[@]}"; echo
"${CMD[@]}"

# ---------------------------------------------------------------------------
# Threshold leakage check.
#
# Computes all-vs-all sequence identity ONCE with EVAL_SIM, then asks of every
# split DataSAIL produced: which genes have a close relative in a DIFFERENT set?
# Paralogs from yeast's genome duplication can share sequence and OE effects;
# with one copy in train and the other in test, a model can predict the test
# gene from its copy.
#
# Every DataSAIL run is scored, and so are BASELINE_REPS random splits of the
# same set sizes. The random numbers are what the check would report with no
# similarity control at all, so they are the yardstick for the real runs.
#
# Writes to ${OUTDIR}:
#   leakage_threshold.tsv  COUNTS per set pair: how many genes in the second set
#                          have a relative in the first at identity >= threshold.
#                          One row per run, DataSAIL runs and random baselines
#                          together, so the comparison is a read down 'kind'.
#   leakage_genes.tsv      the genes to move into '${TRAIN_SET}', DataSAIL runs
#                          only, with parallel comma-separated lists of their
#                          relatives, those relatives' sets and the identities,
#                          closest first. The move list is built from CONNECTED
#                          COMPONENTS, not pairs: a similarity component that
#                          spans more than one split moves whole. 'reason' marks
#                          whether a gene is itself in a cross-set pair or is
#                          carried along by its family. Moving only the paired
#                          genes leaves leakage behind, since a moved gene turns
#                          its same-set relatives into new cross-set pairs.
#                          Also copied into each run's own dir.
#   split_sizes.tsv        requested vs ACTUAL split sizes (cold splits move
#                          whole clusters, so 0.7/0.15/0.15 is a target, not a
#                          result), plus the sizes after moving every gene in
#                          leakage_genes.tsv into '${TRAIN_SET}'.
#   <run>/*_splits_move_to_train.tsv   the run's own split table with those
#                          genes reassigned to '${TRAIN_SET}'. Same ID/Split
#                          columns and row order as DataSAIL's *_splits.tsv, so
#                          it drops straight into whatever reads that file.
#   leakage_after_move.tsv the same check re-run on those moved tables AS READ
#                          BACK FROM DISK, before and after rows side by side.
#                          The after_move counts should be 0; anything else
#                          means the move list is incomplete and the run prints
#                          a WARNING.
#   <run>/cross_set_pairs.tsv   every cross-set gene pair at or above the lowest
#                          threshold, highest identity first.
#   pairwise_identity.tsv.gz    every non-zero gene pair and its identity, one
#                          row per pair (SAVE_ALIGNMENTS=1).
# ---------------------------------------------------------------------------
echo; echo "===== Threshold leakage check ====="

export DS_OUTDIR="${OUTDIR}"
export DS_FASTA="${INPUT_FASTA}"
export DS_SPLIT_NAMES="${NAMES[*]}"
export DS_FRACS="${FRACS[*]}"
export DS_TRAIN_SET="${TRAIN_SET}"
export DS_THRESHOLDS="${THRESHOLDS}"
export DS_EVAL_SIM="${EVAL_SIM}"
export DS_THREADS="${THREADS}"
export DS_BASELINE_REPS="${BASELINE_REPS}"
export DS_BASELINE_SEED="${BASELINE_SEED}"
export DS_SAVE_ALIGNMENTS="${SAVE_ALIGNMENTS}"

if python3 << 'PYEOF'
import glob, itertools, os
from pathlib import Path

import numpy as np
import pandas as pd
from datasail.reader.read import read_data_type
from datasail.cluster.clustering import cluster
from datasail.settings import KW_LINKAGE, KW_LOGDIR, KW_OUTDIR, KW_THREADS

outdir = Path(os.environ["DS_OUTDIR"])
split_names = os.environ["DS_SPLIT_NAMES"].split()
requested = dict(zip(split_names, (float(f) for f in os.environ["DS_FRACS"].split())))
train_name = os.environ["DS_TRAIN_SET"]
thresholds = sorted(float(t) for t in os.environ["DS_THRESHOLDS"].split())
eval_sim = os.environ["DS_EVAL_SIM"]
threads = int(os.environ["DS_THREADS"])
baseline_reps = int(os.environ["DS_BASELINE_REPS"])
_seed = os.environ["DS_BASELINE_SEED"].strip()
baseline_seed = int(_seed) if _seed else None    # None = fresh entropy each run
save_alignments = os.environ["DS_SAVE_ALIGNMENTS"] == "1"
pairs = list(itertools.combinations(split_names, 2))   # train-val, train-test, val-test

# ---------------------------------------------------------------------------
# All-vs-all identity, computed once for every run.
# mmseqspp labels its matrix in FASTA order but sorts it by ID, so we feed it a
# copy of the FASTA sorted by ID and then assert the two orders agree.
# ---------------------------------------------------------------------------
seqs, name = {}, None
for line in open(os.environ["DS_FASTA"]):
    line = line.strip()
    if line.startswith(">"):
        name = line[1:]
        seqs[name] = ""
    elif line:
        seqs[name] += line
sorted_fasta = Path("eval_sorted.fasta")
with sorted_fasta.open("w") as out:
    for n in sorted(seqs):
        out.write(f">{n}\n{seqs[n]}\n")

print(f"Leakage check: computing all-vs-all identity with {eval_sim} "
      f"({len(seqs)} sequences)", flush=True)
ds = read_data_type("P")(data=sorted_fasta, weights=None, sim=eval_sim, dist=None,
                         num_clusters=np.inf, detect_duplicates=False)
ds = cluster(ds, **{KW_THREADS: threads, KW_LOGDIR: None, KW_LINKAGE: "average",
                    KW_OUTDIR: None})
if ds.cluster_similarity is None:
    raise SystemExit(f"ERROR: {eval_sim} produced no similarity matrix.")

names = np.array(ds.cluster_names)
S = np.array(ds.cluster_similarity, dtype=np.float32)   # np.array() also drops read-only
if S.shape[0] != len(names):
    raise SystemExit(f"ERROR: matrix is {S.shape} but there are {len(names)} names.")
if list(names) != sorted(seqs):
    raise SystemExit("ERROR: matrix row order does not match the sorted FASTA. "
                     "The matrix cannot be trusted to be labelled correctly.")

# A vs B and B vs A can differ slightly; keep the higher one and drop self-pairs.
sym = np.maximum(S, S.T)
np.fill_diagonal(sym, 0)
del S

if save_alignments:
    r, c = np.nonzero(np.triu(sym, 1))
    pd.DataFrame({"gene_1": names[r], "gene_2": names[c],
                  "identity": sym[r, c]}).to_csv(
        outdir / "pairwise_identity.tsv.gz", sep="\t", index=False)

# ---------------------------------------------------------------------------
# Per-split reports
# ---------------------------------------------------------------------------
threshold_rows, gene_rows, size_rows, after_rows = [], [], [], []


def pair_counts(label, kind, lab, rows, only=None, extra=None):
    """Append one row per set pair and threshold: how many genes in the second
    set have a relative in the first. `only` restricts it to one threshold."""
    for a, b in pairs:
        ia, ib = lab == a, lab == b
        closest = sym[np.ix_(ib, ia)].max(axis=1) if ia.any() and ib.any() \
            else np.zeros(ib.sum())
        for t in (thresholds if only is None else [only]):
            n_with = int((closest >= t).sum())
            rows.append({
                "run": label, "kind": kind, "set_pair": f"{a}-{b}", "threshold": t,
                "checked_set": b, "n_checked": int(ib.sum()),
                "n_with_relative": n_with,
                "n_without_relative": int(ib.sum()) - n_with,
                "pct_with_relative": round(100 * n_with / max(int(ib.sum()), 1), 2),
                **(extra or {})})


def check_run(label, assign, kind, run_dir=None, split_file=None):
    """Score one split assignment.

    kind is 'datasail' or 'random'. Random baselines only contribute counts to
    leakage_threshold.tsv: naming their offenders is pointless, and their sizes
    match the run they were shuffled from.
    """
    lab = np.array([assign.get(n, "") for n in names])
    in_split = np.isin(lab, split_names)
    # True where two genes are in DIFFERENT splits (both actually assigned)
    cross = (lab[:, None] != lab[None, :]) & in_split[:, None] & in_split[None, :]

    pair_counts(label, kind, lab, threshold_rows)

    if run_dir is None:          # random baseline: counts only
        return

    # --- every offending pair ----------------------------------------------
    i, j = np.nonzero(np.triu((sym >= thresholds[0]) & cross, 1))
    pd.DataFrame({"gene_1": names[i], "set_1": lab[i],
                  "gene_2": names[j], "set_2": lab[j],
                  "identity": np.round(sym[i, j], 4)}).sort_values(
        "identity", ascending=False).to_csv(
        run_dir / "cross_set_pairs.tsv", sep="\t", index=False)

    # --- which genes to move -----------------------------------------------
    # Leakage is a property of CONNECTED COMPONENTS, not of pairs. Flagging only
    # the genes in a cross-set pair and moving them to training does not finish
    # the job: a moved gene drags its same-set relatives across the boundary
    # behind it, turning within-set pairs into new cross-set pairs. So the rule
    # is: build the similarity graph at the threshold, and any component that
    # spans more than one split moves into training WHOLE. A component sitting
    # inside one split is left alone, whatever its internal similarity.
    #
    # Only genes outside training are listed, since the training members of a
    # straddling component are already where they need to be.
    #
    # relatives / relative_sets / identities are parallel lists of that gene's
    # neighbours at or above the threshold, closest first, so position 1 in each
    # is the same neighbour. 'reason' says whether the gene is itself in a
    # cross-set pair or is only being carried along by its family.
    flagged_by_threshold = {}
    for t in thresholds:
        adj = (sym >= t) & in_split[:, None] & in_split[None, :]
        ei, ej = np.nonzero(np.triu(adj, 1))

        # union-find over the edge list, cheaper than materialising components
        parent = np.arange(len(names))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a_, b_ in zip(ei.tolist(), ej.tolist()):
            ra, rb = find(a_), find(b_)
            if ra != rb:
                parent[ra] = rb
        root = np.array([find(k) for k in range(len(names))])

        spans = {}                       # component root -> splits it touches
        for k in np.nonzero(in_split)[0]:
            spans.setdefault(root[k], set()).add(lab[k])

        n_cross = ((sym >= t) & cross).sum(axis=1)
        flagged = []
        for k in np.nonzero(in_split & (lab != train_name))[0]:
            if len(spans[root[k]]) < 2:  # component lives in one split, fine
                continue
            flagged.append(str(names[k]))
            rel = np.nonzero(adj[k])[0]
            rel = rel[np.argsort(-sym[k, rel])]          # closest relative first
            gene_rows.append({
                "run": label,
                "threshold": t,
                "gene": names[k],
                "current_set": lab[k],
                "n_relatives": len(rel),
                "n_cross_set_relatives": int(n_cross[k]),
                "reason": "cross_set_pair" if n_cross[k] else "same_component",
                "relatives": ",".join(names[rel]),
                "relative_sets": ",".join(lab[rel]),
                "identities": ",".join(f"{v:.4g}" for v in sym[k, rel]),
                "sum_identity_over_threshold": round(float(sym[k, rel].sum()), 4),
                "action": f"move_to_{train_name}"})
        flagged_by_threshold[t] = flagged

    # --- the split table you would get after making those moves -------------
    # The file is written, then READ BACK FROM DISK and re-checked, so what gets
    # scored is the file you will actually train on, not an in-memory copy of it.
    # Every straddling component ends up inside training, so no edge can cross a
    # split boundary afterwards and the after_move counts should be 0. A
    # non-zero count means the move list or the file itself is wrong.
    for t in thresholds:
        moved = set(flagged_by_threshold[t]) & set(assign)
        suffix = "_splits_move_to_train.tsv" if len(thresholds) == 1 \
            else f"_splits_move_to_train_t{t:g}.tsv"
        out_name = split_file.name.replace("_splits.tsv", suffix)
        if out_name == split_file.name:      # unexpected name, never overwrite
            out_name = split_file.stem + suffix
        after = {g: (train_name if g in moved else s) for g, s in assign.items()}
        pd.DataFrame({"ID": list(after), "Split": list(after.values())}).to_csv(
            run_dir / out_name, sep="\t", index=False)

        # read the file back and confirm it says what it was meant to say
        back = pd.read_csv(run_dir / out_name, sep="\t", dtype=str,
                           keep_default_na=False)
        reread = dict(zip(back.ID, back.Split))
        changed = {g for g, s in reread.items() if assign.get(g) != s}
        if len(back) != len(assign) or set(reread) != set(assign) \
                or changed != moved:
            print(f"WARNING: {out_name} does not match the intended moves: "
                  f"{len(back)} rows vs {len(assign)}, {len(changed)} genes "
                  f"changed vs {len(moved)} flagged.", flush=True)

        tag = {"n_moved": len(moved), "split_file": out_name}
        pair_counts(label, "before_move", lab, after_rows, only=t, extra=tag)
        pair_counts(label, "after_move",
                    np.array([reread.get(n, "") for n in names]),
                    after_rows, only=t, extra=tag)

    # --- split sizes, as built and after the move ---------------------------
    # Counted from the assignment table itself, so any label DataSAIL writes
    # outside split_names (e.g. 'not selected') shows up as its own row.
    counts = {}
    for v in assign.values():
        counts[v] = counts.get(v, 0) + 1
    total = sum(counts.values())
    for t in thresholds:
        moved = [g for g in flagged_by_threshold[t] if g in assign]
        out_of = {}
        for g in moved:
            out_of[assign[g]] = out_of.get(assign[g], 0) + 1
        for s in sorted(counts):
            n_in = len(moved) if s == train_name else 0
            n_out = out_of.get(s, 0)
            after = counts[s] + n_in - n_out
            size_rows.append({
                "run": label, "threshold": t, "split": s,
                "requested_frac": requested.get(s, float("nan")),
                "n_requested": round(requested[s] * total) if s in requested else "",
                "n_actual": counts[s],
                "pct_actual": round(100 * counts[s] / total, 2),
                "n_moved_in": n_in, "n_moved_out": n_out,
                "n_after_move": after,
                "pct_after_move": round(100 * after / total, 2),
                "n_total": total})


files = sorted(glob.glob(str(outdir / "*" / "*_splits.tsv")))
print(f"Leakage check: found {len(files)} split file(s)", flush=True)
if not files:
    raise SystemExit("ERROR: no *_splits.tsv files found")
template = None
for f in files:
    f = Path(f)
    print(f"Leakage check: scoring {f}", flush=True)
    # keep_default_na so an ID or label that looks like 'NA' survives as text
    sp = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
    if template is None:
        template = (sp.ID.to_numpy(), sp.Split.to_numpy())
    check_run(f.parent.name, dict(zip(sp.ID, sp.Split)), "datasail", f.parent, f)

# Baseline: reshuffle the first run's labels. Set sizes are preserved exactly,
# so the only thing removed is the similarity control, which makes these numbers
# the "no leakage control" reference for the rows above.
ids, labels = template
rng = np.random.default_rng(baseline_seed)
if baseline_reps:
    # works for a given seed and for a drawn one, so the log always says which
    print(f"Leakage check: baseline shuffles use seed "
          f"{rng.bit_generator.seed_seq.entropy}", flush=True)
for rep in range(baseline_reps):
    print(f"Leakage check: scoring random baseline {rep}", flush=True)
    check_run(f"random_{rep}", dict(zip(ids, rng.permutation(labels))), "random")

# ---------------------------------------------------------------------------
# Write the three reports
# ---------------------------------------------------------------------------
th = pd.DataFrame(threshold_rows)
th.to_csv(outdir / "leakage_threshold.tsv", sep="\t", index=False)
print("\nGenes in the second set with a relative in the first set, "
      "at identity >= threshold:")
print(th.to_string(index=False))

gf = pd.DataFrame(gene_rows)
if not gf.empty:
    # 'identities' is already sorted high to low, so its first entry is the
    # closest relative. Sort on that, then drop the helper column.
    gf["_top"] = gf.identities.str.split(",").str[0].astype(float)
    gf = gf.sort_values(["run", "threshold", "_top"],
                        ascending=[True, True, False]).drop(columns="_top")
    for run_label, part in gf.groupby("run"):
        part.to_csv(outdir / run_label / "leakage_genes.tsv", sep="\t", index=False)
gf.to_csv(outdir / "leakage_genes.tsv", sep="\t", index=False)
print(f"\nGenes to move into '{train_name}' (top 20 by closest relative):")
print(gf.head(20).to_string(index=False) if not gf.empty else "  none above threshold")

sf = pd.DataFrame(size_rows)
sf.to_csv(outdir / "split_sizes.tsv", sep="\t", index=False)
print(f"\nSplit sizes, as built and after moving those genes into '{train_name}':")
print(sf.to_string(index=False) if not sf.empty else "  none")

af = pd.DataFrame(after_rows)
af.to_csv(outdir / "leakage_after_move.tsv", sep="\t", index=False)
print("\nThe same check re-run on the moved split tables:")
print(af.to_string(index=False) if not af.empty else "  none")
left = int(af.loc[af.kind == "after_move", "n_with_relative"].sum()) if not af.empty else 0
if left:
    print(f"WARNING: {left} gene(s) still have a cross-set relative after the "
          "move. The move list is incomplete, do not trust it.", flush=True)
else:
    print("No gene has a cross-set relative after the move, as expected.")
PYEOF
then
    echo "Leakage check: DONE -> ${OUTDIR}/{leakage_threshold,leakage_genes,split_sizes,leakage_after_move}.tsv"
else
    echo "WARNING: leakage check FAILED (splits are still packaged)." >&2
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
echo "Splits and reports written under: ${OUTDIR}"
echo "Check leakage_threshold.tsv for how much leaks, DataSAIL rows against the"
echo "random ones; leakage_genes.tsv for which genes cause it; split_sizes.tsv"
echo "for what the splits actually are; and <run>/*_splits_move_to_train.tsv for"
echo "the ready-made split table with those genes already in ${TRAIN_SET}."
date