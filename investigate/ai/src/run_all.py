#!/usr/bin/env python3
"""
run_all.py
----------
Run the pipeline end to end. Each step is a separate script under src/ and can
be run on its own; this just calls them in order and stops on the first failure.

    python run_all.py                 # core pipeline, steps 02 through 07
    python run_all.py --full          # also annotations, enrichments, RNA-seq
    python run_all.py --from 05       # resume partway
    python run_all.py --validate      # compare against Supplementary file 4
    python run_all.py --self-test     # synthetic data plus statistical tests
    python run_all.py --plots         # render the figures

Step 00 (GEO metadata) and step 01 (read structure inspection) are not included
because both need a decision from you before the rest can run. Steps 08 through
10 are behind --full because they need annotation files that step 02 does not;
run `python src/fetch_external_data.py --list` to see what is missing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

CORE_STEPS = [
    ("02", "step02_count_barcodes.py", "count barcodes from FASTQ"),
    ("03", "step03_normalize.py", "library-size normalization"),
    ("04", "step04_impute.py", "pseudocount imputation"),
    ("05", "step05_fitness.py", "fitness scores and FDR"),
    ("06", "step06_gene_lists.py", "commonly deleterious and strain-specific lists"),
    ("07", "step07_core_analyses.py", "figures and reported statistics"),
]

# These need annotation files from data/external. Each one reports what it is
# missing rather than failing, so running them early is harmless.
EXTENDED_STEPS = [
    ("08", "step08_annotations.py", "build gene annotation tables"),
    ("09", "step09_enrichments.py", "functional and biophysical enrichments"),
    ("10", "step10_rnaseq.py", "RNA-seq differential expression and cross-reference"),
]


def run(script: str, extra: list[str] | None = None) -> int:
    cmd = [sys.executable, str(SRC / script)] + (extra or [])
    print(f"\n{'=' * 70}\n$ {' '.join(cmd)}\n{'=' * 70}", flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", default="02",
                    help="First step to run, e.g. 04")
    ap.add_argument("--validate", action="store_true",
                    help="Run validate_against_supp4.py at the end")
    ap.add_argument("--self-test", action="store_true",
                    help="Generate synthetic data first and check the result")
    ap.add_argument("--backend", choices=["python", "edgepython", "edger"],
                    default=None,
                    help="Statistics backend for steps 05 and 10. Default: "
                         "edgepython if installed, otherwise the built-in python "
                         "backend.")
    ap.add_argument("--full", action="store_true",
                    help="Also run steps 08 to 10 (annotations, enrichments, RNA-seq)")
    ap.add_argument("--plots", action="store_true",
                    help="Render the figures at the end")
    args = ap.parse_args()

    if args.backend is None:
        try:
            import edgepython  # noqa: F401
            args.backend = "edgepython"
            print("Using the edgepython backend for steps 05 and 10.")
        except ImportError:
            args.backend = "python"
            print("edgepython not found; using the built-in python backend. "
                  "For edgeR-faithful results: pip install edgepython")

    if args.self_test:
        # The statistical regression tests need no data, so run them first.
        if run("test_statistics.py") != 0:
            print("\nStatistical regression tests failed. Fix those before "
                  "trusting any pipeline output.")
            return 1
        if run("make_test_data.py") != 0:
            return 1

    steps = CORE_STEPS + (EXTENDED_STEPS if args.full else [])

    for number, script, description in steps:
        if number < args.start:
            print(f"skipping step {number} ({description})")
            continue
        extra = ["--backend", args.backend] if number in ("05", "10") else None
        code = run(script, extra)
        if code != 0:
            if number in ("08", "09", "10"):
                # These depend on optional external files; a failure there
                # should not sink the core reproduction.
                print(f"\nStep {number} ({description}) failed with exit code "
                      f"{code}. Continuing, since this step needs external "
                      f"annotation files. Run "
                      f"`python src/fetch_external_data.py --list`.")
                continue
            print(f"\nStep {number} ({description}) failed with exit code {code}.")
            return code

    if args.self_test:
        run("check_against_truth.py")

    if args.plots:
        run("plots.py")

    if args.validate:
        run("validate_against_supp4.py")

    print("\nDone. Outputs are under results/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
