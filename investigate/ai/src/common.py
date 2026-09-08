"""
common.py
---------
Shared helpers used by every step of the pipeline.

Nothing here is specific to the Robinson et al. 2021 analysis. It just handles
config loading, path resolution, logging, and reading/writing the tab-delimited
tables that the steps pass between each other.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# The project root is the directory that contains src/, data/, results/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: str | Path) -> Path:
    """Resolve a user-supplied path.

    Absolute paths are returned as-is. A relative path is tried first against
    the current working directory, then against the project root, so that
    scripts work whether you run them from the project root or from src/.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def get_logger(name: str) -> logging.Logger:
    """Return a logger that prints to stderr with a timestamp.

    Every step calls this so that the console output tells you which step
    produced which message.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_config(path: str | Path | None = None) -> dict:
    """Load config/config.json.

    We use JSON rather than YAML so that the pipeline has no dependency beyond
    the standard library plus the scientific stack.
    """
    if path is None:
        path = PROJECT_ROOT / "config" / "config.json"
    path = Path(path)
    with open(path) as fh:
        cfg = json.load(fh)

    # Turn every relative path in the config into an absolute path, so that
    # scripts can be run from any working directory.
    for key, value in cfg.get("paths", {}).items():
        cfg["paths"][key] = str((PROJECT_ROOT / value).resolve())
    return cfg


def open_maybe_gzip(path: str | Path, mode: str = "rt"):
    """Open a file that may or may not be gzip-compressed.

    FASTQ files from SRA come both ways depending on how you downloaded them,
    so every read of a FASTQ goes through this.
    """
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def read_fastq(path: str | Path):
    """Yield (header, sequence, plus_line, quality) for each record in a FASTQ.

    This is a plain 4-line-at-a-time reader. It does not validate the file;
    a malformed FASTQ will produce garbage rather than an error.
    """
    with open_maybe_gzip(path, "rt") as fh:
        while True:
            header = fh.readline()
            if not header:
                return
            sequence = fh.readline().rstrip("\n")
            plus = fh.readline()
            quality = fh.readline().rstrip("\n")
            yield header.rstrip("\n"), sequence, plus.rstrip("\n"), quality


def read_table(path: str | Path, index_col: int | None = 0) -> pd.DataFrame:
    """Read one of the pipeline's tab-delimited intermediate tables."""
    return pd.read_csv(path, sep="\t", index_col=index_col)


def write_table(df: pd.DataFrame, path: str | Path, index: bool = True) -> None:
    """Write a tab-delimited intermediate table, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=index)


def load_sample_sheet(path: str | Path) -> pd.DataFrame:
    """Load and validate samples.tsv.

    Required columns:
        sample_id   unique name for one sequencing sample
        strain      yeast isolate, e.g. BY4743
        generation  0 or 10
        replicate   biological replicate label, e.g. A/B/C or 1/2/3
        fastq       path to the FASTQ containing this sample's reads
        index_seq   inline multiplex index at the start of the read, or empty
                    if the FASTQ is already demultiplexed
    """
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    required = ["sample_id", "strain", "generation", "replicate", "fastq"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"samples.tsv is missing required columns: {missing}")

    if "index_seq" not in df.columns:
        df["index_seq"] = ""

    df["generation"] = df["generation"].astype(int)
    if df["sample_id"].duplicated().any():
        dupes = df.loc[df["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample_id values in samples.tsv: {dupes}")

    bad_gen = sorted(set(df["generation"]) - {0, 10})
    if bad_gen:
        raise ValueError(
            f"generation column must contain only 0 or 10; found {bad_gen}"
        )
    return df


def resolve_fastq(fastq_field: str, raw_dir: str | Path) -> Path:
    """Turn the samples.tsv fastq field into a real path.

    Accepts either an absolute path or a filename relative to data/raw.
    """
    p = Path(fastq_field)
    if p.is_absolute():
        return p
    return Path(raw_dir) / fastq_field


def sample_columns_for_strain(sample_sheet: pd.DataFrame, strain: str):
    """Return (gen0_sample_ids, gen10_sample_ids) for one strain, sorted."""
    sub = sample_sheet[sample_sheet["strain"] == strain]
    g0 = sorted(sub.loc[sub["generation"] == 0, "sample_id"])
    g10 = sorted(sub.loc[sub["generation"] == 10, "sample_id"])
    return g0, g10


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
