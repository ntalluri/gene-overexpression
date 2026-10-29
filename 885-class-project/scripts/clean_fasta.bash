#!/usr/bin/env bash

set -euo pipefail
export LC_ALL=C   # same sort order for both ID lists (comm needs this)

PREP=data/genes/prepped
IN=data/genes/yorf_filtered_idmapping_2026_10_29.fasta.gz
OUT=$PREP/clean.fasta
MAP=$PREP/mapping_from_yorf_filtered_to_uniprot.tsv
MISSING=$PREP/missing_from_fasta.txt
ID_COL_NAME=Entry   # MAP column holding UniProt accessions

# add the prepped folder if it doesn't exist
mkdir -p "$PREP"

# header ">sp|A2P2R3|YM084_YEAST ..." -> ">A2P2R3" (falls back to first word if no pipes)
# blank line after each accession/sequence pair; existing blank lines are dropped
gzip -dc "$IN" | awk '
  NF == 0 { next }
  /^>/ {
    if (seen) print ""
    seen = 1
    n = split($1, a, "|")
    print ">" (n >= 2 ? a[2] : substr($1, 2))
    next
  }
  { print }
  END { if (seen) print "" }
' > "$OUT"

# ID lists, kept in memory (no intermediate files)
fasta_ids() { sed -n 's/^>//p' "$OUT"; }

# sanity checks on clean.fasta
n_seq=$(fasta_ids | wc -l)
n_dup=$(fasta_ids | sort | uniq -d | wc -l)
echo "clean.fasta: $n_seq sequences, $n_dup duplicate IDs"

# find the accession column in MAP by header name
col=$(head -1 "$MAP" | tr -d '\r' | tr '\t' '\n' | grep -n -x "$ID_COL_NAME" | cut -d: -f1 || true)
if [ -z "$col" ]; then
  echo "No '$ID_COL_NAME' column in $MAP. Header is:" >&2
  head -1 "$MAP" >&2
  exit 1
fi

map_ids() { tail -n +2 "$MAP" | tr -d '\r' | cut -f"$col" | awk 'NF' | sort -u; }

# compare ID sets; only the missing list is written to disk
comm -23 <(map_ids) <(fasta_ids | sort -u) > "$MISSING"

n_map=$(map_ids | wc -l)
n_missing=$(wc -l < "$MISSING")
echo "mapping: $n_map unique accessions, $n_missing missing from clean.fasta"

if [ "$n_missing" -gt 0 ]; then
  echo "Missing IDs written to $MISSING (first 10):"
  head -10 "$MISSING"
  exit 1
fi
echo "All mapping IDs are in clean.fasta."