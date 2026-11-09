#!/usr/bin/env bash
# Maintainer script: builds the archive that `make fetch` downloads, from the directory where the experiments were run.
#
#   bash scripts/pack_artifacts.sh /path/to/original/working/dir
#
# Writes artifacts/vqc-thesis-artifacts.tar.gz and artifacts/SHA256SUMS, and copies the per-round evaluation files
# (metrics.jsonl, small) of the report's runs to data/checkpoints/<run>/, which is tracked by git (Table 8).
set -euo pipefail
SRC="${1:?usage: pack_artifacts.sh SOURCE_DIR}"
cd "$(dirname "$0")/.."
HERE=$(pwd)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

# runs whose final weights (model_latest.pt) are read by the part B scripts and by the A3 analysis
MODEL_RUNS=$(cd "$SRC/checkpoints" && ls -d gen-nav-id-k*-s?-7.5k-cpu-outer gen-four-k*-s?-7.5k-cpu-outer \
  gen-h5-k5-s?-7.5k-cpu-outer gen-h128-k5-s?-7.5k-cpu-outer ent-h5-k25-s?-7.5k-cpu-outer \
  h128-rms-s3-7.5k-cpu-outer 2>/dev/null | grep -v -- '-kinf-' || true)
# runs of the report with a metrics.jsonl (greedy and pool columns of Table 8)
METRIC_RUNS=$(cd "$SRC/checkpoints" && ls -d gen-*-7.5k-cpu-outer ent-*-7.5k-cpu-outer nav-id-hlr2e-3-detc-s*-7.5k-cpu-outer 2>/dev/null || true)

missing=0
for run in $MODEL_RUNS; do
  f="$SRC/checkpoints/$run/model_latest.pt"
  if [ -f "$f" ]; then mkdir -p "$STAGE/checkpoints/$run"; cp "$f" "$STAGE/checkpoints/$run/"; else echo "missing: $f"; missing=1; fi
done
for run in $METRIC_RUNS; do
  f="$SRC/checkpoints/$run/metrics.jsonl"
  if [ -f "$f" ]; then mkdir -p "data/checkpoints/$run"; cp "$f" "data/checkpoints/$run/"; fi
done
A=quantum_heads/analysis
for d in data/a3 data/h128rms_s3 runs/a3/students runs/distill_concentration/students; do
  mkdir -p "$STAGE/$A/$(dirname "$d")"; cp -R "$SRC/$A/$d" "$STAGE/$A/$d"
done

mkdir -p artifacts
tar -czf artifacts/vqc-thesis-artifacts.tar.gz -C "$STAGE" .
( cd artifacts && { shasum -a 256 vqc-thesis-artifacts.tar.gz 2>/dev/null || sha256sum vqc-thesis-artifacts.tar.gz; } > SHA256SUMS )
echo "archive: $(du -h artifacts/vqc-thesis-artifacts.tar.gz | cut -f1); metrics.jsonl copied: $(ls data/checkpoints 2>/dev/null | wc -l) runs; models: $(echo $MODEL_RUNS | wc -w) runs"
cat artifacts/SHA256SUMS
[ $missing -eq 0 ] || { echo "WARNING: some model files were missing (listed above)"; exit 1; }
