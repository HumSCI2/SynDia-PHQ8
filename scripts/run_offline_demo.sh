#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p outputs
syndia-refine \
  --input data/samples/daic_woz_synthetic.jsonl \
  --output outputs/daic_refined.jsonl
syndia-refine \
  --input data/samples/edaic_synthetic.jsonl \
  --output outputs/edaic_refined.jsonl
syndia-train \
  --input data/samples/edaic_synthetic.jsonl \
  --output-dir outputs/training_preview \
  --prepare-only

printf 'Offline demonstration completed. Outputs are under %s/outputs.\n' "$ROOT"
