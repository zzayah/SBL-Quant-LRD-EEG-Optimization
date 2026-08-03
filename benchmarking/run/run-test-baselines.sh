#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${1:?Usage: bash benchmarking/run/run-test-baselines.sh RUN_ID}"
SEED="$(python eeg-parzen/control-logic.py --print-seed)"
RUN_ROOT="data/seed-${SEED}/benchmarking"

python benchmarking/test-baselines.py \
    --bnci-results "$RUN_ROOT/bnci2014_001/$RUN_ID/baseline_results.jsonl" \
    --lee-results "$RUN_ROOT/lee2019_mi/$RUN_ID/baseline_results.jsonl" \
    --physionet-results "$RUN_ROOT/physionet_mi/$RUN_ID/baseline_results.jsonl" \
    --output-dir "$RUN_ROOT/test-best" \
    --run-id "$RUN_ID" \
    --seed "$SEED"
