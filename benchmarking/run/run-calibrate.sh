#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate

SEED="${1:?Usage: bash benchmarking/run/run-calibrate.sh SEED RUN_ID}"
RUN_ID="${2:?Usage: bash benchmarking/run/run-calibrate.sh SEED RUN_ID}"

python benchmarking/calibrate-final-training.py \
    --output-dir "data/seed-${SEED}/benchmarking/calibration" \
    --run-id "$RUN_ID" \
    --seed "$SEED"
