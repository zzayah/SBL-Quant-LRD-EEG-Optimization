#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate

SEED="${1:?Usage: bash benchmarking/run/run-final-training.sh SEED RUN_ID EPOCHS}"
RUN_ID="${2:?Usage: bash benchmarking/run/run-final-training.sh SEED RUN_ID EPOCHS}"
EPOCHS="${3:?Usage: bash benchmarking/run/run-final-training.sh SEED RUN_ID EPOCHS}"

python benchmarking/final-training.py \
    --output-dir "data/seed-${SEED}/benchmarking/final-training" \
    --run-id "$RUN_ID" \
    --seed "$SEED" \
    --epochs "$EPOCHS"
