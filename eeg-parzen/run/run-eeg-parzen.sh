#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python -m pip install -r requirements.txt

SEED="$(python eeg-parzen/control-logic.py --print-seed)"
RUN_ID="${1:-$(python -c 'import secrets; print(secrets.token_hex(4))')}"

echo "Run ID: $RUN_ID"
python eeg-parzen/control-logic.py \
    --output-dir "data/seed-${SEED}/eeg-parzen" \
    --session-id "$RUN_ID"
