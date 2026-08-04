#!/bin/bash
set -e
source .venv/bin/activate
RUN_ID="${1:-nas-seed2-$(python -c 'import secrets; print(secrets.token_hex(4))')}"
echo "Run ID: $RUN_ID"
EEG_SEED=2 python eeg-parzen/control-logic.py --output-dir data/seed-2/eeg-parzen --session-id "$RUN_ID"
