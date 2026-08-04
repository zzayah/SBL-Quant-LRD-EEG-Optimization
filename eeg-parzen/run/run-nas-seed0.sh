#!/bin/bash
set -e
source .venv/bin/activate
EEG_SEED=0 python eeg-parzen/control-logic.py --output-dir data/seed-0/eeg-parzen --session-id nas-seed0
