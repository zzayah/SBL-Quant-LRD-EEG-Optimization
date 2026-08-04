# EEG Parzen NAS

This directory adapts the Accuracy-NAS/Optuna pattern to binary motor-imagery
EEG classification. Each dataset gets an independent TPE study. The objective
maximizes subject-held-out validation balanced accuracy.

## Structure

- `cnn.py`: EEG model blocks, candidate training, artifact logging, and the
  Optuna objective.
- `evaluation.py`: loss, accuracy, balanced accuracy, and per-class recall.
- `control-logic.py`: the single global `SEED = 0` and TPE orchestration.
- `run/run-eeg-parzen.sh`: minimal wrapper for all three studies.
- `run/run-physionet.sh`: focused PhysioNet-only development search.
- `test-best.py`: winner selection, final retraining, and the 3×3
  architecture-transfer test matrix.
- `run/run-test-best.sh`: explicit locked-test launcher used only after NAS.

BNCI candidates are scored by mean balanced accuracy over three subject folds.
OpenBMI and PhysioNet use their fixed subject-held-out validation partitions.
Locked test subjects are not loaded anywhere in the NAS objective.
Changing only `SEED` in `control-logic.py` controls Python, NumPy, PyTorch,
DataLoader shuffling, TPE sampling, and creation/selection of the matching
`data/seed-<seed>/splits_seed<seed>.json` manifest.

## Search protocol

The protocol uses 50 trials per dataset, 50 maximum epochs, patience
10, AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 64, and a hard
350,000-parameter ceiling. The architecture search allows one to six temporal
blocks, caps all learned feature widths at 128, and considers ELU, GELU, and
ReLU. Corrected searches choose either a flatten-and-dense temporal head or a
log-variance power head; global temporal averaging is not used. Fixed filter
ranks are mapped to at least the current width, preserving non-decreasing
channels without pruning. Training hyperparameters are fixed rather than
searched. Given the width cap, the current architecture family remains far
below the parameter ceiling; the ceiling is retained as a defensive check.

Each dataset stores its Optuna study in SQLite. Reusing the same optional
`RUN_ID` resumes that study toward a total of 50 trials. Pruned and failed
trials, including their reasons, are also appended to a separate JSONL file.
Per-trial checkpoints are not retained. The JSONL records preserve sampled
architectures, parameters, fold metrics, parameter counts, training constants,
seed, and runtime information. Same-seed reproduction compares these substantive
fields; timestamps and run identifiers naturally differ. The winning
architecture is retrained from scratch after selection.

To launch only PhysioNet, run:

```bash
bash eeg-parzen/run/run-physionet.sh
```

This creates a new run ID and leaves earlier studies unchanged.

## Complete GPU-node run

On a fresh CUDA node, create the environment, prepare all data, and run all
three studies with:

```bash
bash eeg-parzen/run/run-cuda-nas.sh
```

The script uses Python 3.14, creates `.venv`, installs `requirements.txt`,
checks CUDA, downloads and validates all datasets, and runs all three
100-proposal studies sequentially.

For an already configured environment with prepared data, use:

```bash
bash eeg-parzen/run/run-eeg-parzen.sh
```

Run it inside `tmux`; output is stored under
`data/seed-0/eeg-parzen/<dataset>/<run-id>/`.

## Three-seed 100-trial runs

Run these in three separate GPU-node terminals:

```bash
bash eeg-parzen/run/run-nas-seed0.sh
bash eeg-parzen/run/run-nas-seed1.sh
bash eeg-parzen/run/run-nas-seed2.sh
```

Seed 0 resumes run `nas-seed0` from 50 to 100 total trials. Seeds 1 and 2
generate run IDs in the form `nas-seed<seed>-<id>` and run 100 trials per
dataset under `data/seed-1/` and `data/seed-2/`. Pass a printed run ID back to
the same launcher to resume it. `EEG_SEED` sets the single central seed in
`control-logic.py`, which controls splitting, TPE, initialization, and data
ordering.

## Final retraining and locked testing

After all three studies finish and the trial logs have been reviewed, run the
selected-architecture transfer matrix once:

```bash
bash eeg-parzen/run/run-test-best.sh RUN_ID
```

For each dataset-specific winner, this reconstructs the architecture with the
target dataset's input-channel count and calibrates its training duration using
only the target's original training and validation subjects. Calibration runs
for at most 100 epochs with patience 15; BNCI uses the median best epoch across
its three folds. The model is then reinitialized with the same seed and trained
for exactly that many epochs on all target development subjects before locked
testing. The result is nine independently trained models: three selected
architectures by three target datasets. No weights or normalization statistics
transfer between datasets.

Results and final checkpoints are written under
`data/seed-0/eeg-parzen/test-best/<run-id>/`. The runner refuses to overwrite
an existing `transfer_results.jsonl`, reducing the chance of accidentally
repeating locked-test evaluation.
