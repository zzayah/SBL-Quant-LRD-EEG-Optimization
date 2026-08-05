# EEG Parzen NAS

This directory adapts the Accuracy-NAS/Optuna pattern to binary motor-imagery
EEG classification. Each dataset gets an independent TPE study. The objective
maximizes subject-held-out validation balanced accuracy.

## Structure

- `cnn.py`: EEG model blocks, candidate training, artifact logging, and the
  Optuna objective.
- `evaluation.py`: loss, accuracy, balanced accuracy, and per-class recall.
- `control-logic.py`: the single global `SEED = 0` and TPE orchestration.
- `run/run-nas.sh`: seeded launcher for all three studies.
- `run/run-physionet.sh`: focused PhysioNet-only development search.
- `calibrate-final-training.py`: 10-epoch checkpoint learning curves.
- `final-training.py`: fresh training on all development subjects.
- `final-testing.py`: locked-test evaluation of frozen final models.

BNCI candidates are scored by mean balanced accuracy over three subject folds.
OpenBMI and PhysioNet use their fixed subject-held-out validation partitions.
Locked test subjects are not loaded anywhere in the NAS objective.
Changing only `SEED` in `control-logic.py` controls Python, NumPy, PyTorch,
DataLoader shuffling, TPE sampling, and creation/selection of the matching
`data/seed-<seed>/splits_seed<seed>.json` manifest.

## Search protocol

The protocol uses 100 trials per dataset, 50 maximum epochs, patience
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
`RUN_ID` resumes that study toward a total of 100 trials. Pruned and failed
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

With the environment configured and data prepared, use:

```bash
bash eeg-parzen/run/run-nas.sh SEED [RUN_ID]
```

Run it inside `tmux`; output is stored under
`data/seed-<seed>/eeg-parzen/<dataset>/<run-id>/`.

## Three-seed 100-trial runs

Run these in three separate GPU-node terminals:

```bash
bash eeg-parzen/run/run-nas.sh 0 nas-seed0-10a2117c
bash eeg-parzen/run/run-nas.sh 1
bash eeg-parzen/run/run-nas.sh 2
```

Seed 0 resumes run `nas-seed0-10a2117c` from 50 to 100 total trials. Seeds 1 and 2
generate run IDs in the form `nas-seed<seed>-<id>` and run 100 trials per
dataset under `data/seed-1/` and `data/seed-2/`. Pass a printed run ID back to
the same launcher to resume it. The seed controls splitting, TPE,
initialization, and data ordering.

## Calibration, final training, and locked testing

First, measure development learning curves for the nine architecture-target
pairs. This trains for 100 epochs, saves weights every 10 epochs, and does not
load locked-test subjects. BNCI uses its fixed fold 0 split for this analysis.

```bash
bash eeg-parzen/run/run-calibrate.sh SEED RUN_ID
```

After reviewing the train and validation curves, choose a fixed epoch count.
Train the nine models from fresh weights on all target development subjects:

```bash
bash eeg-parzen/run/run-final-training.sh SEED RUN_ID EPOCHS
```

Only after those models are frozen, evaluate them once on the locked subjects:

```bash
bash eeg-parzen/run/run-final-testing.sh SEED RUN_ID EPOCHS
```

No weights or normalization statistics transfer between datasets. Calibration,
final training, and final testing write to separate directories. Each stage
refuses to overwrite an existing result file.

After all three 100-trial studies and their corresponding baselines have been
reviewed, the three predetermined seed-specific locked evaluations are:

```bash
bash eeg-parzen/run/run-final-testing.sh 0 nas-seed0-10a2117c EPOCHS
bash eeg-parzen/run/run-final-testing.sh 1 nas-seed1-47f127b6 EPOCHS
bash eeg-parzen/run/run-final-testing.sh 2 nas-seed2-61979a0f EPOCHS
```

They may train simultaneously for accuracy evaluation, but concurrent GPU use
invalidates their latency measurements. Run final timing separately.
