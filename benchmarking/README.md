# EEG baselines

The five fixed CNN baselines follow the same three-stage evaluation protocol as
the NAS-selected architectures. Locked test subjects are accessed only during
final testing.

The optional `scaled` suite holds topology constant and varies only width. Its
five models span approximately 1.3k to 86k parameters on BNCI, covering the NAS
capacity range without confounding capacity with architecture family.

## 1. Calibrate final-training length

Calibration trains every baseline for 100 epochs on the development
train/validation split and records train and validation metrics every 10 epochs.
BNCI uses fold 0, matching NAS calibration.

Run one seed per terminal:

```bash
bash benchmarking/run/run-calibrate.sh 0 baseline-seed0
bash benchmarking/run/run-calibrate.sh 1 baseline-seed1
bash benchmarking/run/run-calibrate.sh 2 baseline-seed2
```

To calibrate the size-controlled suite instead:

```bash
bash benchmarking/run/run-calibrate.sh 0 scaled-seed0 scaled
bash benchmarking/run/run-calibrate.sh 1 scaled-seed1 scaled
bash benchmarking/run/run-calibrate.sh 2 scaled-seed2 scaled
```

Each run produces 150 records: five baselines, three datasets, and ten epoch
checkpoints. Results are stored under
`data/seed-<seed>/benchmarking/calibration/<run-id>/`.

## 2. Final training

After choosing an epoch count from validation curves, retrain every baseline on
all development subjects. For example, with 40 epochs:

```bash
bash benchmarking/run/run-final-training.sh 0 baseline-seed0 40
bash benchmarking/run/run-final-training.sh 1 baseline-seed1 40
bash benchmarking/run/run-final-training.sh 2 baseline-seed2 40
```

This stage does not load locked test subjects.

## 3. Final testing

After final training is frozen, evaluate the locked tests once:

```bash
bash benchmarking/run/run-final-testing.sh 0 baseline-seed0 40
bash benchmarking/run/run-final-testing.sh 1 baseline-seed1 40
bash benchmarking/run/run-final-testing.sh 2 baseline-seed2 40
```
