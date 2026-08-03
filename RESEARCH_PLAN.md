# Motor-imagery NAS plan

## Initial scope

- Adapt the CNN block search and Optuna/TPE workflow from
  `NAS-in-the-Loop/accuracy-nas`.
- Use a single optimization objective: subject-held-out validation balanced
  accuracy.
- Run independent studies for BNCI2014_001, Lee2019_MI, and PhysionetMI.
- Keep final test subjects completely outside architecture search and model
  selection.
- Search EEG-appropriate CNN blocks after the data pipeline and splits are
  verified.

## Implementation status

- Complete: MOABB download, preprocessing, per-subject export, and full data
  integrity validation.
- Complete: deterministic seed-0 subject splits and PyTorch data loaders using
  training-subject-only channel normalization.
- Complete: `eeg-parzen/`, a classification-focused adaptation of Accuracy-NAS
  with CNN/search logic, balanced-accuracy evaluation, final winner
  retraining, and a single top-level control seed.
- In progress: initial seed-0 EEG NAS studies using the approved protocol.
- Complete (not yet executed): five fixed validation-only CNN baselines and a
  figures notebook under `benchmarking/`.

## Approved initial NAS protocol

- Run 50 TPE trials for each dataset. Equal trial counts are preferred over
  equalized compute; BNCI therefore receives more compute because each trial
  trains across three subject folds.
- Search architecture only. Hold AdamW, learning rate `1e-3`, weight decay
  `1e-4`, batch size 64, maximum 50 epochs, and early-stopping patience 10
  constant across candidates.
- Maximize validation balanced accuracy only. Record parameter count but do
  not penalize it in the objective.
- Reject architectures exceeding 350,000 trainable parameters.
- Search one to six temporal blocks using the same search space for all three
  datasets. Cap learned feature widths at 128. Dataset-specific input channel
  counts are not searched.
- Let each temporal block propose a fixed width rank corresponding to 16, 24,
  32, 48, 64, 96, or 128 output features. Map the proposal to at least the current
  width, so blocks stay equal or expand without dynamic Optuna distributions
  or channel-progression pruning.
- Search stem pooling over 1, 2, or 4 and block-pooling ranks corresponding to
  1, 2, 4, or 8. Safely saturate a proposed block pool at the remaining
  temporal length, keeping six-block candidates valid without dynamic Optuna
  distributions while including canonical EEGNet-like pooling.
- Use temporal stem widths from 8 through 64 and spatial depth multipliers 1
  or 2, so every sampled stem respects the 128-feature width cap.
- Search ELU, GELU, and ReLU activations.
- Search a flatten-and-dense head that retains pooled temporal positions and a
  log-variance head that directly represents oscillatory power. Do not use
  adaptive global temporal averaging in corrected studies.
- Keep duplicate trials. Record pruned and failed trials with their reasons in
  a separate JSONL file.
- Persist each dataset study in SQLite so interrupted runs resume toward the
  same total of 50 trials.
- Do not retain per-trial checkpoints. The JSONL architecture, parameters,
  fold metrics, parameter count, training constants, seed, and runtime details
  are sufficient for same-seed reproduction checks. Compare substantive fields
  rather than timestamps or run identifiers.
- Run seed 0 initially. Seeds 1, 2, and 3 are future full-study repetitions.
- Retrain the selected architecture from scratch using the combined training
  and validation subjects before evaluating once on the locked test subjects.
- After selecting one winner per dataset, run the predetermined 3×3 transfer
  matrix. For each architecture-target pair, calibrate training duration on
  the original target training/validation partition for at most 100 epochs
  with patience 15; use the median best epoch across BNCI's three folds. Then
  reinitialize and train for exactly that duration on the target dataset's
  combined development subjects, using normalization computed only from those
  subjects. Evaluate each resulting model once on the target dataset's locked
  test subjects and retain the nine final checkpoints. `eeg-parzen/test-best.py`
  implements this procedure and remains separate from the NAS launcher.

The approved `eeg-parzen/cnn.py` search space contains an
EEGNet-inspired temporal convolution, channel-spanning depthwise spatial
filter, and one to six separable temporal blocks. Candidate models use two
logits and cross-entropy. Early stopping and Optuna selection use validation
balanced accuracy. BNCI scores are averaged over its three subject folds;
Lee2019_MI and PhysionetMI use fixed validation-subject groups.

`eeg-parzen/control-logic.py` is the single source of the NAS seed. Changing
its `SEED` controls creation of the corresponding subject split, Python,
NumPy, PyTorch, DataLoader shuffling, model initialization, and the TPE sampler.
Seed-dependent data artifacts live under `data/seed-<seed>/`; for example,
seed 0 uses `data/seed-0/splits_seed0.json`. Raw and normalized MOABB data are
shared because their deterministic preprocessing does not depend on NAS seed.
NAS studies use `data/seed-<seed>/eeg-parzen/<dataset>/<run-id>/`.
`eeg-parzen/run/run-eeg-parzen.sh` is intentionally minimal: it installs requirements
into the active Python environment and runs the three 50-proposal studies
sequentially. Data preparation and environment/GPU setup remain separate. Its
optional run ID can be supplied again to resume the same SQLite studies.

## NAS-in-the-Loop structural alignment

The executable spine mirrors `NAS-in-the-Loop/accuracy-nas`: a thin named run
script calls a hyphenated `control-logic.py`, which owns the global seed and
all study orchestration; `cnn.py` owns the search space, candidate training,
and Optuna objective; and `evaluation.py` owns validation metrics. EEG data
preparation and deterministic splitting remain in `data/` because those
artifacts are shared by all three dataset-specific studies.

Candidate training selects CUDA when available and otherwise uses CPU. The
selected device is recorded in every completed-trial JSONL entry. Apple MPS
is excluded because it did not reproduce the CPU training result.

Intentional differences are limited to the experiment design. The three EEG
studies run sequentially on one GPU rather than as parallel prediction heads.
SQLite is retained for interruption-safe runs, and there is no compare-track
or downstream simulator stage. `eeg-parzen/test-best.py` is the EEG counterpart
to Accuracy-NAS `test-best.py` and remains an explicit post-search command so
locked-test evaluation cannot begin as part of NAS.

The search implementation is single-headed. Joint multi-dataset search and
multi-objective efficiency experiments are possible follow-up experiments,
not requirements for the initial NAS runs.

One top-level `SEED = 0` will control subject splitting, cross-validation,
Python and NumPy randomness, PyTorch initialization and data loading, and the
Optuna TPE sampler.

## Common data representation

- Task: left-hand versus right-hand motor imagery
- Labels: left hand = 0, right hand = 1
- Band-pass filter: 8-35 Hz
- Sampling rate: 128 Hz
- Window: 3 seconds (384 samples)
- Storage: one NPZ and one metadata CSV per subject

Full channel sets remain intact, so each dataset has its own input channel
count. Dataset-specific input handling will be separated from the searched CNN
body if cross-dataset architecture transfer is added later.

## Follow-up experiments

## Seed-0 diagnostic audit before locked testing

The first seed-0 NAS completed, but locked testing remains on hold. Best
validation balanced accuracies were 68.61% for BNCI2014_001, 77.56% for
Lee2019_MI, and 52.89% for PhysionetMI. The best PhysionetMI candidate largely
collapsed to the left class (97.19% left recall, 8.58% right recall).

Data checks found correct 3-second task timing, intact channel ordering, and
near-balanced labels in every PhysionetMI subject group. On the identical
training/validation split, log-variance LDA reached 58.62% and CSP-LDA reached
64.96% balanced accuracy, demonstrating that the exported signals contain
usable cross-subject motor-imagery information. Excluding known problematic
subjects 88, 89, 92, and 100 from CSP training changed performance only to
64.50%, so those subjects are not the primary cause of CNN failure.

The likely architectural error is the final adaptive global temporal average.
Canonical EEGNet retains pooled temporal positions with a flatten-and-dense
classifier, while ShallowConvNet explicitly extracts square/log band-power
features. Global averaging discards much of the temporal power structure used
for motor-imagery discrimination. Search efficiency also needs correction:
only 22 of 50 PhysionetMI proposals completed because 28 were pruned by the
channel-progression rule. Do not run `test-best.py` until a corrected search
space beats the CSP-LDA validation baseline and class recalls are balanced.

The next development run is intentionally PhysionetMI-only. It uses the two
corrected temporal heads and a non-decreasing width encoding in which all 50
ordinary proposals are valid. Earlier seed-0 studies remain diagnostic
artifacts. Once PhysionetMI validation clearly exceeds CSP-LDA, rerun the same
corrected search on BNCI2014_001 and Lee2019_MI for the final comparison.
A validation-only sanity run of a corrected 2,490-parameter EEGNet-like model
reached 72.67% balanced accuracy (83.42% left recall, 61.93% right recall) on
the unchanged PhysionetMI validation subjects. This confirms that retaining
pooled temporal features fixes the near-chance failure well enough to justify
the focused NAS rerun; 75% remains the immediate acceptance target.

The first broadened-width rerun (`2c34b0f2`) remained near chance through ten
completed trials, including five flatten and five log-variance heads. Retain
that run as a second diagnostic. Do not enqueue the validated compact model in
Optuna: doing so would bias TPE and change the experiment from an unseeded
architecture search. Any CPU-versus-MPS reproduction check for that fixed
model must run separately and must not be recorded as a NAS trial.

The complete CUDA-node command is `eeg-parzen/run/run-cuda-nas.sh`. It creates
the environment, prepares all datasets, and runs the three 50-proposal NAS
studies sequentially.

- Fixed-architecture baselines: train five hand-designed sequential CNNs using
  the identical seed, subject splits, normalization, optimizer, batch size,
  epoch limit, and early stopping used by NAS. During development, compare
  validation balanced accuracy, accuracy, macro F1, class precision/recall,
  per-subject accuracy, parameter count, model size, training time, and
  inference latency/throughput. Do not access locked test subjects until the
  NAS winner and baseline protocol are finalized. Implementation lives under
  `benchmarking/`; generated results live under
  `data/seed-<seed>/benchmarking/`.
- Baseline validation training calibrates for at most 100 epochs with patience
  15 and records the best epoch. After review, `benchmarking/test-baselines.py`
  reinitializes each model, trains combined development subjects for the
  median fold-best epoch, and performs one locked-test evaluation, producing
  five baselines by three datasets.

- Cross-session robustness on Lee2019_MI and BNCI2014_001: train using the
  first recording visit and evaluate on the later visit without mixing trials
  across visits. PhysionetMI has no comparable second recording session.
- Architecture transfer: select an architecture on one dataset, then retrain
  its weights from scratch on another dataset.
- Joint multi-dataset NAS: optimize mean validation performance across
  dataset-specific training runs while retaining dataset-specific input
  adapters.

Do not concatenate trials from the three datasets for the initial study. Their
channel layouts, acquisition systems, participant cohorts, and protocols are
different. Initial NAS studies use the same search space and trial budget but
separate data, weights, and Optuna studies.
