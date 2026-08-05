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
  matrix. First train each architecture-target pair for 100 epochs on its fixed
  training/validation split, recording metrics and weights every 10 epochs.
  Choose a fixed epoch count from those learning curves. Then reinitialize and
  train on the target dataset's combined development subjects using
  normalization computed only from those subjects. Freeze the nine resulting
  models before evaluating each once on the target dataset's locked test
  subjects.

The approved `eeg-parzen/cnn.py` search space contains an
EEGNet-inspired temporal convolution, channel-spanning depthwise spatial
filter, and one to six separable temporal blocks. Candidate models use two
logits and cross-entropy. Early stopping and Optuna selection use validation
balanced accuracy. BNCI scores are averaged over its three subject folds;
Lee2019_MI and PhysionetMI use fixed validation-subject groups.

`eeg-parzen/control-logic.py` is the single source of the NAS seed. Setting
`EEG_SEED` controls creation of the corresponding subject split, Python,
NumPy, PyTorch, DataLoader shuffling, model initialization, and the TPE sampler.
Seed-dependent data artifacts live under `data/seed-<seed>/`; for example,
seed 0 uses `data/seed-0/splits_seed0.json`. Raw and normalized MOABB data are
shared because their deterministic preprocessing does not depend on NAS seed.
NAS studies use `data/seed-<seed>/eeg-parzen/<dataset>/<run-id>/`.
Seed-specific JSONL, SQLite, split, and baseline artifacts are tracked in Git.
Only the large shared `data/moabb_raw/` download cache and
`data/moabb_processed/` subject arrays remain ignored; neither should be
deleted because they are the reusable source and prepared EEG data.
`eeg-parzen/run/run-eeg-parzen.sh` is intentionally minimal: it installs requirements
into the active Python environment and runs the three 100-proposal studies
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
or downstream simulator stage. Calibration, final training, and locked testing
are explicit post-search commands so locked-test evaluation cannot begin as
part of NAS.

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
channel-progression rule. Do not run final training until a corrected search
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

## Pre-CUDA-run implementation audit

The active pipeline passed all eight data-loader tests and a dry 50-proposal
Optuna study. Subject partitions are disjoint, normalization uses training
subjects only, locked test subjects are not loaded by NAS, balanced accuracy
is computed correctly, conditional parameter distributions are static, block
widths are non-decreasing, and pooling cannot reduce the temporal dimension
below one. A 5,000-architecture simulation across 22, 62, and 64 input
channels produced no model above 350,000 parameters; median models contained
roughly 31,000--33,000 parameters.

The CUDA run is valid as the planned initial search. Its main limitation is
coverage: 50 proposals are sparse relative to the large conditional space,
and TPE may repeat configurations. Exact continuation after interruption and
cross-machine repetition also depend on Optuna sampler state and package
versions. These are reporting and follow-up concerns, not reasons to stop the
active run.

## Completed seed-0 CUDA NAS

Run `nas-seed0-10a2117c` completed its first 50 proposals for each dataset on CUDA with no
failed or pruned trials. Best validation balanced accuracies were 68.06% for
BNCI2014_001 (trial 34, 47,034 parameters), 80.87% for Lee2019_MI (trial 44,
102,786 parameters), and 78.04% for PhysionetMI (trial 20, 22,050 parameters).
PhysionetMI class recalls were 83.42% left and 72.65% right, resolving the
earlier class-collapse problem. Locked-test evaluation has not yet run.

Post-search review found 44, 47, and 43 unique architectures for BNCI,
Lee2019_MI, and PhysionetMI, respectively. Every repeated architecture
reproduced identical fold metrics on CUDA. TPE improved the best score from
the first ten trials to the final winner on all three datasets. Model size had
no significant monotonic association with validation performance. Lee2019_MI
and PhysionetMI produced balanced, stable winners and converged strongly on
flatten heads, ELU, 32 temporal filters, stem pooling 2, and spatial multiplier
1. BNCI remained less stable because only seven development subjects were
available; its winning mean was 68.06%, with fold scores of 70.49%, 58.68%,
and 75.00%. Keep locked testing closed until fixed baselines are trained and
reviewed under the same CUDA environment.

Run the five fixed baselines in two groups with
`benchmarking/run/run-baselines-2.sh` and
`benchmarking/run/run-baselines-3.sh`. Each group trains sequentially across
all three datasets under the separate run ID `baseline-seed0`, and the runner
skips completed entries when resumed.

The seed-0 CUDA baseline calibration completed all five architectures on all
three datasets under `baseline-seed0`. Best baseline validation balanced
accuracies were 69.27% for BNCI2014_001 (`wide_eegnet`, 7,026 parameters),
79.88% for Lee2019_MI (`dilated_temporal`, 13,250 parameters), and 76.67% for
PhysionetMI (`deep_separable`, 6,530 parameters). The NAS winners scored
68.06%, 80.87%, and 78.04%, respectively. Thus NAS trails the best BNCI
baseline by 1.22 percentage points and leads the best Lee and Physionet
baselines by 1.00 and 1.36 points. These small validation differences are not
treated as test conclusions. Concurrent baseline runs invalidate their timing
measurements, so final latency must be measured sequentially during locked
evaluation.

Before locked testing, expand the architecture-search budget to 100 proposals
per dataset for seeds 0, 1, and 2. Seed 0 resumes `nas-seed0-10a2117c` from 50 to 100;
seeds 1 and 2 generate run IDs of the form `nas-seed<seed>-<id>`. Launch them
with `eeg-parzen/run/run-nas.sh SEED [RUN_ID]`. The seed controls subject
splitting, TPE, initialization, and data ordering. Keep all locked tests
closed until these searches and their corresponding baselines are complete.

After validation review fixes all nine seed-specific NAS winners, calibrate the
training duration, train the frozen final models, and only then run locked
testing.
Concurrent execution is acceptable for
accuracy training, but final accelerator latency must be measured separately
without GPU contention. Never rerun or retune based on locked-test outcomes.

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

## Live run status — 2026-08-04 evening

The canonical seed-0 run `nas-seed0-10a2117c` is complete at 100 COMPLETE trials for
each dataset (300 total). SQLite timestamps confirm final completion at 12:34
for BNCI2014_001, 16:03 for Lee2019_MI, and 17:53 for PhysionetMI. The final
seed-0 PhysionetMI best remains trial 20 at 78.04% validation balanced
accuracy.

At 22:06, one active nonzero-seed run was on its third and final dataset,
PhysionetMI, with trials 0--64 complete (65/100). Its best was trial 57 at
76.5679% validation balanced accuracy. The immediately preceding Lee2019_MI
study reached at least trial 96 and had a best of 83.3125% at trial 39. Confirm
the run ID/seed from its tmux command or SQLite path before recording these as
final seed-specific results; the pasted Optuna lines do not contain the study
name.

At 22:13, seed 2 run `nas-seed2-61979a0f` completed Lee2019_MI at 100/100.
Its winner is trial 54 with 78.125% validation balanced accuracy. It then
created its third and final study, `nas-seed2-61979a0f_physionet_mi`. This
identification is conclusive from the Optuna study-creation line.

Do not access any locked test subjects yet. After seeds 1 and 2 finish, audit
the nine canonical SQLite studies and validation winners, then obtain explicit
user confirmation before any final-testing command. Final evaluation is a 3x3
architecture-transfer grid per seed: each source-dataset NAS winner is
reinitialized and trained on each target dataset's combined train+validation
subjects, then evaluated once on that target's locked test subjects. Average
each grid cell across seeds 0, 1, and 2; diagonal cells are primary
dataset-specific NAS results and off-diagonal cells measure architecture-only
transfer. Never transfer learned weights or concatenate datasets.

### Local artifact audit after GPU pull

The local pull inspected on 2026-08-04 is a partial snapshot of the live GPU
runs. Canonical completed winners at that snapshot are:

- Seed 0: BNCI trial 79 = 68.2099% (48,130 parameters); Lee trial 76 =
  81.3750% (86,274); PhysioNet trial 20 = 78.0363% (22,050).
- Seed 1: BNCI trial 30 = 77.6813% (30,306); Lee trial 39 = 83.3125%
  (86,018). Seed-1 PhysioNet had only 71 COMPLETE plus one RUNNING trial;
  its provisional best was trial 66 = 77.8592% and is not final.
- Seed 2: Lee trial 54 = 78.1250% (56,642). Seed-2 BNCI contains 100 COMPLETE
  trials; its final best is trial 84 = 74.7492%. Seed-2 PhysioNet had only
  one COMPLETE plus one RUNNING trial and no meaningful final result yet.

Reminder: rerun seed-2 BNCI for the separate follow-up task.

All completed canonical JSONL line counts match SQLite COMPLETE counts. The
three completed Lee studies average 80.9375% validation balanced accuracy
(sample SD 2.6213 percentage points). Do not average BNCI or PhysioNet across
seeds until their incomplete studies finish. All listed winners use the
flatten head; activation and block counts vary.
