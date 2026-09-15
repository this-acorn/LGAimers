# Methodology

This document describes the selected `submit14` implementation and the experiments that led to it. It is a guide to the recorded work, rather than a claim that every exploratory branch became part of the deployed model.

## Problem formulation

Each row describes the state immediately before a pitch. The target, `control_success`, indicates whether that pitch satisfied the competition's control criterion. The output is a probability, not a predicted pitch location or pitch type.

Training covers 2019–2024. Inputs include counts, inning, baserunners, handedness, anonymized player/team identifiers, and organizer-provided `asof_*` statistics. `row_id` is used for output matching and excluded from model features.

## Feature engineering

| Group | Count | Implementation and rationale |
| --- | ---: | --- |
| Supplied predictors | 47 | Preserve available pre-pitch context and history. |
| Context and reliability | 18 | Encode count states, handedness, runners, log history counts, smoothed rates, recent-form deviations, and missing history. |
| Current-season progress | 11 | Estimate pitcher/batter season-to-date counts, rates, and deviations using cumulative inputs and fixed earlier-history totals. |
| Pitch mix | 3 | Add fastball/breaking-ball tendencies by pitcher and count group, plus predicted fastball probability. |

The inference implementation is [script.py](../submissions/submit14_src/script.py). Its functions are imported by [the training script](../exp/54_train_mc79.py), keeping the main transformations consistent between training and inference.

### Smoothing sparse history

A rate based on a small number of pitches is noisy. Contextual features use `(n * rate + 200 * prior) / (n + 200)`, where `prior` is the training target mean. Log counts and missing-value indicators describe how much history supports an estimate.

### Current-season features

For a test row, the supplied cumulative count and rate imply a cumulative total. Subtracting a training-derived historical total estimates progress since the training boundary. The implementation clamps negative counts, requires at least five observations for the season rate, and applies shrinkage with weight 50.

Training features use earlier-season constants for each season. Deployment constants are frozen from the complete training history. This produces dynamic features without computing statistics across test rows.

### Pitch-type supervision

Consecutive cumulative fastball and breaking-ball rates provide auxiliary training labels where reconstructable. These support a pitcher-by-count lookup table and a separate CatBoost fastball classifier. Inference uses historical tendencies and the classifier's probability; it does not receive the current pitch's true type.

The full-data refit computes auxiliary fastball predictions on the same training rows used to fit that auxiliary model. This is an implementation detail to consider when designing a stricter out-of-fold reproduction.

## Multiclass target construction

The binary failure class is decomposed using supplied middle-location and reverse-direction indicators:

| Class | Training condition |
| --- | --- |
| 0 | Successful control |
| 1 | Failure with the middle-location indicator only |
| 2 | Failure with the reverse-direction indicator only |
| 3 | Failure with both indicators |
| 4 | Failure with neither indicator; described as “big miss” in the original notes |

The last class is a residual category in the code, not an independently measured physical miss distance.

For consecutive observations from the same pitcher, a cumulative event total is estimated as `rate * count`. A difference between adjacent totals reconstructs an event indicator when the count advances by exactly one. Nonbinary or unavailable reconstructions are marked missing; rows without the required reconstructed labels are excluded from the multiclass fit. The training script also reconstructs the success indicator and asserts greater than 99.9% agreement with the supplied binary target on comparable rows.

Reconstruction is a training-time operation, not an operation across test rows. A fresh evaluation should reconstruct labels within each training partition and check season boundaries explicitly; historical scripts should not be treated as proof of an untouched holdout.

Sources: [target analysis](../exp/52_target_anatomy.py), [multiclass comparison](../exp/53_multiclass.py), [full-data training](../exp/54_train_mc79.py).

## Selected CatBoost configuration

| Parameter | Value |
| --- | --- |
| Objective | `MultiClass` |
| Iterations | 500 |
| Depth | 6 |
| Learning rate | 0.08 |
| L2 leaf regularization | 10.0 |
| Seeds | 42, 7, 123, 2024, 99, 555, 31337, 1 |
| Native categorical columns | `top_bottom`, `game_type`, `base_state` |
| Prediction | Mean class-0 probability from eight models, clipped to [0, 1] |

The selected version does not use a neural network, a TrackMan physics model, or an affine calibration layer in its inference path. Those approaches appear in exploratory work or other versions and should not be attributed to this bundle.

## Validation and development workflow

1. **Measure a baseline.** Inspect target balance, distributions, and variability caused by training seeds.
2. **Form a specific hypothesis.** Examples include isolating current-season form or splitting heterogeneous failures into separate labels.
3. **Compare matching runs.** Keep validation year and random seeds fixed across baseline and candidate; change one feature family or model setting at a time where possible.
4. **Check temporal transfer.** The [multi-year harness](../exp/22_multiyear_harness.py) trains on seasons before each validation year and evaluates 2021–2024. Not every later experiment repeats the full harness.
5. **Explain score changes.** Decompose Brier performance into a mean-bias penalty and the score after analytically centering predictions. This centered quantity uses validation labels for diagnosis; it is not an inference-time correction.
6. **Confirm and package.** Refit a selected model, save its metadata and historical constants, and check output shape, probabilities, and runtime.

Repeated use of 2024 for feature and hyperparameter decisions makes it a development set. Multiple seeds estimate model variability but do not create independent holdout datasets. The original [validation harness](../exp/22_multiyear_harness.py) explicitly discusses this limitation.

## Deployment design

The bundle stores CatBoost models, feature names, a training prior, historical pitcher/batter constants, pitch-mix tables, and the auxiliary classifier. The standalone script reads `data/test.csv` and writes `output/submission.csv`, matching sample-submission order when a compatible file is supplied.

Every feature is a function of one input row and fixed training artifacts. The portable [runner](../tools/run_inference.py) supplies the directory layout expected by the original script. The [packager](../tools/build_submission.py) preserves the archive paths `script.py`, `requirements.txt`, and `model/model.pkl`.
