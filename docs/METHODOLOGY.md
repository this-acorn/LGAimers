# Methodology

## Prediction task

Predict the probability of `control_success` for each pitch using information available before that pitch. Inputs include game context, player and team identifiers, and historical `asof_*` statistics. `row_id` is used only to align outputs.

## Feature engineering

- **Game context:** count state, inning, handedness matchups, and baserunners.
- **Historical reliability:** smoothed success rates, log observation counts, and missing-history indicators to stabilize sparse player records.
- **Current-season form:** season-to-date estimates and deviations from longer-term history.
- **Pitch mix:** pitcher/count tendencies and an auxiliary fastball probability estimate.

Training and inference share feature functions, while prediction uses frozen historical tables. See the [feature implementation](../submissions/submit14_src/script.py).

## Multiclass CatBoost

Split the binary target into successful control and four failure categories using reconstructed middle-location and reverse-direction indicators. This allows the model to distinguish different failure patterns while returning the success-class probability.

Average eight CatBoost models trained with different seeds. Later configurations include categorical team identifiers alongside game-context features. Auxiliary target reconstruction is performed during training; inference requires only the pre-pitch inputs.

Implementation: [target construction](../exp/52_target_anatomy.py) and [multiclass training](../exp/54_train_mc79.py).

## Residual learning

Hierarchical corrections capture pitcher, handedness, game-type, and count-pressure effects. Sparse subgroups are shrunk toward broader parent groups.

A complementary temporal pipeline combines empirically smoothed pitcher/batter rates with LightGBM and histogram-based gradient-boosting residuals, team effects, and low-rank interactions. A further prediction stack adds adaptive gating, game-type experts, and bounded CatBoost residual corrections on regular-season rows.

Implementation: [hierarchical corrections](../exp/103_v18_residual.py), [temporal pipeline](../candidates/last/model/exp021_inference.py), and [regular-season residuals](../candidates/last/model/jm0750_inference.py).

## Calibration and ensembling

Fixed probability shifts and scaling adjust bias and spread. Blend analysis uses the quadratic relationship between mixture weights and Brier loss to compare complementary predictions. The final development stage focused on tuning ensemble weights and packaging the inference pipeline.

Implementation: [calibration experiments](../exp/144_affine_feasible_scan.py) and [ensemble inference](../candidates/last/script.py).

## Validation and inference

Chronological backtests train on earlier seasons and evaluate later seasons. Matched-seed comparisons and ablations test whether a change improves the baseline or adds useful ensemble diversity. Validation seasons and competition feedback informed model selection.

Inference uses one input row and frozen training artifacts. Checks cover row-order invariance, output alignment, and finite probabilities within [0, 1].

Implementation: [temporal evaluation](../exp/22_multiyear_harness.py) and [row-independence checks](../exp/120_row_independence_qa.py).
