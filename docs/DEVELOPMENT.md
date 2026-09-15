# Development journey

This overview records the methods used to reach the final pipeline. It focuses on modeling decisions and engineering work rather than leaderboard results.

## From baseline to final ensemble

| Stage | Methods used | Purpose |
| --- | --- | --- |
| Baselines | Random Forest and histogram-based gradient boosting | Establish the task, feature contract, and probability-evaluation workflow. |
| Contextual features | Count states, handedness, baserunners, missing-history flags, and smoothed rates | Expose useful structure and the reliability of historical estimates. |
| Season-aware features | Career/current-season decomposition, recent-form differences, and pitch-mix estimates | Distinguish long-run player tendencies from current form. |
| Core model | Five-class CatBoost, eight-seed averaging, and native team categorical features | Learn different failure mechanisms and reduce dependence on one fit. |
| Hierarchical correction | Shrunk pitcher/handedness/pressure effects | Model conditional differences while stabilizing sparse groups. |
| Complementary branch | Temporal empirical-Bayes base, LightGBM/HGB residuals, team effects, and low-rank interactions | Add a model with a different inductive structure and error pattern. |
| Calibration and blending | Fixed probability shifts/scaling and quadratic Brier-loss blend analysis | Adjust bias and combine complementary predictions. |
| Final integration | Blend the calibrated pipeline with an adaptive prediction stack and a regular-season CatBoost residual ensemble | Combine complementary predictions in the final local package. |
| Packaging | Frozen artifacts, sequential execution, row-ID alignment, and probability checks | Make the selected computation reproducible and independent of test-batch ordering. |

## How experiments informed development

- Compare candidates with matching seeds and validation years to reduce avoidable differences between runs.
- Train on earlier seasons and evaluate later seasons to expose temporal distribution shift.
- Separate changes in overall probability bias from changes in the shape of predictions.
- Test whether a candidate adds complementary information to an existing ensemble, rather than judging it only in isolation.
- Use ablations and confirmation checks to reject unhelpful additions. Explored directions included expanded historical features, distillation, greater tree capacity, MLPs, and tensor/interaction residuals.
- Keep model parameters, correction tables, calibration constants, and blend coefficients frozen at inference.

Some validation years and public feedback were consulted repeatedly. These were model-development tools, not independent final-test estimates. The final methods should not be confused with every experiment proposed near the deadline.

## Implementation map

| Evidence | What it establishes |
| --- | --- |
| [Shared utilities](../exp/common.py) and [multi-year harness](../exp/22_multiyear_harness.py) | Feature engineering, Brier diagnostics, and chronological comparisons. |
| [Target analysis](../exp/52_target_anatomy.py) and [multiclass training](../exp/54_train_mc79.py) | The five-class, eight-seed CatBoost checkpoint included here. |
| Local `candidate_v18g030_src/script.py` | Later team categorical handling, hierarchical correction, and fixed calibration. |
| `model/exp021_inference.py` inside local `last.zip` | The rebuilt temporal/residual branch and its frozen feature transformations. |
| `model/current_inference.py` inside local `last.zip` | The calibrated combination of the existing branches. |
| `script.py`, `model/jm0750_inference.py`, and its residual manifest inside local `last.zip` | The actual final two-branch blend and the regular-season residual component. |

The later local artifacts are described here but are not bundled into this checkout. The inference commands in the root README run the earlier preserved checkpoint. The final package was inspected directly because a later-numbered builder script alone does not establish which implementation was actually packaged.

The project's work includes feature development, multiclass modeling, experiments, inference implementation, calibration, component integration, and packaging.
