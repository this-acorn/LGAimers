# Results and lessons

## Metric

The repository uses a Brier-based probability score:

```text
Brier = mean((prediction - target)^2)
r = mean(target)
raw_score = 100000 * (1 - Brier / (r * (1 - r)))
```

The local implementation retains negative values for analysis. A constant prediction equal to the evaluation target mean has raw score zero. Higher is better; this is not accuracy or an accuracy percentage. See [exp/common.py](../exp/common.py).

## Historical public-leaderboard progression

| Version | Main change | Recorded score |
| --- | --- | ---: |
| Organizer baseline | Random Forest | 549.51193 |
| `submit4` | 65-feature histogram gradient boosting | 830.322760105 |
| `submit8` | Eight-seed CatBoost | 898.61863054 |
| `submit10` | Add 11 current-season features | 990.957167531 |
| `submit12` | Add three pitch-mix features | 993.6345481775 |
| `submit14` | Five-class outcome decomposition | 1033.9866361712 |

Source: the original [handoff log](archive/HANDOFF.md). These are selected observed milestones from the checked-in history. They do not establish a final score, rank, or independent test-set estimate. Submission archives are in [artifacts/submissions/](../artifacts/submissions/).

## What improved performance

- **Current-season information:** about +92.34 public points; the matching local experiment reported about +89.4. This transferred more strongly than the pitch-mix change.
- **Outcome decomposition:** the recorded local 2024 comparison improved from about 800.9 to 837.0, and the public score by about 40.35. Local gain decomposition attributed about 31.1 of the 36.1 points to the score after removing mean bias and about 5.0 to changed mean bias.
- **Probability averaging:** multiple seeds reduced dependence on a single fit. Increasing the CatBoost ensemble from eight to sixteen seeds did not improve the public result: `submit9` scored 897.6644306949 versus `submit8` at 898.61863054.

Evidence: [season-progress report](../lab/41_result.txt), [multiclass comparison](../lab/53_result.txt), [gain decomposition](../lab/59_result.txt), and [historical public results](archive/HANDOFF.md).

## What did not reliably transfer

| Experiment | Observation | Lesson |
| --- | --- | --- |
| Handedness-history adjustments | Positive historical backtests, but about 22.2 points worse publicly | Earlier-season consistency does not guarantee future transfer. |
| Pitch-mix features | About +18.6 locally, but only +2.68 publicly | A local improvement is not a fixed conversion to leaderboard points. |
| Sixteen-seed CatBoost | About 0.95 below the eight-seed public score | More computation may add no measurable benefit. |
| MLP blend probe | Reports mark the gate as failed, with best permitted mixture gain +0.0 | Model-family diversity alone does not imply useful ensemble diversity. |

The MLP observations are in [the archived report](../lab/deep_learning/exp101_mlp_blend/101_mlp_report.txt). They are experimental results, not part of `submit14`. The experiment-101 training script is not present as a standalone source file in this snapshot.

## Interpretation limits

Validation years were repeatedly consulted during development. Some later ablations use one year, and comparisons with different feature families or seed counts are not interchangeable. Public submissions also influenced model selection. Reported gains are competition development evidence, rather than an unbiased estimate of future-season performance.

Original logs retain unsuccessful hypotheses and revised conclusions, distinguishing observed scores, local diagnostics, and proposed experiments.
