# Experiment guide

Numbered scripts preserve the research sequence. Run from the repository root, since many paths are relative to it. Training can be expensive and can replace corresponding model bundles or predictions.

## Recommended reading order

| Question | Source |
| --- | --- |
| How are predictions scored and features shared? | [common.py](common.py) |
| How is chronological validation organized? | [22_multiyear_harness.py](22_multiyear_harness.py) |
| How were current-season features developed? | [41_season_progress.py](41_season_progress.py) |
| How are auxiliary labels reconstructed? | [45_label_recon.py](45_label_recon.py) |
| Why split the failure target? | [52_target_anatomy.py](52_target_anatomy.py) and [53_multiclass.py](53_multiclass.py) |
| How is the selected model trained? | [54_train_mc79.py](54_train_mc79.py) |
| Did improvement come from more than a mean shift? | [59_gain_decomp.py](59_gain_decomp.py) |

## Experiment families

| IDs | Focus |
| --- | --- |
| 01–18 | Signal discovery, ablations, calibration, tracking history, and season stability |
| 19–35 | Baseline rebuilds, chronological comparisons, CatBoost, LightGBM, and ensemble weights |
| 36–50 | Row matching, player history, season-progress features, and pitch-mix supervision |
| 51–64 | Target anatomy, multiclass training, error analysis, distillation, and capacity sweeps |

Results are in [lab/](../lab/). The checked-in [MLP reports](../lab/deep_learning/exp101_mlp_blend/) extend beyond this standalone source sequence; not every reported experiment has a matching source script here. A historical team package remains in [artifacts/submissions/teampack_100.zip](../artifacts/submissions/teampack_100.zip).

## Running selected work

Install [the selected environment](../submissions/submit14_src/requirements.txt) and provide [the data files](../docs/DATA.md). From the repository root:

```bash
# Multi-year HGB comparison: several model fits per year.
python -X utf8 exp/22_multiyear_harness.py

# Full-data eight-seed, five-class CatBoost training.
python -X utf8 exp/54_train_mc79.py
```

Other scripts may require predictions from earlier experiments, optional libraries, or private data. Original packaging scripts contain historical workstation-specific scratch directories and interpreter paths. For portable packaging and inference, use [tools/run_inference.py](../tools/run_inference.py) and [tools/build_submission.py](../tools/build_submission.py).

Original Korean comments and reports remain as research history. The English [methodology](../docs/METHODOLOGY.md) describes the selected path and its limitations.
