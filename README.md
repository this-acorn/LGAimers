# Baseball Pitch Control Prediction

**LG Aimers 9th Cohort Hackathon**

Predict pitch-control success from pre-pitch game context and player history. Developed on approximately 1.48 million pitches from the 2019–2024 seasons.

## Approach

- **Feature engineering:** encoded count states, handedness, runners, smoothed player history, current-season form, and pitch-mix estimates.
- **Modeling:** trained five-class CatBoost models with eight-seed averaging and categorical team features; predicted the success-class probability.
- **Residual learning:** evaluated hierarchical corrections, LightGBM/HGB residuals, team effects, and low-rank interactions.
- **Ensembling:** combined complementary predictions with fixed calibration and blend weights.
- **Validation:** used chronological backtests, matched-seed comparisons, feature ablations, and Brier-loss analysis.
- **Inference:** packaged shared feature logic and frozen artifacts for independent predictions on each row.

**Stack:** Python, pandas, NumPy, scikit-learn, CatBoost, LightGBM, joblib.

[Methodology](docs/METHODOLOGY.md) · [Development journey](docs/DEVELOPMENT.md) · [Experiment guide](exp/README.md)

## Run the included checkpoint

The runnable example uses the preserved `submit14` multiclass checkpoint. Late-stage source is in `candidates/` and `probes/`; additional weights and ZIPs are kept locally ([artifact guide](artifacts/README.md)).

Use **Python 3.11** in a virtual environment. From the repository root, place the competition `test.csv` and optional `sample_submission.csv` in `data/`, then run:

```bash
python -m pip install -r submissions/submit14_src/requirements.txt
python tools/run_inference.py --data-dir data --output output/submission.csv
```

Output columns: `row_id`, `control_success`. To package the checkpoint:

```bash
python tools/build_submission.py --output output/submit14.zip
```

See the [data guide](docs/DATA.md) and [training instructions](exp/README.md#running-selected-work).

## Repository structure

```text
docs/                   Methodology, development notes, and data guide
exp/                    Experiment scripts and shared utilities
submissions/            Earlier inference packages and preserved checkpoints
candidates/             Late-stage ensemble inference packages
probes/                 Calibration and blend probes
tools/                  Inference and packaging commands
artifacts/              Archive inventory and local artifact locations
```
