# Baseball Pitch Control Prediction

**LG Aimers 9th Cohort Hackathon**

Predict pitch-control success from pre-pitch game context and player history. Developed on approximately 1.48 million pitches from the 2019–2024 seasons.

## Approach

- **Features:** count states, handedness, runners, smoothed player history, current-season form, and pitch-mix estimates.
- **Core model:** five-class CatBoost with eight-seed averaging and native categorical team features; return the success-class probability.
- **Residual learning:** hierarchical corrections, LightGBM/HGB residual models, team effects, and low-rank interactions.
- **Ensembling:** combine complementary prediction pipelines with fixed probability calibration and blend weights.
- **Validation:** chronological backtests, matched-seed comparisons, feature ablations, and Brier-loss analysis.
- **Inference:** shared feature logic, frozen training artifacts, and independent predictions for each input row.

**Stack:** Python, pandas, NumPy, scikit-learn, CatBoost, LightGBM, joblib.

[Methodology](docs/METHODOLOGY.md) · [Development journey](docs/DEVELOPMENT.md) · [Experiment guide](exp/README.md)

## Run the included checkpoint

The runnable example uses the preserved `submit14` multiclass checkpoint. Later ensemble configurations are described in the methodology; their complete model packages are not included here.

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
submissions/            Inference code and preserved model checkpoints
tools/                  Inference and packaging commands
lab/                    Experiment reports
artifacts/submissions/  Historical submission archives
archive/                Earlier local automation
```
