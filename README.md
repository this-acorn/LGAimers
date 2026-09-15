# Baseball Pitch Control Prediction

**LG Aimers 9th Cohort Hackathon**

A machine-learning pipeline for predicting pitch-control success from pre-pitch game context and player history, developed using approximately 1.48 million pitches from the 2019–2024 seasons.

## Approach

- **Feature engineering:** game context, smoothed player history, current-season form, and pitch-mix estimates.
- **Multiclass modeling:** decomposed control outcomes into five classes and averaged eight CatBoost models to estimate success probability.
- **Residual learning:** combined hierarchical corrections, temporal models, LightGBM/HGB residuals, and low-rank interactions.
- **Ensembling:** calibrated probabilities and tuned blend weights across complementary prediction pipelines.
- **Validation:** chronological backtests, feature ablations, matched-seed comparisons, and row-independent inference checks.

**Stack:** Python, pandas, NumPy, scikit-learn, CatBoost, LightGBM, joblib.

[Methodology](docs/METHODOLOGY.md) · [Development process](docs/DEVELOPMENT.md) · [Code overview](docs/REPOSITORY_MAP.md) · [Usage](docs/RUNNING.md)
