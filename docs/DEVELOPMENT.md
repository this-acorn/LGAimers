# Development process

## Establish a baseline

Start with Random Forest and histogram-based gradient boosting to check the data pipeline and probability-based evaluation. Analyze errors by season, player history, and game context to identify useful feature groups.

## Refine features and targets

Add smoothed history, current-season form, and pitch-mix estimates. Decompose failures into multiple classes and train seed-averaged CatBoost models to capture distinct outcome patterns.

## Test complementary models

Evaluate hierarchical residuals, temporal models, low-rank interactions, and neural candidates. Compare each change against matching validation years and seeds, then check whether its predictions improve an ensemble. Use ablations to identify which components contribute.

## Integrate and validate

Calibrate the selected predictions and tune ensemble weights. Freeze feature statistics, model parameters, and correction tables for inference. Package the components with row-ID alignment and probability checks, then test that predictions remain consistent across batch order and composition.

See [methodology](METHODOLOGY.md) for model details and [experiments](../exp/README.md) for implementation entry points.
