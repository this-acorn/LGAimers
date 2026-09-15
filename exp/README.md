# Experiments

Feature, model, and validation experiments supporting the [development process](../docs/DEVELOPMENT.md).

| Area | Entry points |
| --- | --- |
| Features and temporal evaluation | [Shared utilities](common.py), [chronological backtests](22_multiyear_harness.py) |
| Multiclass modeling | [Target analysis](52_target_anatomy.py), [training](54_train_mc79.py) |
| Residual learning | [Hierarchical corrections](103_v18_residual.py), [temporal residuals](143_exp021_component.py) |
| Calibration and ensembling | [Affine calibration](144_affine_feasible_scan.py), [ensemble packaging](188_build_current_calico_jm_blend.py) |
| Inference validation | [Row-independence checks](120_row_independence_qa.py) |

See [usage](../docs/RUNNING.md#training) for training requirements. Inspect each script's inputs before running it; some experiments depend on earlier local results.
