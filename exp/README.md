# Experiments

This directory includes the local experiment series through the late-stage 192 builders, plus shared utilities. Numbers identify development steps and may have multiple branches.

| Stage | Entry points |
| --- | --- |
| Baselines and temporal evaluation | [Shared utilities](common.py), [multi-year harness](22_multiyear_harness.py) |
| Target and multiclass modeling | [Target analysis](52_target_anatomy.py), [multiclass comparison](53_multiclass.py), [full-data fit](54_train_mc79.py) |
| Hierarchical and temporal branches | [Hierarchical residuals](103_v18_residual.py), [temporal endpoint](116_rebuild_exp021_endpoint.py) |
| Calibration and probe analysis | [Affine scan](144_affine_feasible_scan.py), [blend geometry](190_solve_public_geometry.py) |
| Final component integration | [Flat residual blend builder](188_build_current_calico_jm_blend.py) |
| Inference checks | [Row independence](120_row_independence_qa.py), [rebuilt pipeline QA](147_row_independence_ours.py) |

See [methodology](../docs/METHODOLOGY.md) for the modeling details and [development journey](../docs/DEVELOPMENT.md) for the progression. Scripts preserve the exploratory workflow; a higher experiment number does not establish which model was submitted.

## Running selected work

Run scripts from the repository root in the matching Python environment. Training requires competition data and may also require local OOF predictions or supplementary experiment sources. The [artifact guide](../artifacts/README.md) covers inference-package assets. Many historical scripts execute training immediately when run or imported; inspect their arguments and expected inputs first.

```bash
python exp/54_train_mc79.py
```

This refits the earlier multiclass checkpoint and can take several hours. For inference with existing weights, use [the runner](../tools/run_inference.py). Individual experiments have different dependencies; there is no single environment lock covering every exploratory branch.
