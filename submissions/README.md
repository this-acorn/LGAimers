# Submission sources

Each package contains a standalone inference script, pinned dependencies, and a model bundle when available. Version names preserve the experiment history.

| Directory | Role |
| --- | --- |
| `baseline_submit/` | Organizer-provided Random Forest baseline |
| `submit/` | Evolving early HGB/CatBoost package; historical ZIPs preserve exact submitted versions |
| `submit8_extract/` | Extracted eight-seed CatBoost model used in later blends |
| `submit10_src/` | CatBoost with 76 features, including current-season statistics |
| `submit11_src/` | Blend of earlier CatBoost and season-progress models |
| `submit12_src/` | CatBoost with 79 features, including pitch-mix estimates |
| `submit13_src/` | Blend of 79-feature and earlier CatBoost models |
| **`submit14_src/`** | **Reproducible five-class, 79-feature CatBoost checkpoint** |
| `submit15_src/` | Three-way binary-model blend |

See the root [README](../README.md#run-the-included-checkpoint) for checkpoint inference and the [methodology](../docs/METHODOLOGY.md) for later development through the final local ensemble. Historical ZIPs are in [artifacts/submissions/](../artifacts/submissions/); source folders may reflect development after an earlier ZIP was created.
