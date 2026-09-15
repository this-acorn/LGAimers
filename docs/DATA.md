# Data guide

Obtain the competition files separately and place them in a local `data/` directory.

| File | Contents | Used for |
| --- | --- | --- |
| `train.csv` | 1,475,092 rows × 49 columns, including `control_success` | Training and chronological backtests |
| `test.csv` | Distributed five-row schema sample with 48 input columns | Inference smoke checks |
| `sample_submission.csv` | Five-row example: `row_id`, `control_success` | Output schema and ordering |
| `trackman_history.csv` | 1,793,078 records × 30 columns, covering 2019–2024 | Exploratory tracking-history experiments |

Official evaluation replaces the sample with hidden inputs. Passing the five-row sample checks functionality, not predictive quality.

Inputs describe game context, counts, scores, baserunners, player/team identifiers, handedness, and precomputed `asof_*` history. Missing rates can occur when a player has no prior observations. The selected multiclass submission does not require `trackman_history.csv` for inference or full-data training.

The organizer's detailed Korean specification is preserved in [data_description.md](data_description.md). Competition CSVs are not included in this cleanup.
