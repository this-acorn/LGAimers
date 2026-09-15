# Data

The training data covers approximately 1.48 million pitches from 2019–2024. Each row contains pre-pitch context, player/team identifiers, and historical statistics; the target is `control_success`.

| File | Purpose |
| --- | --- |
| `train.csv` | Model training and chronological validation. |
| `test.csv` | Prediction inputs; the distributed sample contains five rows. |
| `sample_submission.csv` | Expected output columns and row order. |
| `trackman_history.csv` | Additional tracking data used in exploratory feature experiments. |

Competition data is stored locally in `data/`. The multiclass inference example does not require tracking data.

See the [official data specification](data_description.md) for field definitions and [usage](RUNNING.md) for execution instructions.
