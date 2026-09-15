# Usage

Use Python 3.11 in a virtual environment and run commands from the repository root. Place `test.csv` and `sample_submission.csv` in `data/`; see the [data guide](DATA.md).

## Inference example

The included `submit14` checkpoint demonstrates the intermediate multiclass model. Ensemble packages require their additional model assets.

```bash
python -m pip install -r submissions/submit14_src/requirements.txt
python tools/run_inference.py --data-dir data --output output/submission.csv
```

The output contains `row_id` and `control_success`. To run another package, install its `requirements.txt` and pass its folder with `--source` to the inference command.

## Model assets

Additional weights and archives are stored locally. The [asset inventory](../artifacts/manifest.json) records the required files. Restore missing assets from a local workspace copy with:

```bash
python tools/restore_artifacts.py --from-dir /path/to/local/workspace
```

The command preserves existing files and reports missing assets; it does not download weights.

## Packaging

```bash
python tools/build_submission.py --output output/submission.zip
```

Use `--source` to package a different inference folder.

## Training

Training requires the competition data and the selected experiment's dependencies. For the multiclass model:

```bash
python exp/54_train_mc79.py
```

This refits the checkpoint and can take several hours. Other [experiments](../exp/README.md) may also require saved OOF predictions or supplementary local sources.
