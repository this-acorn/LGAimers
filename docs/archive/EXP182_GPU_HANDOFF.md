# EXP182 GPU handoff

`exp182` is an official-train-only CUDA/AMP validation run for the fast robust-z MLP endpoint. It does not read `test.csv`. The final epoch of each seed is used; validation labels do not select an epoch.

## Files that must exist on the GPU machine

Place the competition file at `data/train.csv`. The repository checkout must contain:

- `exp/168_elephant_stdmlp_clean_oof.py`
- `exp/179_fast_stdmlp_endpoint.py`
- `exp/182_gpu_stdmlp_handoff.py`
- `lab/179_current_2024.npy` (SHA256 `e78b168244675adea4bd6a3d56f021a9376cd4b78c428d536a9b53b02fcebe9b`)
- `reference/LA9elephantmiracle/LICENSE`
- `reference/LA9elephantmiracle/cowork/cw/v17/src/common.py`
- `reference/LA9elephantmiracle/performance_tracking/models/sj_stdmlp/prep_mlp.py`

The two reference Python files are MIT-licensed source only. No repository prediction, checkpoint, OOF, or competition-data artifact is used.

Install `numpy`, `pandas`, and a CUDA-enabled build of `torch` that matches the lab machine's CUDA driver. Confirm CUDA before the run:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## One-line validation command

```bash
python exp/182_gpu_stdmlp_handoff.py validate --device cuda --amp --epochs 8 --batch 4096 --seeds 0 1 2
```

If GPU memory allows, only the batch size may be raised before the first run (for example `--batch 32768`). Once validation starts, keep the recipe unchanged. Outputs go to `lab/182_gpu_stdmlp/`:

- `endpoint_2024_seed*.npy`
- `validation_seed*.pt`
- `endpoint_2024_seed_ensemble.npy`
- `validation_report.json`, including each seed's history and the ensemble's `d`, `K`, analytic convex weight, gain, early/late gains, and pitcher-cluster bootstrap.

The predeclared pass gate is: analytic blend gain at least 20, early and late gains both positive, and pitcher-bootstrap 2.5th percentile positive. A failed report is not a submission candidate.

## Final training, only after PASS

If `validation_report.json` says `PASS_CANDIDATE_FOR_FINAL_TRAIN`, run the exact same recipe:

```bash
python exp/182_gpu_stdmlp_handoff.py final_train --device cuda --amp --epochs 8 --batch 4096 --seeds 0 1 2
```

This saves three all-through-2024 states. It still does not read test data or build a submission ZIP; inference packaging and row-independence rehearsal remain separate required steps.

## Git staging note

The workspace ignores `reference/`, so the three audited source/license files must be force-added when creating the handoff commit. `data/train.csv` must not be committed.
