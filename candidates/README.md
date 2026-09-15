# Late-stage candidates

This directory preserves locally developed inference packages and source extracted from the final attempts. It does not identify the best-scoring package or the submission order.

- [70% variant](candidate_jm_w070_src/): fixed 30:70 outer blend. The full and compact ZIPs contain matching member paths, sizes, and CRCs.
- [Last variant](last/): the same component model files with revised outer weights and final clipping.
- Other folders preserve affine calibration, hierarchical correction, temporal residual, and tensor candidates.

The top-level `script.py` controls inference. Some descriptive metadata retains an earlier blend setting; use the executable wrapper for the effective weights.

After restoring the [local model assets](../artifacts/README.md), install that package's `requirements.txt` in its own environment and run from the repository root:

```bash
python tools/run_inference.py --source candidates/last --data-dir data --output output/last.csv
python tools/build_submission.py --source candidates/last --output output/last.zip
```
