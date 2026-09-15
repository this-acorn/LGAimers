# -*- coding: utf-8 -*-
"""Sequential, row-local convex blend of two frozen probability endpoints."""

from __future__ import annotations

import gc
import runpy
from pathlib import Path

import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
CURRENT_WEIGHT = 0.30
JM_WEIGHT = 0.70
MODEL_DIR = Path("./model")
OUTPUT_DIR = Path("./output")
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"
SAMPLE_PATH = Path("./data/sample_submission.csv")


def _run_component(filename: str, namespace: str) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
    module = runpy.run_path(str(MODEL_DIR / filename), run_name=namespace)
    component_main = module.get("main")
    if not callable(component_main):
        raise RuntimeError(f"component has no callable main(): {filename}")
    component_main()
    if not OUTPUT_PATH.is_file():
        raise RuntimeError(f"component did not create {OUTPUT_PATH}: {filename}")
    prediction = pd.read_csv(OUTPUT_PATH, encoding="utf-8-sig")
    del component_main, module
    gc.collect()
    return prediction


def _validated_series(frame: pd.DataFrame, name: str) -> pd.Series:
    frame.columns = [str(column).replace("\ufeff", "").strip() for column in frame.columns]
    if list(frame.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"{name}: invalid columns {frame.columns.tolist()}")
    if frame[ID_COL].duplicated().any():
        raise ValueError(f"{name}: duplicate row_id")
    ids = frame[ID_COL].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"{name}: duplicate normalized row_id")
    values = frame[TARGET_COL].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: non-finite probability")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name}: probability outside [0,1]")
    return pd.Series(values, index=ids.to_numpy(), name=name)


def main() -> None:
    current = _validated_series(
        _run_component("current_inference.py", "current_affine_component"), "current")
    current_values = current.copy()
    del current
    gc.collect()

    jm = _validated_series(
        _run_component("jm0750_inference.py", "calico_jm0750_component"), "jm0750")

    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
    sample.columns = [str(column).replace("\ufeff", "").strip() for column in sample.columns]
    if list(sample.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample: invalid columns {sample.columns.tolist()}")
    if sample[ID_COL].duplicated().any():
        raise ValueError("sample: duplicate row_id")
    ids = sample[ID_COL].astype(str)
    if ids.duplicated().any():
        raise ValueError("sample: duplicate normalized row_id")
    if len(current_values) != len(sample) or len(jm) != len(sample):
        raise ValueError("component/sample row-count mismatch")

    current_aligned = current_values.reindex(ids.to_numpy()).to_numpy(np.float64)
    jm_aligned = jm.reindex(ids.to_numpy()).to_numpy(np.float64)
    if not np.isfinite(current_aligned).all() or not np.isfinite(jm_aligned).all():
        raise ValueError("row_id alignment produced missing component predictions")

    blended = CURRENT_WEIGHT * current_aligned + JM_WEIGHT * jm_aligned
    if not np.isfinite(blended).all() or np.any((blended < 0.0) | (blended > 1.0)):
        raise ValueError("invalid blended probabilities")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COL: sample[ID_COL], TARGET_COL: blended}).to_csv(
        OUTPUT_PATH, index=False, encoding="utf-8")
    print(
        f"Saved: {OUTPUT_PATH} | rows={len(sample):,} | "
        f"current={CURRENT_WEIGHT:.9f} jm0750={JM_WEIGHT:.9f} | "
        f"mean={blended.mean():.9f} min={blended.min():.9f} max={blended.max():.9f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
