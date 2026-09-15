"""Sequential, row-local blend of two frozen probability endpoints."""

from __future__ import annotations

import gc
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
CHAMPION_WEIGHT = 0.6434428305247574
EXP021_WEIGHT = 0.35655716947524263
MODEL_DIR = Path("./model")
OUTPUT_DIR = Path("./output")
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"
SAMPLE_PATH = Path("./data/sample_submission.csv")

# --- 최종 혼합 확률에 대한 사후 선형 재조정 (행 단위 상수 연산, 행 간 독립) ---
BLEND_SCALE = 0.96
BLEND_CENTER = 0.44
BLEND_SHIFT = 0.0


def _run_component(filename: str, namespace: str) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
    module = runpy.run_path(str(MODEL_DIR / filename), run_name=namespace)
    main = module.get("main")
    if not callable(main):
        raise RuntimeError(f"component has no callable main(): {filename}")
    main()
    if not OUTPUT_PATH.is_file():
        raise RuntimeError(f"component did not create {OUTPUT_PATH}: {filename}")
    prediction = pd.read_csv(OUTPUT_PATH, encoding="utf-8-sig")
    del main, module
    gc.collect()
    return prediction


def _validate_component(frame: pd.DataFrame, name: str) -> None:
    if list(frame.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"{name}: invalid columns {frame.columns.tolist()}")
    if frame[ID_COL].duplicated().any():
        raise ValueError(f"{name}: duplicate row_id")
    values = frame[TARGET_COL].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: non-finite probability")
    if not ((values >= 0.0).all() and (values <= 1.0).all()):
        raise ValueError(f"{name}: probability outside [0,1]")


def main() -> None:
    champion = _run_component("champion_inference.py", "champ_component")
    _validate_component(champion, "champion")
    champion_values = champion[TARGET_COL].to_numpy(dtype=np.float64, copy=True)
    champion_ids = champion[ID_COL].to_numpy(copy=True)
    del champion
    gc.collect()

    exp021 = _run_component("exp021_inference.py", "exp021_component")
    _validate_component(exp021, "exp021")
    exp021_ids = exp021[ID_COL].to_numpy()
    if not np.array_equal(champion_ids, exp021_ids):
        raise ValueError("component row_id order mismatch")
    exp021_values = exp021[TARGET_COL].to_numpy(dtype=np.float64)

    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
    sample.columns = [column.replace("﻿", "").strip() for column in sample.columns]
    if ID_COL not in sample.columns or len(sample) != len(champion_ids):
        raise ValueError("sample_submission schema/row count mismatch")
    if not np.array_equal(sample[ID_COL].to_numpy(), champion_ids):
        raise ValueError("component order differs from sample_submission")

    blended = CHAMPION_WEIGHT * champion_values + EXP021_WEIGHT * exp021_values
    blended = np.clip(
        BLEND_CENTER + BLEND_SCALE * (blended - BLEND_CENTER) - BLEND_SHIFT, 0.0, 1.0)
    if not np.isfinite(blended).all() or not ((blended >= 0.0).all() and (blended <= 1.0).all()):
        raise ValueError("invalid blended probabilities")
    submission = pd.DataFrame({ID_COL: sample[ID_COL], TARGET_COL: blended})
    submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(
        f"Saved: {OUTPUT_PATH} | rows={len(submission)} | "
        f"champion={CHAMPION_WEIGHT:.2f} exp021={EXP021_WEIGHT:.2f} | "
        f"mean={blended.mean():.9f} min={blended.min():.9f} max={blended.max():.9f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
