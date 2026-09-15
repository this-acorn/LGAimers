"""Sequential row-local blend of two frozen submission endpoints."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
ID_COL = "row_id"
TARGET_COL = "control_success"
ENDPOINT_WEIGHT = 0.5894274349634
BUNDLES = (("current", ROOT / "model" / "current_bundle"),
           ("endpoint", ROOT / "model" / "endpoint_bundle"))


def _attach_data(bundle: Path) -> None:
    target = bundle / "data"
    if target.exists() or target.is_symlink():
        return
    try:
        os.symlink(DATA, target, target_is_directory=True)
        return
    except OSError:
        target.mkdir(parents=True, exist_ok=True)
    for name in ("test.csv", "sample_submission.csv"):
        source = DATA / name
        destination = target / name
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def _run(name: str, bundle: Path) -> pd.DataFrame:
    _attach_data(bundle)
    out_dir = bundle / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "submission.csv"
    if out_path.exists():
        out_path.unlink()
    started = time.time()
    result = subprocess.run(
        [sys.executable, "script.py"], cwd=bundle, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    elapsed = time.time() - started
    print(f"[{name}] rc={result.returncode} elapsed={elapsed:.2f}s", flush=True)
    if result.stdout:
        print(result.stdout[-4000:], flush=True)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr[-8000:], file=sys.stderr, flush=True)
        raise RuntimeError(f"{name} endpoint failed")
    if not out_path.is_file():
        raise RuntimeError(f"{name} endpoint did not create {out_path}")
    return pd.read_csv(out_path, encoding="utf-8-sig")


def _align(frame: pd.DataFrame, ids: pd.Series, name: str) -> np.ndarray:
    if list(frame.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"{name}: invalid columns {frame.columns.tolist()}")
    if frame[ID_COL].duplicated().any():
        raise ValueError(f"{name}: duplicate row_id")
    indexed = frame.set_index(ID_COL)[TARGET_COL]
    missing = pd.Index(ids).difference(indexed.index)
    extra = indexed.index.difference(pd.Index(ids))
    if len(missing) or len(extra):
        raise ValueError(f"{name}: row_id mismatch missing={len(missing)} extra={len(extra)}")
    values = indexed.reindex(ids.to_numpy()).to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not ((values >= 0.0).all() and (values <= 1.0).all()):
        raise ValueError(f"{name}: invalid probability")
    return values


def main() -> None:
    sample = pd.read_csv(DATA / "sample_submission.csv", encoding="utf-8-sig")
    sample.columns = [str(c).replace("\ufeff", "").strip() for c in sample.columns]
    if ID_COL not in sample or sample[ID_COL].duplicated().any():
        raise ValueError("invalid sample_submission row_id")
    current_frame = _run(*BUNDLES[0])
    endpoint_frame = _run(*BUNDLES[1])
    current = _align(current_frame, sample[ID_COL], "current")
    endpoint = _align(endpoint_frame, sample[ID_COL], "endpoint")
    prediction = np.clip((1.0 - ENDPOINT_WEIGHT) * current + ENDPOINT_WEIGHT * endpoint, 0.0, 1.0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({ID_COL: sample[ID_COL], TARGET_COL: prediction})
    out.to_csv(OUTPUT / "submission.csv", index=False, encoding="utf-8")
    print(f"[blend] weight_endpoint={ENDPOINT_WEIGHT:.12g} rows={len(out)} "
          f"current_mean={current.mean():.9f} endpoint_mean={endpoint.mean():.9f} "
          f"blend_mean={prediction.mean():.9f}", flush=True)


if __name__ == "__main__":
    main()
