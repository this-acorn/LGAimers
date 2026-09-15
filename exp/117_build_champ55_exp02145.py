# -*- coding: utf-8 -*-
"""Build the row-local 55% champion + 45% EXP-021 strict submission ZIP."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAMPION = ROOT / 'artifacts/candidates/candidate_v18g030.zip'
EXP021 = ROOT / 'artifacts/candidates/candidate_exp021_strict_rebuilt_v001.zip'
SOURCE = ROOT / 'candidates/candidate_champ55_exp02145_v001_src'
OUTPUT = ROOT / 'artifacts/candidates/candidate_champ55_exp02145_v001.zip'
REPORT = ROOT / "lab" / "132_champ55_exp02145_build.json"

CHAMPION_SHA256 = "ED1B729B9B83EB1F0D344B475939353149D5ADC83009A40FF116DF6E5A168557"
CHAMPION_WEIGHT = 0.55
EXP021_WEIGHT = 0.45
REQUIREMENTS = """numpy==1.26.4
pandas==2.0.3
scikit-learn==1.8.0
joblib==1.5.3
catboost==1.2.10
lightgbm==4.6.0
"""

WRAPPER = r'''"""Sequential, row-local blend of two frozen probability endpoints."""

from __future__ import annotations

import gc
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
CHAMPION_WEIGHT = 0.55
EXP021_WEIGHT = 0.45
MODEL_DIR = Path("./model")
OUTPUT_DIR = Path("./output")
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"
SAMPLE_PATH = Path("./data/sample_submission.csv")


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
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def main() -> None:
    if sha256(CHAMPION).upper() != CHAMPION_SHA256:
        raise AssertionError("locked champion SHA256 mismatch")
    if SOURCE.exists() or OUTPUT.exists():
        raise FileExistsError("refusing to overwrite an existing blend artifact")

    SOURCE.mkdir()
    model_dir = SOURCE / "model"
    model_dir.mkdir()
    (SOURCE / "script.py").write_text(WRAPPER, encoding="utf-8", newline="\n")
    (SOURCE / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8", newline="\n")

    with zipfile.ZipFile(CHAMPION) as archive:
        if archive.testzip() is not None:
            raise AssertionError("champion CRC failure")
        if archive.namelist() != [
            "script.py",
            "requirements.txt",
            "model/model.pkl",
            "model/v18_tables.npz",
        ]:
            raise AssertionError("champion ZIP structure drift")
        write_bytes(model_dir / "champion_inference.py", archive.read("script.py"))
        write_bytes(model_dir / "model.pkl", archive.read("model/model.pkl"))
        write_bytes(model_dir / "v18_tables.npz", archive.read("model/v18_tables.npz"))

    with zipfile.ZipFile(EXP021) as archive:
        if archive.testzip() is not None:
            raise AssertionError("EXP021 CRC failure")
        exp021_requirements = archive.read("requirements.txt").decode("utf-8").replace("\r\n", "\n")
        if exp021_requirements != "lightgbm==4.6.0\n":
            raise AssertionError("EXP021 requirements drift")
        metadata = json.loads(archive.read("model/metadata.json").decode("utf-8"))
        if metadata.get("candidate") != "strict_lowrank_s300_r6":
            raise AssertionError("EXP021 candidate drift")
        write_bytes(model_dir / "exp021_inference.py", archive.read("script.py"))
        for name in archive.namelist():
            if name.startswith("model/") and not name.endswith("/"):
                write_bytes(model_dir / Path(name).name, archive.read(name))

    blend_metadata = {
        "name": "champ55_exp02145_v001",
        "weights": {"champion": CHAMPION_WEIGHT, "exp021_strict": EXP021_WEIGHT},
        "champion_zip": {"name": CHAMPION.name, "sha256": sha256(CHAMPION)},
        "exp021_zip": {"name": EXP021.name, "sha256": sha256(EXP021)},
        "row_local": True,
        "component_execution": "sequential",
    }
    (model_dir / "blend_metadata.json").write_text(
        json.dumps(blend_metadata, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )

    entries = [SOURCE / "script.py", SOURCE / "requirements.txt"] + sorted(model_dir.iterdir())
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in entries:
            arcname = path.name if path.parent == SOURCE else f"model/{path.name}"
            archive.write(path, arcname)
    with zipfile.ZipFile(OUTPUT) as archive:
        bad = archive.testzip()
        names = archive.namelist()
        if bad is not None:
            raise AssertionError(f"blend ZIP CRC failure: {bad}")
        if names[:2] != ["script.py", "requirements.txt"]:
            raise AssertionError("blend ZIP root order drift")
        if any("__pycache__" in name or name.startswith(("data/", "output/")) for name in names):
            raise AssertionError("blend ZIP contains forbidden runtime data")

    report = {
        "artifact": str(OUTPUT),
        "sha256": sha256(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "source": str(SOURCE),
        "weights": blend_metadata["weights"],
        "components": {
            "champion_sha256": sha256(CHAMPION),
            "exp021_sha256": sha256(EXP021),
        },
        "entries": names,
        "crc": "passed",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
