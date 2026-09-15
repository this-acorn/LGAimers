# -*- coding: utf-8 -*-
"""Rebuild the public EXP-021 strict endpoint in its exact training environment.

This wrapper leaves the canonical public clone untouched.  It runs the byte-identical
builder in ``lab/115_mkis_exp021_work`` and only replaces its Unix-only smoke-test
launcher/symlink at runtime with a Windows-safe equivalent.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "lab" / "115_mkis_exp021_work"
CANONICAL = ROOT / "reference" / "mk_isos_lgaimers9"
FINAL_ZIP = ROOT / 'artifacts/candidates/candidate_exp021_strict_rebuilt_v001.zip'
REPORT = ROOT / "lab" / "131_exp021_build.json"

EXPECTED_ENV = {
    "python": "3.12",
    "numpy": "2.5.1",
    "pandas": "3.0.5",
    "sklearn": "1.9.0",
    "lightgbm": "4.6.0",
}
EXPECTED_FILES = [
    "script.py",
    "requirements.txt",
    "model/feature_schemas.json",
    "model/group_effects.json",
    "model/histgradientboosting.json",
    "model/history_state.json",
    "model/lowrank_effects.json",
    "model/metadata.json",
    "model/multirate_state.json",
    "model/pitcher_count_effects.json",
    "model/rfull_lightgbm.txt",
    "model/team_effects.json",
]
EXPECTED_SMOKE = {
    "mean": 0.44332853920463455,
    "min": 0.3729165651008432,
    "max": 0.5162418838966417,
}
SOURCE_FILES = (
    "experiments/build_exp021_final_candidates.py",
    "experiments/exp021_submission_inference.py",
    "experiments/temporal_multirate_features.py",
    "experiments/temporal_residual_features.py",
    "experiments/train_exp019_histgb_residual.py",
    "experiments/train_exp020_low_rank_pitcher_context_eb.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_environment() -> dict[str, str]:
    found = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
    }
    if found != EXPECTED_ENV:
        raise RuntimeError(f"exact build environment mismatch: {found} != {EXPECTED_ENV}")
    return found


def check_sources() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        canonical = CANONICAL / relative
        working = WORK / relative
        if not canonical.is_file() or not working.is_file():
            raise FileNotFoundError(relative)
        canonical_hash = sha256(canonical)
        working_hash = sha256(working)
        if canonical_hash != working_hash:
            raise RuntimeError(f"working source drift: {relative}")
        hashes[relative] = canonical_hash
    return hashes


def windows_smoke_test(archive_path: Path) -> dict[str, object]:
    """Equivalent to the upstream smoke test without a privileged symlink."""
    with tempfile.TemporaryDirectory(prefix="exp021-smoke-win-") as raw:
        stage = Path(raw)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(stage)
        data_dir = stage / "data"
        data_dir.mkdir()
        for filename in ("test.csv", "sample_submission.csv"):
            shutil.copyfile(WORK / "data" / filename, data_dir / filename)
        started = time.time()
        result = subprocess.run(
            [sys.executable, "script.py"],
            cwd=stage,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        runtime = time.time() - started
        prediction = pd.read_csv(stage / "output" / "submission.csv")
        test = pd.read_csv(data_dir / "test.csv", usecols=["row_id"])
        sample = pd.read_csv(data_dir / "sample_submission.csv")
        if len(prediction) != len(test):
            raise AssertionError("smoke row count mismatch")
        if not prediction["row_id"].equals(sample["row_id"]):
            raise AssertionError("smoke row order mismatch")
        if prediction["row_id"].duplicated().any():
            raise AssertionError("smoke duplicate row_id")
        values = prediction["control_success"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not ((values >= 0).all() and (values <= 1).all()):
            raise AssertionError("invalid smoke probabilities")
        return {
            "runtime_seconds": runtime,
            "stdout": result.stdout.strip(),
            "rows": int(len(values)),
            "row_id_order": "passed",
            "row_id_unique": "passed",
            "missing": int(np.isnan(values).sum()),
            "range": "passed",
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "predictions": values.tolist(),
        }


def validate_and_copy(environment: dict[str, str], source_hashes: dict[str, str]) -> None:
    source_zip = WORK / "submit_exp021_strict.zip"
    upstream_report_path = WORK / "artifacts" / "EXP-021" / "final_packages" / "validation_metrics.json"
    upstream_report = json.loads(upstream_report_path.read_text(encoding="utf-8"))
    strict = upstream_report["results"]["strict"]
    if strict["candidate"] != "strict_lowrank_s300_r6":
        raise AssertionError("wrong EXP021 candidate")
    if float(upstream_report["qa"]["hgb_json_export_max_abs"]) != 0.0:
        raise AssertionError("HGB JSON parity is not exact")
    for key, expected in EXPECTED_SMOKE.items():
        actual = float(strict["smoke"][key])
        if abs(actual - expected) > 1e-9:
            raise AssertionError(f"locked smoke {key} drift: {actual} vs {expected}")
    with zipfile.ZipFile(source_zip) as archive:
        if archive.testzip() is not None:
            raise AssertionError("strict ZIP CRC failure")
        names = archive.namelist()
        if names != EXPECTED_FILES:
            raise AssertionError(f"strict ZIP entry drift: {names}")
        requirements = archive.read("requirements.txt").decode("utf-8").replace("\r\n", "\n")
        if requirements != "lightgbm==4.6.0\n":
            raise AssertionError(f"strict requirements drift: {requirements!r}")
        metadata = json.loads(archive.read("model/metadata.json").decode("utf-8"))
        if metadata["candidate"] != "strict_lowrank_s300_r6":
            raise AssertionError("strict ZIP metadata drift")
    if FINAL_ZIP.exists():
        raise FileExistsError(f"refusing to overwrite {FINAL_ZIP}")
    shutil.copyfile(source_zip, FINAL_ZIP)
    report = {
        "artifact": str(FINAL_ZIP),
        "sha256": sha256(FINAL_ZIP),
        "bytes": FINAL_ZIP.stat().st_size,
        "environment": environment,
        "platform": platform.platform(),
        "source_hashes": source_hashes,
        "candidate": strict["candidate"],
        "hgb_json_export_max_abs": upstream_report["qa"]["hgb_json_export_max_abs"],
        "smoke": strict["smoke"],
        "entries": EXPECTED_FILES,
        "public_reference_score": 1043.6074197937,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    environment = check_environment()
    source_hashes = check_sources()
    sys.path.insert(0, str((WORK / "experiments").resolve()))
    builder = importlib.import_module("build_exp021_final_candidates")
    builder.PYTHON = Path(sys.executable)
    builder.smoke_test = windows_smoke_test
    builder.main()
    validate_and_copy(environment, source_hashes)


if __name__ == "__main__":
    main()
