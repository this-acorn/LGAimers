# -*- coding: utf-8 -*-
"""Strengthen EXP189's early-only Tensor scale from 0.20 to measured 0.58.

The executable code and fitted Tensor tables remain byte-identical to EXP189;
only the frozen scalar in tensor_metadata.json changes.  EXP180 directly
measured the 2024 early-period gain at this scale as +11.630026.  Late and F
rows still fall back bit-identically to the affine anchor.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/candidates/candidate_exp189_affine_tensor_early_hold.zip'
ANCHOR = ROOT / 'artifacts/candidates/candidate_affine_opt.zip'
OUTPUT = ROOT / 'artifacts/candidates/candidate_exp191_affine_tensor_early_s058_lottery.zip'
REPORT = ROOT / "lab" / "191_tensor_early_s058_lottery_build.json"
PYTHON = ROOT / "venv311" / "Scripts" / "python.exe"
TEST = ROOT / "data" / "test.csv"
SAMPLE = ROOT / "data" / "sample_submission.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def entry_hash(path: Path, name: str) -> str:
    with zipfile.ZipFile(path) as archive:
        return hashlib.sha256(archive.read(name)).hexdigest().upper()


def build() -> dict[str, object]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    with zipfile.ZipFile(SOURCE) as incoming:
        metadata = json.loads(incoming.read("model/tensor_metadata.json"))
        if float(metadata["scale"]) != 0.20:
            raise AssertionError("EXP189 source scale is not 0.20")
        metadata["scale"] = 0.58
        gate = metadata.setdefault("deployment_gate", {})
        gate["status"] = "HOLD_HIGH_RISK_SINGLE_YEAR_EARLY_SCALE_058_LOTTERY"
        gate["early_scale"] = 0.58
        gate["source_diagnostic"]["early_gain_at_scale_058"] = 11.630026193110098
        replacement = (
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        temporary = OUTPUT.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as outgoing:
            for name in incoming.namelist():
                payload = replacement if name == "model/tensor_metadata.json" else incoming.read(name)
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 1, 8, 55, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                outgoing.writestr(info, payload)
        os.replace(temporary, OUTPUT)
    with zipfile.ZipFile(OUTPUT) as check:
        if check.testzip() is not None:
            raise RuntimeError("output ZIP CRC failure")
    identical_entries = {}
    with zipfile.ZipFile(SOURCE) as source_zip:
        for name in source_zip.namelist():
            if name != "model/tensor_metadata.json":
                identical_entries[name] = entry_hash(SOURCE, name) == entry_hash(OUTPUT, name)
    if not all(identical_entries.values()):
        raise AssertionError("non-metadata entry changed")
    return {
        "non_metadata_entries": len(identical_entries),
        "all_non_metadata_entries_byte_identical": True,
        "script_sha256": entry_hash(OUTPUT, "script.py"),
        "tables_sha256": entry_hash(OUTPUT, "model/tensor_tables.joblib"),
    }


def execute(path: Path, destination: Path) -> tuple[pd.DataFrame, float, str]:
    with zipfile.ZipFile(path) as archive:
        archive.extractall(destination)
    (destination / "data").mkdir()
    pd.read_csv(TEST, encoding="utf-8-sig").to_csv(
        destination / "data" / "test.csv", index=False, encoding="utf-8"
    )
    pd.read_csv(SAMPLE, encoding="utf-8-sig").to_csv(
        destination / "data" / "sample_submission.csv", index=False, encoding="utf-8"
    )
    started = time.time()
    result = subprocess.run(
        [str(PYTHON), "-u", "script.py"], cwd=destination, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=600, check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if result.returncode:
        raise RuntimeError((result.stdout or "")[-3000:] + (result.stderr or "")[-5000:])
    return pd.read_csv(destination / "output" / "submission.csv"), time.time() - started, (result.stdout or "")[-1200:]


def smoke() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="exp191_", dir=ROOT / "lab") as temporary:
        work = Path(temporary)
        anchor, anchor_seconds, _ = execute(ANCHOR, work / "anchor")
        candidate, candidate_seconds, log_tail = execute(OUTPUT, work / "candidate")
    test = pd.read_csv(TEST, encoding="utf-8-sig")
    merged = anchor.merge(candidate, on="row_id", suffixes=("_anchor", "_candidate"))
    merged = merged.merge(test[["row_id", "game_type", "game_month"]], on="row_id")
    active = merged["game_type"].astype(str).eq("R").to_numpy() & (
        pd.to_numeric(merged["game_month"], errors="coerce").to_numpy() <= 6
    )
    difference = (
        merged["control_success_candidate"] - merged["control_success_anchor"]
    ).to_numpy(np.float64)
    if np.abs(difference[~active]).max(initial=0.0) != 0.0:
        raise AssertionError("inactive rows changed")
    if not np.isfinite(merged["control_success_candidate"]).all():
        raise AssertionError("non-finite output")
    return {
        "rows": len(merged),
        "active_rows": int(active.sum()),
        "inactive_max_abs_difference": float(np.abs(difference[~active]).max(initial=0.0)),
        "active_mean_difference": float(difference[active].mean()),
        "anchor_seconds": anchor_seconds,
        "candidate_seconds": candidate_seconds,
        "candidate_log_tail": log_tail,
    }


def main() -> None:
    for path in (SOURCE, ANCHOR, PYTHON, TEST, SAMPLE):
        if not path.exists():
            raise FileNotFoundError(path)
    identity = build()
    qa = smoke()
    payload = {
        "experiment": 191,
        "status": "BUILT_HOLD_EARLY_SCALE_058_LOTTERY",
        "output": {"path": OUTPUT.name, "bytes": OUTPUT.stat().st_size, "sha256": sha256(OUTPUT)},
        "source": {"path": SOURCE.name, "sha256": sha256(SOURCE)},
        "evidence": {
            "2024_early_gain_scale_020": 6.096292926538581,
            "2024_early_gain_scale_058": 11.630026193110098,
            "late_policy": "exact affine anchor",
            "caveat": "single validation season; submit only if EXP189 LB direction is positive",
        },
        "identity": identity,
        "smoke": qa,
        "row_independence_inherited": {
            "basis": "all executable/model-table entries are byte-identical to EXP189; only scalar metadata changed",
            "EXP189_max_absolute_difference": 0.0,
        },
        "submission_performed": False,
    }
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
