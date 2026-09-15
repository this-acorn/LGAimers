# -*- coding: utf-8 -*-
"""Build the two missing cells of the Tensor month-phase factorial.

Frozen axes:
  early scale: 0.00 or +0.20
  late scale:  -0.20

Together with anchor (0,0), EXP189 (+.20,0), EXP191 (+.58,0), and EXP190
(+.58,-.20), these artifacts allow leaderboard decisions without conflating
the early-strength and late-sign hypotheses.  They remain single-season
lottery artifacts and are never submitted by this builder.
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
SOURCE = ROOT / 'artifacts/candidates/candidate_exp190_affine_tensor_phase_lottery.zip'
ANCHOR = ROOT / 'artifacts/candidates/candidate_affine_opt.zip'
PYTHON = ROOT / "venv311" / "Scripts" / "python.exe"
TEST = ROOT / "data" / "test.csv"
SAMPLE = ROOT / "data" / "sample_submission.csv"
REPORT = ROOT / "lab" / "192_tensor_phase_factorial_build.json"
SPECS = (
    {
        "experiment": 192,
        "early": 0.00,
        "late": -0.20,
        "name": 'artifacts/candidates/candidate_exp192_affine_tensor_late_only_m020_lottery.zip',
    },
    {
        "experiment": 193,
        "early": 0.20,
        "late": -0.20,
        "name": 'artifacts/candidates/candidate_exp193_affine_tensor_early020_late_m020_lottery.zip',
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def entry_hash(path: Path, name: str) -> str:
    with zipfile.ZipFile(path) as archive:
        return hashlib.sha256(archive.read(name)).hexdigest().upper()


def build_one(spec: dict[str, object]) -> dict[str, object]:
    output = ROOT / str(spec["name"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    early = float(spec["early"])
    late = float(spec["late"])
    with zipfile.ZipFile(SOURCE) as incoming:
        script = incoming.read("script.py").decode("utf-8")
        metadata = json.loads(incoming.read("model/tensor_metadata.json"))
        old_line = "row_scale = np.where(month <= 6, 0.58, -0.20)"
        new_line = f"row_scale = np.where(month <= 6, {early:.2f}, {late:.2f})"
        if script.count(old_line) != 1:
            raise RuntimeError("phase source line mismatch")
        script = script.replace(old_line, new_line)
        old_print = 'f"Saved: {OUTPUT_PATH} rows={len(prediction)} early_scale=+0.58 late_scale=-0.20 "'
        new_print = (
            'f"Saved: {OUTPUT_PATH} rows={len(prediction)} '
            f'early_scale={early:+.2f} late_scale={late:+.2f} "'
        )
        if script.count(old_print) != 1:
            raise RuntimeError("phase print line mismatch")
        script = script.replace(old_print, new_print)
        gate = metadata["deployment_gate"]
        gate["status"] = f"HOLD_HIGH_RISK_PHASE_FACTORIAL_EXP{spec['experiment']}"
        gate["R_scales"] = {"game_month<=6": early, "game_month>6": late}
        replacement = {
            "script.py": script.encode("utf-8"),
            "model/tensor_metadata.json": (
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
            ).encode("utf-8"),
        }
        temporary = output.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as outgoing:
            for name in incoming.namelist():
                payload = replacement.get(name, incoming.read(name))
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 1, 9, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                outgoing.writestr(info, payload)
        os.replace(temporary, output)
    with zipfile.ZipFile(output) as check:
        if check.testzip() is not None:
            raise RuntimeError(f"CRC failure: {output}")
    with zipfile.ZipFile(SOURCE) as source_zip:
        unchanged = [
            name for name in source_zip.namelist()
            if name not in {"script.py", "model/tensor_metadata.json"}
        ]
    if not all(entry_hash(SOURCE, name) == entry_hash(output, name) for name in unchanged):
        raise AssertionError("unrelated package entry changed")
    return {
        **spec,
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "unchanged_non_runtime_entries": len(unchanged),
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
    return pd.read_csv(destination / "output" / "submission.csv"), time.time() - started, (result.stdout or "")[-800:]


def smoke(artifacts: list[dict[str, object]]) -> dict[str, object]:
    test = pd.read_csv(TEST, encoding="utf-8-sig")
    with tempfile.TemporaryDirectory(prefix="exp192_", dir=ROOT / "lab") as temporary:
        work = Path(temporary)
        anchor, anchor_seconds, _ = execute(ANCHOR, work / "anchor")
        results = {}
        for artifact in artifacts:
            candidate, seconds, log_tail = execute(Path(str(artifact["path"])), work / str(artifact["experiment"]))
            merged = anchor.merge(candidate, on="row_id", suffixes=("_anchor", "_candidate"))
            merged = merged.merge(test[["row_id", "game_type", "game_month"]], on="row_id")
            is_r = merged["game_type"].astype(str).eq("R").to_numpy()
            early = is_r & (pd.to_numeric(merged["game_month"], errors="coerce").to_numpy() <= 6)
            late = is_r & ~early
            difference = (
                merged["control_success_candidate"] - merged["control_success_anchor"]
            ).to_numpy(np.float64)
            expected_active = (early & (float(artifact["early"]) != 0.0)) | (
                late & (float(artifact["late"]) != 0.0)
            )
            if np.abs(difference[~expected_active]).max(initial=0.0) != 0.0:
                raise AssertionError(f"inactive rows changed for {artifact['experiment']}")
            results[str(artifact["experiment"])] = {
                "seconds": seconds,
                "inactive_max_absolute_difference": float(np.abs(difference[~expected_active]).max(initial=0.0)),
                "early_mean_difference": float(difference[early].mean()),
                "late_mean_difference": float(difference[late].mean()),
                "log_tail": log_tail,
            }
    return {"anchor_seconds": anchor_seconds, "artifacts": results}


def main() -> None:
    for path in (SOURCE, ANCHOR, PYTHON, TEST, SAMPLE):
        if not path.exists():
            raise FileNotFoundError(path)
    artifacts = [build_one(spec) for spec in SPECS]
    qa = smoke(artifacts)
    payload = {
        "experiment": "192-193",
        "status": "BUILT_HOLD_PHASE_FACTORIAL_LOTTERIES",
        "source": {"path": SOURCE.name, "sha256": sha256(SOURCE)},
        "artifacts": artifacts,
        "smoke": qa,
        "evidence": {
            "2024_early_gain_s020": 6.096292926538581,
            "2024_late_gain_s020": -5.880302047967007,
            "2024_late_gain_s058": -22.87770741586928,
            "quadratic_estimated_late_gain_at_m020": 3.766025650172852,
            "caveat": "single validation season; branch only after preceding LB result",
        },
        "submission_performed": False,
    }
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
