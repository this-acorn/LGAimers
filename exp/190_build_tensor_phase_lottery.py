# -*- coding: utf-8 -*-
"""Build a high-risk phase-signed Tensor correction on the affine anchor.

The untouched 2024 Tensor scale curve is exactly quadratic absent clipping.
Its two measured scales imply opposite segment slopes:

* months <= 6: observed gain +11.6300 at scale +0.58;
* months > 6: fitted gain +3.7660 at conservative scale -0.20.

This is a single-season, post-validation routing hypothesis, so the resulting
ZIP is explicitly a lottery and not an offline-gate pass.
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
SOURCE = ROOT / 'artifacts/candidates/candidate_exp185_affine_tensor_s020_hold.zip'
ANCHOR = ROOT / 'artifacts/candidates/candidate_affine_opt.zip'
OUTPUT = ROOT / 'artifacts/candidates/candidate_exp190_affine_tensor_phase_lottery.zip'
REPORT = ROOT / "lab" / "190_tensor_phase_lottery_build.json"
PYTHON = ROOT / "venv311" / "Scripts" / "python.exe"
TEST = ROOT / "data" / "test.csv"
SAMPLE = ROOT / "data" / "sample_submission.csv"
EARLY_SCALE = 0.58
LATE_SCALE = -0.20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"patch token count != 1: {old[:100]!r}")
    return source.replace(old, new)


def segment_quadratic(scale1: float, gain1: float, scale2: float, gain2: float):
    curvature = (gain1 / scale1 - gain2 / scale2) / (scale2 - scale1)
    slope = gain1 / scale1 + curvature * scale1
    return slope, curvature


def build() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    with zipfile.ZipFile(SOURCE, "r") as incoming:
        script = incoming.read("script.py").decode("utf-8")
        metadata = json.loads(incoming.read("model/tensor_metadata.json"))
        script = replace_once(
            script,
            'ID_COL, "game_type", "pitcher_id", "pitcher_hand", "batter_hand",\n'
            '        "balls_before", "strikes_before", "base_state",',
            'ID_COL, "game_type", "game_month", "pitcher_id", "pitcher_hand", "batter_hand",\n'
            '        "balls_before", "strikes_before", "base_state",',
        )
        script = replace_once(
            script,
            '    return pd.DataFrame({ID_COL: rows[ID_COL].to_numpy(), "effect": effect,\n'
            '                         "is_r": is_r}), {**coverage, "scale": float(metadata["scale"])}',
            '    month = pd.to_numeric(rows["game_month"], errors="coerce").fillna(99).to_numpy()\n'
            '    return pd.DataFrame({ID_COL: rows[ID_COL].to_numpy(), "effect": effect,\n'
            '                         "is_r": is_r, "game_month": month}), {**coverage, "scale": float(metadata["scale"])}',
        )
        script = replace_once(
            script,
            '    scale = float(diagnostic.pop("scale"))\n'
            '    effect = aligned["effect"].to_numpy(np.float64)\n'
            '    is_r = aligned["is_r"].to_numpy(bool)\n'
            '    prediction = np.clip(anchor + scale * effect, 0.0, 1.0)\n'
            '    if not np.array_equal(prediction[~is_r], anchor[~is_r]):\n'
            '        raise AssertionError("F predictions changed")',
            '    diagnostic.pop("scale")\n'
            '    effect = aligned["effect"].to_numpy(np.float64)\n'
            '    is_r = aligned["is_r"].to_numpy(bool)\n'
            '    month = aligned["game_month"].to_numpy(np.float64)\n'
            '    row_scale = np.where(month <= 6, 0.58, -0.20)\n'
            '    row_scale = np.where(is_r, row_scale, 0.0)\n'
            '    prediction = np.clip(anchor + row_scale * effect, 0.0, 1.0)\n'
            '    if not np.array_equal(prediction[~is_r], anchor[~is_r]):\n'
            '        raise AssertionError("F predictions changed")',
        )
        script = replace_once(
            script,
            'f"Saved: {OUTPUT_PATH} rows={len(prediction)} scale={scale:.2f} "\n'
            '        f"R={int(is_r.sum())} mean_effect_R={effect[is_r].mean():+.8f} "',
            'f"Saved: {OUTPUT_PATH} rows={len(prediction)} early_scale=+0.58 late_scale=-0.20 "\n'
            '        f"R={int(is_r.sum())} mean_effect_R={effect[is_r].mean():+.8f} "',
        )
        early_slope, early_curvature = segment_quadratic(
            0.20, 6.096292926538581, 0.58, 11.630026193110098
        )
        late_slope, late_curvature = segment_quadratic(
            0.20, -5.880302047967007, 0.58, -22.87770741586928
        )
        metadata["deployment_gate"] = {
            "status": "HOLD_HIGH_RISK_SINGLE_YEAR_PHASE_SIGN_LOTTERY",
            "R_scales": {"game_month<=6": EARLY_SCALE, "game_month>6": LATE_SCALE},
            "F_policy": "bit-identical affine anchor",
            "quadratic_diagnostic": {
                "early": {
                    "linear": early_slope,
                    "curvature": early_curvature,
                    "gain_at_scale": early_slope * EARLY_SCALE - early_curvature * EARLY_SCALE**2,
                },
                "late": {
                    "linear": late_slope,
                    "curvature": late_curvature,
                    "estimated_gain_at_scale": late_slope * LATE_SCALE - late_curvature * LATE_SCALE**2,
                },
            },
        }
        replacements = {
            "script.py": script.encode("utf-8"),
            "model/tensor_metadata.json": (
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
            ).encode("utf-8"),
        }
        temporary = OUTPUT.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as outgoing:
            for name in incoming.namelist():
                payload = replacements.get(name, incoming.read(name))
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 1, 8, 40, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                outgoing.writestr(info, payload)
        os.replace(temporary, OUTPUT)
    with zipfile.ZipFile(OUTPUT, "r") as check:
        if check.testzip() is not None:
            raise RuntimeError("output ZIP CRC failure")


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
    return pd.read_csv(destination / "output" / "submission.csv"), time.time() - started, (result.stdout or "")[-1500:]


def smoke() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="exp190_", dir=ROOT / "lab") as temporary:
        work = Path(temporary)
        anchor, anchor_seconds, _ = execute(ANCHOR, work / "anchor")
        candidate, candidate_seconds, log_tail = execute(OUTPUT, work / "candidate")
    test = pd.read_csv(TEST, encoding="utf-8-sig")
    merged = anchor.merge(candidate, on="row_id", suffixes=("_anchor", "_candidate"))
    merged = merged.merge(test[["row_id", "game_type", "game_month"]], on="row_id")
    is_f = ~merged["game_type"].astype(str).eq("R").to_numpy()
    difference = (
        merged["control_success_candidate"] - merged["control_success_anchor"]
    ).to_numpy(np.float64)
    if np.abs(difference[is_f]).max(initial=0.0) != 0.0:
        raise AssertionError("F smoke rows changed")
    if not np.isfinite(merged["control_success_candidate"]).all():
        raise AssertionError("non-finite smoke output")
    return {
        "rows": len(merged),
        "F_max_abs_difference": float(np.abs(difference[is_f]).max(initial=0.0)),
        "early_R_mean_difference": float(difference[(~is_f) & (merged.game_month.to_numpy() <= 6)].mean()),
        "late_R_mean_difference": float(difference[(~is_f) & (merged.game_month.to_numpy() > 6)].mean()),
        "anchor_seconds": anchor_seconds,
        "candidate_seconds": candidate_seconds,
        "candidate_log_tail": log_tail,
    }


def main() -> None:
    for path in (SOURCE, ANCHOR, PYTHON, TEST, SAMPLE):
        if not path.exists():
            raise FileNotFoundError(path)
    build()
    qa = smoke()
    metadata = {
        "experiment": 190,
        "status": "BUILT_HOLD_PHASE_SIGN_LOTTERY",
        "output": {"path": OUTPUT.name, "bytes": OUTPUT.stat().st_size, "sha256": sha256(OUTPUT)},
        "source": {"path": SOURCE.name, "sha256": sha256(SOURCE)},
        "smoke": qa,
        "submission_performed": False,
    }
    REPORT.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
