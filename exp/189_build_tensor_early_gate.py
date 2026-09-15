# -*- coding: utf-8 -*-
"""Build a row-local early-season gate on the existing EXP185 Tensor endpoint.

EXP180's untouched 2024 diagnostics were +6.0963 in months <= 6 and -5.8803
in months > 6.  This packaging-only candidate retains the already-fitted
official-data Tensor correction for early regular-season rows and makes every
other row bit-identical to the submitted affine anchor.

No model is fitted and no label or leaderboard value is read by this builder.
The output remains a high-risk HOLD candidate because the month split was
selected from 2024 diagnostics and has not been independently confirmed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/candidates/candidate_exp185_affine_tensor_s020_hold.zip'
ANCHOR = ROOT / 'artifacts/candidates/candidate_affine_opt.zip'
OUTPUT = ROOT / 'artifacts/candidates/candidate_exp189_affine_tensor_early_hold.zip'
REPORT = ROOT / "lab" / "189_tensor_early_gate_build.json"
PYTHON = ROOT / "venv311" / "Scripts" / "python.exe"
TEST = ROOT / "data" / "test.csv"
SAMPLE = ROOT / "data" / "sample_submission.csv"


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
            '    is_r = rows["game_type"].to_numpy() == "R"\n'
            '    effect[~is_r] = 0.0\n'
            '    if not np.isfinite(effect).all() or np.any(effect[~is_r] != 0.0):\n'
            '        raise AssertionError("invalid tensor effect or F fallback")\n'
            '    return pd.DataFrame({ID_COL: rows[ID_COL].to_numpy(), "effect": effect,\n'
            '                         "is_r": is_r}), {**coverage, "scale": float(metadata["scale"])}',
            '    is_r = rows["game_type"].to_numpy() == "R"\n'
            '    month = pd.to_numeric(rows["game_month"], errors="coerce").fillna(99).to_numpy()\n'
            '    active = is_r & (month <= 6)\n'
            '    effect[~active] = 0.0\n'
            '    if not np.isfinite(effect).all() or np.any(effect[~active] != 0.0):\n'
            '        raise AssertionError("invalid tensor effect or early-season fallback")\n'
            '    return pd.DataFrame({ID_COL: rows[ID_COL].to_numpy(), "effect": effect,\n'
            '                         "is_r": is_r, "active": active}), {**coverage, "scale": float(metadata["scale"])}',
        )
        script = replace_once(
            script,
            '    is_r = aligned["is_r"].to_numpy(bool)\n'
            '    prediction = np.clip(anchor + scale * effect, 0.0, 1.0)\n'
            '    if not np.array_equal(prediction[~is_r], anchor[~is_r]):\n'
            '        raise AssertionError("F predictions changed")',
            '    is_r = aligned["is_r"].to_numpy(bool)\n'
            '    active = aligned["active"].to_numpy(bool)\n'
            '    prediction = np.clip(anchor + scale * effect, 0.0, 1.0)\n'
            '    if not np.array_equal(prediction[~active], anchor[~active]):\n'
            '        raise AssertionError("inactive predictions changed")',
        )
        script = replace_once(
            script,
            'f"R={int(is_r.sum())} mean_effect_R={effect[is_r].mean():+.8f} "',
            'f"R={int(is_r.sum())} active_early_R={int(active.sum())} "\n'
            '        f"mean_effect_active={effect[active].mean() if active.any() else 0.0:+.8f} "',
        )
        metadata["deployment_gate"] = {
            "active": "game_type == R and numeric game_month <= 6",
            "inactive": "bit-identical affine anchor",
            "source_diagnostic": {
                "year": 2024,
                "early_gain": 6.096292926538581,
                "late_gain_before_gate": -5.880302047967007,
            },
            "status": "HOLD_HIGH_RISK_SINGLE_YEAR_MONTH_GATE",
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
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 1, 8, 30, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                outgoing.writestr(info, payload)
        os.replace(temporary, OUTPUT)
    with zipfile.ZipFile(OUTPUT, "r") as check:
        if check.testzip() is not None:
            raise RuntimeError("output ZIP CRC failure")


def run_package(path: Path, root: Path) -> tuple[pd.DataFrame, str, float]:
    with zipfile.ZipFile(path, "r") as archive:
        archive.extractall(root)
    (root / "data").mkdir(exist_ok=True)
    pd.read_csv(TEST, encoding="utf-8-sig").to_csv(
        root / "data" / "test.csv", index=False, encoding="utf-8"
    )
    pd.read_csv(SAMPLE, encoding="utf-8-sig").to_csv(
        root / "data" / "sample_submission.csv", index=False, encoding="utf-8"
    )
    started = time.time()
    result = subprocess.run(
        [str(PYTHON), "-u", "script.py"],
        cwd=root,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    elapsed = time.time() - started
    if result.returncode != 0:
        raise RuntimeError((result.stdout or "")[-3000:] + (result.stderr or "")[-5000:])
    output = pd.read_csv(root / "output" / "submission.csv")
    return output, (result.stdout or "")[-2000:], elapsed


def qa() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="exp189_", dir=ROOT / "lab") as temp:
        temporary = Path(temp)
        anchor, anchor_log, anchor_seconds = run_package(ANCHOR, temporary / "anchor")
        candidate, candidate_log, candidate_seconds = run_package(OUTPUT, temporary / "candidate")
    test = pd.read_csv(TEST, encoding="utf-8-sig")
    comparison = anchor.merge(candidate, on="row_id", suffixes=("_anchor", "_candidate"))
    comparison = comparison.merge(test[["row_id", "game_month", "game_type"]], on="row_id")
    active = comparison["game_type"].astype(str).eq("R") & (
        pd.to_numeric(comparison["game_month"], errors="coerce") <= 6
    )
    difference = (
        comparison["control_success_candidate"] - comparison["control_success_anchor"]
    ).to_numpy(np.float64)
    if np.any(difference[~active.to_numpy()] != 0.0):
        raise AssertionError("inactive smoke rows differ from affine anchor")
    if not np.isfinite(comparison["control_success_candidate"]).all():
        raise AssertionError("non-finite candidate output")
    return {
        "rows": len(comparison),
        "active_rows": int(active.sum()),
        "inactive_rows": int((~active).sum()),
        "inactive_max_abs_difference": float(np.abs(difference[~active.to_numpy()]).max(initial=0.0)),
        "active_mean_difference": float(difference[active.to_numpy()].mean()),
        "anchor_seconds": anchor_seconds,
        "candidate_seconds": candidate_seconds,
        "anchor_log_tail": anchor_log,
        "candidate_log_tail": candidate_log,
    }


def main() -> None:
    for path in (SOURCE, ANCHOR, PYTHON, TEST, SAMPLE):
        if not path.exists():
            raise FileNotFoundError(path)
    build()
    smoke = qa()
    payload = {
        "experiment": 189,
        "status": "BUILT_HOLD_HIGH_RISK_MONTH_GATE",
        "source": {"path": SOURCE.name, "sha256": sha256(SOURCE)},
        "anchor": {"path": ANCHOR.name, "sha256": sha256(ANCHOR)},
        "output": {
            "path": OUTPUT.name,
            "bytes": OUTPUT.stat().st_size,
            "sha256": sha256(OUTPUT),
        },
        "evidence": {
            "2024_early_gain_before_gate": 6.096292926538581,
            "2024_late_gain_before_gate": -5.880302047967007,
            "expected_late_gain_after_gate": 0.0,
            "caveat": "month gate chosen on one validation season; no independent confirmation",
        },
        "smoke": smoke,
        "submission_performed": False,
    }
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
