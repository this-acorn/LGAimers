# -*- coding: utf-8 -*-
"""Build and rehearse a clean-room R-only CAT5 anti-blend submission.

The package contains only our existing independently reconstructed components.
It runs the champion endpoint, the same champion model with V18/affine disabled
to expose its raw CAT5 probability, and EXP021.  On R rows only it applies the
signed weight frozen from pooled 2022--2024 forward OOF geometry.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'candidates/candidate_exp021_ours_v001_src'
NAME = "candidate_cat5anti_r054"
DEST = ROOT / f"{NAME}_src"
ZIP_PATH = ROOT / f"{NAME}.zip"
STAGE = ROOT / 'archive/scratch/_rehearse_161'
PYTHON = ROOT / "venv311" / "Scripts" / "python.exe"
ANTI_WEIGHT = -0.5397043598776958


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: expected one anchor, got {text.count(old)}")
    return text.replace(old, new, 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_source() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    shutil.copytree(SOURCE, DEST, ignore=shutil.ignore_patterns("__pycache__"))

    champion_path = DEST / "model" / "champion_inference.py"
    raw_path = DEST / "model" / "cat5_raw_inference.py"
    raw = champion_path.read_text(encoding="utf-8")
    raw = replace_once(raw, "SCALE_S = 1.06", "SCALE_S = 1.0", "raw scale")
    raw = replace_once(raw, "SHIFT_DELTA = 0.0066", "SHIFT_DELTA = 0.0", "raw shift")
    raw = replace_once(raw, "V18_GAMMA = 0.30", "V18_GAMMA = 0.0", "raw v18")
    raw_path.write_text(raw, encoding="utf-8")

    script_path = DEST / "script.py"
    script = script_path.read_text(encoding="utf-8")
    constant_anchor = 'SAMPLE_PATH = Path("./data/sample_submission.csv")\n'
    script = replace_once(
        script, constant_anchor,
        constant_anchor + f"ANTI_WEIGHT = {ANTI_WEIGHT!r}\nTEST_PATH = Path(\"./data/test.csv\")\n",
        "outer constants",
    )
    champion_anchor = (
        "    champion_ids = champion[ID_COL].to_numpy(copy=True)\n"
        "    del champion\n"
        "    gc.collect()\n\n"
        "    exp021 = _run_component(\"exp021_inference.py\", \"exp021_component\")\n"
    )
    replacement = (
        "    champion_ids = champion[ID_COL].to_numpy(copy=True)\n"
        "    del champion\n"
        "    gc.collect()\n\n"
        "    raw_cat5 = _run_component(\"cat5_raw_inference.py\", \"raw_cat5_component\")\n"
        "    _validate_component(raw_cat5, \"raw_cat5\")\n"
        "    if not np.array_equal(champion_ids, raw_cat5[ID_COL].to_numpy()):\n"
        "        raise ValueError(\"raw CAT5 row_id order mismatch\")\n"
        "    raw_cat5_values = raw_cat5[TARGET_COL].to_numpy(dtype=np.float64, copy=True)\n"
        "    del raw_cat5\n"
        "    gc.collect()\n\n"
        "    exp021 = _run_component(\"exp021_inference.py\", \"exp021_component\")\n"
    )
    script = replace_once(script, champion_anchor, replacement, "raw component")
    blend_anchor = "    blended = CHAMPION_WEIGHT * champion_values + EXP021_WEIGHT * exp021_values\n"
    anti_apply = (
        blend_anchor
        + "    test = pd.read_csv(TEST_PATH, encoding=\"utf-8-sig\", usecols=[ID_COL, \"game_type\"])\n"
        + "    test.columns = [column.replace(\"ï»¿\", \"\").strip() for column in test.columns]\n"
        + "    if test[ID_COL].duplicated().any():\n"
        + "        raise ValueError(\"duplicate test row_id\")\n"
        + "    game_by_id = test.set_index(ID_COL)[\"game_type\"]\n"
        + "    if not pd.Index(champion_ids).isin(game_by_id.index).all():\n"
        + "        raise ValueError(\"missing test row_id for game_type routing\")\n"
        + "    is_r = game_by_id.reindex(champion_ids).astype(str).eq(\"R\").to_numpy()\n"
        + "    blended[is_r] = np.clip(\n"
        + "        blended[is_r] + ANTI_WEIGHT * (raw_cat5_values[is_r] - blended[is_r]), 0.0, 1.0\n"
        + "    )\n"
    )
    script = replace_once(script, blend_anchor, anti_apply, "anti application")
    script_path.write_text(script, encoding="utf-8")


def make_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(DEST.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(DEST).as_posix())


def rehearse(rows: int = 245789, seed: int = 20260901) -> float:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = archive.namelist()
        tops = {name.split("/")[0] for name in names}
        if tops != {"model", "script.py", "requirements.txt"}:
            raise RuntimeError(f"bad zip topology: {tops}")
        archive.extractall(STAGE)
    (STAGE / "data").mkdir()
    sys.path.insert(0, str(ROOT / "exp"))
    build_fake_test = __import__("145_rehearse_reimpl").build_fake_test
    real = pd.read_csv(ROOT / "data" / "test.csv", encoding="utf-8-sig")
    fake = build_fake_test(real, rows, seed)
    fake.to_csv(STAGE / "data" / "test.csv", index=False, encoding="utf-8")
    pd.DataFrame({"row_id": fake["row_id"], "control_success": 0.0}).to_csv(
        STAGE / "data" / "sample_submission.csv", index=False, encoding="utf-8"
    )
    started = time.time()
    result = subprocess.run(
        [str(PYTHON), "script.py"], cwd=STAGE, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    elapsed = time.time() - started
    if result.returncode != 0:
        print((result.stdout or "")[-4000:])
        print((result.stderr or "")[-4000:], file=sys.stderr)
        raise SystemExit("rehearsal failed")
    out = pd.read_csv(STAGE / "output" / "submission.csv")
    if list(out.columns) != ["row_id", "control_success"] or len(out) != rows:
        raise RuntimeError("bad rehearsal output")
    probability = out["control_success"]
    if probability.isna().any() or not probability.between(0.0, 1.0).all():
        raise RuntimeError("invalid rehearsal probability")
    tail = (result.stdout or "")[-2500:]
    print(tail.encode("ascii", "backslashreplace").decode("ascii"))
    print(f"rehearsal rows={rows:,} elapsed={elapsed:.1f}s mean={probability.mean():.9f} "
          f"min={probability.min():.9f} max={probability.max():.9f}")
    return elapsed


def main() -> None:
    build_source()
    make_zip()
    print(f"built={ZIP_PATH.name} bytes={ZIP_PATH.stat().st_size} sha256={sha256(ZIP_PATH)}")
    print(f"R-only signed CAT5 weight={ANTI_WEIGHT:+.12f}")
    rehearse()


if __name__ == "__main__":
    main()
