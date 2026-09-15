# -*- coding: utf-8 -*-
"""
[105] Build and validate a frozen V18-style residual candidate bundle.

The source is the measured 1081.67 submit22_s106 bundle.  Its model.pkl is
copied byte-for-byte.  This builder adds only a small train-derived NPZ lookup
and row-independent inference code:

    raw_cat5 <- existing 8-seed model
    raw_corrected = clip(raw_cat5 + 0.30 * frozen_effect, 0, 1)
    final = existing affine calibration(raw_corrected)

No test aggregation, fitting, distribution statistic, or cross-row operation
is used.  This script creates a candidate ZIP but never submits it.
"""

from __future__ import annotations

_PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd


ap = argparse.ArgumentParser()
ap.add_argument("--src-zip", default="submit22_s106.zip")
ap.add_argument("--out-zip", default='artifacts/candidates/candidate_v18g030.zip')
ap.add_argument("--src-dir", default='candidates/candidate_v18g030_src')
ap.add_argument("--gamma", type=float, default=0.30)
ap.add_argument(
    "--allow-unlocked-gamma",
    action="store_true",
    help="Explicitly allow an LB bracket probe whose gamma differs from the validation lock.",
)
ap.add_argument("--report-path", default="lab/105_build_v18_bundle.txt")
ap.add_argument("--meta-path", default="lab/105_v18_tables.json")
args = ap.parse_args()

ROOT = Path(str(_PROJECT_ROOT)).resolve()
SRC_ZIP = (ROOT / args.src_zip).resolve()
OUT_ZIP = (ROOT / args.out_zip).resolve()
SRC_DIR = (ROOT / args.src_dir).resolve()
TABLE_PATH = SRC_DIR / "model" / "v18_tables.npz"
REPORT_PATH = (ROOT / args.report_path).resolve()
META_PATH = (ROOT / args.meta_path).resolve()
GAMMA = float(args.gamma)

LOCK_PATH = ROOT / "lab" / "103_v18_pregate.json"
EXACT_PATH = ROOT / "lab" / "104_cat5_yearfold_summary.json"
VENV_PY = Path(
    str(_PROJECT_ROOT / "venv311/Scripts/python.exe")
)

ROOT_PRIOR = 0.53
K_ROOT = 1200.0
K_PITCHER = 220.0
K_HAND = 110.0
K_DETAIL = 220.0
N_EVAL = 245_789

SIM_FULL = ROOT / "_tmp_sim_v18_full"
SIM_HEAD = ROOT / "_tmp_sim_v18_head"
SIM_REAL = ROOT / "_tmp_sim_v18_real"
SIM_ORIG = ROOT / "_tmp_sim_v18_orig"

LINES: list[str] = []
OK = True


def log(message: str = "") -> None:
    print(message, flush=True)
    LINES.append(message)


def check(condition: bool, good: str, bad: str) -> bool:
    global OK
    log(f"    [{'OK' if condition else '!!'}] {good if condition else bad}")
    if not condition:
        OK = False
    return condition


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["game_type"] = (
        output["game_type"].astype("string").fillna("__MISSING__").astype(str)
    )
    output["pitcher_id"] = (
        pd.to_numeric(output["pitcher_id"], errors="coerce")
        .fillna(-1)
        .astype("int64")
    )
    output["batter_hand"] = (
        output["batter_hand"].astype("string").fillna("__MISSING__").astype(str)
    )
    balls = output["balls_before"].fillna(0).to_numpy(np.int8)
    strikes = output["strikes_before"].fillna(0).to_numpy(np.int8)
    output["pressure"] = np.where(
        (balls == 3) & (strikes == 2),
        2,
        np.where((balls == 3) | (strikes == 2), 1, 0),
    ).astype(np.int8)
    return output


def grouped(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(keys, sort=False, observed=True)["control_success"]
        .agg(success="sum", n="size")
        .reset_index()
    )


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    columns = [
        "game_type",
        "pitcher_id",
        "batter_hand",
        "balls_before",
        "strikes_before",
        "control_success",
    ]
    frame = normalized(
        pd.read_csv(
            ROOT / "data" / "train.csv",
            encoding="utf-8-sig",
            usecols=columns,
            low_memory=False,
        )
    )

    root = grouped(frame, ["game_type"])
    root["root_rate"] = (root["success"] + K_ROOT * ROOT_PRIOR) / (
        root["n"] + K_ROOT
    )

    pitcher = grouped(frame, ["game_type", "pitcher_id"]).merge(
        root[["game_type", "root_rate"]], on="game_type", how="left", validate="m:1"
    )
    pitcher["pitcher_rate"] = (
        pitcher["success"] + K_PITCHER * pitcher["root_rate"]
    ) / (pitcher["n"] + K_PITCHER)

    hand = grouped(frame, ["game_type", "pitcher_id", "batter_hand"]).merge(
        pitcher[["game_type", "pitcher_id", "pitcher_rate"]],
        on=["game_type", "pitcher_id"],
        how="left",
        validate="m:1",
    )
    hand["hand_rate"] = (
        hand["success"] + K_HAND * hand["pitcher_rate"]
    ) / (hand["n"] + K_HAND)
    hand["corr_hand"] = (
        hand["n"] / (hand["n"] + K_HAND)
    ) * (hand["hand_rate"] - hand["pitcher_rate"])

    detail_keys = ["game_type", "pitcher_id", "pressure", "batter_hand"]
    detail = grouped(frame, detail_keys).merge(
        hand[["game_type", "pitcher_id", "batter_hand", "hand_rate"]],
        on=["game_type", "pitcher_id", "batter_hand"],
        how="left",
        validate="m:1",
    )
    detail["detail_rate"] = (
        detail["success"] + K_DETAIL * detail["hand_rate"]
    ) / (detail["n"] + K_DETAIL)
    detail["corr_detail"] = (
        detail["n"] / (detail["n"] + K_DETAIL)
    ) * (detail["detail_rate"] - detail["hand_rate"])

    hand_out = hand[
        ["game_type", "pitcher_id", "batter_hand", "corr_hand"]
    ].copy()
    detail_out = detail[detail_keys + ["corr_detail"]].copy()
    hand_out["corr_hand"] = hand_out["corr_hand"].astype(np.float32)
    detail_out["corr_detail"] = detail_out["corr_detail"].astype(np.float32)
    if hand_out.duplicated(["game_type", "pitcher_id", "batter_hand"]).any():
        raise AssertionError("duplicate hand table key")
    if detail_out.duplicated(detail_keys).any():
        raise AssertionError("duplicate detail table key")

    metadata = {
        "source": "data/train.csv only",
        "train_rows": int(len(frame)),
        "gamma": GAMMA,
        "root_prior": ROOT_PRIOR,
        "k_root": K_ROOT,
        "k_pitcher": K_PITCHER,
        "k_hand": K_HAND,
        "k_detail": K_DETAIL,
        "hand_rows": int(len(hand_out)),
        "detail_rows": int(len(detail_out)),
        "hand_corr_mean": float(hand_out["corr_hand"].mean()),
        "hand_corr_std": float(hand_out["corr_hand"].std()),
        "detail_corr_mean": float(detail_out["corr_detail"].mean()),
        "detail_corr_std": float(detail_out["corr_detail"].std()),
    }
    return hand_out, detail_out, metadata


def save_tables(hand: pd.DataFrame, detail: pd.DataFrame) -> None:
    np.savez_compressed(
        TABLE_PATH,
        hand_game=hand["game_type"].to_numpy(str),
        hand_pitcher=hand["pitcher_id"].to_numpy(np.int64),
        hand_batter=hand["batter_hand"].to_numpy(str),
        hand_corr=hand["corr_hand"].to_numpy(np.float32),
        detail_game=detail["game_type"].to_numpy(str),
        detail_pitcher=detail["pitcher_id"].to_numpy(np.int64),
        detail_pressure=detail["pressure"].to_numpy(np.int8),
        detail_batter=detail["batter_hand"].to_numpy(str),
        detail_corr=detail["corr_detail"].to_numpy(np.float32),
    )


INFERENCE_FUNCTION = r'''

# Frozen public-V18-style conditional residual.  The NPZ was built only from
# train.csv.  Each test row independently looks up two fixed corrections.
V18_GAMMA = __V18_GAMMA__


def attach_v18_residual(rows, tables):
    game = rows["game_type"].astype("string").fillna("__MISSING__").astype(str)
    pitcher = pd.to_numeric(rows["pitcher_id"], errors="coerce").fillna(-1).astype("int64")
    batter = rows["batter_hand"].astype("string").fillna("__MISSING__").astype(str)
    balls = rows["balls_before"].fillna(0).to_numpy("int8")
    strikes = rows["strikes_before"].fillna(0).to_numpy("int8")
    pressure = np.where((balls == 3) & (strikes == 2), 2,
                        np.where((balls == 3) | (strikes == 2), 1, 0)).astype("int8")
    keys = pd.DataFrame({"game_type": game.to_numpy(),
                         "pitcher_id": pitcher.to_numpy(),
                         "batter_hand": batter.to_numpy(),
                         "pressure": pressure})
    hand = pd.DataFrame({"game_type": tables["hand_game"],
                         "pitcher_id": tables["hand_pitcher"],
                         "batter_hand": tables["hand_batter"],
                         "corr_hand": tables["hand_corr"]})
    detail = pd.DataFrame({"game_type": tables["detail_game"],
                           "pitcher_id": tables["detail_pitcher"],
                           "pressure": tables["detail_pressure"],
                           "batter_hand": tables["detail_batter"],
                           "corr_detail": tables["detail_corr"]})
    h = keys.merge(hand, on=["game_type", "pitcher_id", "batter_hand"],
                   how="left", sort=False, validate="m:1")["corr_hand"].fillna(0.0).to_numpy("float64")
    q = keys.merge(detail, on=["game_type", "pitcher_id", "pressure", "batter_hand"],
                   how="left", sort=False, validate="m:1")["corr_detail"].fillna(0.0).to_numpy("float64")
    return h + q
'''


def patch_script(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    function_anchor = "# =====================================================================\n# 기본 엔지니어링 18개"
    check(function_anchor in source, "inference function anchor found", "function anchor missing")
    inference_function = INFERENCE_FUNCTION.replace("__V18_GAMMA__", repr(GAMMA))
    source = source.replace(function_anchor, inference_function + "\n\n" + function_anchor, 1)

    load_anchor = '    b = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))\n'
    load_patch = load_anchor + '    v18_tables = np.load(os.path.join(MODEL_DIR, "v18_tables.npz"), allow_pickle=False)\n'
    check(source.count(load_anchor) == 1, "model load anchor found", "model load anchor ambiguous")
    source = source.replace(load_anchor, load_patch, 1)

    affine = "    preds = np.clip(CENTER_C + SCALE_S * (preds - CENTER_C) - SHIFT_DELTA, 0.0, 1.0)"
    correction = (
        '    v18_effect = attach_v18_residual(test, v18_tables)\n'
        '    preds = np.clip(preds + V18_GAMMA * v18_effect, 0.0, 1.0)\n'
        '    if len(preds):\n'
        '        print(f"  V18 frozen residual gamma={V18_GAMMA} coverage={(v18_effect != 0).mean()*100:.1f}% "\n'
        '              f"effect mean={v18_effect.mean():+.6f} std={v18_effect.std():.6f}")\n'
        + affine
    )
    check(source.count(affine) == 1, "affine anchor found", "affine anchor missing/ambiguous")
    source = source.replace(affine, correction, 1)
    path.write_text(source, encoding="utf-8")


def run_script(directory: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(VENV_PY), "-u", "script.py"],
        cwd=directory,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def extract(zip_path: Path, directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(directory)


def write_fake_data(directory: Path, rows: pd.DataFrame) -> None:
    (directory / "data").mkdir(parents=True, exist_ok=True)
    rows.to_csv(directory / "data" / "test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {"row_id": rows["row_id"], "control_success": np.full(len(rows), 0.5)}
    ).to_csv(
        directory / "data" / "sample_submission.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    started = time.time()
    log("=" * 88)
    log("exp/105 — frozen V18-style residual candidate build")
    log("=" * 88)
    check(SRC_ZIP.is_file(), f"source exists: {SRC_ZIP.name}", "source ZIP missing")
    check(VENV_PY.is_file(), "server-equivalent venv311 exists", "venv311 missing")
    check(LOCK_PATH.is_file() and EXACT_PATH.is_file(), "validation records exist", "validation records missing")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    exact = json.loads(EXACT_PATH.read_text(encoding="utf-8"))
    check(lock.get("pregate_pass") is True, "exp/103 PASS", "exp/103 not PASS")
    check(exact.get("all_requested_pass") is True, "exp/104 exact folds PASS", "exp/104 not PASS")
    locked_gamma = float(lock["selected_gamma"])
    gamma_is_locked = bool(np.isclose(locked_gamma, GAMMA))
    if not gamma_is_locked and not args.allow_unlocked_gamma:
        raise RuntimeError(
            f"gamma={GAMMA:g} differs from validation lock={locked_gamma:g}; "
            "use --allow-unlocked-gamma only for a separately named LB bracket probe"
        )
    if not gamma_is_locked:
        protected_defaults = {
            OUT_ZIP: ROOT / 'artifacts/candidates/candidate_v18g030.zip',
            SRC_DIR: ROOT / 'candidates/candidate_v18g030_src',
            REPORT_PATH: ROOT / "lab" / "105_build_v18_bundle.txt",
            META_PATH: ROOT / "lab" / "105_v18_tables.json",
        }
        reused = [str(path) for path, protected in protected_defaults.items() if path == protected]
        if reused:
            raise RuntimeError(
                "unlocked-gamma probe must use separate --out-zip, --src-dir, "
                f"--report-path and --meta-path; protected paths: {reused}"
            )
        log(f"    [PROBE] unlocked gamma={GAMMA:g}; validation lock remains {locked_gamma:g}")
    else:
        check(True, f"locked gamma={GAMMA:.2f}", "unreachable")

    for path in (SRC_DIR, REPORT_PATH, META_PATH, SIM_FULL, SIM_HEAD, SIM_REAL, SIM_ORIG):
        try:
            path.relative_to(ROOT)
        except ValueError as error:
            raise RuntimeError(f"unsafe path outside workspace: {path}") from error

    with zipfile.ZipFile(SRC_ZIP) as archive:
        original_names = sorted(archive.namelist())
    check(original_names == ["model/model.pkl", "requirements.txt", "script.py"],
          f"source structure {original_names}", "unexpected source structure")
    extract(SRC_ZIP, SRC_DIR)
    original_model_hash = sha256(SRC_DIR / "model" / "model.pkl")
    patch_script(SRC_DIR / "script.py")

    log("\nBuild frozen tables from train.csv only...")
    hand, detail, metadata = build_tables()
    save_tables(hand, detail)
    META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    log(
        f"    hand={len(hand):,} detail={len(detail):,} table={TABLE_PATH.stat().st_size/1024:.1f} KiB"
    )
    check(np.isfinite(hand["corr_hand"]).all() and np.isfinite(detail["corr_detail"]).all(),
          "all corrections finite", "non-finite correction")
    check(sha256(SRC_DIR / "model" / "model.pkl") == original_model_hash,
          "model.pkl byte-identical after patch", "model.pkl changed")

    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in (
            "script.py",
            "requirements.txt",
            "model/model.pkl",
            "model/v18_tables.npz",
        ):
            archive.write(SRC_DIR / relative, relative)
    with zipfile.ZipFile(OUT_ZIP) as archive:
        names = sorted(archive.namelist())
        corrupt = archive.testzip()
    expected = [
        "model/model.pkl",
        "model/v18_tables.npz",
        "requirements.txt",
        "script.py",
    ]
    check(names == expected and corrupt is None,
          f"candidate ZIP valid ({OUT_ZIP.stat().st_size/1024**2:.1f} MiB)", "candidate ZIP invalid")

    log("\nServer-equivalent environment check...")
    version = subprocess.run(
        [str(VENV_PY), "-c", "import sys,numpy,pandas,catboost;print(sys.version.split()[0],numpy.__version__,pandas.__version__,catboost.__version__)"],
        capture_output=True,
        text=True,
    )
    log(f"    {version.stdout.strip()}")
    check(version.stdout.startswith("3.11") and "1.26.4" in version.stdout and "2.0.3" in version.stdout and "1.2.10" in version.stdout,
          "server versions match", "server version mismatch")

    log(f"\nFull fake-server run ({N_EVAL:,} rows)...")
    for directory in (SIM_FULL, SIM_HEAD, SIM_REAL, SIM_ORIG):
        if directory.exists():
            shutil.rmtree(directory)
    extract(OUT_ZIP, SIM_FULL)
    test_columns = [
        column.replace("\ufeff", "").strip()
        for column in pd.read_csv(ROOT / "data" / "test.csv", encoding="utf-8-sig", nrows=1).columns
    ]
    source = pd.read_csv(ROOT / "data" / "train.csv", encoding="utf-8-sig", nrows=N_EVAL)
    source.columns = [column.replace("\ufeff", "").strip() for column in source.columns]
    fake = source[[column for column in test_columns if column != "row_id"]].copy()
    fake.insert(0, "row_id", [f"TEST_{i:06d}" for i in range(len(fake))])
    write_fake_data(SIM_FULL, fake)
    run_started = time.time()
    full_run = run_script(SIM_FULL)
    elapsed = time.time() - run_started
    for line in (full_run.stdout or "").splitlines():
        log(f"    | {line}")
    if full_run.returncode != 0:
        for line in (full_run.stderr or "").splitlines()[-30:]:
            log(f"    | {line}")
    check(full_run.returncode == 0, f"full inference OK ({elapsed:.1f}s)", f"full inference exit={full_run.returncode}")
    gamma_log = f"V18 frozen residual gamma={GAMMA:g}"
    check(gamma_log in (full_run.stdout or ""), f"correction path confirmed ({gamma_log})", "correction log missing")
    check(elapsed < 600.0, "runtime below 10 minutes", "runtime limit exceeded")
    full_sub = pd.read_csv(SIM_FULL / "output" / "submission.csv")
    full_probability = full_sub["control_success"].to_numpy(np.float64)
    check(
        len(full_sub) == N_EVAL
        and full_sub["row_id"].tolist() == fake["row_id"].tolist()
        and np.isfinite(full_probability).all()
        and full_probability.min() >= 0.0
        and full_probability.max() <= 1.0,
        f"full output valid mean={full_probability.mean():.6f}",
        "full output invalid",
    )

    log("\nRow-independence: first 1,000 rows alone vs full run...")
    extract(OUT_ZIP, SIM_HEAD)
    write_fake_data(SIM_HEAD, fake.head(1000).copy())
    head_run = run_script(SIM_HEAD)
    check(head_run.returncode == 0, "head inference OK", f"head inference exit={head_run.returncode}")
    if head_run.returncode == 0:
        head_probability = pd.read_csv(SIM_HEAD / "output" / "submission.csv")[
            "control_success"
        ].to_numpy(np.float64)
        maximum_difference = float(np.max(np.abs(head_probability - full_probability[:1000])))
        check(maximum_difference == 0.0, f"row-independent maxdiff={maximum_difference:.2e}", f"row dependence maxdiff={maximum_difference:.3e}")

    log("\nReal five-row test: candidate vs source bundle...")
    extract(OUT_ZIP, SIM_REAL)
    extract(SRC_ZIP, SIM_ORIG)
    for directory in (SIM_REAL, SIM_ORIG):
        (directory / "data").mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "data" / "test.csv", directory / "data" / "test.csv")
        shutil.copy(ROOT / "data" / "sample_submission.csv", directory / "data" / "sample_submission.csv")
    real_run = run_script(SIM_REAL)
    orig_run = run_script(SIM_ORIG)
    check(real_run.returncode == 0 and orig_run.returncode == 0,
          "candidate/source real test inference OK", f"real exits={real_run.returncode}/{orig_run.returncode}")
    if real_run.returncode == 0 and orig_run.returncode == 0:
        real_probability = pd.read_csv(SIM_REAL / "output" / "submission.csv")["control_success"].to_numpy(np.float64)
        orig_probability = pd.read_csv(SIM_ORIG / "output" / "submission.csv")["control_success"].to_numpy(np.float64)
        difference = real_probability - orig_probability
        log(f"    source    {orig_probability.round(7).tolist()}")
        log(f"    candidate {real_probability.round(7).tolist()}")
        log(f"    delta     {difference.round(7).tolist()}")
        check(np.isfinite(real_probability).all(), "real candidate finite", "real candidate non-finite")

    for directory in (SIM_FULL, SIM_HEAD, SIM_REAL, SIM_ORIG):
        shutil.rmtree(directory, ignore_errors=True)

    log("")
    log("=" * 88)
    log(("ALL CHECKS PASS — candidate ready, not submitted: " if OK else "CHECK FAILURE — do not submit: ") + str(OUT_ZIP))
    log(f"sha256={sha256(OUT_ZIP)}")
    log(f"elapsed={time.time() - started:.1f}s")
    log("=" * 88)
    REPORT_PATH.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    raise SystemExit(0 if OK else 1)


if __name__ == "__main__":
    main()
