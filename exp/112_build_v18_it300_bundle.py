# -*- coding: utf-8 -*-
"""Forensic reconstruction of the rejected IT300 + frozen V18 cross.

This is deployment engineering only.  It starts from the already submitted
``candidate_v18g030.zip`` and changes exactly one inference constant:

    NTREE_END = 500  ->  NTREE_END = 300

The model, frozen V18 lookup table, affine calibration, and every other byte of
the inference source remain unchanged.  The existing champion ZIP/source are
read-only and are checked again after the build.

IMPORTANT: the IT300 backbone already scored 1040.0366 on the 2025 LB, which
was -19.0 versus the then champion.  The positive 2024 result reproduced here
is known non-transfer and must never authorize a submission.
"""

from __future__ import annotations

_PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ZIP = ROOT / 'artifacts/candidates/candidate_v18g030.zip'
SOURCE_DIR = ROOT / 'candidates/candidate_v18g030_src'
REJECTED_DIR = ROOT / "archive" / "rejected" / "v18_it300"
OUTPUT_ZIP = REJECTED_DIR / "REJECTED_DO_NOT_SUBMIT_v18g030_it300.zip"
OUTPUT_DIR = REJECTED_DIR / "REJECTED_DO_NOT_SUBMIT_v18g030_it300_src"
REPORT = ROOT / "lab" / "112_build_v18_it300.txt"
META = ROOT / "lab" / "112_build_v18_it300.json"

EXPECTED_SOURCE_ZIP_SHA = "ed1b729b9b83eb1f0d344b475939353149d5adc83009a40ff116df6e5a168557"
EXPECTED_MODEL_SHA = "382ed9775302a2b57fa69fc568203a8c8cc3d4f8d210dae2b673074f00e006bc"
EXPECTED_TABLE_SHA = "6bb99c9f2c367c65f0551a2b43a00af99f5a7b3df01769d082ed744c5dc178a1"
OLD_LINE = "NTREE_END = 500"
NEW_LINE = "NTREE_END = 300"

VENV_PY = Path(
    str(_PROJECT_ROOT / "venv311/Scripts/python.exe")
)

LINES: list[str] = []


def log(message: str = "") -> None:
    print(message, flush=True)
    LINES.append(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
    log(f"[OK] {message}")


def score(probability: np.ndarray, target: np.ndarray) -> float:
    rate = float(target.mean())
    mse = float(np.mean((probability - target) ** 2))
    return 100000.0 * (1.0 - mse / (rate * (1.0 - rate)))


def equal_mean_gain(base: np.ndarray, candidate: np.ndarray, target: np.ndarray) -> float:
    same_mean = candidate - candidate.mean() + base.mean()
    return score(same_mean, target) - score(base, target)


def contribution(
    base: np.ndarray, candidate: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> float:
    rate = float(target.mean())
    row_gain = (base - target) ** 2 - (candidate - target) ** 2
    return 100000.0 * float(row_gain[mask].sum()) / (
        len(target) * rate * (1.0 - rate)
    )


def cluster_bootstrap(
    base: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    pitcher: np.ndarray,
    draws: int = 5000,
) -> dict[str, float]:
    work = pd.DataFrame(
        {
            "pitcher": pitcher,
            "n": np.ones(len(target), dtype=np.int32),
            "y": target,
            "base_se": (base - target) ** 2,
            "candidate_se": (candidate - target) ** 2,
        }
    )
    grouped = work.groupby("pitcher", sort=False, observed=True).agg(
        n=("n", "sum"),
        y=("y", "sum"),
        base_se=("base_se", "sum"),
        candidate_se=("candidate_se", "sum"),
    )
    values = grouped[["n", "y", "base_se", "candidate_se"]].to_numpy(float)
    rng = np.random.default_rng(112)
    gains = np.empty(draws, dtype=float)
    for start in range(0, draws, 250):
        size = min(250, draws - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        sampled = values[indices].sum(axis=1)
        rate = sampled[:, 1] / sampled[:, 0]
        gains[start : start + size] = 100000.0 * (
            sampled[:, 2] - sampled[:, 3]
        ) / (sampled[:, 0] * rate * (1.0 - rate))
    return {
        "draws": draws,
        "pitchers": int(len(values)),
        "median": float(np.median(gains)),
        "p025": float(np.quantile(gains, 0.025)),
        "p975": float(np.quantile(gains, 0.975)),
        "prob_positive": float(np.mean(gains > 0.0)),
    }


def validate_local() -> dict:
    frame = pd.read_csv(ROOT / "data" / "train.csv", encoding="utf-8-sig")
    frame.columns = [column.replace("ï»¿", "").strip() for column in frame.columns]
    rows = frame.loc[frame["season"] == 2024].reset_index(drop=True)
    target = rows["control_success"].to_numpy(float)
    effect = np.load(ROOT / "lab" / "103_v18_effect_2024.npy").astype(float)
    raw500 = np.load(ROOT / "lab" / "89_cat5.npy").astype(float)
    raw300 = np.load(ROOT / "lab" / "100_IT300.npy").astype(float)
    require(len(rows) == len(effect) == len(raw500) == len(raw300), "2024 arrays align")

    def deployed(raw: np.ndarray) -> np.ndarray:
        corrected = np.clip(raw + 0.30 * effect, 0.0, 1.0)
        return np.clip(0.49 + 1.06 * (corrected - 0.49) - 0.0066, 0.0, 1.0)

    base = deployed(raw500)
    candidate = deployed(raw300)
    game_type = rows["game_type"].astype(str).to_numpy()
    result = {
        "rows": int(len(rows)),
        "base_score": float(score(base, target)),
        "candidate_score": float(score(candidate, target)),
        "raw_gain": float(score(candidate, target) - score(base, target)),
        "shape_gain": float(equal_mean_gain(base, candidate, target)),
        "mean_shift": float(candidate.mean() - base.mean()),
        "F_contribution": float(
            contribution(base, candidate, target, game_type == "F")
        ),
        "R_contribution": float(
            contribution(base, candidate, target, game_type != "F")
        ),
        "bootstrap": cluster_bootstrap(
            base,
            candidate,
            target,
            rows["pitcher_id"].to_numpy(),
        ),
    }
    log(
        "2024 current(it500+V18) -> candidate(it300+V18): "
        f"{result['base_score']:.3f} -> {result['candidate_score']:.3f}, "
        f"gain={result['raw_gain']:+.3f}, shape={result['shape_gain']:+.3f}, "
        f"F/R={result['F_contribution']:+.3f}/{result['R_contribution']:+.3f}"
    )
    boot = result["bootstrap"]
    log(
        "pitcher bootstrap: "
        f"median={boot['median']:+.3f} 95%=[{boot['p025']:+.3f},{boot['p975']:+.3f}] "
        f"P(>0)={boot['prob_positive']:.4f}"
    )
    # These diagnostics explain why the locally attractive arm was tempting;
    # they are not deployment gates.  The direct 2025 LB result dominates them.
    return result


def make_simulation(source: Path, name: str) -> Path:
    destination = ROOT / name
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite existing simulation directory: {destination}")
    shutil.copytree(source, destination)
    data_dir = destination / "data"
    data_dir.mkdir()
    shutil.copy2(ROOT / "data" / "test.csv", data_dir / "test.csv")
    shutil.copy2(
        ROOT / "data" / "sample_submission.csv", data_dir / "sample_submission.csv"
    )
    return destination


def run_bundle(source: Path, name: str) -> tuple[np.ndarray, str]:
    simulation = make_simulation(source, name)
    python = VENV_PY if VENV_PY.is_file() else Path("py")
    command = [str(python), "-u", "script.py"]
    if python.name.lower() == "py":
        command = ["py", "-3.12", "-u", "script.py"]
    result = subprocess.run(
        command,
        cwd=simulation,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"simulation failed:\n{result.stdout}\n{result.stderr}")
    output = pd.read_csv(simulation / "output" / "submission.csv")
    probability = output["control_success"].to_numpy(float)
    return probability, result.stdout


def main() -> None:
    started = time.time()
    log("=" * 88)
    log("exp/112 — FORENSIC ONLY: rejected IT300 + frozen V18 correction")
    log("DO NOT SUBMIT: IT300 direct 2025 LB=1040.0366 (-19.0 vs then champion)")
    log("=" * 88)

    require(SOURCE_ZIP.is_file(), "protected champion ZIP exists")
    require(sha256(SOURCE_ZIP) == EXPECTED_SOURCE_ZIP_SHA, "protected champion ZIP hash")
    require(not OUTPUT_ZIP.exists(), "new candidate ZIP path is unused")
    require(not OUTPUT_DIR.exists(), "new candidate source path is unused")

    local = validate_local()

    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        require(
            sorted(archive.namelist())
            == ["model/model.pkl", "model/v18_tables.npz", "requirements.txt", "script.py"],
            "champion archive structure",
        )
        archive.extractall(OUTPUT_DIR)

    script_path = OUTPUT_DIR / "script.py"
    original_script = script_path.read_text(encoding="utf-8")
    require(original_script.count(OLD_LINE) == 1, "single NTREE_END=500 patch anchor")
    patched_script = original_script.replace(OLD_LINE, NEW_LINE, 1)
    script_path.write_text(patched_script, encoding="utf-8")
    require(
        original_script.replace(OLD_LINE, NEW_LINE, 1) == patched_script,
        "only intended source replacement applied",
    )
    require(sha256(OUTPUT_DIR / "model" / "model.pkl") == EXPECTED_MODEL_SHA, "model byte identity")
    require(sha256(OUTPUT_DIR / "model" / "v18_tables.npz") == EXPECTED_TABLE_SHA, "V18 table byte identity")

    current_probability, current_log = run_bundle(
        SOURCE_DIR, "_tmp_sim_112_current_it500"
    )
    candidate_probability, candidate_log = run_bundle(
        OUTPUT_DIR, "_tmp_sim_112_candidate_it300"
    )
    require(len(current_probability) == len(candidate_probability) > 0, "real test outputs align")
    require(np.isfinite(candidate_probability).all(), "candidate output is finite")
    require(
        float(np.max(np.abs(candidate_probability - current_probability))) > 0.0,
        "IT300 changes real test predictions",
    )
    require("ntree_end=500" in current_log, "current runtime confirms 500 trees")
    require("ntree_end=300" in candidate_log, "candidate runtime confirms 300 trees")
    require("V18 frozen residual gamma=0.3" in candidate_log, "candidate runtime confirms V18 path")

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(OUTPUT_DIR).as_posix())

    output_names: list[str]
    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        output_names = sorted(archive.namelist())
        test_result = archive.testzip()
    require(test_result is None, "candidate ZIP CRC")
    require(
        output_names
        == ["model/model.pkl", "model/v18_tables.npz", "requirements.txt", "script.py"],
        "candidate archive structure",
    )
    require(sha256(SOURCE_ZIP) == EXPECTED_SOURCE_ZIP_SHA, "protected champion still unchanged")

    metadata = {
        "experiment": 112,
        "description": "IT300 inference + frozen V18 gamma .30 + existing affine",
        "source_zip": SOURCE_ZIP.name,
        "source_sha256": EXPECTED_SOURCE_ZIP_SHA,
        "candidate_zip": OUTPUT_ZIP.name,
        "candidate_sha256": sha256(OUTPUT_ZIP),
        "model_sha256": sha256(OUTPUT_DIR / "model" / "model.pkl"),
        "v18_tables_sha256": sha256(OUTPUT_DIR / "model" / "v18_tables.npz"),
        "single_change": f"{OLD_LINE} -> {NEW_LINE}",
        "local_validation": local,
        "real_test_rows": int(len(candidate_probability)),
        "real_test_max_abs_delta_vs_current": float(
            np.max(np.abs(candidate_probability - current_probability))
        ),
        "elapsed_seconds": time.time() - started,
        "direct_it300_lb": 1040.0366,
        "direct_it300_lb_delta": -19.0,
        "status": "REJECTED_DO_NOT_SUBMIT",
    }
    META.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    log("")
    log(f"REJECTED, DO NOT SUBMIT: {OUTPUT_ZIP}")
    log(f"sha256={metadata['candidate_sha256']}")
    log(f"real test max |delta|={metadata['real_test_max_abs_delta_vs_current']:.8f}")
    log(f"elapsed={metadata['elapsed_seconds']:.1f}s")
    REPORT.write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"FAIL: {type(error).__name__}: {error}")
        REPORT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
        raise
