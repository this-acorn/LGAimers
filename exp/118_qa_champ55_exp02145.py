# -*- coding: utf-8 -*-
"""Server-environment and row-independence QA for an EXP021 blend."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import catboost
import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_NAME = os.environ.get("EXP021_QA_BUNDLE", 'artifacts/candidates/candidate_champ55_exp02145_v001.zip')
BUNDLE = Path(BUNDLE_NAME)
if not BUNDLE.is_absolute():
    BUNDLE = ROOT / BUNDLE
CHAMPION = ROOT / 'artifacts/candidates/candidate_v18g030.zip'
EXP021 = ROOT / 'artifacts/candidates/candidate_exp021_strict_rebuilt_v001.zip'
DATA = ROOT / "data"
REPORT_STEM = os.environ.get("EXP021_QA_REPORT_STEM", "132_champ55_exp02145_qa")
REPORT_JSON = ROOT / "lab" / f"{REPORT_STEM}.json"
REPORT_TEXT = ROOT / "lab" / f"{REPORT_STEM}.txt"
TARGET = "control_success"
ID = "row_id"
FULL_ROWS = 245_789
TOLERANCE = 1e-12
CHAMPION_WEIGHT = float(os.environ.get("EXP021_QA_CHAMPION_WEIGHT", "0.55"))
EXP021_WEIGHT = float(os.environ.get("EXP021_QA_ENDPOINT_WEIGHT", "0.45"))

EXPECTED_ENV = {
    "python": "3.11",
    "numpy": "1.26.4",
    "pandas": "2.0.3",
    "sklearn": "1.8.0",
    "joblib": "1.5.3",
    "catboost": "1.2.10",
    "lightgbm": "4.6.0",
}


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
        "joblib": joblib.__version__,
        "catboost": catboost.__version__,
        "lightgbm": lightgbm.__version__,
    }
    if found != EXPECTED_ENV:
        raise RuntimeError(f"server QA environment mismatch: {found} != {EXPECTED_ENV}")
    return found


def inspect_zip(path: Path, kind: str) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        names = archive.namelist()
        if bad is not None:
            raise AssertionError(f"{kind} CRC failure: {bad}")
        if names[:2] != ["script.py", "requirements.txt"]:
            raise AssertionError(f"{kind} root structure drift")
        if any(name.startswith(("data/", "output/")) or "__pycache__" in name for name in names):
            raise AssertionError(f"{kind} contains runtime/generated data")
        if any(name.lower().endswith(("train.csv", "test.csv", "submission.csv")) for name in names):
            raise AssertionError(f"{kind} contains a data CSV")
        return {"sha256": sha256(path), "bytes": path.stat().st_size, "entries": names, "crc": "passed"}


def stage_zip(path: Path, destination: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        archive.extractall(destination)
    (destination / "data").mkdir()
    (destination / "output").mkdir()


def run_case(stage: Path, test: pd.DataFrame, sample: pd.DataFrame) -> tuple[pd.DataFrame, float, str]:
    output_path = stage / "output" / "submission.csv"
    if output_path.exists():
        output_path.unlink()
    test.to_csv(stage / "data" / "test.csv", index=False, encoding="utf-8-sig")
    sample.to_csv(stage / "data" / "sample_submission.csv", index=False, encoding="utf-8-sig")
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
    prediction = pd.read_csv(output_path, encoding="utf-8-sig")
    if list(prediction.columns) != [ID, TARGET]:
        raise AssertionError(f"invalid output columns: {prediction.columns.tolist()}")
    if not prediction[ID].equals(sample[ID]):
        raise AssertionError("output row_id order differs from sample")
    if prediction[ID].duplicated().any():
        raise AssertionError("output contains duplicate row_id")
    values = prediction[TARGET].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((values >= 0).all() and (values <= 1).all()):
        raise AssertionError("invalid output probability")
    return prediction, runtime, result.stdout


def sample_for(test: pd.DataFrame) -> pd.DataFrame:
    ids = test[ID].reset_index(drop=True)
    return pd.DataFrame({ID: ids, TARGET: np.zeros(len(ids), dtype=float)})


def unique_repeated_test(source: pd.DataFrame, rows: int, prefix: str) -> tuple[pd.DataFrame, np.ndarray]:
    positions = np.arange(rows, dtype=np.int64) % len(source)
    repeated = source.iloc[positions].reset_index(drop=True).copy()
    repeated[ID] = [f"{prefix}_{index:06d}" for index in range(rows)]
    return repeated, positions


def max_abs_by_id(reference: pd.DataFrame, candidate: pd.DataFrame) -> float:
    left = reference.set_index(ID)[TARGET].sort_index()
    right = candidate.set_index(ID)[TARGET].sort_index()
    if not left.index.equals(right.index):
        raise AssertionError("row_id sets differ")
    return float(np.max(np.abs(left.to_numpy(dtype=float) - right.to_numpy(dtype=float)))) if len(left) else 0.0


def main() -> None:
    environment = check_environment()
    archives = {
        "champion": inspect_zip(CHAMPION, "champion"),
        "exp021": inspect_zip(EXP021, "exp021"),
        "blend": inspect_zip(BUNDLE, "blend"),
    }
    genuine = pd.read_csv(DATA / "test.csv", encoding="utf-8-sig")
    genuine.columns = [column.replace("﻿", "").strip() for column in genuine.columns]
    genuine_sample = pd.read_csv(DATA / "sample_submission.csv", encoding="utf-8-sig")
    genuine_sample.columns = [column.replace("﻿", "").strip() for column in genuine_sample.columns]

    with tempfile.TemporaryDirectory(prefix="qa-champ55-exp02145-") as raw:
        temporary = Path(raw)
        champion_stage = temporary / "champion"
        exp021_stage = temporary / "exp021"
        blend_stage = temporary / "blend"
        for archive, stage in ((CHAMPION, champion_stage), (EXP021, exp021_stage), (BUNDLE, blend_stage)):
            stage.mkdir()
            stage_zip(archive, stage)

        champion5, champion5_seconds, _ = run_case(champion_stage, genuine, genuine_sample)
        exp0215, exp0215_seconds, _ = run_case(exp021_stage, genuine, genuine_sample)
        blend5, blend5_seconds, blend5_stdout = run_case(blend_stage, genuine, genuine_sample)
        expected5 = (
            CHAMPION_WEIGHT * champion5[TARGET].to_numpy(dtype=float)
            + EXP021_WEIGHT * exp0215[TARGET].to_numpy(dtype=float)
        )
        component_parity = float(np.max(np.abs(blend5[TARGET].to_numpy(dtype=float) - expected5)))
        if component_parity > TOLERANCE:
            raise AssertionError(f"component blend parity failed: {component_parity}")

        small, small_positions = unique_repeated_test(genuine, 40, "SMALL")
        small_sample = sample_for(small)
        baseline, small_seconds, _ = run_case(blend_stage, small, small_sample)
        expected_small = expected5[small_positions]
        small_pattern_diff = float(np.max(np.abs(baseline[TARGET].to_numpy(dtype=float) - expected_small)))
        if small_pattern_diff > TOLERANCE:
            raise AssertionError(f"row-local repeated-pattern parity failed: {small_pattern_diff}")

        repeated, repeat_seconds, _ = run_case(blend_stage, small, small_sample)
        repeat_diff = max_abs_by_id(baseline, repeated)

        reverse_index = np.arange(len(small) - 1, -1, -1)
        reverse_test = small.iloc[reverse_index].reset_index(drop=True)
        reversed_output, reverse_seconds, _ = run_case(blend_stage, reverse_test, sample_for(reverse_test))
        reverse_diff = max_abs_by_id(baseline, reversed_output)

        rng = np.random.default_rng(20260831)
        permutation = rng.permutation(len(small))
        permuted_test = small.iloc[permutation].reset_index(drop=True)
        permuted_output, permutation_seconds, _ = run_case(blend_stage, permuted_test, sample_for(permuted_test))
        permutation_diff = max_abs_by_id(baseline, permuted_output)

        first_output, split1_seconds, _ = run_case(blend_stage, small.iloc[:20].copy(), sample_for(small.iloc[:20]))
        second_output, split2_seconds, _ = run_case(blend_stage, small.iloc[20:].copy(), sample_for(small.iloc[20:]))
        split_output = pd.concat([first_output, second_output], ignore_index=True)
        split_diff = max_abs_by_id(baseline, split_output)

        singleton_differences = []
        singleton_seconds = 0.0
        for index in range(len(genuine)):
            one = genuine.iloc[[index]].copy()
            one_output, seconds, _ = run_case(blend_stage, one, sample_for(one))
            singleton_seconds += seconds
            singleton_differences.append(
                abs(float(one_output.iloc[0][TARGET]) - float(blend5.iloc[index][TARGET]))
            )
        singleton_diff = float(max(singleton_differences, default=0.0))

        full_test, full_positions = unique_repeated_test(genuine, FULL_ROWS, "FULL")
        full_output, full_seconds, full_stdout = run_case(blend_stage, full_test, sample_for(full_test))
        if full_seconds >= 600.0:
            raise AssertionError(f"full fake-server runtime exceeded limit: {full_seconds:.1f}s")
        expected_full = expected5[full_positions]
        full_pattern_diff = float(np.max(np.abs(full_output[TARGET].to_numpy(dtype=float) - expected_full)))

    independence = {
        "repeat_max_abs": repeat_diff,
        "reverse_max_abs": reverse_diff,
        "permutation_max_abs": permutation_diff,
        "split_max_abs": split_diff,
        "singleton_max_abs": singleton_diff,
        "small_repeated_pattern_max_abs": small_pattern_diff,
        "full_repeated_pattern_max_abs": full_pattern_diff,
    }
    if any(value > TOLERANCE for value in independence.values()):
        raise AssertionError(f"row-independence QA failed: {independence}")

    report = {
        "artifact": str(BUNDLE),
        "sha256": sha256(BUNDLE),
        "environment": environment,
        "archives": archives,
        "weights": {"champion": CHAMPION_WEIGHT, "exp021_strict": EXP021_WEIGHT},
        "genuine_5": {
            "champion": champion5[TARGET].astype(float).tolist(),
            "exp021": exp0215[TARGET].astype(float).tolist(),
            "blend": blend5[TARGET].astype(float).tolist(),
            "blend_mean": float(blend5[TARGET].mean()),
            "component_parity_max_abs": component_parity,
            "stdout_tail": blend5_stdout.strip().splitlines()[-1],
        },
        "row_independence": independence,
        "runtime_seconds": {
            "champion_5": champion5_seconds,
            "exp021_5": exp0215_seconds,
            "blend_5": blend5_seconds,
            "small_40": small_seconds,
            "repeat_40": repeat_seconds,
            "reverse_40": reverse_seconds,
            "permutation_40": permutation_seconds,
            "split_20_plus_20": split1_seconds + split2_seconds,
            "singletons_5_total": singleton_seconds,
            "fake_server_245789": full_seconds,
        },
        "fake_server": {
            "rows": FULL_ROWS,
            "runtime_below_600": True,
            "stdout_tail": full_stdout.strip().splitlines()[-1],
            "row_id_order": "passed",
            "row_id_unique": "passed",
            "finite_range": "passed",
        },
        "tolerance": TOLERANCE,
        "status": "PASS",
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"EXP021 {EXP021_WEIGHT:.9%} blend QA: PASS",
        f"artifact={BUNDLE}",
        f"sha256={report['sha256']}",
        f"component_parity_max_abs={component_parity:.3e}",
        *(f"{key}={value:.3e}" for key, value in independence.items()),
        f"fake_server_rows={FULL_ROWS}",
        f"fake_server_runtime_seconds={full_seconds:.3f}",
    ]
    REPORT_TEXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
