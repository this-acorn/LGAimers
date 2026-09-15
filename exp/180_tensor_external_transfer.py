# -*- coding: utf-8 -*-
"""Build and audit official-only 2024 -> hidden-2025 Tensor-EB candidates."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "lab"
TRAIN = ROOT / "data" / "train.csv"
TEST = ROOT / "data" / "test.csv"
SAMPLE = ROOT / "data" / "sample_submission.csv"
ANCHOR_SOURCE = ROOT / 'candidates/candidate_exp021_w0356557_v002_src'
RUNTIME_SOURCE = ROOT / "exp" / "180_tensor_runtime.py"
OUT_JSON = LAB / "180_tensor_external_transfer_audit.json"
OUT_TXT = LAB / "180_tensor_external_transfer_audit.txt"
SCALES = (0.20, 0.58)
SPECS = (
    ("pitcher_batter_hand", ("pitcher_id", "batter_hand"), 0.50, 500.0),
    ("pitcher_count_base", ("pitcher_id", "_count_state", "base_state"), 0.40, 1000.0),
    ("hand_count_base", ("pitcher_hand", "batter_hand", "_count_state", "base_state"), 0.10, 500.0),
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def build_table(source: pd.DataFrame, residual: np.ndarray,
                keys: tuple[str, ...], alpha: float) -> pd.DataFrame:
    work = source.loc[:, list(keys)].copy()
    work["residual"] = np.asarray(residual, np.float64)
    table = work.groupby(
        list(keys), sort=False, observed=True, dropna=False
    )["residual"].agg(["sum", "size"]).reset_index()
    table["correction"] = table["sum"] / (table["size"] + alpha)
    output = table.loc[:, list(keys) + ["correction"]].copy()
    if output.duplicated(list(keys)).any() or not np.isfinite(output["correction"]).all():
        raise AssertionError("invalid EB table")
    return output


def lookup(rows: pd.DataFrame, table: pd.DataFrame,
           keys: tuple[str, ...]) -> tuple[np.ndarray, float]:
    left = rows.loc[:, list(keys)].copy()
    left["order"] = np.arange(len(left), dtype=np.int64)
    joined = left.merge(table, on=list(keys), how="left", sort=False,
                        validate="many_to_one").sort_values("order", kind="stable")
    if not np.array_equal(joined["order"], np.arange(len(rows))):
        raise AssertionError("lookup order changed")
    return joined["correction"].fillna(0.0).to_numpy(np.float64), float(
        joined["correction"].notna().mean()
    )


def tables_and_effect(source: pd.DataFrame, residual: np.ndarray,
                      validation: pd.DataFrame):
    tables = {}
    effect = np.zeros(len(validation), np.float64)
    diagnostics = {}
    for name, keys, weight, alpha in SPECS:
        table = build_table(source, residual, keys, alpha)
        correction, coverage = lookup(validation, table, keys)
        effect += weight * correction
        tables[name] = table
        diagnostics[name] = {
            "keys": list(keys), "weight": weight, "alpha": alpha,
            "groups": int(len(table)), "coverage": coverage,
            "mean_abs": float(np.abs(correction).mean()),
            "max_abs": float(np.abs(correction).max(initial=0.0)),
        }
    is_r = validation["game_type"].to_numpy() == "R"
    effect[~is_r] = 0.0
    return tables, effect, diagnostics


def zip_directory(source: Path, output: Path) -> None:
    temporary = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())
    os.replace(temporary, output)


def build_packages(tables: dict[str, pd.DataFrame], source_meta: dict):
    packages = {}
    for scale in SCALES:
        tag = f"s{int(round(scale * 100)):03d}"
        directory = ROOT / f"candidate_tensor_{tag}_src"
        archive = ROOT / f"candidate_tensor_{tag}.zip"
        if directory.exists() or archive.exists():
            if not directory.is_dir() or not archive.is_file():
                raise FileExistsError(f"partial pre-existing package: {directory} / {archive}")
            metadata_path = directory / "model" / "tensor_metadata.json"
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if float(existing.get("scale", -1.0)) != scale:
                raise ValueError(f"pre-existing package scale mismatch: {directory}")
            packages[tag] = {
                "source": str(directory), "zip": str(archive),
                "size": archive.stat().st_size, "sha256": sha256(archive),
                "reused_after_interrupted_QA": True,
            }
            continue
        shutil.copytree(ANCHOR_SOURCE, directory)
        shutil.copy2(RUNTIME_SOURCE, directory / "script.py")
        specifications = [
            {"name": name, "keys": list(keys), "weight": weight, "alpha": alpha}
            for name, keys, weight, alpha in SPECS
        ]
        joblib.dump({"specifications": specifications, "tables": tables},
                    directory / "model" / "tensor_tables.joblib", compress=3)
        metadata = {
            "method": "official 2024 R residual zero-prior Tensor-EB",
            "scale": scale, "specifications": specifications,
            "source": source_meta,
            "F_fallback": "exact current clean anchor",
            "row_independent": True,
        }
        write_json(directory / "model" / "tensor_metadata.json", metadata)
        zip_directory(directory, archive)
        packages[tag] = {
            "source": str(directory), "zip": str(archive),
            "size": archive.stat().st_size, "sha256": sha256(archive),
        }
    return packages


def run_fake(zip_path: Path, test: pd.DataFrame, sample: pd.DataFrame,
             label: str):
    with tempfile.TemporaryDirectory(prefix=f"exp180_{label}_", dir=LAB) as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(root)
        (root / "data").mkdir()
        (root / "output").mkdir()
        test.to_csv(root / "data" / "test.csv", index=False, encoding="utf-8")
        sample.to_csv(root / "data" / "sample_submission.csv", index=False,
                      encoding="utf-8")
        started = time.time()
        result = subprocess.run(
            [sys.executable, "-u", "script.py"], cwd=root,
            text=True, capture_output=True, timeout=600, check=False,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        elapsed = time.time() - started
        if result.returncode != 0:
            raise RuntimeError(f"fake server {label} failed: {result.stderr[-4000:]}")
        output = pd.read_csv(root / "output" / "submission.csv",
                             encoding="utf-8-sig")
        return output, elapsed, result.stdout[-3000:]


def main() -> None:
    started = time.time()
    for path in (TRAIN, TEST, SAMPLE, ANCHOR_SOURCE, RUNTIME_SOURCE):
        if not path.exists():
            raise FileNotFoundError(path)
    e158 = load_module(ROOT / "exp" / "158_robust_conditional_tensor_eb.py", "e180_158")
    e155 = load_module(ROOT / "exp" / "155_cause_runner_currentblend_gate.py", "e180_155")
    columns = [
        "season", "game_month", "game_type", "pitcher_id", "pitcher_hand",
        "batter_hand", "balls_before", "strikes_before", "base_state",
        "control_success",
    ]
    frame = e158.prepare_keys(pd.read_csv(
        TRAIN, encoding="utf-8-sig", usecols=columns, low_memory=False
    ))
    rows23 = frame.loc[frame["season"].eq(2023)].reset_index(drop=True)
    rows24 = frame.loc[frame["season"].eq(2024)].reset_index(drop=True)
    y23 = rows23["control_success"].to_numpy(np.float64)
    y24 = rows24["control_success"].to_numpy(np.float64)
    anchor23 = e158.load_current_baseline(e155, rows23, 2023)
    anchor24 = e158.load_current_baseline(e155, rows24, 2024)

    source23 = rows23.loc[rows23["game_type"].eq("R")].reset_index(drop=True)
    residual23 = (y23 - anchor23)[rows23["game_type"].to_numpy() == "R"]
    _, effect24, forward_diagnostics = tables_and_effect(source23, residual23, rows24)
    forward = {}
    for scale in SCALES:
        candidate = np.clip(anchor24 + scale * effect24, 0.0, 1.0)
        metric = e155.metrics(anchor24, scale * effect24, y24,
                              rows24["game_type"].to_numpy())
        month = rows24["game_month"].to_numpy()
        forward[str(scale)] = {
            **metric,
            "early_gain": e158.subset_gain(e155, anchor24, candidate, y24,
                                             month <= 6)["gain"],
            "late_gain": e158.subset_gain(e155, anchor24, candidate, y24,
                                            month > 6)["gain"],
            "pitcher_bootstrap": e158.cluster_bootstrap(
                anchor24, candidate, y24, rows24["pitcher_id"].to_numpy(np.int64),
                draws=5000, seed=180 + int(scale * 100),
            ),
        }

    source24_mask = rows24["game_type"].to_numpy() == "R"
    source24 = rows24.loc[source24_mask].reset_index(drop=True)
    residual24 = (y24 - anchor24)[source24_mask]
    test_raw = pd.read_csv(TEST, encoding="utf-8-sig", low_memory=False)
    test_raw.columns = test_raw.columns.str.replace("\ufeff", "", regex=False).str.strip()
    test_keys = e158.prepare_keys(test_raw)
    tables, test_effect, test_diagnostics = tables_and_effect(
        source24, residual24, test_keys
    )
    source_meta = {
        "official_train_sha256": sha256(TRAIN),
        "source_season": 2024, "source_R_rows": int(source24_mask.sum()),
        "residual": "control_success - own current-clean OOF",
        "current_anchor_score_2024": e155.score(anchor24, y24),
    }
    packages = build_packages(tables, source_meta)

    sample = pd.read_csv(SAMPLE, encoding="utf-8-sig")
    regular, runtime_regular, stdout_regular = run_fake(
        Path(packages["s020"]["zip"]), test_raw, sample, "regular"
    )
    reverse, runtime_reverse, stdout_reverse = run_fake(
        Path(packages["s020"]["zip"]), test_raw.iloc[::-1].reset_index(drop=True),
        sample.iloc[::-1].reset_index(drop=True), "reverse"
    )
    aligned = regular.merge(reverse, on="row_id", suffixes=("_a", "_b"),
                            validate="one_to_one")
    independence_max_abs = float(np.max(np.abs(
        aligned["control_success_a"] - aligned["control_success_b"]
    )))
    qa_pass = bool(
        len(regular) == len(test_raw)
        and independence_max_abs <= 1e-15
        and max(runtime_regular, runtime_reverse) < 600.0
        and all(value["size"] < 500_000_000 for value in packages.values())
    )
    offline_gate = {
        str(scale): bool(
            forward[str(scale)]["raw_gain"] >= 8.0
            and forward[str(scale)]["early_gain"] > 0.0
            and forward[str(scale)]["late_gain"] > 0.0
            and forward[str(scale)]["pitcher_bootstrap"]["p025"] > 0.0
        )
        for scale in SCALES
    }
    payload = {
        "experiment": 180,
        "status": (
            "BUILT_HIGH_RISK_OFFLINE_GATE_FAIL_NOT_SUBMITTED"
            if qa_pass and not any(offline_gate.values())
            else "BUILT_NOT_SUBMITTED"
            if qa_pass
            else "FAIL_QA"
        ),
        "data_boundary": "official train/test + own current OOF/current clean anchor only",
        "external_code_models_predictions": False,
        "specifications": [
            {"name": n, "keys": list(k), "weight": w, "alpha": a}
            for n, k, w, a in SPECS
        ],
        "forward_2023_to_2024": forward,
        "offline_deployment_gate": offline_gate,
        "forward_components": forward_diagnostics,
        "external_2024_to_test": {
            "source": source_meta, "lookup": test_diagnostics,
            "effect_R_mean": float(test_effect[test_keys["game_type"].eq("R")].mean()),
            "effect_R_std": float(test_effect[test_keys["game_type"].eq("R")].std()),
            "F_effect_exact_zero": bool(np.all(test_effect[test_keys["game_type"].eq("F")] == 0.0)),
        },
        "packages": packages,
        "fake_server": {
            "local_public_test_rows": int(len(test_raw)),
            "hidden_full_runtime_verified": False,
            "regular_seconds": runtime_regular, "reverse_seconds": runtime_reverse,
            "row_independence_max_abs": independence_max_abs,
            "pass": qa_pass, "regular_stdout_tail": stdout_regular,
            "reverse_stdout_tail": stdout_reverse,
        },
        "submission_performed": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(OUT_JSON, payload)
    lines = [
        "EXP180 Tensor external transfer",
        *[
            f"scale {scale:.2f}: 2024 raw={forward[str(scale)]['raw_gain']:+.6f} "
            f"early={forward[str(scale)]['early_gain']:+.6f} "
            f"late={forward[str(scale)]['late_gain']:+.6f} "
            f"CI025={forward[str(scale)]['pitcher_bootstrap']['p025']:+.6f}"
            for scale in SCALES
        ],
        f"test effect R mean/std={payload['external_2024_to_test']['effect_R_mean']:+.8f}/"
        f"{payload['external_2024_to_test']['effect_R_std']:.8f}",
        f"fake runtime={runtime_regular:.2f}/{runtime_reverse:.2f}s "
        f"row_independence={independence_max_abs:.3e} pass={qa_pass}",
        *[f"{key}: {value['zip']} size={value['size']} sha256={value['sha256']}"
          for key, value in packages.items()],
        "NOT SUBMITTED",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
