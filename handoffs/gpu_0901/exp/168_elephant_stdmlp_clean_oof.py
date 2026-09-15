# -*- coding: utf-8 -*-
"""Official-data-only forward OOF for the reproducible Elephant std-z MLP idea.

This is deliberately a *partial endpoint*, not a claim that the repository's
1082.106 ``sj_stdmlp`` submission is reproducible.  That endpoint depends on
an absent ``run_arm.py``, an absent SJ 272-feature deployment pipeline, and an
absent base champion archive.  This experiment reconstructs only the auditable
CW base features, train-only platoon encodings, id-frequency atoms, robust-z
preprocessing, and residual MLP architecture from the licensed source code.

Protocol:
  discovery: official seasons <=2022 -> 2023; lock the analytic convex blend
             weight against our current clean OOF.
  confirm:   only if discovery blend gain is positive, refit <=2023 -> 2024
             and apply the 2023 weight unchanged.

No test.csv, repository data/predictions, archive, weight, OOF, PT/NPZ, or
leaderboard result is opened.  Query features and inference are row-local;
all lookup/preprocessing state is fitted from earlier official train seasons.

Code provenance / license:
  Adapted from LA9elephantmiracle, commit
  57783aa88ccc30915a57abe887c8b8125408d483.
  Copyright (c) 2026 LA9elephantmiracle, MIT License.
  The repository LICENSE explicitly limits its MIT grant to code.  Accordingly
  only the listed Python source files are imported; no repository artifact is
  consumed.  Preserve the repository LICENSE notice with redistributions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train.csv"
REFERENCE = ROOT / "reference" / "LA9elephantmiracle"
LICENSE = REFERENCE / "LICENSE"
COMMON_SOURCE = REFERENCE / "cowork" / "cw" / "v17" / "src" / "common.py"
DL_SOURCE = REFERENCE / "cowork" / "cw" / "v17" / "src" / "dl.py"
PREP_SOURCE = (
    REFERENCE / "performance_tracking" / "models" / "sj_stdmlp" / "prep_mlp.py"
)
ATOMS_SOURCE = (
    REFERENCE / "performance_tracking" / "models" / "sj_stdmlp" / "atoms.py"
)

LAB = ROOT / "lab"
STEM = "168_elephant_stdmlp_clean_oof"
AUDIT_PATH = LAB / f"{STEM}_repo_audit.json"
LOCK_PATH = LAB / f"{STEM}_lock.json"
OUT_JSON = LAB / f"{STEM}.json"
OUT_TXT = LAB / f"{STEM}.txt"

AUDITED_COMMIT = "57783aa88ccc30915a57abe887c8b8125408d483"
THREADS = 6
SEED = 0
EPOCHS = 8
BATCH = 1024
WIDTH = 384
DEPTH = 3
DROP = 0.30
LR = 2.0e-3
WEIGHT_DECAY = 3.0e-4
SCORE_SCALE = 100000.0
PREDICT_BATCH = 8192
BOOTSTRAP_DRAWS = 5000

ID_COLUMNS = ("pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id")
SOURCE_FILES = (LICENSE, COMMON_SOURCE, DL_SOURCE, PREP_SOURCE, ATOMS_SOURCE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("discovery", "confirm"))
    return parser.parse_args()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load code source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray, dtype=np.float64) -> str:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def row_id_sha256(rows: pd.DataFrame) -> str:
    return hashlib.sha256(
        "\n".join(rows["row_id"].astype(str).tolist()).encode("utf-8")
    ).hexdigest()


def save_vector(kind: str, year: int, value: np.ndarray) -> dict[str, Any]:
    path = LAB / f"168_elephant_stdmlp_{kind}_{year}.npy"
    np.save(path, np.asarray(value, np.float32), allow_pickle=False)
    return {
        "path": str(path),
        "rows": int(len(value)),
        "dtype": "float32",
        "sha256": file_sha256(path),
    }


def source_audit() -> dict[str, Any]:
    missing = [str(path) for path in SOURCE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing audited code sources: {missing}")
    license_text = LICENSE.read_text(encoding="utf-8")
    if "MIT License" not in license_text or "Permission is hereby granted" not in license_text:
        raise RuntimeError("repository LICENSE is not the audited MIT text")
    audit = {
        "repository": "https://github.com/whdpdms2004-bot/LA9elephantmiracle.git",
        "audited_commit": AUDITED_COMMIT,
        "license": "MIT",
        "license_scope": (
            "repository LICENSE grants MIT rights to code only; repository-derived "
            "competition data is outside that grant"
        ),
        "exact_1082_106_reproduction": False,
        "exact_reproduction_blockers": [
            "run_arm.py imported by final builders is absent",
            "SJ 272-feature training/deployment pipeline is incomplete",
            "base champion archive required by build_submit_zip.py is absent",
        ],
        "partial_endpoint": (
            "CW row-local base features + strict platoon encoding + id_freq + "
            "robust-z residual MLP, retrained from official train.csv"
        ),
        "opened_reference_content": "listed Python source files and LICENSE only",
        "forbidden_reference_content_opened": False,
        "forbidden_extensions_not_opened": [
            ".zip", ".pt", ".npz", "prediction/OOF .csv", ".gz"
        ],
        "source_sha256": {str(path.relative_to(REFERENCE)): file_sha256(path) for path in SOURCE_FILES},
        "test_csv_read": False,
        "threads": THREADS,
    }
    LAB.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def load_official_through(year: int, common) -> pd.DataFrame:
    # No path other than official train.csv is permitted here.  Reading the
    # entire CSV is necessary to select rows, but targets after ``year`` are
    # immediately discarded and never placed in a Python result object.
    columns = list(dict.fromkeys(
        ["row_id", "control_success", *common.NUM_COLS, *common.CAT_LEVELS.keys()]
    ))
    frame = pd.read_csv(TRAIN, encoding="utf-8-sig", usecols=columns, low_memory=False)
    frame.columns = [column.replace("\ufeff", "").strip() for column in frame.columns]
    frame = frame.loc[pd.to_numeric(frame["season"], errors="coerce") <= year].reset_index(drop=True)
    if frame.empty or not frame["season"].eq(year).any():
        raise ValueError(f"official season {year} absent")
    target = pd.to_numeric(frame["control_success"], errors="coerce")
    if target.isna().any() or not target.isin((0, 1)).all():
        raise ValueError("invalid official target")
    return frame


def integer_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").fillna(-1).to_numpy(np.int64)


def strict_platoon_features(frame: pd.DataFrame, target: np.ndarray, common) -> np.ndarray:
    """For every season S, fit the four EB lookup columns using seasons <S."""
    seasons = integer_array(frame, "season")
    pitcher = integer_array(frame, "pitcher_id")
    batter = integer_array(frame, "batter_id")
    pitcher_hand = integer_array(frame, "pitcher_hand")
    batter_hand = integer_array(frame, "batter_hand")
    out = np.zeros((len(frame), 4), dtype=np.float32)
    out[:, :2] = np.nan
    for season in sorted(np.unique(seasons).tolist()):
        history = seasons < season
        query = seasons == season
        if not history.any():
            continue
        enc = common.build_encodings(
            pitcher[history], batter[history], pitcher_hand[history], batter_hand[history],
            target[history],
        )
        out[query] = common.encode_rows(
            pitcher[query], batter[query], pitcher_hand[query], batter_hand[query], enc
        )
    return out


def id_frequency_features(
    frame: pd.DataFrame, train_mask: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for column in ID_COLUMNS:
        values = integer_array(frame, column)
        unique, counts = np.unique(values[train_mask], return_counts=True)
        position = np.searchsorted(unique, values)
        clipped = np.clip(position, 0, len(unique) - 1)
        hit = unique[clipped] == values
        frequency = np.where(hit, counts[clipped], 0).astype(np.float32)
        blocks.extend((np.log1p(frequency), (frequency == 0).astype(np.float32)))
        names.extend((f"pa_{column}_logfreq", f"pa_{column}_unseen"))
    return np.column_stack(blocks).astype(np.float32), names


def build_fold(year: int) -> dict[str, Any]:
    common = load_module(COMMON_SOURCE, f"elephant_common_{year}")
    prep_code = load_module(PREP_SOURCE, f"elephant_prep_{year}")
    frame = load_official_through(year, common)
    seasons = integer_array(frame, "season")
    train_mask = seasons < year
    valid_mask = seasons == year
    target_all = frame["control_success"].to_numpy(np.float32)
    started = time.time()
    base, names = common.build_features(frame)
    platoon = strict_platoon_features(frame, target_all, common)
    names = list(names) + list(common.ENC_NAMES)
    matrix72 = np.ascontiguousarray(np.column_stack((base, platoon)), dtype=np.float32)
    frequency, frequency_names = id_frequency_features(frame, train_mask)
    raw = np.ascontiguousarray(np.column_stack((matrix72, frequency)), dtype=np.float32)
    names += frequency_names
    del base, platoon, matrix72, frequency
    gc.collect()
    train_raw = np.ascontiguousarray(raw[train_mask])
    valid_raw = np.ascontiguousarray(raw[valid_mask])
    del raw
    prep = prep_code.make_prep(train_raw, "std")
    z_train = prep_code.apply_prep(train_raw, prep, True)
    z_valid = prep_code.apply_prep(valid_raw, prep, True)
    del train_raw, valid_raw
    rows = frame.loc[valid_mask].reset_index(drop=True)
    y_train = target_all[train_mask].copy()
    y_valid = target_all[valid_mask].astype(np.float64)
    train_season_counts = {
        str(int(s)): int(np.sum(seasons[train_mask] == s))
        for s in np.unique(seasons[train_mask])
    }
    metadata = {
        "year": year,
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "train_seasons": train_season_counts,
        "raw_features": int(len(names)),
        "network_inputs_with_missing_mask": int(z_train.shape[1]),
        "feature_names": names,
        "feature_build_seconds": float(time.time() - started),
        "row_id_order_sha256": row_id_sha256(rows),
        "target_float64_sha256": array_sha256(y_valid),
    }
    del frame, target_all, seasons
    gc.collect()
    return {
        "z_train": z_train,
        "z_valid": z_valid,
        "y_train": y_train,
        "y_valid": y_valid,
        "rows": rows,
        "metadata": metadata,
    }


def predict_network(network, matrix: np.ndarray) -> np.ndarray:
    import torch

    network.eval()
    source = torch.from_numpy(matrix)
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(source), PREDICT_BATCH):
            output.append(
                torch.sigmoid(network(source[start : start + PREDICT_BATCH]).squeeze(-1))
                .cpu().numpy()
            )
    return np.concatenate(output).astype(np.float64)


def fit_predict(z_train: np.ndarray, y_train: np.ndarray, z_valid: np.ndarray, y_valid: np.ndarray):
    import torch
    import torch.nn as nn

    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    dl = load_module(DL_SOURCE, "elephant_dl_168")
    network = dl.build_mlp(torch, nn, z_train.shape[1], WIDTH, DEPTH, DROP)
    optimizer = torch.optim.AdamW(network.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    batches = math.ceil(len(z_train) / BATCH)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, LR, total_steps=batches * EPOCHS, pct_start=0.1
    )
    x_tensor = torch.from_numpy(z_train)
    y_tensor = torch.from_numpy(y_train.astype(np.float32))
    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(EPOCHS):
        network.train()
        permutation = torch.randperm(len(x_tensor))
        epoch_started = time.time()
        loss_sum = 0.0
        seen = 0
        for batch_index, start in enumerate(range(0, len(x_tensor), BATCH)):
            index = permutation[start : start + BATCH]
            xb = x_tensor[index]
            yb = y_tensor[index]
            probability = torch.sigmoid(network(xb).squeeze(-1))
            loss = ((probability - yb) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            count = int(len(index))
            loss_sum += float(loss.detach()) * count
            seen += count
            if epoch == 0 and batch_index == 49:
                elapsed50 = time.time() - epoch_started
                estimate = elapsed50 * batches * EPOCHS / 50.0
                print(
                    f"[speed] 50_batches={elapsed50:.1f}s "
                    f"epoch_estimate={elapsed50 * batches / 50.0:.1f}s "
                    f"total_estimate={estimate / 60.0:.1f}m",
                    flush=True,
                )
        elapsed = time.time() - epoch_started
        record = {"epoch": epoch + 1, "train_brier": loss_sum / seen, "seconds": elapsed}
        history.append(record)
        print(f"[epoch {epoch + 1}/{EPOCHS}] {record}", flush=True)
    prediction = predict_network(network, z_valid)
    # Dynamic query-batch independence: the same trained state, 64 rows at once
    # versus one row at a time.  No validation label is involved in this check.
    probe = min(64, len(z_valid))
    batched = predict_network(network, z_valid[:probe])
    singleton = np.concatenate(
        [predict_network(network, z_valid[i : i + 1]) for i in range(probe)]
    )
    row_independence_max_abs = float(np.max(np.abs(batched - singleton), initial=0.0))
    del network, x_tensor, y_tensor
    gc.collect()
    return prediction, history, float(time.time() - started), row_independence_max_abs


def score(probability: np.ndarray, target: np.ndarray) -> float:
    rate = float(target.mean())
    return float(
        SCORE_SCALE
        * (1.0 - float(np.mean(np.square(probability - target))) / (rate * (1.0 - rate)))
    )


def blend_geometry(current: np.ndarray, endpoint: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    direction = endpoint - current
    curvature_sum = float(direction @ direction)
    if curvature_sum <= 0.0:
        weight_raw = 0.0
    else:
        weight_raw = float(((target - current) @ direction) / curvature_sum)
    weight = float(np.clip(weight_raw, 0.0, 1.0))
    candidate = current + weight * direction
    rate = float(target.mean())
    endpoint_delta = float(score(endpoint, target) - score(current, target))
    k = float(SCORE_SCALE * np.mean(np.square(direction)) / (rate * (1.0 - rate)))
    formula_weight = float((endpoint_delta + k) / (2.0 * k)) if k > 0.0 else 0.0
    return {
        "definition": "gain(w)=w*(d+K)-K*w^2, p=current+w*(endpoint-current)",
        "d_endpoint_score_minus_current": endpoint_delta,
        "K_direction_curvature": k,
        "optimal_weight_unconstrained": weight_raw,
        "optimal_weight_from_d_K": formula_weight,
        "locked_convex_weight": weight,
        "current_score": score(current, target),
        "endpoint_score": score(endpoint, target),
        "optimal_blend_score": score(candidate, target),
        "optimal_blend_gain": float(score(candidate, target) - score(current, target)),
        "endpoint_correlation_with_current": float(np.corrcoef(endpoint, current)[0, 1]),
        "direction_mean_square": float(np.mean(np.square(direction))),
    }


def load_current(rows: pd.DataFrame, year: int) -> tuple[Any, Any, np.ndarray]:
    e155 = load_module(ROOT / "exp" / "155_cause_runner_currentblend_gate.py", f"e155_168_{year}")
    e158 = load_module(ROOT / "exp" / "158_robust_conditional_tensor_eb.py", f"e158_168_{year}")
    current = e158.load_current_baseline(e155, rows, year).astype(np.float64)
    if current.shape != (len(rows),) or not np.isfinite(current).all():
        raise ValueError(f"invalid current clean OOF for {year}")
    return e155, e158, current


def subset_gain(
    current: np.ndarray, candidate: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    return {
        "rows": int(mask.sum()),
        "current_score": score(current[mask], target[mask]),
        "candidate_score": score(candidate[mask], target[mask]),
        "gain": float(score(candidate[mask], target[mask]) - score(current[mask], target[mask])),
    }


def discovery() -> None:
    if LOCK_PATH.exists():
        raise FileExistsError(f"discovery lock exists: {LOCK_PATH}")
    audit = source_audit()
    print("[discovery] build official <=2022 -> 2023 fold", flush=True)
    fold = build_fold(2023)
    print(f"[features] {fold['metadata']}", flush=True)
    endpoint, history, fit_seconds, independence = fit_predict(
        fold["z_train"], fold["y_train"], fold["z_valid"], fold["y_valid"]
    )
    if independence > 1e-5:
        raise AssertionError(f"row-independence failure: {independence}")
    _, _, current = load_current(fold["rows"], 2023)
    geometry = blend_geometry(current, endpoint, fold["y_valid"])
    candidate = current + geometry["locked_convex_weight"] * (endpoint - current)
    endpoint_artifact = save_vector("endpoint", 2023, endpoint)
    current_artifact = save_vector("current", 2023, current)
    candidate_artifact = save_vector("locked_blend", 2023, candidate)
    lock = {
        "experiment": 168,
        "phase": "DISCOVERY_LOCKED_ON_2023_BEFORE_2024_EVALUATION",
        "status": (
            "ELIGIBLE_FOR_2024_CONFIRMATION"
            if geometry["optimal_blend_gain"] > 0.0
            else "FAIL_STOP_BEFORE_2024"
        ),
        "audit": audit,
        "protocol": "official <=2022 fit -> 2023 eval; analytic convex weight",
        "model": {
            "kind": "partial Elephant robust-z MLP",
            "seed": SEED,
            "epochs": EPOCHS,
            "batch": BATCH,
            "width": WIDTH,
            "depth": DEPTH,
            "dropout": DROP,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "threads": THREADS,
        },
        "fold": fold["metadata"],
        "fit_seconds": fit_seconds,
        "epoch_history": history,
        "row_independence": {
            "probe_rows": min(64, len(endpoint)),
            "batch_vs_singleton_max_abs": independence,
            "pass_tolerance": 1e-5,
            "passed": True,
        },
        "geometry_2023": geometry,
        "endpoint_2023": endpoint_artifact,
        "current_2023": current_artifact,
        "locked_blend_2023": candidate_artifact,
        "row_id_order_sha256": row_id_sha256(fold["rows"]),
        "target_float64_sha256": array_sha256(fold["y_valid"]),
        "reads_test": False,
        "reads_reference_artifact": False,
        "confirmation_2024_opened": False,
        "zip_created": False,
    }
    LOCK_PATH.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[LOCK] {json.dumps(geometry, ensure_ascii=False)}", flush=True)


def pitcher_bootstrap(
    current: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    pitcher: np.ndarray,
) -> dict[str, Any]:
    work = pd.DataFrame({
        "pitcher": pitcher,
        "n": np.ones(len(target), dtype=np.int32),
        "y": target,
        "current_se": np.square(current - target),
        "candidate_se": np.square(candidate - target),
    })
    grouped = work.groupby("pitcher", sort=False).agg(
        n=("n", "sum"), y=("y", "sum"),
        current_se=("current_se", "sum"), candidate_se=("candidate_se", "sum")
    ).to_numpy(np.float64)
    rng = np.random.default_rng(168)
    gains = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for start in range(0, BOOTSTRAP_DRAWS, 200):
        width = min(200, BOOTSTRAP_DRAWS - start)
        sampled = grouped[rng.integers(0, len(grouped), size=(width, len(grouped)))].sum(axis=1)
        rate = sampled[:, 1] / sampled[:, 0]
        gains[start : start + width] = (
            SCORE_SCALE * (sampled[:, 2] - sampled[:, 3])
            / (sampled[:, 0] * rate * (1.0 - rate))
        )
    return {
        "draws": BOOTSTRAP_DRAWS,
        "clusters": int(len(grouped)),
        "p025": float(np.quantile(gains, 0.025)),
        "median": float(np.median(gains)),
        "p975": float(np.quantile(gains, 0.975)),
        "prob_positive": float(np.mean(gains > 0.0)),
    }


def confirm() -> None:
    if not LOCK_PATH.is_file():
        raise FileNotFoundError("run discovery first")
    if OUT_JSON.exists():
        raise FileExistsError(f"confirmation exists: {OUT_JSON}")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock["status"] != "ELIGIBLE_FOR_2024_CONFIRMATION":
        raise RuntimeError("2023 discovery failed; 2024 confirmation is forbidden")
    locked_weight = float(lock["geometry_2023"]["locked_convex_weight"])
    print(f"[confirm] locked weight={locked_weight:.12g}; build official <=2023 -> 2024", flush=True)
    fold = build_fold(2024)
    endpoint, history, fit_seconds, independence = fit_predict(
        fold["z_train"], fold["y_train"], fold["z_valid"], fold["y_valid"]
    )
    if independence > 1e-5:
        raise AssertionError(f"row-independence failure: {independence}")
    _, _, current = load_current(fold["rows"], 2024)
    target = fold["y_valid"]
    candidate = current + locked_weight * (endpoint - current)
    geometry = blend_geometry(current, endpoint, target)
    locked_gain = float(score(candidate, target) - score(current, target))
    month = integer_array(fold["rows"], "game_month")
    bootstrap = pitcher_bootstrap(
        current, candidate, target, integer_array(fold["rows"], "pitcher_id")
    )
    endpoint_artifact = save_vector("endpoint", 2024, endpoint)
    current_artifact = save_vector("current", 2024, current)
    candidate_artifact = save_vector("locked_blend", 2024, candidate)
    report = {
        "experiment": 168,
        "status": "ANALYSIS_ONLY_NO_TEST_NO_ZIP",
        "exact_1082_106_reproduction": False,
        "lock_path": str(LOCK_PATH),
        "lock_sha256": file_sha256(LOCK_PATH),
        "protocol": "2023 analytic blend weight transferred unchanged to untouched 2024",
        "locked_weight_from_2023": locked_weight,
        "fold": fold["metadata"],
        "fit_seconds": fit_seconds,
        "epoch_history": history,
        "row_independence_batch_vs_singleton_max_abs": independence,
        "geometry_2024_oracle_diagnostic_only": geometry,
        "locked_2024": {
            "current_score": score(current, target),
            "endpoint_score": score(endpoint, target),
            "candidate_score": score(candidate, target),
            "gain": locked_gain,
            "early": subset_gain(current, candidate, target, month <= 6),
            "late": subset_gain(current, candidate, target, month > 6),
            "pitcher_cluster_bootstrap": bootstrap,
        },
        "artifacts": {
            "endpoint_2024": endpoint_artifact,
            "current_2024": current_artifact,
            "locked_blend_2024": candidate_artifact,
        },
        "row_id_order_sha256": row_id_sha256(fold["rows"]),
        "target_float64_sha256": array_sha256(target),
        "post_2024_tuning": False,
        "reads_test": False,
        "reads_reference_artifact": False,
        "zip_created": False,
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "=== exp168 Elephant licensed-code partial std-z MLP ===",
        "exact 1082.106 reproduction: IMPOSSIBLE FROM REPOSITORY COMMIT",
        f"2023 d={lock['geometry_2023']['d_endpoint_score_minus_current']:+.4f} "
        f"K={lock['geometry_2023']['K_direction_curvature']:.4f} "
        f"w={locked_weight:.8f} gain={lock['geometry_2023']['optimal_blend_gain']:+.4f}",
        f"2024 current={score(current, target):.4f} endpoint={score(endpoint, target):.4f} "
        f"locked_gain={locked_gain:+.4f}",
        f"2024 early={report['locked_2024']['early']['gain']:+.4f} "
        f"late={report['locked_2024']['late']['gain']:+.4f} "
        f"pitcher_p025={bootstrap['p025']:+.4f}",
        f"row_independence_max_abs={independence:.3e}",
        "FINAL: ANALYSIS ONLY; NO TEST / NO ZIP",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.phase == "discovery":
        discovery()
    else:
        confirm()
    print(f"elapsed_seconds={time.time() - started:.2f}", flush=True)


if __name__ == "__main__":
    main()
