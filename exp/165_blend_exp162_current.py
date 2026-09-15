# -*- coding: utf-8 -*-
"""Forward-only blend gate for the clean-room exp162 endpoint.

The two blend endpoints are:

* the frozen current OOF probability reconstructed only through the existing
  exp158 -> exp155 loader; and
* the official-data-only exp162 robust LightGBM + Tensor prediction artifact.

The exp162 weight is selected once from 2023 Brier loss.  The lock is written
before any 2024 prediction, target, or confirmation metadata is loaded.  The
same weight is then transferred to 2024 without tuning or clipping.

This diagnostic never reads reference/calico, archives, third-party code,
third-party predictions, test data, a submission ZIP, or leaderboard values.
It does not create deployment code unless the untouched 2024 PASS gate is met.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train.csv"
EXP162_LAB = ROOT / "lab" / "162_robust_cleanroom"
EXP162_CONFIRMATION = EXP162_LAB / "confirmation_2024.json"
EXP162_ARRAYS = {
    2023: {
        "anchor": EXP162_LAB / "anchor_2023.npy",
        "tensor": EXP162_LAB / "tensor_candidate_2023.npy",
    },
    2024: {
        "anchor": EXP162_LAB / "anchor_2024.npy",
        "tensor": EXP162_LAB / "tensor_candidate_2024.npy",
    },
}

LOCK_PATH = ROOT / "lab" / "165_blend_exp162_current_lock.json"
OUT_JSON = ROOT / "lab" / "165_blend_exp162_current.json"
OUT_TXT = ROOT / "lab" / "165_blend_exp162_current.txt"
BUILD_SPEC = ROOT / "lab" / "165_blend_exp162_current_build_spec.json"

SCORE_SCALE = 100000.0
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 165
RANGE_TOLERANCE = 2e-7  # exp162 artifacts are stored as float32.
LOCK_ABS_TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=1800.0,
        help="maximum time to wait for the exp162 confirmation artifacts",
    )
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    return parser.parse_args()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import loader module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def row_id_hash(rows: pd.DataFrame) -> str:
    return hashlib.sha256(
        "\n".join(rows["row_id"].astype(str).tolist()).encode("utf-8")
    ).hexdigest()


def target_hash(target: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(target, dtype=np.float64)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def wait_for_exp162(timeout: float, poll: float) -> None:
    required = [
        EXP162_CONFIRMATION,
        *(EXP162_ARRAYS[year][kind] for year in (2023, 2024) for kind in ("anchor", "tensor")),
    ]
    started = time.monotonic()
    last_notice = -math.inf
    while True:
        missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
        if not missing:
            # The confirmation JSON is written after the four np.save calls.
            # Its nonempty existence is the completion barrier.  Do not parse
            # its 2024-bearing contents until after the 2023 lock is persisted.
            return
        elapsed = time.monotonic() - started
        if elapsed >= timeout:
            names = ", ".join(str(path.relative_to(ROOT)) for path in missing)
            raise TimeoutError(f"timed out waiting for exp162 after {elapsed:.1f}s: {names}")
        if elapsed - last_notice >= 30.0:
            names = ", ".join(path.name for path in missing)
            print(f"[wait] exp162 incomplete at {elapsed:.0f}s: {names}", flush=True)
            last_notice = elapsed
        time.sleep(max(0.25, min(poll, 30.0)))


def load_year_rows(year: int) -> pd.DataFrame:
    columns = [
        "season",
        "row_id",
        "control_success",
        "game_type",
        "game_month",
        "pitcher_id",
    ]
    frame = pd.read_csv(
        TRAIN,
        encoding="utf-8-sig",
        usecols=columns,
        low_memory=False,
    )
    rows = frame.loc[frame["season"].eq(year)].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"official train has no rows for {year}")
    target = rows["control_success"].to_numpy(np.float64)
    if not np.isfinite(target).all() or not np.isin(target, (0.0, 1.0)).all():
        raise ValueError(f"invalid target for {year}")
    return rows


def load_probability(path: Path, rows: int) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.ndim != 1 or value.shape != (rows,):
        raise ValueError(f"bad probability shape {path}: {value.shape}, expected {(rows,)}")
    probability = value.astype(np.float64)
    if not np.isfinite(probability).all():
        raise ValueError(f"nonfinite probability: {path}")
    if probability.min() < -RANGE_TOLERANCE or probability.max() > 1.0 + RANGE_TOLERANCE:
        raise ValueError(
            f"probability outside [0,1] in {path}: "
            f"min={probability.min()} max={probability.max()}"
        )
    return probability


def load_current(e158, e155, rows: pd.DataFrame, year: int) -> np.ndarray:
    probability = e158.load_current_baseline(e155, rows, year).astype(np.float64)
    if probability.shape != (len(rows),) or not np.isfinite(probability).all():
        raise ValueError(f"invalid current baseline for {year}: {probability.shape}")
    if probability.min() < 0.0 or probability.max() > 1.0:
        raise ValueError(f"current baseline outside [0,1] for {year}")
    return probability


def score(probability: np.ndarray, target: np.ndarray) -> float:
    rate = float(target.mean())
    denominator = rate * (1.0 - rate)
    if denominator <= 0.0:
        raise ValueError("score undefined for a constant target")
    return float(
        SCORE_SCALE
        * (1.0 - float(np.mean(np.square(probability - target))) / denominator)
    )


def feasible_weight_interval(current: np.ndarray, endpoint: np.ndarray) -> tuple[float, float]:
    """Return all endpoint weights w keeping current + w*(endpoint-current) in [0,1]."""
    direction = endpoint - current
    positive = direction > 0.0
    negative = direction < 0.0
    lower = -math.inf
    upper = math.inf
    if positive.any():
        lower = max(lower, float(np.max(-current[positive] / direction[positive])))
        upper = min(
            upper,
            float(np.min((1.0 - current[positive]) / direction[positive])),
        )
    if negative.any():
        lower = max(
            lower,
            float(np.max((1.0 - current[negative]) / direction[negative])),
        )
        upper = min(upper, float(np.min(-current[negative] / direction[negative])))
    if lower > upper + LOCK_ABS_TOLERANCE:
        raise AssertionError(f"empty no-clipping weight interval: [{lower}, {upper}]")
    if lower > 0.0 + LOCK_ABS_TOLERANCE or upper < 1.0 - LOCK_ABS_TOLERANCE:
        raise AssertionError(f"convex weights unexpectedly infeasible: [{lower}, {upper}]")
    return lower, upper


def combine_no_clip(current: np.ndarray, endpoint: np.ndarray, weight: float) -> np.ndarray:
    candidate = current + float(weight) * (endpoint - current)
    minimum = float(candidate.min())
    maximum = float(candidate.max())
    if minimum < -RANGE_TOLERANCE or maximum > 1.0 + RANGE_TOLERANCE:
        raise ValueError(
            f"locked blend needs clipping: weight={weight:.17g}, "
            f"range=[{minimum:.17g}, {maximum:.17g}]"
        )
    # No np.clip: the exact linear prediction is the object being validated.
    return candidate


def discover_weight(
    current: np.ndarray,
    endpoint: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    direction = endpoint - current
    denominator = float(direction @ direction)
    if denominator <= 0.0:
        unconstrained = 0.0
    else:
        unconstrained = float(((target - current) @ direction) / denominator)
    convex = float(np.clip(unconstrained, 0.0, 1.0))
    lower, upper = feasible_weight_interval(current, endpoint)
    feasible = float(np.clip(unconstrained, lower, upper))
    signed_needed = bool(
        unconstrained < -LOCK_ABS_TOLERANCE
        or unconstrained > 1.0 + LOCK_ABS_TOLERANCE
    )
    selected_kind = "signed_no_clip" if signed_needed else "convex"
    selected = feasible if signed_needed else convex
    selected_prediction = combine_no_clip(current, endpoint, selected)
    convex_prediction = combine_no_clip(current, endpoint, convex)
    return {
        "parameterization": "p = current + w_exp162 * (exp162_tensor - current)",
        "brier_direction_sum_squares": denominator,
        "brier_direction_mean_squares": float(np.mean(np.square(direction))),
        "unconstrained_quadratic_optimum": unconstrained,
        "convex_weight": convex,
        "no_clip_feasible_interval_2023": [lower, upper],
        "signed_needed": signed_needed,
        "signed_no_clip_weight": feasible if signed_needed else None,
        "selected_kind": selected_kind,
        "selected_weight_exp162": selected,
        "selected_weight_current": 1.0 - selected,
        "current_brier_2023": float(np.mean(np.square(current - target))),
        "convex_brier_2023": float(np.mean(np.square(convex_prediction - target))),
        "selected_brier_2023": float(np.mean(np.square(selected_prediction - target))),
    }


def validate_or_write_lock(candidate: dict[str, Any]) -> dict[str, Any]:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOCK_PATH.exists():
        LOCK_PATH.write_text(
            json.dumps(candidate, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return candidate
    locked = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    exact_keys = (
        "experiment",
        "phase",
        "endpoint_2023_sha256",
        "current_2023_float64_sha256",
        "row_id_order_sha256",
        "target_float64_sha256",
        "selected_kind",
    )
    for key in exact_keys:
        if locked.get(key) != candidate.get(key):
            raise RuntimeError(
                f"existing 2023 lock mismatch for {key}: "
                f"locked={locked.get(key)!r}, reconstructed={candidate.get(key)!r}"
            )
    numeric_keys = (
        "unconstrained_quadratic_optimum",
        "convex_weight",
        "selected_weight_exp162",
        "selected_weight_current",
    )
    for key in numeric_keys:
        if not math.isclose(
            float(locked[key]),
            float(candidate[key]),
            rel_tol=0.0,
            abs_tol=LOCK_ABS_TOLERANCE,
        ):
            raise RuntimeError(
                f"existing 2023 lock mismatch for {key}: "
                f"locked={locked[key]!r}, reconstructed={candidate[key]!r}"
            )
    return locked


def subset_score(
    current: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    count = int(mask.sum())
    if count < 2:
        return {"rows": count, "gain": float("nan")}
    base_score = score(current[mask], target[mask])
    candidate_score = score(candidate[mask], target[mask])
    return {
        "rows": count,
        "target_rate": float(target[mask].mean()),
        "current_score": base_score,
        "candidate_score": candidate_score,
        "gain": float(candidate_score - base_score),
    }


def evaluate(
    current: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    rows: pd.DataFrame,
) -> dict[str, Any]:
    game_type = rows["game_type"].astype(str).to_numpy()
    is_f = game_type == "F"
    is_r = game_type == "R"
    if not np.all(is_f | is_r):
        unknown = sorted(pd.unique(game_type[~(is_f | is_r)]).tolist())
        raise ValueError(f"unknown game_type values: {unknown}")
    rate = float(target.mean())
    row_gain = (
        SCORE_SCALE
        * (np.square(current - target) - np.square(candidate - target))
        / (len(target) * rate * (1.0 - rate))
    )
    raw_gain = float(score(candidate, target) - score(current, target))
    f_contribution = float(row_gain[is_f].sum())
    r_contribution = float(row_gain[is_r].sum())
    if abs(raw_gain - f_contribution - r_contribution) > 2e-7:
        raise AssertionError("F/R contributions do not sum to the total gain")
    month = pd.to_numeric(rows["game_month"], errors="coerce").to_numpy(np.float64)
    return {
        "rows": int(len(target)),
        "current_score": score(current, target),
        "candidate_score": score(candidate, target),
        "raw_gain_vs_current": raw_gain,
        "brier": float(np.mean(np.square(candidate - target))),
        "prediction_mean": float(candidate.mean()),
        "prediction_min": float(candidate.min()),
        "prediction_max": float(candidate.max()),
        "mean_shift_vs_current": float(candidate.mean() - current.mean()),
        "F_contribution": f_contribution,
        "R_contribution": r_contribution,
        "F_subset": subset_score(current, candidate, target, is_f),
        "R_subset": subset_score(current, candidate, target, is_r),
        "early_month_le_6": subset_score(current, candidate, target, month <= 6),
        "late_month_gt_6": subset_score(current, candidate, target, month > 6),
        "no_clipping": bool(candidate.min() >= -RANGE_TOLERANCE and candidate.max() <= 1.0 + RANGE_TOLERANCE),
    }


def pitcher_cluster_bootstrap(
    current: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    pitcher: np.ndarray,
    draws: int,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    work = pd.DataFrame(
        {
            "pitcher": pd.to_numeric(pitcher, errors="coerce").fillna(-1).astype("int64"),
            "n": np.ones(len(target), dtype=np.int32),
            "y": target,
            "current_se": np.square(current - target),
            "candidate_se": np.square(candidate - target),
        }
    )
    grouped = work.groupby("pitcher", sort=False, observed=True).agg(
        n=("n", "sum"),
        y=("y", "sum"),
        current_se=("current_se", "sum"),
        candidate_se=("candidate_se", "sum"),
    )
    values = grouped[["n", "y", "current_se", "candidate_se"]].to_numpy(np.float64)
    if len(values) == 0:
        raise ValueError("no pitcher clusters")
    rng = np.random.default_rng(seed)
    gains = np.empty(draws, dtype=np.float64)
    cursor = 0
    while cursor < draws:
        width = min(200, draws - cursor)
        indices = rng.integers(0, len(values), size=(width, len(values)))
        sampled = values[indices].sum(axis=1)
        rate = sampled[:, 1] / sampled[:, 0]
        denominator = sampled[:, 0] * rate * (1.0 - rate)
        valid = denominator > 0.0
        chunk = np.full(width, np.nan, dtype=np.float64)
        chunk[valid] = (
            SCORE_SCALE
            * (sampled[valid, 2] - sampled[valid, 3])
            / denominator[valid]
        )
        gains[cursor : cursor + width] = chunk
        cursor += width
    finite = gains[np.isfinite(gains)]
    if len(finite) < max(100, int(0.95 * draws)):
        raise RuntimeError(f"too many invalid bootstrap draws: {len(finite)}/{draws}")
    return {
        "draws_requested": int(draws),
        "draws_valid": int(len(finite)),
        "pitchers": int(len(values)),
        "seed": int(seed),
        "median": float(np.median(finite)),
        "p025": float(np.quantile(finite, 0.025)),
        "p975": float(np.quantile(finite, 0.975)),
        "prob_positive": float(np.mean(finite > 0.0)),
    }


def verify_exp162_metadata(
    confirmation: dict[str, Any],
    rows_by_year: dict[int, pd.DataFrame],
) -> dict[str, Any]:
    artifacts = confirmation.get("official_yearfold_prediction_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("exp162 confirmation is missing prediction artifact metadata")
    verified: dict[str, Any] = {}
    for year in (2023, 2024):
        rows = rows_by_year[year]
        target = rows["control_success"].to_numpy(np.float64)
        for kind, artifact_key in (("anchor", f"anchor_{year}"), ("tensor", f"tensor_candidate_{year}")):
            expected_path = EXP162_ARRAYS[year][kind]
            metadata = artifacts.get(artifact_key)
            if not isinstance(metadata, dict):
                raise ValueError(f"missing exp162 metadata: {artifact_key}")
            claimed_path = Path(str(metadata.get("path", "")))
            if claimed_path.resolve() != expected_path.resolve():
                raise ValueError(
                    f"exp162 metadata path escaped fixed clean-room location for {artifact_key}: "
                    f"{claimed_path}"
                )
            checks = {
                "rows": int(metadata.get("rows", -1)) == len(rows),
                "sha256": metadata.get("sha256") == sha256_file(expected_path),
                "row_id_order_sha256": metadata.get("row_id_order_sha256") == row_id_hash(rows),
                "target_float64_sha256": metadata.get("target_float64_sha256") == target_hash(target),
            }
            if not all(checks.values()):
                raise ValueError(f"exp162 artifact integrity failure {artifact_key}: {checks}")
            verified[artifact_key] = {
                "path": str(expected_path),
                "sha256": sha256_file(expected_path),
                "rows": int(len(rows)),
                "row_id_order_sha256": row_id_hash(rows),
                "target_float64_sha256": target_hash(target),
                "checks": checks,
            }
    return verified


def candidate_set(
    current: np.ndarray,
    anchor: np.ndarray,
    tensor: np.ndarray,
    lock: dict[str, Any],
) -> dict[str, np.ndarray]:
    convex_weight = float(lock["convex_weight"])
    selected_weight = float(lock["selected_weight_exp162"])
    candidates = {
        "current_endpoint": current,
        "exp162_anchor_component": anchor,
        "exp162_tensor_endpoint": tensor,
        "convex_locked_2023": combine_no_clip(current, tensor, convex_weight),
    }
    if bool(lock["signed_needed"]):
        candidates["signed_no_clip_locked_2023"] = combine_no_clip(
            current, tensor, selected_weight
        )
    candidates["selected_locked_2023"] = combine_no_clip(
        current, tensor, selected_weight
    )
    return candidates


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.bootstrap_draws < 100:
        raise ValueError("--bootstrap-draws must be at least 100")
    wait_for_exp162(args.wait_seconds, args.poll_seconds)

    # The only frozen-current access is the existing exp158 -> exp155 loader.
    e158 = load_module(ROOT / "exp" / "158_robust_conditional_tensor_eb.py", "e158_for_165")
    e155 = load_module(ROOT / "exp" / "155_cause_runner_currentblend_gate.py", "e155_for_165")

    # Discovery: do not parse confirmation metadata or load either 2024 array.
    rows23 = load_year_rows(2023)
    y23 = rows23["control_success"].to_numpy(np.float64)
    current23 = load_current(e158, e155, rows23, 2023)
    anchor23 = load_probability(EXP162_ARRAYS[2023]["anchor"], len(rows23))
    tensor23 = load_probability(EXP162_ARRAYS[2023]["tensor"], len(rows23))
    discovered = discover_weight(current23, tensor23, y23)
    lock_candidate = {
        "experiment": 165,
        "phase": "2023_weight_locked_before_2024_open",
        "protocol": {
            "endpoints": "frozen current OOF versus official-only exp162 Tensor endpoint",
            "selection_year": 2023,
            "objective": "analytic Brier quadratic optimum",
            "signed_policy": "if optimum is outside [0,1], project only to the 2023 no-clipping feasible interval",
            "transfer": "same exact weight on 2024; no target-based tuning and no prediction clipping",
        },
        "endpoint_2023_path": str(EXP162_ARRAYS[2023]["tensor"]),
        "endpoint_2023_sha256": sha256_file(EXP162_ARRAYS[2023]["tensor"]),
        "anchor_component_2023_sha256": sha256_file(EXP162_ARRAYS[2023]["anchor"]),
        "current_2023_float64_sha256": sha256_array(current23),
        "row_id_order_sha256": row_id_hash(rows23),
        "target_float64_sha256": target_hash(y23),
        **discovered,
    }
    lock = validate_or_write_lock(lock_candidate)
    print(
        f"[lock] kind={lock['selected_kind']} "
        f"w_exp162={float(lock['selected_weight_exp162']):.12g}",
        flush=True,
    )

    # Confirmation: only after the persisted 2023 lock exists do we open 2024.
    confirmation = json.loads(EXP162_CONFIRMATION.read_text(encoding="utf-8"))
    rows24 = load_year_rows(2024)
    y24 = rows24["control_success"].to_numpy(np.float64)
    current24 = load_current(e158, e155, rows24, 2024)
    anchor24 = load_probability(EXP162_ARRAYS[2024]["anchor"], len(rows24))
    tensor24 = load_probability(EXP162_ARRAYS[2024]["tensor"], len(rows24))
    verified = verify_exp162_metadata(confirmation, {2023: rows23, 2024: rows24})

    candidates23 = candidate_set(current23, anchor23, tensor23, lock)
    candidates24: dict[str, np.ndarray] = {}
    invalid_transfer: dict[str, str] = {}
    for name, candidate23 in candidates23.items():
        del candidate23
        try:
            if name == "current_endpoint":
                value = current24
            elif name == "exp162_anchor_component":
                value = anchor24
            elif name == "exp162_tensor_endpoint":
                value = tensor24
            elif name == "convex_locked_2023":
                value = combine_no_clip(current24, tensor24, float(lock["convex_weight"]))
            else:
                value = combine_no_clip(
                    current24, tensor24, float(lock["selected_weight_exp162"])
                )
            candidates24[name] = value
        except ValueError as error:
            invalid_transfer[name] = str(error)

    metrics23 = {
        name: evaluate(current23, value, y23, rows23)
        for name, value in candidates23.items()
    }
    metrics24 = {
        name: evaluate(current24, value, y24, rows24)
        for name, value in candidates24.items()
    }
    bootstrap24 = {
        name: pitcher_cluster_bootstrap(
            current24,
            value,
            y24,
            rows24["pitcher_id"],
            draws=args.bootstrap_draws,
        )
        for name, value in candidates24.items()
        if name != "current_endpoint"
    }

    selected_name = "selected_locked_2023"
    selected_valid = selected_name in metrics24 and selected_name in bootstrap24
    selected_metrics = metrics24.get(selected_name, {})
    selected_bootstrap = bootstrap24.get(selected_name, {})
    gates = {
        "selected_transfer_no_clipping": selected_valid,
        "raw_gain_ge_12_vs_current": bool(
            selected_valid and float(selected_metrics["raw_gain_vs_current"]) >= 12.0
        ),
        "early_gain_positive": bool(
            selected_valid and float(selected_metrics["early_month_le_6"]["gain"]) > 0.0
        ),
        "late_gain_positive": bool(
            selected_valid and float(selected_metrics["late_month_gt_6"]["gain"]) > 0.0
        ),
        "pitcher_cluster_p025_positive": bool(
            selected_valid and float(selected_bootstrap["p025"]) > 0.0
        ),
    }
    passed = bool(all(gates.values()))
    status = "PASS_PREPARE_DEPLOY_BUILDER" if passed else "FAIL_NO_DEPLOY_BUILDER"

    payload = {
        "experiment": 165,
        "status": status,
        "analysis_only": True,
        "reads_test": False,
        "uses_lb": False,
        "post_2024_weight_tuning": False,
        "data_boundary": {
            "official_train": str(TRAIN),
            "current_endpoint": "existing exp158 -> exp155 frozen-current loader only",
            "new_endpoint": "exp162 official-data-only saved predictions only",
            "reference_or_calico_read": False,
            "external_code_model_prediction_or_zip_read": False,
        },
        "lock_path": str(LOCK_PATH),
        "lock": lock,
        "exp162_confirmation_path": str(EXP162_CONFIRMATION),
        "exp162_confirmation_sha256": sha256_file(EXP162_CONFIRMATION),
        "exp162_artifacts_verified": verified,
        "alignment": {
            "2023": {
                "rows": int(len(rows23)),
                "row_id_order_sha256": row_id_hash(rows23),
                "target_float64_sha256": target_hash(y23),
                "target_rate": float(y23.mean()),
            },
            "2024": {
                "rows": int(len(rows24)),
                "row_id_order_sha256": row_id_hash(rows24),
                "target_float64_sha256": target_hash(y24),
                "target_rate": float(y24.mean()),
            },
        },
        "metrics_2023_selection_year": metrics23,
        "metrics_2024_untouched_transfer": metrics24,
        "pitcher_cluster_bootstrap_2024": bootstrap24,
        "invalid_no_clip_transfers": invalid_transfer,
        "pass_gates": gates,
        "deployment_builder_prepared": passed,
        "elapsed_seconds": float(time.time() - started),
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if passed:
        build_spec = {
            "experiment": 165,
            "status": "AUTHORIZED_BY_UNTOUCHED_2024_GATE",
            "builder_not_yet_executed": True,
            "selected_kind": lock["selected_kind"],
            "weight_exp162": lock["selected_weight_exp162"],
            "weight_current": lock["selected_weight_current"],
            "blend_formula": lock["parameterization"],
            "no_clipping_required": True,
            "validation_report": str(OUT_JSON),
            "validation_report_sha256_after_write": sha256_file(OUT_JSON),
        }
        BUILD_SPEC.write_text(
            json.dumps(build_spec, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif BUILD_SPEC.exists():
        raise RuntimeError(
            f"stale PASS-only build spec exists despite failed gate: {BUILD_SPEC}"
        )

    selected23 = metrics23[selected_name]
    lines = [
        "=== exp165 exp162/current forward blend gate ===",
        "BOUNDARY: official train + exp158/155 current loader + exp162 generated arrays only",
        "2023 SELECT/LOCK; 2024 TRANSFER ONLY; no 2024 weight tuning; no clipping",
        (
            f"lock: kind={lock['selected_kind']} "
            f"w_exp162={float(lock['selected_weight_exp162']):.12g} "
            f"w_current={float(lock['selected_weight_current']):.12g} "
            f"unconstrained={float(lock['unconstrained_quadratic_optimum']):.12g} "
            f"convex={float(lock['convex_weight']):.12g}"
        ),
        (
            "2023 selected: "
            f"gain={float(selected23['raw_gain_vs_current']):+.6f} "
            f"F={float(selected23['F_contribution']):+.6f} "
            f"R={float(selected23['R_contribution']):+.6f}"
        ),
    ]
    for name, metric in metrics24.items():
        bootstrap = bootstrap24.get(name)
        ci = ""
        if bootstrap is not None:
            ci = (
                f" bootstrap=[{float(bootstrap['p025']):+.6f},"
                f"{float(bootstrap['p975']):+.6f}]"
            )
        lines.append(
            f"2024 {name}: gain={float(metric['raw_gain_vs_current']):+.6f} "
            f"F={float(metric['F_contribution']):+.6f} "
            f"R={float(metric['R_contribution']):+.6f} "
            f"early={float(metric['early_month_le_6']['gain']):+.6f} "
            f"late={float(metric['late_month_gt_6']['gain']):+.6f}{ci}"
        )
    if invalid_transfer:
        lines.append(f"invalid no-clip transfers: {invalid_transfer}")
    lines.extend(
        [
            f"PASS gates: {gates}",
            f"deployment builder prepared: {passed}",
            f"FINAL: {status}",
            f"elapsed={payload['elapsed_seconds']:.2f}s",
        ]
    )
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
