# -*- coding: utf-8 -*-
"""Exhaustive forward meta-screen of reusable local OOF directions.

Boundary
--------
Only official ``data/train.csv``, top-level ``lab/*.npy`` files, and the
existing exp158 -> exp155 frozen-current loader are used.  No nested artifact,
archive, reference repository, model file, test row, ZIP, or leaderboard value
is read.

Protocol
--------
* Fit every scalar/ridge correction on 2022 only.
* Use 2023 only to choose a predeclared shrink/ridge configuration and persist
  the complete lock.
* Load 2024 arrays and targets only after that lock is written, then transfer
  every locked family once without retuning or clipping.
* Quantify novelty against the known current/champion affine-plane direction.

The complete top-level NPY/log inventory is included so that candidates lacking
a 2022/2023/2024 row-aligned family are explicitly excluded rather than silently
ignored.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "lab"
TRAIN = ROOT / "data" / "train.csv"
LOCK_PATH = LAB / "166_existing_oof_third_endpoint_lock.json"
OUT_JSON = LAB / "166_existing_oof_third_endpoint_meta.json"
OUT_TXT = LAB / "166_existing_oof_third_endpoint_meta.txt"
BUILD_SPEC = LAB / "166_existing_oof_third_endpoint_build_spec.json"

YEAR_PATHS = {
    2022: {
        "cat42": LAB / "104_cat5_y2022_probs_seed42.npy",
        "cat7": LAB / "104_cat5_y2022_probs_seed7.npy",
        "v18": LAB / "104_v18_effect_y2022.npy",
    },
    2023: {
        "cat42": LAB / "104_cat5_y2023_probs_seed42.npy",
        "cat7": LAB / "104_cat5_y2023_probs_seed7.npy",
        "v18": LAB / "104_v18_effect_y2023.npy",
    },
    2024: {
        "cat42": LAB / "89_cat5_probs_seed42.npy",
        "cat7": LAB / "89_cat5_probs_seed7.npy",
        "v18": LAB / "103_v18_effect_2024.npy",
    },
}

SHRINKS = (0.125, 0.25, 0.50, 0.75, 1.0)
RIDGES = (0.01, 0.10, 1.0, 10.0)
BOOTSTRAP_DRAWS = 5000
RANGE_TOLERANCE = 2e-7
NOVELTY_MINIMUM = 0.15


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_inventory() -> dict[str, Any]:
    eligible = {
        path.resolve()
        for paths in YEAR_PATHS.values()
        for path in paths.values()
    }
    rows: list[dict[str, Any]] = []
    for path in sorted(LAB.glob("*.npy")):
        try:
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            shape: list[int] | None = list(value.shape)
            dtype = str(value.dtype)
            error = None
        except (OSError, ValueError) as exc:
            shape = None
            dtype = None
            error = str(exc)
        if path.resolve() in eligible:
            disposition = "used_complete_2022_2023_2024_family"
        elif shape and shape[0] == 253507:
            disposition = "excluded_2024_only_or_no_preconfirmation_pair"
        elif shape and shape[0] == 245525:
            disposition = "excluded_2023_only_or_missing_2022"
        elif shape and shape[0] == 247472:
            disposition = "excluded_2022_only_or_missing_later_pair"
        else:
            disposition = "excluded_non_yearfold_shape_or_nonprediction"
        rows.append(
            {
                "name": path.name,
                "shape": shape,
                "dtype": dtype,
                "disposition": disposition,
                "header_error": error,
            }
        )
    return {
        "top_level_npy_count": len(rows),
        "used_count": sum(row["disposition"].startswith("used_") for row in rows),
        "files": rows,
    }


def log_inventory() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    forbidden_markers = (
        "reference/calico",
        "reference\\calico",
        "third_party_artifact",
        "external_model",
    )
    for path in sorted([*LAB.glob("*.txt"), *LAB.glob("*.json")]):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            rows.append({"name": path.name, "read_error": str(exc)})
            continue
        lowered = text.lower()
        rows.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "mentions_2022": "2022" in text,
                "mentions_2023": "2023" in text,
                "mentions_2024": "2024" in text,
                "mentions_pass": "pass" in lowered,
                "mentions_fail": "fail" in lowered,
                "forbidden_provenance_marker": any(marker in lowered for marker in forbidden_markers),
                "used_for_numeric_selection": False,
            }
        )
    return {
        "top_level_log_count": len(rows),
        "logs_mentioning_all_three_years": sum(
            row.get("mentions_2022", False)
            and row.get("mentions_2023", False)
            and row.get("mentions_2024", False)
            for row in rows
        ),
        "files": rows,
    }


def load_rows(year: int) -> pd.DataFrame:
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
        raise ValueError(f"missing official rows for {year}")
    return rows


def load_full_probability(path: Path, rows: int) -> np.ndarray:
    value = np.load(path, allow_pickle=False).astype(np.float64)
    if value.shape != (rows, 5) or not np.isfinite(value).all():
        raise ValueError(f"invalid five-class probability {path}: {value.shape}")
    if value.min() < -RANGE_TOLERANCE or value.max() > 1.0 + RANGE_TOLERANCE:
        raise ValueError(f"out-of-range five-class probability: {path}")
    if np.max(np.abs(value.sum(axis=1) - 1.0)) > 2e-5:
        raise ValueError(f"five-class rows do not sum to one: {path}")
    return value


def load_vector(path: Path, rows: int) -> np.ndarray:
    value = np.load(path, allow_pickle=False).astype(np.float64)
    if value.shape != (rows,) or not np.isfinite(value).all():
        raise ValueError(f"invalid vector {path}: {value.shape}")
    return value


def load_source(year: int, rows: pd.DataFrame, e155, e158) -> dict[str, np.ndarray]:
    n = len(rows)
    current = e158.load_current_baseline(e155, rows, year).astype(np.float64)
    full42 = load_full_probability(YEAR_PATHS[year]["cat42"], n)
    full7 = load_full_probability(YEAR_PATHS[year]["cat7"], n)
    fullavg = 0.5 * (full42 + full7)
    v18 = load_vector(YEAR_PATHS[year]["v18"], n)
    current_cat = e155.load_probability(e155.CAT5[year], n)
    plane = e155.champion_probability(current_cat, v18)
    game_type = rows["game_type"].astype(str).to_numpy()
    is_f = game_type == "F"
    is_r = game_type == "R"
    if not np.all(is_f | is_r):
        raise ValueError(f"unexpected game types in {year}")
    return {
        "current": current,
        "plane": plane,
        "full42": full42,
        "full7": full7,
        "fullavg": fullavg,
        "v18": v18,
        "is_f": is_f,
        "is_r": is_r,
    }


def affine_endpoint(e155, probability: np.ndarray, effect: np.ndarray | None) -> np.ndarray:
    corrected = probability if effect is None else np.clip(
        probability + e155.V18_GAMMA * effect, 0.0, 1.0
    )
    return np.clip(
        e155.AFFINE_CENTER
        + e155.AFFINE_SCALE * (corrected - e155.AFFINE_CENTER)
        + e155.AFFINE_SHIFT,
        0.0,
        1.0,
    )


def fit_shape_centers(source22: dict[str, np.ndarray]) -> dict[str, float]:
    probability = source22["fullavg"]
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=1)
    margin = probability[:, 0] - np.max(probability[:, 1:], axis=1)
    values = {
        "middle": probability[:, 1],
        "reverse": probability[:, 2],
        "overlap": probability[:, 3],
        "bigmiss": probability[:, 4],
        "middle_minus_reverse": probability[:, 1] - probability[:, 2],
        "bigmiss_minus_reverse": probability[:, 4] - probability[:, 2],
        "entropy": entropy,
        "success_margin": margin,
    }
    return {name: float(value.mean()) for name, value in values.items()}


def scalar_directions(
    source: dict[str, np.ndarray],
    centers: dict[str, float],
    e155,
) -> dict[str, np.ndarray]:
    current = source["current"]
    full42 = source["full42"]
    full7 = source["full7"]
    fullavg = source["fullavg"]
    v18 = source["v18"]
    is_f = source["is_f"].astype(np.float64)
    is_r = source["is_r"].astype(np.float64)
    champion7 = affine_endpoint(e155, full7[:, 0], v18)
    championavg = affine_endpoint(e155, fullavg[:, 0], v18)
    affine42 = affine_endpoint(e155, full42[:, 0], None)
    affine7 = affine_endpoint(e155, full7[:, 0], None)
    affineavg = affine_endpoint(e155, fullavg[:, 0], None)
    entropy = -np.sum(fullavg * np.log(np.maximum(fullavg, 1e-12)), axis=1)
    margin = fullavg[:, 0] - np.max(fullavg[:, 1:], axis=1)
    shapes = {
        "middle_shape": fullavg[:, 1] - centers["middle"],
        "reverse_shape": fullavg[:, 2] - centers["reverse"],
        "overlap_shape": fullavg[:, 3] - centers["overlap"],
        "bigmiss_shape": fullavg[:, 4] - centers["bigmiss"],
        "middle_minus_reverse_shape": (
            fullavg[:, 1] - fullavg[:, 2] - centers["middle_minus_reverse"]
        ),
        "bigmiss_minus_reverse_shape": (
            fullavg[:, 4] - fullavg[:, 2] - centers["bigmiss_minus_reverse"]
        ),
        "entropy_shape": entropy - centers["entropy"],
        "success_margin_shape": margin - centers["success_margin"],
    }
    directions = {
        "plane_champion_control": source["plane"] - current,
        "cat42_raw_endpoint": full42[:, 0] - current,
        "cat7_raw_endpoint": full7[:, 0] - current,
        "cat2seed_raw_endpoint": fullavg[:, 0] - current,
        "cat42_affine_no_v18": affine42 - current,
        "cat7_affine_no_v18": affine7 - current,
        "cat2seed_affine_no_v18": affineavg - current,
        "cat7_champion": champion7 - current,
        "cat2seed_champion": championavg - current,
        "v18_all": v18,
        "v18_F_only": v18 * is_f,
        "v18_R_only": v18 * is_r,
        "seed7_minus_seed42_all": full7[:, 0] - full42[:, 0],
        "seed7_minus_seed42_F": (full7[:, 0] - full42[:, 0]) * is_f,
        "seed7_minus_seed42_R": (full7[:, 0] - full42[:, 0]) * is_r,
        **shapes,
    }
    for name in (
        "cat42_raw_endpoint",
        "cat7_raw_endpoint",
        "cat2seed_raw_endpoint",
        "cat7_champion",
        "cat2seed_champion",
    ):
        directions[f"{name}_F_only"] = directions[name] * is_f
        directions[f"{name}_R_only"] = directions[name] * is_r
    return directions


def fit_feature_state(source22: dict[str, np.ndarray]) -> dict[str, list[float]]:
    avg = source22["fullavg"][:, 1:]
    spread = (source22["full7"] - source22["full42"])[:, :4]
    raw = np.column_stack([avg, spread, source22["v18"]])
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std[std < 1e-8] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist()}


def standardized_blocks(
    source: dict[str, np.ndarray], state: dict[str, list[float]]
) -> dict[str, np.ndarray]:
    raw = np.column_stack(
        [
            source["fullavg"][:, 1:],
            (source["full7"] - source["full42"])[:, :4],
            source["v18"],
        ]
    )
    z = (raw - np.asarray(state["mean"])) / np.asarray(state["std"])
    avg = z[:, :4]
    spread = z[:, 4:8]
    v18 = z[:, 8:9]
    is_f = source["is_f"].astype(np.float64)[:, None]
    is_r = source["is_r"].astype(np.float64)[:, None]
    return {
        "mc2seed_shape": avg,
        "mc_seed_spread": spread,
        "mc_shape_plus_spread": np.column_stack([avg, spread]),
        "mc_shape_plus_v18": np.column_stack([avg, v18]),
        "mc_shape_spread_v18": np.column_stack([avg, spread, v18]),
        "mc_shape_by_game": np.column_stack([avg * is_f, avg * is_r]),
        "mc_shape_v18_by_game": np.column_stack(
            [avg * is_f, avg * is_r, v18 * is_f, v18 * is_r]
        ),
    }


def ridge_fit(matrix: np.ndarray, residual: np.ndarray, ridge: float) -> np.ndarray:
    xtx = matrix.T @ matrix
    scale = float(np.trace(xtx) / matrix.shape[1])
    return np.linalg.solve(
        xtx + ridge * max(scale, 1e-12) * np.eye(matrix.shape[1]),
        matrix.T @ residual,
    )


def feasible_interval(current: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
    positive = direction > 0.0
    negative = direction < 0.0
    lower = -math.inf
    upper = math.inf
    if positive.any():
        lower = max(lower, float(np.max(-current[positive] / direction[positive])))
        upper = min(upper, float(np.min((1.0 - current[positive]) / direction[positive])))
    if negative.any():
        lower = max(lower, float(np.max((1.0 - current[negative]) / direction[negative])))
        upper = min(upper, float(np.min(-current[negative] / direction[negative])))
    return lower, upper


def combine(current: np.ndarray, direction: np.ndarray, weight: float) -> np.ndarray:
    candidate = current + weight * direction
    if candidate.min() < -RANGE_TOLERANCE or candidate.max() > 1.0 + RANGE_TOLERANCE:
        raise ValueError(
            f"candidate requires clipping: range=[{candidate.min()}, {candidate.max()}]"
        )
    return candidate


def analytic_weight(current: np.ndarray, direction: np.ndarray, target: np.ndarray) -> float:
    denominator = float(direction @ direction)
    return float(((target - current) @ direction) / denominator) if denominator else 0.0


def novelty(direction: np.ndarray, plane_direction: np.ndarray) -> dict[str, float]:
    dn = float(np.linalg.norm(direction))
    pn = float(np.linalg.norm(plane_direction))
    if dn == 0.0:
        return {"cosine_with_plane": 0.0, "orthogonal_fraction": 0.0, "direction_std": 0.0}
    cosine = float((direction @ plane_direction) / (dn * pn)) if pn else 0.0
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return {
        "cosine_with_plane": cosine,
        "orthogonal_fraction": float(math.sqrt(max(0.0, 1.0 - cosine * cosine))),
        "direction_std": float(direction.std()),
    }


def select_scale(
    name: str,
    direction22: np.ndarray,
    direction23: np.ndarray,
    source22: dict[str, np.ndarray],
    source23: dict[str, np.ndarray],
    rows22: pd.DataFrame,
    rows23: pd.DataFrame,
    e165,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    current22 = source22["current"]
    current23 = source23["current"]
    y22 = rows22["control_success"].to_numpy(np.float64)
    y23 = rows23["control_success"].to_numpy(np.float64)
    oracle22 = analytic_weight(current22, direction22, y22)
    low22, high22 = feasible_interval(current22, direction22)
    low23, high23 = feasible_interval(current23, direction23)
    lower = max(low22, low23)
    upper = min(high22, high23)
    base_weight = float(np.clip(oracle22, lower, upper))
    trials: list[dict[str, Any]] = []
    predictions: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for shrink in SHRINKS:
        weight = shrink * base_weight
        try:
            candidate22 = combine(current22, direction22, weight)
            candidate23 = combine(current23, direction23, weight)
        except ValueError:
            continue
        metric22 = e165.evaluate(current22, candidate22, y22, rows22)
        metric23 = e165.evaluate(current23, candidate23, y23, rows23)
        trials.append(
            {
                "shrink": shrink,
                "weight": weight,
                "gain_2022": metric22["raw_gain_vs_current"],
                "gain_2023": metric23["raw_gain_vs_current"],
                "robust_min_gain": min(
                    metric22["raw_gain_vs_current"], metric23["raw_gain_vs_current"]
                ),
            }
        )
        predictions[shrink] = (candidate22, candidate23)
    if not trials:
        raise ValueError(f"no clipping-free shrink for {name}")
    selected = max(
        trials,
        key=lambda row: (
            row["robust_min_gain"],
            0.5 * (row["gain_2022"] + row["gain_2023"]),
            -row["shrink"],
        ),
    )
    candidate22, candidate23 = predictions[float(selected["shrink"])]
    metric22 = e165.evaluate(current22, candidate22, y22, rows22)
    metric23 = e165.evaluate(current23, candidate23, y23, rows23)
    bootstrap23 = (
        e165.pitcher_cluster_bootstrap(
            current23,
            candidate23,
            y23,
            rows23["pitcher_id"],
            draws=bootstrap_draws,
            seed=166,
        )
        if bootstrap_draws
        else None
    )
    novelty23 = novelty(
        float(selected["weight"]) * direction23,
        source23["plane"] - current23,
    )
    discovery_gates = {
        "gain_2022_positive": bool(metric22["raw_gain_vs_current"] > 0.0),
        "gain_2023_positive": bool(metric23["raw_gain_vs_current"] > 0.0),
        "early_2023_positive": bool(metric23["early_month_le_6"]["gain"] > 0.0),
        "late_2023_positive": bool(metric23["late_month_gt_6"]["gain"] > 0.0),
        "F_2023_nonnegative": bool(metric23["F_contribution"] >= 0.0),
        "R_2023_nonnegative": bool(metric23["R_contribution"] >= 0.0),
        "bootstrap_2023_p025_positive": bool(
            bootstrap23 is None or bootstrap23["p025"] > 0.0
        ),
        "outside_plane_novelty": bool(novelty23["orthogonal_fraction"] >= NOVELTY_MINIMUM),
    }
    lock = {
        "name": name,
        "oracle_weight_fit_2022": oracle22,
        "joint_no_clip_interval_2022_2023": [lower, upper],
        "base_weight_after_geometry_only_cap": base_weight,
        "selected_shrink_on_2022_2023": selected["shrink"],
        "locked_weight": selected["weight"],
        "trials": trials,
        "metrics_2022": metric22,
        "metrics_2023": metric23,
        "bootstrap_2023": bootstrap23,
        "novelty_2023": novelty23,
        "discovery_gates": discovery_gates,
        "eligible_for_2024_confirmation": bool(all(discovery_gates.values())),
    }
    return lock, candidate22, candidate23


def main() -> None:
    started = time.time()
    e155 = load_module(ROOT / "exp" / "155_cause_runner_currentblend_gate.py", "e155_166")
    e158 = load_module(ROOT / "exp" / "158_robust_conditional_tensor_eb.py", "e158_166")
    e165 = load_module(ROOT / "exp" / "165_blend_exp162_current.py", "e165_166")

    inventory = array_inventory()
    logs = log_inventory()
    # A completed pre-2024 lock with zero eligible families is terminal.  It is
    # neither useful nor authorized to bootstrap every failed family on 2024.
    if LOCK_PATH.is_file():
        prior_lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        prior_candidates = prior_lock.get("candidates", [])
        if (
            prior_lock.get("phase")
            == "LOCKED_AFTER_2022_FIT_AND_2023_TRANSFER_BEFORE_2024_LOAD"
            and prior_candidates
            and not any(
                bool(row.get("eligible_for_2024_confirmation"))
                for row in prior_candidates
            )
        ):
            status = "FAIL_NO_THIRD_ENDPOINT_BUILD"
            ranked_2023 = sorted(
                [row for row in prior_candidates if "metrics_2023" in row],
                key=lambda row: row["metrics_2023"]["raw_gain_vs_current"],
                reverse=True,
            )
            payload = {
                "experiment": 166,
                "status": status,
                "analysis_only": True,
                "reads_test": False,
                "uses_lb": False,
                "external_artifact_read": False,
                "post_2024_tuning": False,
                "inventory": inventory,
                "log_inventory": logs,
                "lock_path": str(LOCK_PATH),
                "lock_sha256": sha256_file(LOCK_PATH),
                "families_locked": len(prior_candidates),
                "eligible_after_2022_fit_2023_transfer": 0,
                "confirmation_2024_skipped": True,
                "confirmation_skip_reason": (
                    "zero families passed the precommitted 2022->2023 discovery gate"
                ),
                "ranked_2023": [
                    {
                        "name": row["name"],
                        "gain_2022": row["metrics_2022"]["raw_gain_vs_current"],
                        "gain_2023": row["metrics_2023"]["raw_gain_vs_current"],
                        "early_2023": row["metrics_2023"]["early_month_le_6"]["gain"],
                        "late_2023": row["metrics_2023"]["late_month_gt_6"]["gain"],
                        "F_2023": row["metrics_2023"]["F_contribution"],
                        "R_2023": row["metrics_2023"]["R_contribution"],
                        "p025_2023": row["bootstrap_2023"]["p025"],
                        "orthogonal_fraction_2023": row["novelty_2023"]["orthogonal_fraction"],
                    }
                    for row in ranked_2023
                ],
                "stable_positive_candidates": [],
                "submission_strength_candidates": [],
                "deployment_builder_possible": False,
                "elapsed_seconds": float(time.time() - started),
            }
            OUT_JSON.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if BUILD_SPEC.exists():
                raise RuntimeError(f"stale PASS-only build spec exists: {BUILD_SPEC}")
            lines = [
                "=== exp166 existing-OOF third-endpoint discovery gate ===",
                "2022 FIT -> 2023 CONFIG LOCK; 2024 confirmation skipped because eligible=0",
                f"inventory: top-level npy={inventory['top_level_npy_count']} used_complete_family_files={inventory['used_count']} logs={logs['top_level_log_count']}",
                f"families={len(prior_candidates)} discovery_eligible=0",
            ]
            for row in ranked_2023[:10]:
                metric = row["metrics_2023"]
                bootstrap = row["bootstrap_2023"]
                lines.append(
                    f"{row['name']}: g22={float(row['metrics_2022']['raw_gain_vs_current']):+.4f} "
                    f"g23={float(metric['raw_gain_vs_current']):+.4f} "
                    f"F/R={float(metric['F_contribution']):+.4f}/{float(metric['R_contribution']):+.4f} "
                    f"early/late={float(metric['early_month_le_6']['gain']):+.4f}/{float(metric['late_month_gt_6']['gain']):+.4f} "
                    f"p025={float(bootstrap['p025']):+.4f}"
                )
            lines.extend(
                [
                    "2024-only and incomplete-year arrays excluded from the forward gate.",
                    "stable positive: []",
                    "submission strength: []",
                    "builder possible/prepared: False",
                    f"FINAL: {status}",
                    f"elapsed={payload['elapsed_seconds']:.2f}s",
                ]
            )
            OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print("\n".join(lines), flush=True)
            return
    rows22 = load_rows(2022)
    rows23 = load_rows(2023)
    source22 = load_source(2022, rows22, e155, e158)
    source23 = load_source(2023, rows23, e155, e158)
    centers = fit_shape_centers(source22)
    feature_state = fit_feature_state(source22)
    directions22 = scalar_directions(source22, centers, e155)
    directions23 = scalar_directions(source23, centers, e155)

    runtime: dict[str, dict[str, Any]] = {}
    locks: list[dict[str, Any]] = []
    for name in directions22:
        try:
            lock, _, _ = select_scale(
                name,
                directions22[name],
                directions23[name],
                source22,
                source23,
                rows22,
                rows23,
                e165,
            )
        except ValueError as exc:
            locks.append({"name": name, "lock_error": str(exc), "eligible_for_2024_confirmation": False})
            continue
        lock["kind"] = "scalar_existing_oof_direction"
        locks.append(lock)
        runtime[name] = {"kind": "scalar", "lock": lock}

    blocks22 = standardized_blocks(source22, feature_state)
    blocks23 = standardized_blocks(source23, feature_state)
    residual22 = rows22["control_success"].to_numpy(np.float64) - source22["current"]
    for family in blocks22:
        configurations: list[
            tuple[dict[str, Any], float, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        for ridge in RIDGES:
            coefficient = ridge_fit(blocks22[family], residual22, ridge)
            effect22 = blocks22[family] @ coefficient
            effect23 = blocks23[family] @ coefficient
            try:
                lock, _, _ = select_scale(
                    f"{family}|ridge={ridge}",
                    effect22,
                    effect23,
                    source22,
                    source23,
                    rows22,
                    rows23,
                    e165,
                    bootstrap_draws=0,
                )
            except ValueError:
                continue
            configurations.append((lock, ridge, coefficient, effect22, effect23))
        if not configurations:
            locks.append({"name": family, "lock_error": "no valid ridge configuration", "eligible_for_2024_confirmation": False})
            continue
        preliminary_lock, chosen_ridge, coefficient, effect22, effect23 = max(
            configurations,
            key=lambda item: (
                min(
                    item[0]["metrics_2022"]["raw_gain_vs_current"],
                    item[0]["metrics_2023"]["raw_gain_vs_current"],
                ),
                item[0]["metrics_2023"]["raw_gain_vs_current"],
            ),
        )
        del preliminary_lock
        selected_lock, _, _ = select_scale(
            family,
            effect22,
            effect23,
            source22,
            source23,
            rows22,
            rows23,
            e165,
        )
        family_name = family
        selected_lock["name"] = family_name
        selected_lock["kind"] = "ridge_fit_2022_existing_oof"
        selected_lock["selected_ridge"] = chosen_ridge
        selected_lock["coefficient_fit_2022"] = coefficient.tolist()
        # Deep snapshots avoid embedding selected_lock back into itself.
        selected_lock["ridge_configurations"] = [
            json.loads(json.dumps(item[0])) for item in configurations
        ]
        locks.append(selected_lock)
        runtime[family_name] = {
            "kind": "ridge",
            "lock": selected_lock,
            "coefficient": coefficient,
            "feature_family": family,
        }

    source_hashes = {
        str(year): {
            kind: sha256_file(path) for kind, path in YEAR_PATHS[year].items()
        }
        for year in (2022, 2023)
    }
    lock_payload = {
        "experiment": 166,
        "phase": "LOCKED_AFTER_2022_FIT_AND_2023_TRANSFER_BEFORE_2024_LOAD",
        "data_boundary": "official train + top-level lab NPY + exp158/155 current loader only",
        "protocol": {
            "fit": "analytic scalar or ridge coefficient on 2022 only",
            "selection": "predeclared shrink/ridge by max min(2022 gain, 2023 gain)",
            "confirmation": "same family/config/weight transferred once to 2024",
            "no_clipping": True,
            "shrink_grid": list(SHRINKS),
            "ridge_grid": list(RIDGES),
            "novelty_minimum": NOVELTY_MINIMUM,
        },
        "shape_centers_fit_2022": centers,
        "feature_standardization_fit_2022": feature_state,
        "source_hashes_preconfirmation": source_hashes,
        "candidates": locks,
    }
    LOCK_PATH.write_text(
        json.dumps(lock_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[lock] {len(locks)} families; "
        f"eligible={sum(bool(row.get('eligible_for_2024_confirmation')) for row in locks)}",
        flush=True,
    )

    # Untouched confirmation is loaded only after the full family/config lock.
    rows24 = load_rows(2024)
    source24 = load_source(2024, rows24, e155, e158)
    directions24 = scalar_directions(source24, centers, e155)
    blocks24 = standardized_blocks(source24, feature_state)
    y24 = rows24["control_success"].to_numpy(np.float64)
    confirmations: list[dict[str, Any]] = []
    for locked in locks:
        name = str(locked["name"])
        if name not in runtime:
            confirmations.append({"name": name, "confirmation_error": locked.get("lock_error", "missing runtime")})
            continue
        item = runtime[name]
        if item["kind"] == "scalar":
            direction24 = directions24[name]
        else:
            direction24 = blocks24[item["feature_family"]] @ item["coefficient"]
        weight = float(locked["locked_weight"])
        try:
            candidate24 = combine(source24["current"], direction24, weight)
        except ValueError as exc:
            confirmations.append(
                {
                    "name": name,
                    "eligible_from_2022_2023": locked["eligible_for_2024_confirmation"],
                    "confirmation_error": str(exc),
                    "pass_stable_positive": False,
                    "pass_submission_strength": False,
                }
            )
            continue
        metrics24 = e165.evaluate(source24["current"], candidate24, y24, rows24)
        bootstrap24 = e165.pitcher_cluster_bootstrap(
            source24["current"],
            candidate24,
            y24,
            rows24["pitcher_id"],
            draws=BOOTSTRAP_DRAWS,
            seed=166,
        )
        novelty24 = novelty(direction24 * weight, source24["plane"] - source24["current"])
        stable_gates = {
            "eligible_from_2022_2023": bool(locked["eligible_for_2024_confirmation"]),
            "raw_2024_positive": bool(metrics24["raw_gain_vs_current"] > 0.0),
            "early_2024_positive": bool(metrics24["early_month_le_6"]["gain"] > 0.0),
            "late_2024_positive": bool(metrics24["late_month_gt_6"]["gain"] > 0.0),
            "F_2024_nonnegative": bool(metrics24["F_contribution"] >= 0.0),
            "R_2024_nonnegative": bool(metrics24["R_contribution"] >= 0.0),
            "bootstrap_2024_p025_positive": bool(bootstrap24["p025"] > 0.0),
            "outside_plane_novelty_2024": bool(
                novelty24["orthogonal_fraction"] >= NOVELTY_MINIMUM
            ),
        }
        strong_gates = {
            **stable_gates,
            "raw_2024_ge_12": bool(metrics24["raw_gain_vs_current"] >= 12.0),
        }
        confirmations.append(
            {
                "name": name,
                "kind": locked.get("kind"),
                "locked_weight": weight,
                "eligible_from_2022_2023": locked["eligible_for_2024_confirmation"],
                "metrics_2023": locked["metrics_2023"],
                "bootstrap_2023": locked["bootstrap_2023"],
                "novelty_2023": locked["novelty_2023"],
                "metrics_2024": metrics24,
                "bootstrap_2024": bootstrap24,
                "novelty_2024": novelty24,
                "stable_gates": stable_gates,
                "pass_stable_positive": bool(all(stable_gates.values())),
                "strong_gates": strong_gates,
                "pass_submission_strength": bool(all(strong_gates.values())),
            }
        )

    stable = [row for row in confirmations if row.get("pass_stable_positive")]
    strong = [row for row in confirmations if row.get("pass_submission_strength")]
    ranked = sorted(
        [row for row in confirmations if "metrics_2024" in row],
        key=lambda row: row["metrics_2024"]["raw_gain_vs_current"],
        reverse=True,
    )
    status = "PASS_THIRD_ENDPOINT_BUILD_POSSIBLE" if strong else "FAIL_NO_THIRD_ENDPOINT_BUILD"
    payload = {
        "experiment": 166,
        "status": status,
        "analysis_only": True,
        "reads_test": False,
        "uses_lb": False,
        "external_artifact_read": False,
        "post_2024_tuning": False,
        "inventory": inventory,
        "log_inventory": logs,
        "lock_path": str(LOCK_PATH),
        "lock_sha256": sha256_file(LOCK_PATH),
        "source_hashes_2024": {
            kind: sha256_file(path) for kind, path in YEAR_PATHS[2024].items()
        },
        "confirmations": confirmations,
        "stable_positive_candidates": [row["name"] for row in stable],
        "submission_strength_candidates": [row["name"] for row in strong],
        "ranked_by_2024_gain": [
            {
                "name": row["name"],
                "gain_2023": row["metrics_2023"]["raw_gain_vs_current"],
                "gain_2024": row["metrics_2024"]["raw_gain_vs_current"],
                "p025_2024": row["bootstrap_2024"]["p025"],
                "orthogonal_fraction_2024": row["novelty_2024"]["orthogonal_fraction"],
                "pass_stable_positive": row["pass_stable_positive"],
                "pass_submission_strength": row["pass_submission_strength"],
            }
            for row in ranked
        ],
        "deployment_builder_possible": bool(strong),
        "elapsed_seconds": float(time.time() - started),
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if strong:
        BUILD_SPEC.write_text(
            json.dumps(
                {
                    "experiment": 166,
                    "status": "BUILD_REVIEW_AUTHORIZED_NOT_EXECUTED",
                    "candidates": [row["name"] for row in strong],
                    "validation_report": str(OUT_JSON),
                    "validation_report_sha256": sha256_file(OUT_JSON),
                    "requirements": "reproduce exact current + CAT5 full probabilities/V18 inputs; apply locked correction without clipping",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    elif BUILD_SPEC.exists():
        raise RuntimeError(f"stale PASS-only build spec exists: {BUILD_SPEC}")

    lines = [
        "=== exp166 existing-OOF third-endpoint meta-screen ===",
        "2022 FIT -> 2023 CONFIG LOCK -> 2024 ONE-SHOT; no clipping/LB/test/external artifact",
        f"inventory: top-level npy={inventory['top_level_npy_count']} used_complete_family_files={inventory['used_count']} logs={logs['top_level_log_count']}",
        f"families={len(locks)} discovery_eligible={sum(bool(row.get('eligible_for_2024_confirmation')) for row in locks)}",
    ]
    for row in ranked[:15]:
        m23 = row["metrics_2023"]
        m24 = row["metrics_2024"]
        b24 = row["bootstrap_2024"]
        lines.append(
            f"{row['name']}: g23={float(m23['raw_gain_vs_current']):+.4f} "
            f"g24={float(m24['raw_gain_vs_current']):+.4f} "
            f"F/R={float(m24['F_contribution']):+.4f}/{float(m24['R_contribution']):+.4f} "
            f"early/late={float(m24['early_month_le_6']['gain']):+.4f}/{float(m24['late_month_gt_6']['gain']):+.4f} "
            f"p025={float(b24['p025']):+.4f} "
            f"orth={float(row['novelty_2024']['orthogonal_fraction']):.3f} "
            f"stable={row['pass_stable_positive']} strong={row['pass_submission_strength']}"
        )
    lines.extend(
        [
            f"stable positive: {[row['name'] for row in stable]}",
            f"submission strength: {[row['name'] for row in strong]}",
            f"builder possible/prepared: {bool(strong)}",
            f"FINAL: {status}",
            f"elapsed={payload['elapsed_seconds']:.2f}s",
        ]
    )
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
