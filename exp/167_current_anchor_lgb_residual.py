# -*- coding: utf-8 -*-
"""Strict R-only LightGBM residual model on the current clean anchor.

Allowed inputs are official data/train.csv and our own frozen current-clean OOF
loaded through the exp158/e155 contract.  No reference directory, external
model/code/weight, test data, leaderboard value, or archive is read.

Protocol:
  discovery: fit 2022 R residuals; select leaves 15/31 and scale on 2023 only;
  confirm:   refit locked configuration on 2022+2023 R residuals with 0.65
             weight on 2022, then evaluate untouched 2024 once.

The correction is clipped to +/-0.08 before scaling.  F rows are bit-identical
to the current clean anchor.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train.csv"
LAB = ROOT / "lab"
LOCK_PATH = LAB / "167_current_anchor_lgb_residual_lock.json"
OUT_JSON = LAB / "167_current_anchor_lgb_residual.json"
OUT_TXT = LAB / "167_current_anchor_lgb_residual.txt"

THREADS = 6
ROUNDS = 500
LEAF_CANDIDATES = (15, 31)
YEAR_DECAY = 0.65
CORRECTION_CLIP = 0.08
SCALE_GRID = (0.0, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0)

CATEGORICAL = (
    "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "base_state",
    "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "_count_state", "_base_out",
    "_count_hand", "_team_matchup",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("discovery", "confirm"))
    return parser.parse_args()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    balls = pd.to_numeric(out["balls_before"], errors="coerce").fillna(-1).astype("int16")
    strikes = pd.to_numeric(out["strikes_before"], errors="coerce").fillna(-1).astype("int16")
    outs = pd.to_numeric(out["outs_before"], errors="coerce").fillna(-1).astype("int16")
    pitcher_hand = pd.to_numeric(out["pitcher_hand"], errors="coerce").fillna(-1).astype("int16")
    batter_hand = pd.to_numeric(out["batter_hand"], errors="coerce").fillna(-1).astype("int16")
    top_bottom = out["top_bottom"].astype("string").fillna("M").map({"T": 0, "B": 1}).fillna(2)
    game_type = out["game_type"].astype("string").fillna("M").map({"R": 0, "F": 1}).fillna(2)
    base_code = (
        out["base_state"].astype("string").fillna("M")
        .map({"___": 0, "1__": 1, "_2_": 2, "__3": 3, "12_": 4, "1_3": 5, "_23": 6, "123": 7})
        .fillna(8).astype("int16")
    )
    out["top_bottom"] = top_bottom.astype("int16")
    out["game_type"] = game_type.astype("int16")
    out["base_state"] = base_code
    out["pitcher_hand"] = pitcher_hand
    out["batter_hand"] = batter_hand
    out["_count_state"] = (balls * 3 + strikes).astype("int16")
    out["_base_out"] = (base_code * 3 + outs).astype("int16")
    out["_count_hand"] = (
        out["_count_state"] * 16 + (pitcher_hand + 1) * 4 + batter_hand + 1
    ).astype("int16")
    pitcher_team = pd.to_numeric(out["pitcher_team_id"], errors="coerce").fillna(-1).astype("int32")
    batter_team = pd.to_numeric(out["batter_team_id"], errors="coerce").fillna(-1).astype("int32")
    out["_team_matchup"] = ((pitcher_team + 1) * 128 + batter_team + 1).astype("int32")
    out["_same_hand"] = (pitcher_hand == batter_hand).astype("int8")
    out["_count_diff"] = (balls - strikes).astype("int8")
    out["_is_full_count"] = ((balls == 3) & (strikes == 2)).astype("int8")
    out["_any_runner"] = (
        pd.to_numeric(out["num_runners_on"], errors="coerce").fillna(0) > 0
    ).astype("int8")
    out["_scoring_position"] = (
        pd.to_numeric(out["runner_on_2b"], errors="coerce").fillna(0)
        + pd.to_numeric(out["runner_on_3b"], errors="coerce").fillna(0) > 0
    ).astype("int8")
    pitcher_n = pd.to_numeric(out["asof_pitcher_n"], errors="coerce").fillna(0).clip(lower=0)
    batter_n = pd.to_numeric(out["asof_batter_n"], errors="coerce").fillna(0).clip(lower=0)
    out["_log_pitcher_n"] = np.log1p(pitcher_n).astype("float32")
    out["_log_batter_n"] = np.log1p(batter_n).astype("float32")
    pitcher_rate = pd.to_numeric(out["asof_pitcher_success_rate"], errors="coerce")
    batter_rate = pd.to_numeric(out["asof_batter_success_rate"], errors="coerce")
    out["_pitcher_form1"] = (
        pd.to_numeric(out["asof_pitcher_prev1_game_success_rate"], errors="coerce") - pitcher_rate
    ).astype("float32")
    out["_pitcher_form3"] = (
        pd.to_numeric(out["asof_pitcher_prev3_game_success_rate"], errors="coerce") - pitcher_rate
    ).astype("float32")
    out["_pitcher_trend"] = (
        pd.to_numeric(out["asof_pitcher_prev1_game_success_rate"], errors="coerce")
        - pd.to_numeric(out["asof_pitcher_prev5_game_success_rate"], errors="coerce")
    ).astype("float32")
    out["_pitcher_batter_gap"] = (pitcher_rate - batter_rate).astype("float32")
    out["_win_expectancy_gap"] = (
        pd.to_numeric(out["home_win_expectancy"], errors="coerce")
        - pd.to_numeric(out["away_win_expectancy"], errors="coerce")
    ).astype("float32")
    out["_score_pressure"] = (
        pd.to_numeric(out["score_diff_pitcher_team"], errors="coerce").abs()
        * np.log1p(pd.to_numeric(out["li"], errors="coerce").clip(lower=0))
    ).astype("float32")
    return out


def build_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(index=frame.index)
    for column in features:
        value = pd.to_numeric(frame[column], errors="coerce")
        if column in CATEGORICAL:
            matrix[column] = value.fillna(-1).astype("int32")
        else:
            matrix[column] = value.astype("float32")
    return matrix


def model_params(num_leaves: int) -> dict[str, object]:
    return {
        "objective": "regression_l2",
        "metric": "l2",
        "learning_rate": 0.035,
        "num_leaves": int(num_leaves),
        "min_data_in_leaf": 900,
        "lambda_l2": 20.0,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        "max_bin": 127,
        "max_cat_to_onehot": 8,
        "cat_l2": 20.0,
        "cat_smooth": 50.0,
        "verbosity": -1,
        "seed": 167,
        "feature_fraction_seed": 167,
        "bagging_seed": 167,
        "num_threads": THREADS,
        "force_col_wise": True,
    }


def fit_predict(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    residual: np.ndarray,
    weights: np.ndarray,
    num_leaves: int,
) -> tuple[np.ndarray, float]:
    started = time.time()
    x_train = build_matrix(train, features)
    x_valid = build_matrix(valid, features)
    dataset = lgb.Dataset(
        x_train,
        label=np.asarray(residual, np.float32),
        weight=np.asarray(weights, np.float32),
        categorical_feature=list(CATEGORICAL),
        free_raw_data=True,
    )
    model = lgb.train(model_params(num_leaves), dataset, num_boost_round=ROUNDS)
    prediction = np.asarray(model.predict(x_valid, num_threads=THREADS), np.float64)
    prediction = np.clip(prediction, -CORRECTION_CLIP, CORRECTION_CLIP)
    del model, dataset, x_train, x_valid
    gc.collect()
    return prediction, float(time.time() - started)


def curve(e155, baseline, effect, target, game_type) -> list[dict[str, float]]:
    return [
        {"scale": scale, **e155.metrics(baseline, scale * effect, target, game_type)}
        for scale in SCALE_GRID
    ]


def save_vector(name: str, value: np.ndarray) -> dict[str, object]:
    path = LAB / f"167_{name}.npy"
    np.save(path, np.asarray(value, np.float32), allow_pickle=False)
    return {"path": str(path), "rows": int(len(value)), "sha256": file_sha256(path)}


def load_context():
    e158 = load_module(ROOT / "exp" / "158_robust_conditional_tensor_eb.py", "e158_167")
    e155 = load_module(ROOT / "exp" / "155_cause_runner_currentblend_gate.py", "e155_167")
    frame = pd.read_csv(TRAIN, encoding="utf-8-sig", low_memory=False)
    frame.columns = [column.replace("\ufeff", "").strip() for column in frame.columns]
    frame = add_features(frame)
    rows = {
        year: frame.loc[frame["season"].eq(year)].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    target = {year: rows[year]["control_success"].to_numpy(np.float64) for year in rows}
    baseline = {year: e158.load_current_baseline(e155, rows[year], year) for year in rows}
    raw_features = [
        column for column in frame.columns
        if column not in {"row_id", "season", "control_success"}
    ]
    # Current clean probability is an explicit row-local residual feature.
    for year in rows:
        rows[year]["_base_probability"] = baseline[year].astype("float32")
    features = raw_features + ["_base_probability"]
    if "season" in features or "row_id" in features or "control_success" in features:
        raise AssertionError("forbidden feature present")
    if sorted(set(CATEGORICAL) - set(features)):
        raise AssertionError("categorical feature missing")
    return e158, e155, rows, target, baseline, features


def discovery() -> dict[str, object]:
    if LOCK_PATH.exists():
        raise FileExistsError(f"lock exists: {LOCK_PATH}")
    _, e155, rows, target, baseline, features = load_context()
    is_r22 = rows[2022]["game_type"].to_numpy() == 0
    game_type23 = np.where(rows[2023]["game_type"].to_numpy() == 0, "R", "F")
    alternatives: list[dict[str, object]] = []
    effects: dict[int, np.ndarray] = {}
    for leaves in LEAF_CANDIDATES:
        print(f"[discovery] leaves={leaves}", flush=True)
        effect_r, elapsed = fit_predict(
            rows[2022].loc[is_r22].reset_index(drop=True), rows[2023], features,
            target[2022][is_r22] - baseline[2022][is_r22],
            np.ones(int(is_r22.sum()), np.float64), leaves,
        )
        effect = np.where(rows[2023]["game_type"].to_numpy() == 0, effect_r, 0.0)
        effects[leaves] = effect
        scale_curve = curve(e155, baseline[2023], effect, target[2023], game_type23)
        selected_scale = max(
            scale_curve,
            key=lambda row: (row["raw_gain"], row["equal_mean_shape_gain"], -row["scale"]),
        )
        alternatives.append({
            "num_leaves": leaves,
            "fit_seconds": elapsed,
            "scale_curve_2023": scale_curve,
            "selected_scale_2023": selected_scale,
        })
        print(f"  selected={selected_scale}", flush=True)
    selected = max(
        alternatives,
        key=lambda row: (
            row["selected_scale_2023"]["raw_gain"],
            row["selected_scale_2023"]["equal_mean_shape_gain"],
            -row["num_leaves"],
        ),
    )
    leaves = int(selected["num_leaves"])
    scale = float(selected["selected_scale_2023"]["scale"])
    effect = effects[leaves]
    candidate = np.clip(baseline[2023] + scale * effect, 0.0, 1.0)
    payload = {
        "experiment": 167,
        "phase": "discovery_locked_before_2024",
        "allowed_inputs": "official train + own current-clean OOF via exp158/e155",
        "reads_test_or_external_reference_artifact": False,
        "threads": THREADS,
        "rounds": ROUNDS,
        "leaf_candidates_fixed": list(LEAF_CANDIDATES),
        "scale_grid_fixed": list(SCALE_GRID),
        "year_decay_locked": YEAR_DECAY,
        "correction_clip": CORRECTION_CLIP,
        "selection_rule": "max 2023 raw gain, tie equal-mean gain, smaller leaves",
        "features": features,
        "categorical": list(CATEGORICAL),
        "alternatives": alternatives,
        "selected_num_leaves": leaves,
        "locked_scale": scale,
        "selected_effect_2023": save_vector("effect_2023", effect),
        "selected_candidate_2023": save_vector("candidate_2023", candidate),
    }
    LOCK_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"LOCKED leaves={leaves} scale={scale} gain={selected['selected_scale_2023']['raw_gain']:+.4f}", flush=True)
    return payload


def confirm() -> dict[str, object]:
    if not LOCK_PATH.is_file():
        raise FileNotFoundError("run discovery first")
    if OUT_JSON.exists():
        raise FileExistsError(f"confirmation already exists: {OUT_JSON}")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    e158, e155, rows, target, baseline, features = load_context()
    leaves = int(lock["selected_num_leaves"])
    scale = float(lock["locked_scale"])
    combined = pd.concat([rows[2022], rows[2023]], ignore_index=True)
    combined_target = np.concatenate([target[2022], target[2023]])
    combined_baseline = np.concatenate([baseline[2022], baseline[2023]])
    is_r = combined["game_type"].to_numpy() == 0
    weights = np.where(combined["season"].to_numpy() == 2022, YEAR_DECAY, 1.0)
    print(f"[confirm] locked leaves={leaves} scale={scale}; opening 2024 once", flush=True)
    effect_r, elapsed = fit_predict(
        combined.loc[is_r].reset_index(drop=True), rows[2024], features,
        combined_target[is_r] - combined_baseline[is_r], weights[is_r], leaves,
    )
    is_r24 = rows[2024]["game_type"].to_numpy() == 0
    effect = np.where(is_r24, effect_r, 0.0)
    correction = scale * effect
    candidate = np.clip(baseline[2024] + correction, 0.0, 1.0)
    game_type24 = np.where(is_r24, "R", "F")
    metrics = e155.metrics(baseline[2024], correction, target[2024], game_type24)
    month = rows[2024]["game_month"].to_numpy()
    early = e158.subset_gain(e155, baseline[2024], candidate, target[2024], month <= 6)
    late = e158.subset_gain(e155, baseline[2024], candidate, target[2024], month > 6)
    bootstrap = e158.cluster_bootstrap(
        baseline[2024], candidate, target[2024],
        rows[2024]["pitcher_id"].to_numpy(np.int64), draws=5000, seed=167,
    )
    f_exact = bool(np.array_equal(candidate[~is_r24], baseline[2024][~is_r24]))
    gates = {
        "discovery_positive_and_nonzero_scale": bool(
            scale > 0.0
            and max(row["selected_scale_2023"]["raw_gain"] for row in lock["alternatives"]) > 0.0
        ),
        "raw_gain_ge_12": bool(metrics["raw_gain"] >= 12.0),
        "early_positive": bool(early["gain"] > 0.0),
        "late_positive": bool(late["gain"] > 0.0),
        "pitcher_cluster_p025_positive": bool(bootstrap["p025"] > 0.0),
        "F_bit_exact": f_exact,
    }
    passed = bool(all(gates.values()))
    payload = {
        "experiment": 167,
        "status": "PASS_FOR_DEPLOY_REVIEW" if passed else "FAIL_NO_TEST_NO_ZIP",
        "analysis_only": True,
        "reads_test_or_external_reference_artifact": False,
        "lock_sha256": file_sha256(LOCK_PATH),
        "selected_num_leaves": leaves,
        "locked_scale": scale,
        "year_decay": YEAR_DECAY,
        "correction_clip": CORRECTION_CLIP,
        "fit_seconds_2024": elapsed,
        "confirmation_2024": metrics,
        "early_2024": early,
        "late_2024": late,
        "pitcher_cluster_bootstrap_2024": bootstrap,
        "F_exact_unchanged": f_exact,
        "gates": gates,
        "effect_2024": save_vector("effect_2024", effect),
        "candidate_2024": save_vector("candidate_2024", candidate),
        "post_2024_tuning": False,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "=== exp167 current clean anchor R-only LightGBM residual ===",
        f"locked leaves={leaves} scale={scale:.3f} decay={YEAR_DECAY} clip={CORRECTION_CLIP}",
        f"2024 fixed={metrics}",
        f"early={early['gain']:+.4f} late={late['gain']:+.4f}",
        f"pitcher bootstrap={bootstrap}",
        f"F exact={f_exact} gates={gates}",
        f"FINAL: {payload['status']}",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return payload


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
