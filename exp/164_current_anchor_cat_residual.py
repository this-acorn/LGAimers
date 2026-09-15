# -*- coding: utf-8 -*-
"""Strict temporal CatBoost residual model on the current clean anchor.

This experiment is written from scratch and uses only official train rows plus
our own frozen OOF predictions.  It tests the general high-level pattern of a
small R-only residual model while keeping F rows bit-identical to the anchor.

Selection protocol:
  1. fit 2022 OOF residuals, choose a correction scale on 2023;
  2. freeze model configuration and scale;
  3. refit on 2022+2023 OOF residuals with fixed year weights;
  4. evaluate once on 2024.

No test data, external artifact, leaderboard score, or deployable ZIP is read
or produced here.  A separate builder is permitted only after the final gate.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train.csv"
OUT_JSON = ROOT / "lab" / "164_current_anchor_cat_residual.json"
OUT_TXT = ROOT / "lab" / "164_current_anchor_cat_residual.txt"

# A single locked seed is used for the time-critical gate.  Three seeds are
# trained only by the deployment builder if this direction survives 2024.
SEEDS = (42,)
PARAMS = {
    "iterations": 500,
    "depth": 7,
    "learning_rate": 0.035,
    "l2_leaf_reg": 20.0,
    "loss_function": "RMSE",
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": 14,
}
YEAR_DECAY = 0.65
CORRECTION_CLIP = 0.08
SCALE_GRID = (0.0, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0)

CATEGORICAL = [
    "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "base_state",
    "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "_count_state", "_base_out",
    "_count_hand", "_team_matchup",
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    b = pd.to_numeric(out["balls_before"], errors="coerce").fillna(-1).astype("int16")
    s = pd.to_numeric(out["strikes_before"], errors="coerce").fillna(-1).astype("int16")
    o = pd.to_numeric(out["outs_before"], errors="coerce").fillna(-1).astype("int16")
    ph = out["pitcher_hand"].astype("string").fillna("M")
    bh = out["batter_hand"].astype("string").fillna("M")
    out["_count_state"] = (b.astype(str) + "_" + s.astype(str))
    out["_base_out"] = out["base_state"].astype("string").fillna("M") + "_" + o.astype(str)
    out["_count_hand"] = out["_count_state"] + "_" + ph + "_" + bh
    out["_team_matchup"] = (
        out["pitcher_team_id"].astype("string").fillna("M") + "_" +
        out["batter_team_id"].astype("string").fillna("M")
    )
    out["_same_hand"] = (ph == bh).astype("int8")
    out["_count_diff"] = (b - s).astype("int8")
    out["_any_runner"] = (
        pd.to_numeric(out["num_runners_on"], errors="coerce").fillna(0) > 0
    ).astype("int8")
    out["_scoring_position"] = (
        pd.to_numeric(out["runner_on_2b"], errors="coerce").fillna(0)
        + pd.to_numeric(out["runner_on_3b"], errors="coerce").fillna(0) > 0
    ).astype("int8")
    out["_log_pitcher_n"] = np.log1p(
        pd.to_numeric(out["asof_pitcher_n"], errors="coerce").fillna(0).clip(lower=0)
    ).astype("float32")
    out["_log_batter_n"] = np.log1p(
        pd.to_numeric(out["asof_batter_n"], errors="coerce").fillna(0).clip(lower=0)
    ).astype("float32")
    out["_pitcher_form1"] = (
        pd.to_numeric(out["asof_pitcher_prev1_game_success_rate"], errors="coerce")
        - pd.to_numeric(out["asof_pitcher_success_rate"], errors="coerce")
    ).astype("float32")
    out["_pitcher_form3"] = (
        pd.to_numeric(out["asof_pitcher_prev3_game_success_rate"], errors="coerce")
        - pd.to_numeric(out["asof_pitcher_success_rate"], errors="coerce")
    ).astype("float32")
    out["_pitcher_trend"] = (
        pd.to_numeric(out["asof_pitcher_prev1_game_success_rate"], errors="coerce")
        - pd.to_numeric(out["asof_pitcher_prev5_game_success_rate"], errors="coerce")
    ).astype("float32")
    out["_pitcher_batter_gap"] = (
        pd.to_numeric(out["asof_pitcher_success_rate"], errors="coerce")
        - pd.to_numeric(out["asof_batter_success_rate"], errors="coerce")
    ).astype("float32")
    return out


def matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for column in features:
        if column in CATEGORICAL:
            out[column] = frame[column].astype("string").fillna("__MISSING__").astype(str)
        else:
            out[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    return out


def fit_predict(train, valid, features, residual, weights):
    x_train = matrix(train, features)
    x_valid = matrix(valid, features)
    train_pool = Pool(
        x_train, residual.astype(np.float32), cat_features=CATEGORICAL,
        weight=weights.astype(np.float32),
    )
    valid_pool = Pool(x_valid, cat_features=CATEGORICAL)
    del x_train, x_valid
    gc.collect()
    predictions = []
    elapsed = []
    for seed in SEEDS:
        tick = time.time()
        model = CatBoostRegressor(**PARAMS, random_seed=seed)
        model.fit(train_pool)
        pred = np.asarray(model.predict(valid_pool), dtype=np.float64)
        predictions.append(np.clip(pred, -CORRECTION_CLIP, CORRECTION_CLIP))
        elapsed.append(time.time() - tick)
        del model
        gc.collect()
    return np.mean(predictions, axis=0), elapsed


def curve(e155, baseline, effect, target, game_type):
    rows = []
    for scale in SCALE_GRID:
        metric = e155.metrics(baseline, scale * effect, target, game_type)
        rows.append({"scale": scale, **metric})
    return rows


def main():
    started = time.time()
    e158 = load_module(ROOT / "exp" / "158_robust_conditional_tensor_eb.py", "e158_164")
    e155 = load_module(ROOT / "exp" / "155_cause_runner_currentblend_gate.py", "e155_164")
    frame = pd.read_csv(TRAIN, encoding="utf-8-sig", low_memory=False)
    frame.columns = [column.replace("\ufeff", "").strip() for column in frame.columns]
    frame = add_features(frame)
    rows = {
        year: frame.loc[frame["season"].eq(year)].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    target = {
        year: rows[year]["control_success"].to_numpy(np.float64) for year in rows
    }
    baseline = {
        year: e158.load_current_baseline(e155, rows[year], year) for year in rows
    }
    raw_features = [
        column for column in frame.columns
        if column not in {"row_id", "season", "control_success"}
    ]
    features = [column for column in raw_features if column not in {"_base_probability"}]
    # Include the current anchor probability as an explicit residual surface.
    for year in rows:
        rows[year]["_base_probability"] = baseline[year].astype("float32")
    features.append("_base_probability")
    missing_cats = sorted(set(CATEGORICAL) - set(features))
    if missing_cats:
        raise AssertionError(f"missing categorical features: {missing_cats}")

    train22 = rows[2022]
    valid23 = rows[2023]
    mask22 = train22["game_type"].to_numpy() == "R"
    effect23_r, times23 = fit_predict(
        train22.loc[mask22].reset_index(drop=True), valid23, features,
        target[2022][mask22] - baseline[2022][mask22],
        np.ones(int(mask22.sum()), dtype=np.float64),
    )
    effect23 = np.where(valid23["game_type"].to_numpy() == "R", effect23_r, 0.0)
    discovery_curve = curve(
        e155, baseline[2023], effect23, target[2023],
        valid23["game_type"].to_numpy(),
    )
    selected = max(
        discovery_curve,
        key=lambda item: (item["raw_gain"], item["equal_mean_shape_gain"], -item["scale"]),
    )
    locked_scale = float(selected["scale"])

    combined = pd.concat([rows[2022], rows[2023]], ignore_index=True)
    combined_target = np.concatenate([target[2022], target[2023]])
    combined_baseline = np.concatenate([baseline[2022], baseline[2023]])
    is_r = combined["game_type"].to_numpy() == "R"
    weights = np.where(combined["season"].to_numpy() == 2022, YEAR_DECAY, 1.0)
    effect24_r, times24 = fit_predict(
        combined.loc[is_r].reset_index(drop=True), rows[2024], features,
        combined_target[is_r] - combined_baseline[is_r], weights[is_r],
    )
    effect24 = np.where(rows[2024]["game_type"].to_numpy() == "R", effect24_r, 0.0)
    confirmation = e155.metrics(
        baseline[2024], locked_scale * effect24, target[2024],
        rows[2024]["game_type"].to_numpy(),
    )
    diagnostic_curve = curve(
        e155, baseline[2024], effect24, target[2024],
        rows[2024]["game_type"].to_numpy(),
    )
    candidate24 = np.clip(baseline[2024] + locked_scale * effect24, 0.0, 1.0)
    month = rows[2024]["game_month"].to_numpy()
    early = e158.subset_gain(e155, baseline[2024], candidate24, target[2024], month <= 6)
    late = e158.subset_gain(e155, baseline[2024], candidate24, target[2024], month > 6)
    bootstrap = e158.cluster_bootstrap(
        baseline[2024], candidate24, target[2024],
        rows[2024]["pitcher_id"].to_numpy(np.int64), draws=3000, seed=164,
    )
    passed = bool(
        locked_scale > 0.0
        and selected["raw_gain"] > 0.0
        and confirmation["raw_gain"] >= 12.0
        and confirmation["equal_mean_shape_gain"] >= 8.0
        and early["gain"] > 0.0
        and late["gain"] > 0.0
        and bootstrap["p025"] > 0.0
        and confirmation["F_contribution"] == 0.0
    )
    status = "PASS_FOR_FINAL_REFIT" if passed else "FAIL_NO_SUBMISSION"
    payload = {
        "experiment": 164,
        "status": status,
        "analysis_only": True,
        "official_train_and_own_oof_only": True,
        "reads_test_or_lb": False,
        "params": PARAMS,
        "seeds": SEEDS,
        "year_decay": YEAR_DECAY,
        "correction_clip": CORRECTION_CLIP,
        "features": features,
        "categorical": CATEGORICAL,
        "discovery_curve_2023": discovery_curve,
        "selected_2023": selected,
        "locked_scale": locked_scale,
        "confirmation_2024": confirmation,
        "diagnostic_curve_2024": diagnostic_curve,
        "early_2024": early,
        "late_2024": late,
        "bootstrap_2024": bootstrap,
        "fit_seconds_2023": times23,
        "fit_seconds_2024": times24,
        "elapsed_seconds": time.time() - started,
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "=== exp164 current-anchor R-only CatBoost residual ===",
        f"params={PARAMS} seeds={SEEDS} year_decay={YEAR_DECAY}",
        "2023 curve: " + " ".join(
            f"s={row['scale']:.3f}:{row['raw_gain']:+.3f}" for row in discovery_curve
        ),
        f"locked scale={locked_scale:.3f} selected={selected}",
        f"2024 fixed confirmation={confirmation}",
        "2024 diagnostic curve: " + " ".join(
            f"s={row['scale']:.3f}:{row['raw_gain']:+.3f}" for row in diagnostic_curve
        ),
        f"2024 early={early['gain']:+.3f} late={late['gain']:+.3f}",
        f"bootstrap={bootstrap}",
        f"fit seconds 2023={times23} 2024={times24}",
        f"FINAL: {status}",
        f"elapsed={payload['elapsed_seconds']:.2f}s",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
