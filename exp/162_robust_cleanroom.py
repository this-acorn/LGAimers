# -*- coding: utf-8 -*-
"""Clean-room forward validation and deployment for robust conditional EB.

Data boundary: this program reads only data/train.csv for fitting/validation and
data/test.csv for the final local inference smoke test.  It never reads saved
predictions, OOF arrays, reference repositories, archives, or third-party
weights/code.

The protocol is intentionally split:

  discovery: train 2022/2023 forward anchors, select an anchor on 2023, lock it.
  confirm:   reconstruct the lock and open 2024 exactly once.
  deploy:    fit the locked 2025 anchor, learn 2024 residual tables, build ZIP.

Every conditional residual table is learned on R rows from the immediately
preceding season and shrunk exactly toward zero.  F predictions are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"
LAB = ROOT / "lab" / "162_robust_cleanroom"
LOCK_PATH = LAB / "discovery_lock.json"
CONFIRM_PATH = LAB / "confirmation_2024.json"
REPORT_PATH = LAB / "report.json"
REPORT_TXT = LAB / "report.txt"
BUNDLE = LAB / 'candidates/candidate_exp162_robust_cleanroom_src'
ZIP_PATH = ROOT / 'artifacts/candidates/candidate_exp162_robust_cleanroom.zip'

TARGET = "control_success"
ID_COL = "row_id"
SCORE_SCALE = 100000.0
TENSOR_SPECS = (
    ("pitcher_batter_hand", ("pitcher_id", "batter_hand"), 0.25, 500.0),
    (
        "pitcher_count_base",
        ("pitcher_id", "_count_state", "base_state"),
        0.75,
        1000.0,
    ),
    (
        "hand_count_base",
        ("pitcher_hand", "batter_hand", "_count_state", "base_state"),
        0.05,
        500.0,
    ),
)
RELIABILITY_ALPHA = 500.0
RELIABLE_GATE = 1.0
UNSTABLE_GATE = 0.25
UNSEEN_GATE = 0.5

# All alternatives and their ordering are fixed before 2024 is opened.  They
# differ only in regularisation/capacity of the official-data anchor.
ANCHORS = {
    "stable_l15": {"num_leaves": 15, "min_data_in_leaf": 1500, "lambda_l2": 40.0},
    "balanced_l31": {"num_leaves": 31, "min_data_in_leaf": 900, "lambda_l2": 25.0},
    "flex_l63": {"num_leaves": 63, "min_data_in_leaf": 600, "lambda_l2": 30.0},
}

RAW_NUMERIC = (
    "game_month", "game_dayofweek", "inning", "balls_before", "strikes_before",
    "outs_before", "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b",
    "runner_on_3b", "num_runners_on", "home_win_expectancy",
    "away_win_expectancy", "li", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "asof_pitcher_n",
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_n",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("discovery", "confirm", "deploy", "all"))
    parser.add_argument("--threads", type=int, default=max(1, min(14, os.cpu_count() or 1)))
    parser.add_argument("--rounds", type=int, default=320)
    return parser.parse_args()


def score(probability: np.ndarray, target: np.ndarray) -> float:
    p = np.asarray(probability, np.float64)
    y = np.asarray(target, np.float64)
    rate = float(y.mean())
    return float(SCORE_SCALE * (1.0 - np.mean(np.square(p - y)) / (rate * (1.0 - rate))))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_prediction_artifact(
    name: str,
    prediction: np.ndarray,
    rows: pd.DataFrame,
    target: np.ndarray,
) -> dict[str, object]:
    """Persist only predictions generated in this run from official train."""
    path = LAB / f"{name}.npy"
    value = np.asarray(prediction, dtype=np.float32)
    np.save(path, value, allow_pickle=False)
    row_hash = hashlib.sha256(
        "\n".join(rows[ID_COL].astype(str).tolist()).encode("utf-8")
    ).hexdigest()
    target64 = np.ascontiguousarray(target, dtype=np.float64)
    return {
        "path": str(path),
        "rows": int(len(value)),
        "dtype": str(value.dtype),
        "sha256": sha256(path),
        "row_id_order_sha256": row_hash,
        "target_float64_sha256": hashlib.sha256(target64.view(np.uint8)).hexdigest(),
    }


def prepare_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["pitcher_id"] = pd.to_numeric(out["pitcher_id"], errors="coerce").fillna(-1).astype("int64")
    for column in ("pitcher_hand", "batter_hand"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(-1).astype("int16")
    for column in ("base_state", "game_type", "top_bottom"):
        out[column] = out[column].astype("string").fillna("__MISSING__").astype(str)
    balls = pd.to_numeric(out["balls_before"], errors="coerce").fillna(-1).astype("int16")
    strikes = pd.to_numeric(out["strikes_before"], errors="coerce").fillna(-1).astype("int16")
    out["_count_state"] = (balls * 3 + strikes).astype("int16")
    return out


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Row-local features; no raw player id and no validation distribution use."""
    x = frame.loc[:, list(RAW_NUMERIC)].copy()
    for column in x.columns:
        x[column] = pd.to_numeric(x[column], errors="coerce").astype("float32")
    balls = frame["balls_before"].to_numpy(np.float32)
    strikes = frame["strikes_before"].to_numpy(np.float32)
    pn = np.nan_to_num(frame["asof_pitcher_n"].to_numpy(np.float64), nan=0.0)
    bn = np.nan_to_num(frame["asof_batter_n"].to_numpy(np.float64), nan=0.0)
    x["f_count_state"] = (balls * 3.0 + strikes).astype("float32")
    x["f_count_diff"] = (balls - strikes).astype("float32")
    x["f_same_hand"] = (
        frame["pitcher_hand"].to_numpy() == frame["batter_hand"].to_numpy()
    ).astype("float32")
    x["f_is_r"] = (frame["game_type"].to_numpy() == "R").astype("float32")
    x["f_bottom"] = (frame["top_bottom"].to_numpy() == "B").astype("float32")
    x["f_log_pitcher_n"] = np.log1p(pn).astype("float32")
    x["f_log_batter_n"] = np.log1p(bn).astype("float32")
    x["f_pb_gap"] = (
        frame["asof_pitcher_success_rate"].to_numpy(np.float32)
        - frame["asof_batter_success_rate"].to_numpy(np.float32)
    )
    x["f_recent1_gap"] = (
        frame["asof_pitcher_prev1_game_success_rate"].to_numpy(np.float32)
        - frame["asof_pitcher_success_rate"].to_numpy(np.float32)
    )
    x["f_recent3_gap"] = (
        frame["asof_pitcher_prev3_game_success_rate"].to_numpy(np.float32)
        - frame["asof_pitcher_success_rate"].to_numpy(np.float32)
    )
    x["f_recent_trend"] = (
        frame["asof_pitcher_prev1_game_success_rate"].to_numpy(np.float32)
        - frame["asof_pitcher_prev5_game_success_rate"].to_numpy(np.float32)
    )
    return x


def load_train() -> pd.DataFrame:
    frame = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", low_memory=False)
    frame.columns = [column.replace("ï»¿", "").strip() for column in frame.columns]
    return prepare_keys(frame)


def fit_anchor(
    frame: pd.DataFrame,
    target_year: int,
    anchor_name: str,
    rounds: int,
    threads: int,
) -> tuple[lgb.Booster, np.ndarray]:
    train_mask = frame["season"].to_numpy() < target_year
    valid_mask = frame["season"].to_numpy() == target_year
    x_train = feature_matrix(frame.loc[train_mask])
    y_train = frame.loc[train_mask, TARGET].to_numpy(np.float32)
    x_valid = feature_matrix(frame.loc[valid_mask])
    config = ANCHORS[anchor_name]
    params = {
        "objective": "regression_l2",
        "metric": "l2",
        "learning_rate": 0.035,
        "num_leaves": config["num_leaves"],
        "min_data_in_leaf": config["min_data_in_leaf"],
        "lambda_l2": config["lambda_l2"],
        "feature_fraction": 0.82,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        "max_bin": 127,
        "verbosity": -1,
        "seed": 162,
        "feature_fraction_seed": 162,
        "bagging_seed": 162,
        "num_threads": threads,
        "force_col_wise": True,
    }
    dataset = lgb.Dataset(x_train, label=y_train, free_raw_data=True)
    model = lgb.train(params, dataset, num_boost_round=rounds)
    prediction = np.clip(model.predict(x_valid, num_threads=threads), 0.0, 1.0)
    return model, prediction.astype(np.float64)


def lookup_eb(
    source: pd.DataFrame,
    residual: np.ndarray,
    validation: pd.DataFrame,
    keys: tuple[str, ...],
    alpha: float,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, float]]:
    work = source.loc[:, list(keys)].copy()
    work["_residual"] = residual
    table = (
        work.groupby(list(keys), sort=False, observed=True, dropna=False)["_residual"]
        .agg(["sum", "size"])
        .reset_index()
    )
    table["correction"] = table["sum"] / (table["size"] + alpha)
    left = validation.loc[:, list(keys)].copy()
    left["_row"] = np.arange(len(left), dtype=np.int64)
    joined = left.merge(
        table.loc[:, list(keys) + ["correction", "size"]],
        on=list(keys), how="left", sort=False, validate="m:1",
    ).sort_values("_row")
    correction = joined["correction"].fillna(0.0).to_numpy(np.float64)
    compact = table.loc[:, list(keys) + ["correction", "size"]].copy()
    diagnostic = {
        "groups": int(len(table)),
        "coverage": float(joined["size"].notna().mean()),
        "mean_abs": float(np.abs(correction).mean()),
        "std": float(correction.std()),
    }
    return correction, compact, diagnostic


def tensor_effect(
    source: pd.DataFrame,
    source_target: np.ndarray,
    source_anchor: np.ndarray,
    validation: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, pd.DataFrame], dict[str, object]]:
    is_r_source = source["game_type"].to_numpy() == "R"
    source_r = source.loc[is_r_source].reset_index(drop=True)
    residual = source_target[is_r_source] - source_anchor[is_r_source]
    effect = np.zeros(len(validation), dtype=np.float64)
    tables: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, object] = {}
    for name, keys, scale, alpha in TENSOR_SPECS:
        correction, table, diagnostic = lookup_eb(source_r, residual, validation, keys, alpha)
        effect += scale * correction
        tables[name] = table
        diagnostics[name] = {"keys": list(keys), "scale": scale, "alpha": alpha, **diagnostic}
    is_r_valid = validation["game_type"].to_numpy() == "R"
    effect[~is_r_valid] = 0.0
    return effect, tables, diagnostics


def learn_gate(
    rows: pd.DataFrame,
    target: np.ndarray,
    anchor: np.ndarray,
    effect: np.ndarray,
) -> tuple[dict[int, float], dict[str, float]]:
    is_r = rows["game_type"].to_numpy() == "R"
    candidate = np.clip(anchor + effect, 0.0, 1.0)
    row_gain = np.square(anchor - target) - np.square(candidate - target)
    global_mean = float(row_gain[is_r].mean())
    work = pd.DataFrame({
        "pitcher_id": rows.loc[is_r, "pitcher_id"].to_numpy(np.int64),
        "gain": row_gain[is_r],
    })
    grouped = work.groupby("pitcher_id", sort=False)["gain"].agg(["sum", "size"])
    grouped["shrunk"] = (grouped["sum"] + RELIABILITY_ALPHA * global_mean) / (
        grouped["size"] + RELIABILITY_ALPHA
    )
    grouped["gate"] = np.where(grouped["shrunk"] > 0.0, RELIABLE_GATE, UNSTABLE_GATE)
    mapping = {int(key): float(value) for key, value in grouped["gate"].items()}
    diagnostic = {
        "pitchers": int(len(grouped)),
        "global_mean_row_gain": global_mean,
        "reliable_pitchers": int((grouped["gate"] == RELIABLE_GATE).sum()),
        "unstable_pitchers": int((grouped["gate"] == UNSTABLE_GATE).sum()),
    }
    return mapping, diagnostic


def apply_gate(rows: pd.DataFrame, effect: np.ndarray, mapping: dict[int, float]) -> tuple[np.ndarray, dict[str, float]]:
    ids = rows["pitcher_id"].to_numpy(np.int64)
    gate = np.fromiter((mapping.get(int(value), UNSEEN_GATE) for value in ids), np.float64, len(rows))
    gated = effect * gate
    is_r = rows["game_type"].to_numpy() == "R"
    gated[~is_r] = 0.0
    return gated, {
        "gate_1_share_R": float(np.mean(gate[is_r] == 1.0)),
        "gate_025_share_R": float(np.mean(gate[is_r] == 0.25)),
        "gate_unseen_share_R": float(np.mean(gate[is_r] == 0.5)),
    }


def evaluate(anchor: np.ndarray, effect: np.ndarray, target: np.ndarray, game_type: np.ndarray) -> dict[str, float]:
    candidate = np.clip(anchor + effect, 0.0, 1.0)
    equal_mean = np.clip(candidate + anchor.mean() - candidate.mean(), 0.0, 1.0)
    is_f = game_type == "F"
    rate = float(target.mean())
    row_gain = SCORE_SCALE * (
        np.square(anchor - target) - np.square(candidate - target)
    ) / (len(target) * rate * (1.0 - rate))
    return {
        "anchor_score": score(anchor, target),
        "candidate_score": score(candidate, target),
        "raw_gain": score(candidate, target) - score(anchor, target),
        "equal_mean_shape_gain": score(equal_mean, target) - score(anchor, target),
        "F_contribution": float(row_gain[is_f].sum()),
        "R_contribution": float(row_gain[~is_f].sum()),
        "mean_shift": float(candidate.mean() - anchor.mean()),
        "effect_std": float(effect.std()),
    }


def subset_gain(
    anchor: np.ndarray, effect: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    candidate = np.clip(anchor + effect, 0.0, 1.0)
    return {
        "rows": int(mask.sum()),
        "anchor_score": score(anchor[mask], target[mask]),
        "candidate_score": score(candidate[mask], target[mask]),
        "gain": score(candidate[mask], target[mask]) - score(anchor[mask], target[mask]),
    }


def pitcher_cluster_bootstrap(
    anchor: np.ndarray,
    effect: np.ndarray,
    target: np.ndarray,
    pitcher: np.ndarray,
    draws: int = 5000,
) -> dict[str, float | int]:
    candidate = np.clip(anchor + effect, 0.0, 1.0)
    work = pd.DataFrame({
        "pitcher": pitcher,
        "n": np.ones(len(target), dtype=np.int32),
        "y": target,
        "anchor_se": np.square(anchor - target),
        "candidate_se": np.square(candidate - target),
    })
    grouped = work.groupby("pitcher", sort=False).agg(
        n=("n", "sum"), y=("y", "sum"),
        anchor_se=("anchor_se", "sum"), candidate_se=("candidate_se", "sum"),
    )
    values = grouped[["n", "y", "anchor_se", "candidate_se"]].to_numpy(np.float64)
    rng = np.random.default_rng(162)
    gains = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 200):
        width = min(200, draws - start)
        sample = values[rng.integers(0, len(values), size=(width, len(values)))].sum(axis=1)
        rate = sample[:, 1] / sample[:, 0]
        gains[start : start + width] = SCORE_SCALE * (
            sample[:, 2] - sample[:, 3]
        ) / (sample[:, 0] * rate * (1.0 - rate))
    return {
        "draws": draws,
        "pitchers": int(len(values)),
        "median": float(np.median(gains)),
        "p025": float(np.quantile(gains, 0.025)),
        "p975": float(np.quantile(gains, 0.975)),
        "prob_positive": float(np.mean(gains > 0.0)),
    }


def rows_for(frame: pd.DataFrame, year: int) -> tuple[pd.DataFrame, np.ndarray]:
    rows = frame.loc[frame["season"].eq(year)].reset_index(drop=True)
    return rows, rows[TARGET].to_numpy(np.float64)


def reconstruct_discovery(
    frame: pd.DataFrame, anchor_name: str, rounds: int, threads: int
) -> dict[str, object]:
    rows22, y22 = rows_for(frame, 2022)
    rows23, y23 = rows_for(frame, 2023)
    _, p22 = fit_anchor(frame, 2022, anchor_name, rounds, threads)
    _, p23 = fit_anchor(frame, 2023, anchor_name, rounds, threads)
    effect23, _, tensor_diag = tensor_effect(rows22, y22, p22, rows23)
    gate, gate_diag = learn_gate(rows23, y23, p23, effect23)
    metrics = evaluate(p23, effect23, y23, rows23["game_type"].to_numpy())
    return {
        "anchor": anchor_name,
        "p22": p22,
        "p23": p23,
        "effect23": effect23,
        "gate": gate,
        "tensor": tensor_diag,
        "gate_fit": gate_diag,
        "metrics": metrics,
    }


def discovery(frame: pd.DataFrame, rounds: int, threads: int) -> dict[str, object]:
    if LOCK_PATH.exists():
        raise FileExistsError(f"discovery lock already exists: {LOCK_PATH}")
    alternatives: list[dict[str, object]] = []
    for name in ANCHORS:
        print(f"[discovery] fitting {name}", flush=True)
        result = reconstruct_discovery(frame, name, rounds, threads)
        alternatives.append({
            "anchor": name,
            "metrics_2023": result["metrics"],
            "tensor_2022_to_2023": result["tensor"],
            "gate_fit_2023": result["gate_fit"],
        })
        print(f"  {name}: {result['metrics']}", flush=True)
    # Primary selection is correction gain on 2023.  This line and all configs
    # are committed before any 2024 target is loaded by the confirmation phase.
    selected = max(alternatives, key=lambda row: (row["metrics_2023"]["raw_gain"], row["metrics_2023"]["anchor_score"]))
    payload = {
        "experiment": 162,
        "phase": "discovery_locked",
        "data_boundary": "official data/train.csv only; no saved OOF/model/archive/reference",
        "selection_rule": "max 2023 raw correction gain; tie by 2023 anchor score",
        "rounds": rounds,
        "anchors_fixed_before_2024": ANCHORS,
        "tensor_specs": [
            {"name": n, "keys": list(k), "scale": s, "alpha": a}
            for n, k, s, a in TENSOR_SPECS
        ],
        "gate": {"alpha": 500.0, "positive": 1.0, "nonpositive": 0.25, "unseen": 0.5},
        "alternatives": alternatives,
        "selected_anchor": selected["anchor"],
    }
    LAB.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def confirm(frame: pd.DataFrame, threads: int) -> dict[str, object]:
    if not LOCK_PATH.is_file():
        raise FileNotFoundError("run discovery first")
    if CONFIRM_PATH.exists():
        raise FileExistsError(f"2024 confirmation already exists: {CONFIRM_PATH}")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    anchor_name = str(lock["selected_anchor"])
    rounds = int(lock["rounds"])
    print(f"[confirm] reconstructing locked discovery anchor={anchor_name}", flush=True)
    d = reconstruct_discovery(frame.loc[frame["season"] <= 2023].copy(), anchor_name, rounds, threads)
    rows23, y23 = rows_for(frame, 2023)
    rows24, y24 = rows_for(frame, 2024)
    print("[confirm] fitting strict train<=2023 anchor; opening 2024 once", flush=True)
    _, p24 = fit_anchor(frame, 2024, anchor_name, rounds, threads)
    effect24, _, tensor_diag = tensor_effect(rows23, y23, d["p23"], rows24)
    gated24, gate_apply = apply_gate(rows24, effect24, d["gate"])
    metrics = evaluate(p24, gated24, y24, rows24["game_type"].to_numpy())
    month = rows24["game_month"].to_numpy(np.int16)
    periods = {
        "early_month_le_6": subset_gain(p24, gated24, y24, month <= 6),
        "late_month_gt_6": subset_gain(p24, gated24, y24, month > 6),
    }
    bootstrap = pitcher_cluster_bootstrap(
        p24, gated24, y24, rows24["pitcher_id"].to_numpy(np.int64)
    )
    is_f = rows24["game_type"].to_numpy() == "F"
    f_exact = bool(np.array_equal(np.clip(p24 + gated24, 0.0, 1.0)[is_f], p24[is_f]))
    gates = {
        "raw_gain_ge_12": bool(metrics["raw_gain"] >= 12.0),
        "early_gain_positive": bool(periods["early_month_le_6"]["gain"] > 0.0),
        "late_gain_positive": bool(periods["late_month_gt_6"]["gain"] > 0.0),
        "pitcher_cluster_p025_positive": bool(bootstrap["p025"] > 0.0),
        "F_exact_unchanged": f_exact,
    }
    artifacts = {
        "anchor_2023": save_prediction_artifact("anchor_2023", d["p23"], rows23, y23),
        "tensor_candidate_2023": save_prediction_artifact(
            "tensor_candidate_2023", np.clip(d["p23"] + d["effect23"], 0.0, 1.0), rows23, y23
        ),
        "anchor_2024": save_prediction_artifact("anchor_2024", p24, rows24, y24),
        "tensor_candidate_2024": save_prediction_artifact(
            "tensor_candidate_2024", np.clip(p24 + gated24, 0.0, 1.0), rows24, y24
        ),
    }
    payload = {
        "experiment": 162,
        "phase": "confirmation_2024_once",
        "lock_sha256": sha256(LOCK_PATH),
        "selected_anchor": anchor_name,
        "discovery_reconstruction_2023": d["metrics"],
        "tensor_2023_to_2024": tensor_diag,
        "gate_application_2024": gate_apply,
        "metrics_2024": metrics,
        "periods_2024": periods,
        "pitcher_cluster_bootstrap_2024": bootstrap,
        "F_exact_unchanged": f_exact,
        "deployment_gates": gates,
        "official_yearfold_prediction_artifacts": artifacts,
        "pass_for_deploy": bool(all(gates.values())),
        "post_2024_tuning": False,
    }
    CONFIRM_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[confirm] {metrics} pass={payload['pass_for_deploy']}", flush=True)
    return payload


DEPLOY_SCRIPT = r'''"""Official-data clean-room LightGBM + frozen robust conditional EB."""
from __future__ import annotations
import json
import os
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET = "control_success"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./model"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./output"))
RAW_NUMERIC = __RAW_NUMERIC__

def prepare(frame):
    out = frame.copy()
    out["pitcher_id"] = pd.to_numeric(out["pitcher_id"], errors="coerce").fillna(-1).astype("int64")
    for column in ("pitcher_hand", "batter_hand"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(-1).astype("int16")
    for column in ("base_state", "game_type", "top_bottom"):
        out[column] = out[column].astype("string").fillna("__MISSING__").astype(str)
    out["_count_state"] = (pd.to_numeric(out["balls_before"], errors="coerce").fillna(-1).astype("int16") * 3 + pd.to_numeric(out["strikes_before"], errors="coerce").fillna(-1).astype("int16")).astype("int16")
    return out

def matrix(frame):
    x = frame.loc[:, list(RAW_NUMERIC)].copy()
    for column in x.columns:
        x[column] = pd.to_numeric(x[column], errors="coerce").astype("float32")
    balls = frame["balls_before"].to_numpy(np.float32); strikes = frame["strikes_before"].to_numpy(np.float32)
    pn = np.nan_to_num(frame["asof_pitcher_n"].to_numpy(np.float64), nan=0.0); bn = np.nan_to_num(frame["asof_batter_n"].to_numpy(np.float64), nan=0.0)
    x["f_count_state"] = (balls * 3.0 + strikes).astype("float32"); x["f_count_diff"] = (balls - strikes).astype("float32")
    x["f_same_hand"] = (frame["pitcher_hand"].to_numpy() == frame["batter_hand"].to_numpy()).astype("float32")
    x["f_is_r"] = (frame["game_type"].to_numpy() == "R").astype("float32"); x["f_bottom"] = (frame["top_bottom"].to_numpy() == "B").astype("float32")
    x["f_log_pitcher_n"] = np.log1p(pn).astype("float32"); x["f_log_batter_n"] = np.log1p(bn).astype("float32")
    x["f_pb_gap"] = frame["asof_pitcher_success_rate"].to_numpy(np.float32) - frame["asof_batter_success_rate"].to_numpy(np.float32)
    x["f_recent1_gap"] = frame["asof_pitcher_prev1_game_success_rate"].to_numpy(np.float32) - frame["asof_pitcher_success_rate"].to_numpy(np.float32)
    x["f_recent3_gap"] = frame["asof_pitcher_prev3_game_success_rate"].to_numpy(np.float32) - frame["asof_pitcher_success_rate"].to_numpy(np.float32)
    x["f_recent_trend"] = frame["asof_pitcher_prev1_game_success_rate"].to_numpy(np.float32) - frame["asof_pitcher_prev5_game_success_rate"].to_numpy(np.float32)
    return x

def main():
    frame = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8-sig", low_memory=False)
    frame.columns = [c.replace("ï»¿", "").strip() for c in frame.columns]
    frame = prepare(frame)
    model = lgb.Booster(model_file=str(MODEL_DIR / "anchor.txt"))
    anchor = np.clip(model.predict(matrix(frame)), 0.0, 1.0)
    state = json.loads((MODEL_DIR / "eb_state.json").read_text(encoding="utf-8"))
    effect = np.zeros(len(frame), np.float64)
    for spec in state["tables"]:
        keys = spec["keys"]
        table = pd.DataFrame(spec["records"])
        left = frame.loc[:, keys].copy(); left["_row"] = np.arange(len(frame), dtype=np.int64)
        joined = left.merge(table, on=keys, how="left", sort=False, validate="m:1").sort_values("_row")
        effect += float(spec["scale"]) * joined["correction"].fillna(0.0).to_numpy(np.float64)
    mapping = {int(k): float(v) for k, v in state["pitcher_gate"].items()}
    gate = np.fromiter((mapping.get(int(v), 0.5) for v in frame["pitcher_id"].to_numpy()), np.float64, len(frame))
    is_r = frame["game_type"].to_numpy() == "R"
    effect *= gate; effect[~is_r] = 0.0
    prediction = np.clip(anchor + effect, 0.0, 1.0)
    if not np.array_equal(prediction[~is_r], anchor[~is_r]): raise AssertionError("F changed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame({ID_COL: frame[ID_COL], TARGET: prediction})
    output.to_csv(OUTPUT_DIR / "submission.csv", index=False, encoding="utf-8")
    print(f"Saved: {OUTPUT_DIR / 'submission.csv'} rows={len(output)} mean={prediction.mean():.9f} F_exact=True", flush=True)

if __name__ == "__main__": main()
'''


def deploy(frame: pd.DataFrame, threads: int) -> dict[str, object]:
    if not LOCK_PATH.is_file() or not CONFIRM_PATH.is_file():
        raise FileNotFoundError("run discovery and confirm first")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    confirmation = json.loads(CONFIRM_PATH.read_text(encoding="utf-8"))
    if not confirmation.get("pass_for_deploy"):
        raise RuntimeError("locked method did not pass untouched 2024; refusing deploy")
    anchor_name = str(lock["selected_anchor"]); rounds = int(lock["rounds"])
    # Reconstruct gate without reading or persisting an OOF vector.
    d = reconstruct_discovery(frame.loc[frame["season"] <= 2023].copy(), anchor_name, rounds, threads)
    rows24, y24 = rows_for(frame, 2024)
    print("[deploy] fitting 2024 strict residual anchor", flush=True)
    _, p24 = fit_anchor(frame, 2024, anchor_name, rounds, threads)
    print("[deploy] fitting final official train<=2024 anchor", flush=True)
    final_model, _ = fit_anchor(frame, 2025, anchor_name, rounds, threads)
    # The validation argument is irrelevant to saved tables; using the official
    # test only computes deployment coverage and is not used for training/tuning.
    test = prepare_keys(pd.read_csv(TEST_PATH, encoding="utf-8-sig", low_memory=False))
    effect_test, tables, tensor_diag = tensor_effect(rows24, y24, p24, test)
    gated_test, gate_diag = apply_gate(test, effect_test, d["gate"])
    if np.any(gated_test[test["game_type"].to_numpy() != "R"] != 0.0):
        raise AssertionError("F correction is nonzero")
    BUNDLE.mkdir(parents=True, exist_ok=True); (BUNDLE / "model").mkdir(exist_ok=True)
    final_model.save_model(str(BUNDLE / "model" / "anchor.txt"))
    state_tables = []
    for name, keys, scale, alpha in TENSOR_SPECS:
        table = tables[name].loc[:, list(keys) + ["correction"]].copy()
        records = json.loads(table.to_json(orient="records"))
        state_tables.append({"name": name, "keys": list(keys), "scale": scale, "alpha": alpha, "records": records})
    state = {
        "experiment": 162,
        "source_season": 2024,
        "residual_anchor": "strict model train seasons < 2024",
        "pitcher_gate_source": "2023 fixed discovery",
        "pitcher_gate": {str(k): v for k, v in d["gate"].items()},
        "tables": state_tables,
    }
    (BUNDLE / "model" / "eb_state.json").write_text(json.dumps(state, separators=(",", ":")) + "\n", encoding="utf-8")
    script = DEPLOY_SCRIPT.replace("__RAW_NUMERIC__", repr(RAW_NUMERIC))
    (BUNDLE / "script.py").write_text(script, encoding="utf-8")
    (BUNDLE / "requirements.txt").write_text("numpy==1.26.4\npandas==2.0.3\nlightgbm==4.6.0\n", encoding="utf-8")
    manifest = {
        "experiment": 162,
        "data_boundary": "official train.csv/test.csv only",
        "anchor": anchor_name,
        "rounds": rounds,
        "confirmation_sha256": sha256(CONFIRM_PATH),
        "tensor_deploy_diagnostics": tensor_diag,
        "gate_deploy_diagnostics": gate_diag,
        "files": {},
    }
    for path in sorted(p for p in BUNDLE.rglob("*") if p.is_file()):
        manifest["files"][path.relative_to(BUNDLE).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (BUNDLE / "model" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    # Validate unpacked source first.  No archive is consumed during QA.
    env = os.environ.copy(); env["DATA_DIR"] = str(ROOT / "data"); env["MODEL_DIR"] = str(BUNDLE / "model"); env["OUTPUT_DIR"] = str(BUNDLE / "output")
    run = subprocess.run([sys.executable, str(BUNDLE / "script.py")], cwd=BUNDLE, env=env, text=True, capture_output=True, check=True)
    submission = pd.read_csv(BUNDLE / "output" / "submission.csv")
    if list(submission.columns) != [ID_COL, TARGET] or len(submission) != len(test):
        raise AssertionError("local inference output schema mismatch")
    values = submission[TARGET].to_numpy(np.float64)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise AssertionError("invalid local inference probabilities")
    # Full local row-independence rehearsal: reverse order plus every singleton.
    raw_test = pd.read_csv(TEST_PATH, encoding="utf-8-sig", low_memory=False)
    baseline_by_id = dict(zip(submission[ID_COL].astype(str), values))
    qa_max_abs = 0.0
    qa_runs = 0
    qa_inputs = [("reverse", raw_test.iloc[::-1].reset_index(drop=True))]
    qa_inputs.extend(
        (f"single_{index}", raw_test.iloc[[index]].reset_index(drop=True))
        for index in range(len(raw_test))
    )
    for qa_name, qa_frame in qa_inputs:
        qa_root = LAB / "row_independence_qa" / qa_name
        qa_data = qa_root / "data"; qa_output = qa_root / "output"
        qa_data.mkdir(parents=True, exist_ok=True); qa_output.mkdir(parents=True, exist_ok=True)
        qa_frame.to_csv(qa_data / "test.csv", index=False, encoding="utf-8")
        qa_env = os.environ.copy(); qa_env["DATA_DIR"] = str(qa_data); qa_env["MODEL_DIR"] = str(BUNDLE / "model"); qa_env["OUTPUT_DIR"] = str(qa_output)
        subprocess.run([sys.executable, str(BUNDLE / "script.py")], cwd=BUNDLE, env=qa_env, text=True, capture_output=True, check=True)
        qa_prediction = pd.read_csv(qa_output / "submission.csv")
        expected = qa_prediction[ID_COL].astype(str).map(baseline_by_id).to_numpy(np.float64)
        difference = np.abs(qa_prediction[TARGET].to_numpy(np.float64) - expected)
        qa_max_abs = max(qa_max_abs, float(difference.max(initial=0.0)))
        qa_runs += 1
    if qa_max_abs != 0.0:
        raise AssertionError(f"row independence failed max_abs={qa_max_abs}")
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in ("script.py", "requirements.txt", "model/anchor.txt", "model/eb_state.json", "model/manifest.json"):
            archive.write(BUNDLE / relative, relative)
    payload = {
        "experiment": 162,
        "status": "READY_TO_SUBMIT",
        "zip": str(ZIP_PATH),
        "zip_bytes": ZIP_PATH.stat().st_size,
        "zip_sha256": sha256(ZIP_PATH),
        "bundle": str(BUNDLE),
        "selected_anchor": anchor_name,
        "discovery_gain_2023": lock["alternatives"][[row["anchor"] for row in lock["alternatives"]].index(anchor_name)]["metrics_2023"],
        "confirmation_2024": confirmation["metrics_2024"],
        "F_exact_unchanged": confirmation["F_exact_unchanged"],
        "local_test_rows": int(len(submission)),
        "local_test_mean": float(values.mean()),
        "local_test_min": float(values.min()),
        "local_test_max": float(values.max()),
        "local_inference_stdout": run.stdout.strip(),
        "row_independence": {
            "runs": qa_runs,
            "reverse_and_every_singleton": True,
            "max_abs_difference": qa_max_abs,
            "passed": True,
        },
        "data_compliance": {
            "train": str(TRAIN_PATH), "test": str(TEST_PATH),
            "saved_OOF_read": False, "existing_zip_read": False,
            "reference_directory_read": False, "external_model_or_code_read": False,
        },
    }
    REPORT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    REPORT_TXT.write_text("\n".join([
        "=== exp162 robust clean-room ===", f"status={payload['status']}",
        f"selected_anchor={anchor_name}", f"2023={payload['discovery_gain_2023']}",
        f"2024={payload['confirmation_2024']}", f"F_exact={payload['F_exact_unchanged']}",
        f"zip={ZIP_PATH}", f"sha256={payload['zip_sha256']}",
        f"local_test={run.stdout.strip()}",
        "compliance=official train/test only; no saved OOF/reference/external weights/code/zip read",
    ]) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    args = parse_args(); started = time.time()
    LAB.mkdir(parents=True, exist_ok=True)
    frame = load_train()
    if args.phase in ("discovery", "all") and not LOCK_PATH.exists(): discovery(frame, args.rounds, args.threads)
    if args.phase in ("confirm", "all") and not CONFIRM_PATH.exists(): confirm(frame, args.threads)
    if args.phase in ("deploy", "all"): deploy(frame, args.threads)
    print(f"elapsed_seconds={time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
