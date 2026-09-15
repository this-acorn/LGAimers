# -*- coding: utf-8 -*-
"""EXP-127: dynamic hierarchical base + shared residual regressor.

Both the 20k/20k smoke and the reviewed full discovery use the strict
``train <= 2022 -> valid == 2023`` fold.  Neither mode reads test.csv, builds
a submission, packages an inference model, or reads any EXP-021 Python source.

The experiment has two pieces:

1. A row-local hierarchical probability ``p0``.  Exact success counts are
   recovered from official as-of counts/rates and a train-only prior-season
   snapshot.  Pitcher shrinkage increases when the official previous 1/3/5
   game rates disagree.
2. A CatBoostRegressor trained on ``control_success - p0`` over all R/F rows.

Prepared features and models are cached under mode-distinct paths.  Cache reuse requires
exact row-id/order fingerprints, the same feature schema, and the same source
file identity.  The primary comparator is exact ``Acur``: corrected seed-42
CAT5+CS+V18+affine blended with the saved EXP-020 rank-6/s300 OOF at the
already-frozen weights.  The EXP-122 endpoint is reported only as a diagnostic.

Usage:

    python -u exp/127_dynamic_hier_residual.py --smoke
    python -u exp/127_dynamic_hier_residual.py --full

Full mode is deliberately fixed to all fold rows, seed 42 and 500 trees.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.csv"
S12_PATH = ROOT / "archive" / "src" / 'submissions/submit12_src' / "script.py"
LAB = ROOT / "lab"
LAB.mkdir(parents=True, exist_ok=True)

YEAR = 2023
TRAIN_END = 2022
SMOKE_ROWS = 20_000
SMOKE_PFB_ROWS = 50_000
SCHEMA_VERSION_SMOKE = "127-stage1-smoke-v1"
SCHEMA_VERSION_FULL = "127-full-discovery-v2"

AFFINE_CENTER = 0.49
AFFINE_SCALE = 1.06
AFFINE_SHIFT = -0.0066
V18_GAMMA = 0.30
EXP020_WEIGHT = 0.35655716947524263
OWN_WEIGHT = 1.0 - EXP020_WEIGHT
EXPECTED_ACUR_FULL_SCORE = -22.742734501002282

K_CAREER = 200.0
K_SEASON_BASE = 50.0
PITCHER_WEIGHT = 0.70
BATTER_WEIGHT = 0.30
EPS = 1e-5
GLOBAL_EFFECT_CLIP = 0.12

CAT_FEATURES = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_team_id",
    "batter_team_id",
]
ENG18 = [
    "f_count_state",
    "f_count_diff",
    "f_is_3ball",
    "f_is_2strike",
    "f_is_full",
    "f_same_hand",
    "f_scoring_pos",
    "f_any_runner",
    "f_log_pn",
    "f_log_bn",
    "f_smooth_p",
    "f_smooth_b",
    "f_form_dev1",
    "f_form_dev3",
    "f_form_trend",
    "f_pb_diff",
    "f_miss_p",
    "f_miss_prev1",
]
HIER_FEATURES = [
    "h_game_prior",
    "h_p_career",
    "h_p_season_logn",
    "h_p_success_raw",
    "h_p_success_post",
    "h_p_success_rel",
    "h_p_success_sd",
    "h_p_recent_vol",
    "h_p_recent_vol_missing",
    "h_p_kseason_log",
    "h_p_bridge_reverse_raw",
    "h_p_bridge_reverse_post",
    "h_p_bridge_middle_raw",
    "h_p_bridge_middle_post",
    "h_b_career",
    "h_b_season_logn",
    "h_b_success_raw",
    "h_b_success_post",
    "h_b_success_rel",
    "h_b_success_sd",
    "h_base",
    "h_base_sd",
]

RECENT_COLUMNS = [
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
]
COMPONENT_RATES = {
    "reverse": "asof_pitcher_reverse_rate",
    "middle": "asof_pitcher_middle_rate",
}
COMPONENT_LABELS = {"reverse": "lab_rev", "middle": "lab_mid"}

OWN_BASE_PATH = LAB / "122_y2023_seed42_base_final.npy"
V18_EFFECT_PATH = LAB / "104_v18_effect_y2023.npy"
EXP020_ROOT = (
    LAB
    / "115_mkis_exp021_work"
    / "artifacts"
    / "EXP-020"
    / "low_rank_pitcher_context_eb"
)
EXP020_OOF_PATH = EXP020_ROOT / "predictions_lowrank_s300_r6_2023.npy"
EXP020_TARGET_PATH = EXP020_ROOT / "targets_2023.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    return parser.parse_args()


ARGS = parse_args()
IS_SMOKE = bool(ARGS.smoke)
if ARGS.full and ARGS.seed != 42:
    raise ValueError("full discovery is frozen to seed 42")
MODE = "smoke" if IS_SMOKE else "full"
SCHEMA_VERSION = SCHEMA_VERSION_SMOKE if IS_SMOKE else SCHEMA_VERSION_FULL
if IS_SMOKE:
    CACHE_PATH = LAB / "127_y2023_smoke_features.joblib"
    CACHE_META_PATH = LAB / "127_y2023_smoke_features_meta.json"
    MODEL_PATH = LAB / "127_y2023_seed42_smoke.cbm"
    MODEL_META_PATH = LAB / "127_y2023_seed42_smoke_model_meta.json"
    PRED_PATH = LAB / "127_y2023_seed42_smoke_pred.npy"
    P0_PATH = LAB / "127_y2023_smoke_p0.npy"
    RESULT_JSON = LAB / "127_dynamic_hier_residual_smoke.json"
    RESULT_TXT = LAB / "127_dynamic_hier_residual_smoke.txt"
    LIVE_TXT = LAB / "127_dynamic_hier_residual_smoke_live.txt"
else:
    CACHE_PATH = LAB / "127_y2023_full_features.joblib"
    CACHE_META_PATH = LAB / "127_y2023_full_features_meta.json"
    MODEL_PATH = LAB / "127_y2023_seed42_full500.cbm"
    MODEL_META_PATH = LAB / "127_y2023_seed42_full500_model_meta.json"
    PRED_PATH = LAB / "127_y2023_seed42_full500_pred.npy"
    P0_PATH = LAB / "127_y2023_full_p0.npy"
    RESULT_JSON = LAB / "127_dynamic_hier_residual_full.json"
    RESULT_TXT = LAB / "127_dynamic_hier_residual_full.txt"
    LIVE_TXT = LAB / "127_dynamic_hier_residual_full_live.txt"
STARTED = time.time()
LINES: list[str] = []


def log(message: str = "") -> None:
    line = str(message)
    print(line, flush=True)
    LINES.append(line)
    with LIVE_TXT.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def tick(message: str) -> None:
    log(f"  [{time.time() - STARTED:8.1f}s] {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def ordered_value_fingerprint(values: pd.Series | np.ndarray) -> str:
    """Stable, order-sensitive fingerprint with unambiguous length framing."""

    digest = hashlib.sha256()
    for value in np.asarray(values, dtype=object):
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def schema_fingerprint(feature_names: list[str]) -> str:
    payload = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "features": feature_names,
            "categorical": CAT_FEATURES,
            "hierarchical": HIER_FEATURES,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_s12():
    if not S12_PATH.is_file():
        raise FileNotFoundError(S12_PATH)
    spec = importlib.util.spec_from_file_location("exp127_s12", S12_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(S12_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S12 = load_s12()


def load_scope_index() -> tuple[pd.DataFrame, int]:
    """Read no future targets; use season only to stop exactly after 2023."""

    index = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
        usecols=["row_id", "season"],
        low_memory=False,
    )
    index.columns = index.columns.str.replace("\ufeff", "", regex=False).str.strip()
    seasons = pd.to_numeric(index["season"], errors="raise").astype(int)
    if not seasons.is_monotonic_increasing:
        raise AssertionError("train.csv must be season-sorted for strict staged loading")
    allowed = seasons <= YEAR
    if not allowed.any() or int(seasons[allowed].max()) != YEAR:
        raise AssertionError(f"could not locate validation season {YEAR}")
    nrows = int(allowed.sum())
    scope = index.iloc[:nrows].copy().reset_index(drop=True)
    if (pd.to_numeric(scope["season"]) > YEAR).any():
        raise AssertionError("future season entered strict scope")
    return scope, nrows


def deterministic_sample(indices: np.ndarray, count: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= count:
        return np.sort(indices)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=count, replace=False))


def selected_positions(scope: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seasons = pd.to_numeric(scope["season"], errors="raise").to_numpy(np.int16)
    train_global = np.flatnonzero(seasons <= TRAIN_END)
    valid_global_all = np.flatnonzero(seasons == YEAR)
    if IS_SMOKE:
        train_selected = deterministic_sample(
            train_global, SMOKE_ROWS, 127_000 + YEAR
        )
        valid_selected_global = deterministic_sample(
            valid_global_all, SMOKE_ROWS, 127_500 + YEAR
        )
    else:
        train_selected = train_global
        valid_selected_global = valid_global_all
    valid_selected_local = np.searchsorted(valid_global_all, valid_selected_global)
    if not np.array_equal(
        valid_global_all[valid_selected_local], valid_selected_global
    ):
        raise AssertionError("validation global/local position mapping failed")
    return train_selected, valid_selected_global, valid_selected_local


def load_strict_frame(nrows: int) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
        nrows=nrows,
        low_memory=False,
    )
    frame.columns = frame.columns.str.replace("\ufeff", "", regex=False).str.strip()
    if int(frame["season"].max()) != YEAR or (frame["season"] > YEAR).any():
        raise AssertionError("strict frame boundary failed")
    if frame["row_id"].duplicated().any():
        raise AssertionError("row_id must be unique")
    base_columns = [
        column
        for column in frame.columns
        if column not in ("row_id", "control_success")
    ]
    if len(base_columns) != 47:
        raise AssertionError(f"base feature contract changed: {len(base_columns)} != 47")
    return frame, base_columns


def recover_pitch_labels(history: pd.DataFrame) -> pd.DataFrame:
    """Recover only within the training scope; validation rows are never used."""

    history = history.copy()
    ordered = history.sort_values(
        ["pitcher_id", "asof_pitcher_n"], kind="mergesort"
    ).reset_index()
    pitcher = ordered["pitcher_id"].to_numpy()
    count = ordered["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    next_pitch = (pitcher[1:] == pitcher[:-1]) & (
        np.abs(count[1:] - count[:-1] - 1.0) <= 1e-6
    )
    for name, column in (
        ("lab_mid", "asof_pitcher_middle_rate"),
        ("lab_rev", "asof_pitcher_reverse_rate"),
        ("lab_fb", "asof_pitcher_fastball_rate"),
        ("lab_brk", "asof_pitcher_breaking_rate"),
    ):
        cumulative = ordered[column].fillna(0).to_numpy(np.float64) * count
        label = np.full(len(ordered), np.nan, dtype=np.float64)
        difference = np.round(cumulative[1:] - cumulative[:-1])
        label[:-1] = np.where(next_pitch, difference, np.nan)
        ordered[name] = np.where((label == 0.0) | (label == 1.0), label, np.nan)
    restored = ordered.set_index("index").sort_index()
    for name in ("lab_mid", "lab_rev", "lab_fb", "lab_brk"):
        history[name] = restored[name].to_numpy()
    history["_cg"] = np.where(
        history["strikes_before"].to_numpy() > history["balls_before"].to_numpy(),
        2,
        np.where(
            history["balls_before"].to_numpy()
            > history["strikes_before"].to_numpy(),
            0,
            1,
        ),
    ).astype(np.int8)
    return history


def add_count_group(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["_cg"] = np.where(
        rows["strikes_before"].to_numpy() > rows["balls_before"].to_numpy(),
        2,
        np.where(
            rows["balls_before"].to_numpy()
            > rows["strikes_before"].to_numpy(),
            0,
            1,
        ),
    ).astype(np.int8)
    return rows


def build_const(
    source: pd.DataFrame,
    id_column: str,
    n_column: str,
    rates: dict[str, str],
) -> pd.DataFrame:
    if source.empty:
        return pd.DataFrame(
            {"id": [], "N_end": [], **{f"S_{name}": [] for name in rates}}
        )
    last = source.sort_values([id_column, n_column], kind="mergesort").groupby(
        id_column, sort=False
    ).tail(1)
    output = pd.DataFrame({"id": last[id_column].to_numpy()})
    n_last = last[n_column].fillna(0).to_numpy(np.float64)
    output["N_end"] = n_last + 1.0
    for name, column in rates.items():
        rate = last[column].fillna(0).to_numpy(np.float64)
        if name == "succ":
            output[f"S_{name}"] = np.round(rate * n_last) + last[
                "control_success"
            ].to_numpy(np.float64)
        else:
            # Preserve the accepted current-CS feature contract.  The new
            # integer bridge features below do not use this approximation.
            output[f"S_{name}"] = rate * (n_last + 1.0)
    return output


def mix_asof_train(history: pd.DataFrame) -> pd.DataFrame:
    rows = history.dropna(subset=["lab_fb", "lab_brk"])
    conditional = (
        rows.groupby(["pitcher_id", "season", "_cg"])
        .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
        .reset_index()
        .sort_values(["pitcher_id", "_cg", "season"])
    )
    group = conditional.groupby(["pitcher_id", "_cg"])
    for column in ("n", "fb", "brk"):
        conditional[f"p_{column}"] = group[column].cumsum() - conditional[column]
    overall = (
        rows.groupby(["pitcher_id", "season"])
        .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
        .reset_index()
        .sort_values(["pitcher_id", "season"])
    )
    overall_group = overall.groupby("pitcher_id")
    for column in ("n", "fb", "brk"):
        overall[f"po_{column}"] = overall_group[column].cumsum() - overall[column]
    table = conditional.merge(
        overall[["pitcher_id", "season", "po_n", "po_fb", "po_brk"]],
        on=["pitcher_id", "season"],
        how="left",
        validate="many_to_one",
    )
    k = 50.0
    overall_fb = (table["po_fb"] + 0.5 * k) / (table["po_n"] + k)
    overall_brk = (table["po_brk"] + 0.3 * k) / (table["po_n"] + k)
    table["mix_fb"] = np.where(
        table["po_n"] > 0,
        (table["p_fb"] + k * overall_fb) / (table["p_n"] + k),
        np.nan,
    ).astype(np.float32)
    table["mix_brk"] = np.where(
        table["po_n"] > 0,
        (table["p_brk"] + k * overall_brk) / (table["p_n"] + k),
        np.nan,
    ).astype(np.float32)
    return table[["pitcher_id", "season", "_cg", "mix_fb", "mix_brk"]]


def mix_career(history: pd.DataFrame) -> pd.DataFrame:
    rows = history.dropna(subset=["lab_fb", "lab_brk"])
    table = (
        rows.groupby(["pitcher_id", "_cg"])
        .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
        .reset_index()
    )
    overall = (
        rows.groupby("pitcher_id")
        .agg(on=("lab_fb", "size"), ofb=("lab_fb", "sum"), obrk=("lab_brk", "sum"))
        .reset_index()
    )
    table = table.merge(overall, on="pitcher_id", how="left", validate="many_to_one")
    k = 50.0
    overall_fb = (table["ofb"] + 0.5 * k) / (table["on"] + k)
    overall_brk = (table["obrk"] + 0.3 * k) / (table["on"] + k)
    table["mix_fb"] = ((table["fb"] + k * overall_fb) / (table["n"] + k)).astype(
        np.float32
    )
    table["mix_brk"] = (
        (table["brk"] + k * overall_brk) / (table["n"] + k)
    ).astype(np.float32)
    return table[["pitcher_id", "_cg", "mix_fb", "mix_brk"]]


def row_volatility(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = rows[RECENT_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(
        np.float64
    )
    finite_count = np.isfinite(values).sum(axis=1)
    with np.errstate(invalid="ignore"):
        volatility = np.nanstd(values, axis=1, ddof=0)
    missing = finite_count < 2
    volatility[missing] = np.nan
    return volatility, missing


def game_prior(source: pd.DataFrame, label: str) -> tuple[dict[str, float], float]:
    if source.empty or label not in source:
        return {}, 0.5
    valid = source[label].notna()
    if not valid.any():
        return {}, 0.5
    overall = float(source.loc[valid, label].mean())
    grouped = (
        source.loc[valid, ["game_type", label]]
        .assign(game_type=lambda frame: frame["game_type"].astype(str))
        .groupby("game_type", sort=False)[label]
        .mean()
        .to_dict()
    )
    return {str(key): float(value) for key, value in grouped.items()}, overall


def last_success_snapshot(
    source: pd.DataFrame, id_column: str, n_column: str, rate_column: str
) -> pd.DataFrame:
    if source.empty:
        return pd.DataFrame(columns=["id", "N_end", "S_end"])
    last = source.sort_values([id_column, n_column], kind="mergesort").groupby(
        id_column, sort=False
    ).tail(1)
    n_last = last[n_column].fillna(0).to_numpy(np.float64)
    rate_last = last[rate_column].fillna(0).to_numpy(np.float64)
    output = pd.DataFrame(
        {
            "id": last[id_column].to_numpy(),
            "N_end": n_last + 1.0,
            "S_end": np.round(rate_last * n_last)
            + last["control_success"].to_numpy(np.float64),
        }
    )
    if output["id"].duplicated().any():
        raise AssertionError(f"duplicate success snapshot for {id_column}")
    return output


def last_component_snapshot(source: pd.DataFrame) -> pd.DataFrame:
    if source.empty:
        return pd.DataFrame(
            columns=["id", "N_anchor", "C_reverse", "C_middle"]
        )
    last = source.sort_values(
        ["pitcher_id", "asof_pitcher_n"], kind="mergesort"
    ).groupby("pitcher_id", sort=False).tail(1)
    n_last = last["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    output = pd.DataFrame(
        {"id": last["pitcher_id"].to_numpy(), "N_anchor": n_last}
    )
    for name, rate_column in COMPONENT_RATES.items():
        rate = last[rate_column].fillna(0).to_numpy(np.float64)
        output[f"C_{name}"] = np.round(rate * n_last)
    if output["id"].duplicated().any():
        raise AssertionError("duplicate component snapshot")
    return output


def merge_snapshot(
    rows: pd.DataFrame, id_column: str, snapshot: pd.DataFrame
) -> pd.DataFrame:
    key = pd.DataFrame(
        {
            "_order": np.arange(len(rows), dtype=np.int64),
            "id": rows[id_column].to_numpy(),
        }
    )
    merged = key.merge(snapshot, on="id", how="left", validate="many_to_one")
    return merged.sort_values("_order", kind="stable").reset_index(drop=True)


def attach_hierarchical(
    rows: pd.DataFrame,
    source: pd.DataFrame,
    volatility_scale: float,
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach only train-derived, row-local hierarchical state."""

    rows = rows.reset_index(drop=True)
    p_snapshot = last_success_snapshot(
        source,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    )
    b_snapshot = last_success_snapshot(
        source,
        "batter_id",
        "asof_batter_n",
        "asof_batter_success_rate",
    )
    component_snapshot = last_component_snapshot(source)
    p_state = merge_snapshot(rows, "pitcher_id", p_snapshot)
    b_state = merge_snapshot(rows, "batter_id", b_snapshot)
    component_state = merge_snapshot(rows, "pitcher_id", component_snapshot)

    prior_map, prior_global = game_prior(source, "control_success")
    row_game = rows["game_type"].astype(str)
    mu = row_game.map(prior_map).fillna(prior_global).to_numpy(np.float64)
    mu = np.clip(mu, EPS, 1.0 - EPS)

    p_n_now = np.rint(
        pd.to_numeric(rows["asof_pitcher_n"], errors="coerce")
        .fillna(0)
        .to_numpy(np.float64)
    )
    b_n_now = np.rint(
        pd.to_numeric(rows["asof_batter_n"], errors="coerce")
        .fillna(0)
        .to_numpy(np.float64)
    )
    p_rate_now = pd.to_numeric(
        rows["asof_pitcher_success_rate"], errors="coerce"
    ).to_numpy(np.float64)
    b_rate_now = pd.to_numeric(
        rows["asof_batter_success_rate"], errors="coerce"
    ).to_numpy(np.float64)
    p_rate_now = np.where(np.isfinite(p_rate_now), p_rate_now, mu)
    b_rate_now = np.where(np.isfinite(b_rate_now), b_rate_now, mu)

    p_seen = p_state["N_end"].notna().to_numpy()
    b_seen = b_state["N_end"].notna().to_numpy()
    p_n_end = p_state["N_end"].fillna(0).to_numpy(np.float64)
    b_n_end = b_state["N_end"].fillna(0).to_numpy(np.float64)
    p_s_end = p_state["S_end"].fillna(0).to_numpy(np.float64)
    b_s_end = b_state["S_end"].fillna(0).to_numpy(np.float64)

    p_window_ok = (~p_seen) | (p_n_now >= p_n_end - 1e-9)
    b_window_ok = (~b_seen) | (b_n_now >= b_n_end - 1e-9)
    p_n_season = np.where(p_window_ok, np.maximum(p_n_now - p_n_end, 0.0), 0.0)
    b_n_season = np.where(b_window_ok, np.maximum(b_n_now - b_n_end, 0.0), 0.0)
    p_s_season_raw = np.round(p_rate_now * p_n_now) - p_s_end
    b_s_season_raw = np.round(b_rate_now * b_n_now) - b_s_end

    p_count_bad = p_window_ok & (
        (p_s_season_raw < -1e-9) | (p_s_season_raw > p_n_season + 1e-9)
    )
    b_count_bad = b_window_ok & (
        (b_s_season_raw < -1e-9) | (b_s_season_raw > b_n_season + 1e-9)
    )
    if p_count_bad.any() or b_count_bad.any():
        raise ValueError(
            f"{label}: invalid exact success counts pitcher={int(p_count_bad.sum())} "
            f"batter={int(b_count_bad.sum())}"
        )
    p_s_season = np.clip(p_s_season_raw, 0.0, p_n_season)
    b_s_season = np.clip(b_s_season_raw, 0.0, b_n_season)

    volatility, volatility_missing = row_volatility(rows)
    volatility_filled = np.where(
        volatility_missing, 3.0 * volatility_scale, volatility
    )
    volatility_ratio = np.clip(volatility_filled / volatility_scale, 0.0, 3.0)
    k_pitcher = K_SEASON_BASE * (1.0 + volatility_ratio)

    p_career = (p_s_end + K_CAREER * mu) / (p_n_end + K_CAREER)
    b_career = (b_s_end + K_CAREER * mu) / (b_n_end + K_CAREER)
    p_post = (p_s_season + k_pitcher * p_career) / (p_n_season + k_pitcher)
    b_post = (b_s_season + K_SEASON_BASE * b_career) / (
        b_n_season + K_SEASON_BASE
    )
    p_rel = p_n_season / (p_n_season + k_pitcher)
    b_rel = b_n_season / (b_n_season + K_SEASON_BASE)
    p_sd = np.sqrt(
        np.clip(p_post * (1.0 - p_post), 0.0, None)
        / (p_n_season + k_pitcher + 1.0)
    )
    b_sd = np.sqrt(
        np.clip(b_post * (1.0 - b_post), 0.0, None)
        / (b_n_season + K_SEASON_BASE + 1.0)
    )
    p0 = np.clip(
        PITCHER_WEIGHT * p_post + BATTER_WEIGHT * b_post, EPS, 1.0 - EPS
    )
    p0_sd = np.sqrt(
        PITCHER_WEIGHT**2 * p_sd**2 + BATTER_WEIGHT**2 * b_sd**2
    )

    component_seen = component_state["N_anchor"].notna().to_numpy()
    component_anchor_n = component_state["N_anchor"].fillna(0).to_numpy(np.float64)
    bridge_window_ok = (~component_seen) | (p_n_now >= component_anchor_n - 1e-9)
    bridge_n = np.where(
        bridge_window_ok, np.maximum(p_n_now - component_anchor_n, 0.0), 0.0
    )
    component_diag: dict[str, Any] = {}
    component_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, rate_column in COMPONENT_RATES.items():
        current_rate = (
            pd.to_numeric(rows[rate_column], errors="coerce")
            .fillna(0)
            .to_numpy(np.float64)
        )
        anchor_count = component_state[f"C_{name}"].fillna(0).to_numpy(np.float64)
        bridge_count_raw = np.round(current_rate * p_n_now) - anchor_count
        bad = bridge_window_ok & (
            (bridge_count_raw < -1e-9) | (bridge_count_raw > bridge_n + 1e-9)
        )
        component_diag[name] = {
            "negative_window_rows": int((~bridge_window_ok).sum()),
            "invalid_count_rows": int(bad.sum()),
            "bridge_n_min": float(np.min(bridge_n)) if len(bridge_n) else None,
            "bridge_n_max": float(np.max(bridge_n)) if len(bridge_n) else None,
            "count_min": float(np.min(bridge_count_raw)) if len(bridge_count_raw) else None,
            "count_max": float(np.max(bridge_count_raw)) if len(bridge_count_raw) else None,
        }
        if bad.any():
            raise ValueError(f"{label}: invalid {name} bridge counts={int(bad.sum())}")
        bridge_count = np.clip(bridge_count_raw, 0.0, bridge_n)
        raw = np.full(len(rows), np.nan, dtype=np.float64)
        positive = bridge_n > 0
        raw[positive] = bridge_count[positive] / bridge_n[positive]

        component_prior_map, component_prior_global = game_prior(
            source, COMPONENT_LABELS[name]
        )
        component_mu = row_game.map(component_prior_map).fillna(
            component_prior_global
        ).to_numpy(np.float64)
        component_mu = np.clip(component_mu, EPS, 1.0 - EPS)
        component_career = (
            anchor_count + K_CAREER * component_mu
        ) / (component_anchor_n + K_CAREER)
        component_post = (
            bridge_count + k_pitcher * component_career
        ) / (bridge_n + k_pitcher)
        component_values[name] = (raw, component_post)

    features = pd.DataFrame(
        {
            "h_game_prior": mu,
            "h_p_career": p_career,
            "h_p_season_logn": np.log1p(p_n_season),
            "h_p_success_raw": np.where(
                p_n_season > 0, p_s_season / np.maximum(p_n_season, 1.0), np.nan
            ),
            "h_p_success_post": p_post,
            "h_p_success_rel": p_rel,
            "h_p_success_sd": p_sd,
            "h_p_recent_vol": volatility_filled,
            "h_p_recent_vol_missing": volatility_missing.astype(np.int8),
            "h_p_kseason_log": np.log1p(k_pitcher),
            "h_p_bridge_reverse_raw": component_values["reverse"][0],
            "h_p_bridge_reverse_post": component_values["reverse"][1],
            "h_p_bridge_middle_raw": component_values["middle"][0],
            "h_p_bridge_middle_post": component_values["middle"][1],
            "h_b_career": b_career,
            "h_b_season_logn": np.log1p(b_n_season),
            "h_b_success_raw": np.where(
                b_n_season > 0, b_s_season / np.maximum(b_n_season, 1.0), np.nan
            ),
            "h_b_success_post": b_post,
            "h_b_success_rel": b_rel,
            "h_b_success_sd": b_sd,
            "h_base": p0,
            "h_base_sd": p0_sd,
        }
    ).astype({name: np.float32 for name in HIER_FEATURES})
    if features[HIER_FEATURES].isna().all(axis=0).any():
        missing_columns = features.columns[features.isna().all(axis=0)].tolist()
        raise AssertionError(f"{label}: all-missing hierarchical features {missing_columns}")
    diagnostics = {
        "label": label,
        "rows": int(len(rows)),
        "pitcher_seen_rate": float(p_seen.mean()),
        "batter_seen_rate": float(b_seen.mean()),
        "pitcher_negative_window_rows": int((~p_window_ok).sum()),
        "batter_negative_window_rows": int((~b_window_ok).sum()),
        "pitcher_invalid_success_rows": int(p_count_bad.sum()),
        "batter_invalid_success_rows": int(b_count_bad.sum()),
        "pitcher_season_n_min": float(np.min(p_n_season)),
        "pitcher_season_n_max": float(np.max(p_n_season)),
        "volatility_missing_rate": float(volatility_missing.mean()),
        "p0_min": float(np.min(p0)),
        "p0_max": float(np.max(p0)),
        "p0_mean": float(np.mean(p0)),
        "components": component_diag,
    }
    return features, diagnostics


def attach_existing_features(
    sampled_train_raw: pd.DataFrame,
    sampled_valid_raw: pd.DataFrame,
    full_history: pd.DataFrame,
    base_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prior = float(full_history["control_success"].mean())
    pfb_rows = full_history.dropna(subset=["lab_fb"])
    if IS_SMOKE and len(pfb_rows) > SMOKE_PFB_ROWS:
        pfb_rows = pfb_rows.sample(SMOKE_PFB_ROWS, random_state=127)
    pfb = CatBoostClassifier(
        iterations=3 if IS_SMOKE else 300,
        depth=6,
        learning_rate=0.1,
        verbose=False,
        thread_count=ARGS.threads,
        allow_writing_files=False,
        random_seed=42,
    )
    pfb.fit(
        pfb_rows[S12.PFB_IN].fillna(-999),
        pfb_rows["lab_fb"].astype(int),
    )
    mix_train = mix_asof_train(full_history)
    mix_valid = mix_career(full_history)

    train_parts: list[pd.DataFrame] = []
    for season in sorted(sampled_train_raw["season"].unique()):
        source = full_history[full_history["season"] < season]
        rows = sampled_train_raw[sampled_train_raw["season"] == season]
        pitcher_const = build_const(
            source, "pitcher_id", "asof_pitcher_n", S12.P_RATES
        )
        batter_const = build_const(
            source, "batter_id", "asof_batter_n", S12.B_RATES
        )
        train_parts.append(S12.attach_cs(rows, pitcher_const, batter_const))
    train = pd.concat(train_parts).sort_index(kind="stable")
    train = S12.add_features(train, prior)
    train_key = train[["pitcher_id", "season", "_cg"]].merge(
        mix_train,
        on=["pitcher_id", "season", "_cg"],
        how="left",
        validate="many_to_one",
    )
    train["f_mixcg_fb"] = train_key["mix_fb"].to_numpy(np.float32)
    train["f_mixcg_brk"] = train_key["mix_brk"].to_numpy(np.float32)
    train["f_pfb"] = pfb.predict_proba(train[S12.PFB_IN].fillna(-999))[
        :, 1
    ].astype(np.float32)

    pitcher_const = build_const(
        full_history, "pitcher_id", "asof_pitcher_n", S12.P_RATES
    )
    batter_const = build_const(
        full_history, "batter_id", "asof_batter_n", S12.B_RATES
    )
    valid = S12.attach_cs(sampled_valid_raw, pitcher_const, batter_const)
    valid = S12.add_features(valid, prior)
    valid_key = valid[["pitcher_id", "_cg"]].merge(
        mix_valid,
        on=["pitcher_id", "_cg"],
        how="left",
        validate="many_to_one",
    )
    valid["f_mixcg_fb"] = valid_key["mix_fb"].to_numpy(np.float32)
    valid["f_mixcg_brk"] = valid_key["mix_brk"].to_numpy(np.float32)
    valid["f_pfb"] = pfb.predict_proba(valid[S12.PFB_IN].fillna(-999))[
        :, 1
    ].astype(np.float32)

    existing_features = base_columns + ENG18 + list(S12.CS_FEATS) + list(S12.PT_FEATS)
    if len(existing_features) != 79 or len(set(existing_features)) != 79:
        raise AssertionError(f"existing feature contract changed: {len(existing_features)}")
    direct_train_same = (
        pd.to_numeric(train["pitcher_hand"], errors="coerce")
        == pd.to_numeric(train["batter_hand"], errors="coerce")
    ).astype(np.int8)
    direct_valid_same = (
        pd.to_numeric(valid["pitcher_hand"], errors="coerce")
        == pd.to_numeric(valid["batter_hand"], errors="coerce")
    ).astype(np.int8)
    if not np.array_equal(direct_train_same.to_numpy(), train["f_same_hand"].to_numpy()):
        raise AssertionError("training f_same_hand does not match numeric hands")
    if not np.array_equal(direct_valid_same.to_numpy(), valid["f_same_hand"].to_numpy()):
        raise AssertionError("validation f_same_hand does not match numeric hands")
    train_same_rate = float(train["f_same_hand"].mean())
    valid_same_rate = float(valid["f_same_hand"].mean())
    if not 0.30 <= train_same_rate <= 0.75 or not 0.30 <= valid_same_rate <= 0.75:
        raise AssertionError(
            f"same-hand QA failed train={train_same_rate} valid={valid_same_rate}"
        )
    diagnostics = {
        "existing_feature_count": len(existing_features),
        "train_same_hand_rate": train_same_rate,
        "valid_same_hand_rate": valid_same_rate,
        "pfb_fit_rows": int(len(pfb_rows)),
        "pfb_iterations": 3 if IS_SMOKE else 300,
        "train_mix_coverage": float(train["f_mixcg_fb"].notna().mean()),
        "valid_mix_coverage": float(valid["f_mixcg_fb"].notna().mean()),
    }
    return train, valid, diagnostics


def model_frame(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    output = frame[[name for name in feature_names if name not in CAT_FEATURES]].copy()
    for name in CAT_FEATURES:
        output[name] = (
            frame[name]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )
    return output[feature_names]


def atomic_joblib_dump(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def build_cache(
    scope: pd.DataFrame,
    nrows: int,
    train_positions: np.ndarray,
    valid_positions: np.ndarray,
    valid_local_positions: np.ndarray,
) -> dict[str, Any]:
    tick("load strict frame through 2023 (no 2024 targets)")
    frame, base_columns = load_strict_frame(nrows)
    if not np.array_equal(frame["row_id"].astype(str), scope["row_id"].astype(str)):
        raise AssertionError("minimal/full row-id order mismatch")
    history = frame[frame["season"] <= TRAIN_END].copy()
    validation = frame[frame["season"] == YEAR].copy()
    if len(validation) != len(np.flatnonzero(scope["season"].to_numpy() == YEAR)):
        raise AssertionError("validation row count mismatch")

    tick("recover train-only pitch labels and force numeric hand representation")
    history = recover_pitch_labels(history)
    validation = add_count_group(validation)
    for local in (history, validation):
        local["pitcher_hand"] = pd.to_numeric(local["pitcher_hand"], errors="coerce")
        local["batter_hand"] = pd.to_numeric(local["batter_hand"], errors="coerce")

    train_scope_positions = np.flatnonzero(scope["season"].to_numpy() <= TRAIN_END)
    train_local_positions = np.searchsorted(train_scope_positions, train_positions)
    if not np.array_equal(train_scope_positions[train_local_positions], train_positions):
        raise AssertionError("training global/local position mapping failed")
    sampled_train_raw = history.iloc[train_local_positions].copy()
    sampled_valid_raw = validation.iloc[valid_local_positions].copy()
    sampled_train_raw.index = train_positions
    sampled_valid_raw.index = valid_positions

    volatility, volatility_missing = row_volatility(history)
    finite = volatility[~volatility_missing & np.isfinite(volatility)]
    if len(finite) < 1000:
        raise AssertionError("too few finite volatility rows")
    volatility_scale = max(float(np.quantile(finite, 0.75)), 0.01)
    tick(f"volatility scale frozen from train-only Q75={volatility_scale:.6f}")

    tick(
        "attach accepted 79-feature contract to 20k/20k smoke rows"
        if IS_SMOKE
        else "attach accepted 79-feature contract to all fold rows"
    )
    train_enriched, valid_enriched, existing_diag = attach_existing_features(
        sampled_train_raw,
        sampled_valid_raw,
        history,
        base_columns,
    )
    tick("attach dynamic hierarchical state and integer bridge features")
    train_hier_parts: list[pd.DataFrame] = []
    train_hier_diag: list[dict[str, Any]] = []
    for season in sorted(sampled_train_raw["season"].unique()):
        mask = sampled_train_raw["season"].to_numpy() == season
        rows = sampled_train_raw.loc[mask]
        source = history[history["season"] < season]
        features, diagnostics = attach_hierarchical(
            rows, source, volatility_scale, f"train-season-{int(season)}"
        )
        features.index = rows.index
        train_hier_parts.append(features)
        train_hier_diag.append(diagnostics)
    train_hier = pd.concat(train_hier_parts).sort_index(kind="stable")
    valid_hier, valid_hier_diag = attach_hierarchical(
        sampled_valid_raw,
        history,
        volatility_scale,
        "valid-2023",
    )
    valid_hier.index = sampled_valid_raw.index
    train_enriched = train_enriched.sort_index(kind="stable")
    valid_enriched = valid_enriched.sort_index(kind="stable")
    if not np.array_equal(train_enriched.index, train_hier.index):
        raise AssertionError("training hierarchy alignment failed")
    if not np.array_equal(valid_enriched.index, valid_hier.index):
        raise AssertionError("validation hierarchy alignment failed")
    for name in HIER_FEATURES:
        train_enriched[name] = train_hier[name].to_numpy()
        valid_enriched[name] = valid_hier[name].to_numpy()

    feature_names = base_columns + ENG18 + list(S12.CS_FEATS) + list(S12.PT_FEATS) + HIER_FEATURES
    if len(feature_names) != 101 or len(set(feature_names)) != 101:
        raise AssertionError(f"full feature contract changed: {len(feature_names)}")
    train_x = model_frame(train_enriched, feature_names)
    valid_x = model_frame(valid_enriched, feature_names)
    train_y = sampled_train_raw["control_success"].to_numpy(np.float64)
    valid_y = sampled_valid_raw["control_success"].to_numpy(np.float64)
    train_p0 = train_hier["h_base"].to_numpy(np.float64)
    valid_p0 = valid_hier["h_base"].to_numpy(np.float64)
    if not np.isfinite(train_p0).all() or not np.isfinite(valid_p0).all():
        raise AssertionError("non-finite hierarchical base")
    if not ((train_p0 >= EPS) & (train_p0 <= 1.0 - EPS)).all():
        raise AssertionError("training p0 outside probability bounds")
    if not ((valid_p0 >= EPS) & (valid_p0 <= 1.0 - EPS)).all():
        raise AssertionError("validation p0 outside probability bounds")

    train_ids = scope.iloc[train_positions]["row_id"]
    valid_ids = scope.iloc[valid_positions]["row_id"]
    source_identity = {
        "path": str(DATA_PATH),
        "size": int(DATA_PATH.stat().st_size),
        "mtime_ns": int(DATA_PATH.stat().st_mtime_ns),
        "scope_rows": int(nrows),
        "scope_row_id_fingerprint": ordered_value_fingerprint(scope["row_id"]),
        "train_row_id_fingerprint": ordered_value_fingerprint(train_ids),
        "valid_row_id_fingerprint": ordered_value_fingerprint(valid_ids),
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_identity": source_identity,
        "feature_names": feature_names,
        "schema_fingerprint": schema_fingerprint(feature_names),
        "cat_features": CAT_FEATURES,
        "train_positions": train_positions,
        "valid_positions": valid_positions,
        "valid_local_positions": valid_local_positions,
        "train_x": train_x,
        "valid_x": valid_x,
        "train_y": train_y,
        "valid_y": valid_y,
        "train_p0": train_p0,
        "valid_p0": valid_p0,
        "valid_game_type": sampled_valid_raw["game_type"].astype(str).to_numpy(),
        "valid_pitcher_id": sampled_valid_raw["pitcher_id"].to_numpy(),
        "volatility_scale": volatility_scale,
        "diagnostics": {
            "existing": existing_diag,
            "hierarchical_train": train_hier_diag,
            "hierarchical_valid": valid_hier_diag,
            "train_season_counts": {
                str(int(key)): int(value)
                for key, value in sampled_train_raw["season"].value_counts().sort_index().items()
            },
            "valid_rows": int(len(sampled_valid_raw)),
        },
    }
    tick(f"write resumable prepared-feature cache {CACHE_PATH.name}")
    atomic_joblib_dump(payload, CACHE_PATH)
    write_json(
        CACHE_META_PATH,
        {
            "schema_version": payload["schema_version"],
            "source_identity": source_identity,
            "schema_fingerprint": payload["schema_fingerprint"],
            "feature_names": feature_names,
            "cat_features": CAT_FEATURES,
            "train_positions_sha256": hashlib.sha256(
                train_positions.astype("<i8").tobytes()
            ).hexdigest(),
            "valid_positions_sha256": hashlib.sha256(
                valid_positions.astype("<i8").tobytes()
            ).hexdigest(),
            "diagnostics": payload["diagnostics"],
        },
    )
    del frame, history, validation, train_enriched, valid_enriched
    gc.collect()
    return payload


def validate_cached_payload(
    payload: dict[str, Any],
    scope: pd.DataFrame,
    nrows: int,
    train_positions: np.ndarray,
    valid_positions: np.ndarray,
    valid_local_positions: np.ndarray,
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("prepared cache schema version mismatch")
    if payload.get("schema_fingerprint") != schema_fingerprint(payload["feature_names"]):
        raise ValueError("prepared cache feature schema fingerprint mismatch")
    if list(payload.get("cat_features", [])) != CAT_FEATURES:
        raise ValueError("prepared cache categorical schema mismatch")
    for name, expected in (
        ("train_positions", train_positions),
        ("valid_positions", valid_positions),
        ("valid_local_positions", valid_local_positions),
    ):
        if not np.array_equal(np.asarray(payload[name]), expected):
            raise ValueError(f"prepared cache {name} mismatch")
    identity = payload["source_identity"]
    if int(identity["size"]) != int(DATA_PATH.stat().st_size):
        raise ValueError("prepared cache source size mismatch")
    if int(identity["mtime_ns"]) != int(DATA_PATH.stat().st_mtime_ns):
        raise ValueError("prepared cache source mtime mismatch")
    if int(identity["scope_rows"]) != nrows:
        raise ValueError("prepared cache source row count mismatch")
    current_scope_hash = ordered_value_fingerprint(scope["row_id"])
    current_train_hash = ordered_value_fingerprint(scope.iloc[train_positions]["row_id"])
    current_valid_hash = ordered_value_fingerprint(scope.iloc[valid_positions]["row_id"])
    if identity["scope_row_id_fingerprint"] != current_scope_hash:
        raise ValueError("prepared cache scope row-id/order fingerprint mismatch")
    if identity["train_row_id_fingerprint"] != current_train_hash:
        raise ValueError("prepared cache training row-id/order fingerprint mismatch")
    if identity["valid_row_id_fingerprint"] != current_valid_hash:
        raise ValueError("prepared cache validation row-id/order fingerprint mismatch")
    if len(payload["train_x"]) != len(train_positions):
        raise ValueError("prepared cache training row count mismatch")
    if len(payload["valid_x"]) != len(valid_positions):
        raise ValueError("prepared cache validation row count mismatch")


def load_or_build_cache() -> tuple[dict[str, Any], bool]:
    scope, nrows = load_scope_index()
    train_positions, valid_positions, valid_local_positions = selected_positions(scope)
    if CACHE_PATH.is_file() and not ARGS.rebuild_cache:
        tick(f"load prepared-feature cache {CACHE_PATH.name}")
        payload = joblib.load(CACHE_PATH)
        validate_cached_payload(
            payload,
            scope,
            nrows,
            train_positions,
            valid_positions,
            valid_local_positions,
        )
        tick("cache row-id/order/schema validation passed")
        return payload, True
    payload = build_cache(
        scope,
        nrows,
        train_positions,
        valid_positions,
        valid_local_positions,
    )
    validate_cached_payload(
        payload,
        scope,
        nrows,
        train_positions,
        valid_positions,
        valid_local_positions,
    )
    return payload, False


def raw_score(probability: np.ndarray, target: np.ndarray) -> float:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rate = float(target.mean())
    denominator = rate * (1.0 - rate)
    return float(
        100000.0
        * (1.0 - float(np.mean((probability - target) ** 2)) / denominator)
    )


def equal_mean_gain(
    base: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    *,
    clip: bool,
) -> float:
    shifted = candidate - float(np.mean(candidate)) + float(np.mean(base))
    if clip:
        shifted = np.clip(shifted, EPS, 1.0 - EPS)
    return raw_score(shifted, target) - raw_score(base, target)


def domain_contribution(
    base: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    target = np.asarray(target, np.float64)
    rate = float(target.mean())
    row_gain = (base - target) ** 2 - (candidate - target) ** 2
    return float(
        100000.0
        * float(np.sum(row_gain[np.asarray(mask, bool)]))
        / (len(target) * rate * (1.0 - rate))
    )


def blend_geometry(
    base: np.ndarray, candidate: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    base = np.asarray(base, np.float64)
    candidate = np.asarray(candidate, np.float64)
    target = np.asarray(target, np.float64)
    rate = float(target.mean())
    denominator = rate * (1.0 - rate)
    base_score = raw_score(base, target)
    candidate_score = raw_score(candidate, target)
    d = candidate_score - base_score
    k = float(100000.0 * np.mean((candidate - base) ** 2) / denominator)
    if k <= 1e-15:
        weight = 1.0 if d > 0.0 else 0.0
    else:
        weight = float(np.clip((d + k) / (2.0 * k), 0.0, 1.0))
    blended = np.clip(base + weight * (candidate - base), EPS, 1.0 - EPS)
    return {
        "base_score": base_score,
        "candidate_score": candidate_score,
        "d": d,
        "K": k,
        "d_plus_K": d + k,
        "optimal_weight_candidate": weight,
        "analytic_gain": raw_score(blended, target) - base_score,
        # The historical discovery gate uses the unconstrained affine-shape
        # diagnostic.  Report the probability-clipped variant separately so
        # the semantics cannot be mistaken for each other.
        "equal_mean_gain": equal_mean_gain(base, blended, target, clip=False),
        "equal_mean_unclipped_gain": equal_mean_gain(
            base, blended, target, clip=False
        ),
        "equal_mean_clipped_gain": equal_mean_gain(
            base, blended, target, clip=True
        ),
        "equal_mean_semantics": (
            "candidate shifted to exact baseline mean; primary is not re-clipped; "
            "clipped probability variant also reported"
        ),
    }


def pitcher_bootstrap(
    base: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    pitcher_id: np.ndarray,
    seed: int,
    draws: int = 300,
) -> dict[str, float]:
    target = np.asarray(target, np.float64)
    rate = float(target.mean())
    scale = 100000.0 / (len(target) * rate * (1.0 - rate))
    row_gain = scale * ((base - target) ** 2 - (candidate - target) ** 2)
    grouped = (
        pd.DataFrame({"pitcher_id": pitcher_id, "gain": row_gain})
        .groupby("pitcher_id", sort=False)["gain"]
        .sum()
        .to_numpy(np.float64)
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 100):
        size = min(100, draws - start)
        indices = rng.integers(0, len(grouped), size=(size, len(grouped)))
        samples[start : start + size] = grouped[indices].sum(axis=1)
    quantiles = np.quantile(samples, [0.025, 0.5, 0.975])
    return {
        "pitchers": int(len(grouped)),
        "draws": int(draws),
        "p025": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p975": float(quantiles[2]),
        "prob_positive": float(np.mean(samples > 0.0)),
    }


def model_contract(payload: dict[str, Any]) -> dict[str, Any]:
    params = {
        "loss_function": "RMSE",
        "iterations": 20 if IS_SMOKE else 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 20.0,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.0,
        "verbose": False,
        "allow_writing_files": False,
        "random_seed": int(ARGS.seed),
        "thread_count": int(ARGS.threads),
    }
    stable_params = {key: value for key, value in params.items() if key != "thread_count"}
    digest = hashlib.sha256(
        json.dumps(
            {
                "schema": payload["schema_fingerprint"],
                "train_rows": payload["source_identity"]["train_row_id_fingerprint"],
                "params": stable_params,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {"contract_sha256": digest, "params": params, "stable_params": stable_params}


def load_or_train_model(
    payload: dict[str, Any]
) -> tuple[CatBoostRegressor, bool, dict[str, Any]]:
    contract = model_contract(payload)
    if MODEL_PATH.is_file() and MODEL_META_PATH.is_file() and not ARGS.retrain:
        metadata = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
        if metadata.get("contract_sha256") == contract["contract_sha256"]:
            model = CatBoostRegressor()
            model.load_model(MODEL_PATH)
            if list(model.feature_names_) != list(payload["feature_names"]):
                raise AssertionError("cached model feature-name/order mismatch")
            tick(f"loaded resumable {MODE} model {MODEL_PATH.name}")
            return model, True, contract
        tick(f"cached {MODE} model contract mismatch; retraining")

    residual = np.asarray(payload["train_y"], np.float64) - np.asarray(
        payload["train_p0"], np.float64
    )
    train_pool = Pool(
        payload["train_x"],
        residual,
        cat_features=payload["cat_features"],
    )
    tick(
        "fit authorised 20-tree CatBoostRegressor smoke"
        if IS_SMOKE
        else "fit reviewed 500-tree CatBoostRegressor full discovery"
    )
    model = CatBoostRegressor(**contract["params"])
    model.fit(train_pool)
    if list(model.feature_names_) != list(payload["feature_names"]):
        raise AssertionError("trained model feature-name/order mismatch")
    model.save_model(MODEL_PATH)
    write_json(
        MODEL_META_PATH,
        {
            "contract_sha256": contract["contract_sha256"],
            "stable_params": contract["stable_params"],
            "feature_names": payload["feature_names"],
            "schema_fingerprint": payload["schema_fingerprint"],
            "train_row_id_fingerprint": payload["source_identity"][
                "train_row_id_fingerprint"
            ],
            "smoke_only": IS_SMOKE,
            "mode": MODE,
        },
    )
    return model, False, contract


def evaluate(payload: dict[str, Any], model: CatBoostRegressor) -> dict[str, Any]:
    valid_pool = Pool(
        payload["valid_x"], cat_features=payload["cat_features"]
    )
    residual_prediction = np.asarray(model.predict(valid_pool), np.float64)
    clipped_effect = np.clip(
        residual_prediction, -GLOBAL_EFFECT_CLIP, GLOBAL_EFFECT_CLIP
    )
    candidate = np.clip(
        np.asarray(payload["valid_p0"], np.float64) + clipped_effect,
        EPS,
        1.0 - EPS,
    )
    np.save(PRED_PATH, candidate.astype(np.float32))
    np.save(P0_PATH, np.asarray(payload["valid_p0"], np.float32))

    valid_local_positions = np.asarray(payload["valid_local_positions"], np.int64)
    official_scope = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
        usecols=["row_id", "season", "control_success"],
        nrows=int(payload["source_identity"]["scope_rows"]),
        low_memory=False,
    )
    official_scope.columns = official_scope.columns.str.replace(
        "\ufeff", "", regex=False
    ).str.strip()
    official_valid = official_scope.loc[
        pd.to_numeric(official_scope["season"], errors="raise").eq(YEAR)
    ].reset_index(drop=True)
    official_target = official_valid["control_success"].to_numpy(np.float64)
    expected_full_rows = len(official_valid)

    own_base_full = np.load(OWN_BASE_PATH, allow_pickle=False).astype(np.float64)
    v18_full = np.load(V18_EFFECT_PATH, allow_pickle=False).astype(np.float64)
    exp020_full = np.load(EXP020_OOF_PATH, allow_pickle=False).astype(np.float64)
    exp020_target = np.load(EXP020_TARGET_PATH, allow_pickle=False).astype(np.float64)
    for label, value in (
        ("EXP122 base", own_base_full),
        ("V18 effect", v18_full),
        ("EXP020 rank6/s300 OOF", exp020_full),
        ("EXP020 saved target", exp020_target),
    ):
        if value.shape != (expected_full_rows,) or not np.isfinite(value).all():
            raise AssertionError(
                f"{label} shape/finite mismatch: {value.shape} vs {(expected_full_rows,)}"
            )
    if not np.array_equal(exp020_target, official_target):
        raise AssertionError("EXP020 saved target/order != exact official 2023 target/order")
    if not np.array_equal(
        official_target[valid_local_positions],
        np.asarray(payload["valid_y"], np.float64),
    ):
        raise AssertionError("cached validation target/order != exact official order")
    if not ((own_base_full >= 0.0) & (own_base_full <= 1.0)).all():
        raise AssertionError("EXP122 base outside probability range")
    if not ((exp020_full >= 0.0) & (exp020_full <= 1.0)).all():
        raise AssertionError("EXP020 OOF outside probability range")

    own_corrected_full = np.clip(
        AFFINE_CENTER
        + AFFINE_SCALE
        * (np.clip(own_base_full + V18_GAMMA * v18_full, 0.0, 1.0) - AFFINE_CENTER)
        + AFFINE_SHIFT,
        0.0,
        1.0,
    )
    baseline_full = OWN_WEIGHT * own_corrected_full + EXP020_WEIGHT * exp020_full
    acur_full_score = raw_score(baseline_full, official_target)
    if abs(acur_full_score - EXPECTED_ACUR_FULL_SCORE) > 1e-8:
        raise AssertionError(
            f"Acur exact-score assertion failed: {acur_full_score} "
            f"!= {EXPECTED_ACUR_FULL_SCORE}"
        )
    baseline = baseline_full[valid_local_positions]
    own_endpoint = own_corrected_full[valid_local_positions]
    target = np.asarray(payload["valid_y"], np.float64)
    game_type = np.asarray(payload["valid_game_type"], dtype=str)
    pitcher = np.asarray(payload["valid_pitcher_id"])

    geometry = blend_geometry(baseline, candidate, target)
    optimal_weight = geometry["optimal_weight_candidate"]
    optimal_blend = np.clip(
        baseline + optimal_weight * (candidate - baseline), EPS, 1.0 - EPS
    )
    f_mask = game_type == "F"
    result = {
        "baseline": {
            "name": "Acur exact current blend",
            "formula": (
                f"{OWN_WEIGHT:.16f}*corrected_EXP122+"
                f"{EXP020_WEIGHT:.16f}*EXP020_rank6_s300"
            ),
            "full_score_asserted": acur_full_score,
            "expected_full_score": EXPECTED_ACUR_FULL_SCORE,
            "official_valid_row_id_fingerprint": ordered_value_fingerprint(
                official_valid["row_id"]
            ),
            "official_vs_exp020_target_order_exact": True,
            "sources": {
                "exp122_base": {
                    "path": str(OWN_BASE_PATH),
                    "sha256": sha256_file(OWN_BASE_PATH),
                },
                "v18_effect": {
                    "path": str(V18_EFFECT_PATH),
                    "sha256": sha256_file(V18_EFFECT_PATH),
                },
                "exp020_rank6_s300_oof": {
                    "path": str(EXP020_OOF_PATH),
                    "sha256": sha256_file(EXP020_OOF_PATH),
                },
                "exp020_target": {
                    "path": str(EXP020_TARGET_PATH),
                    "sha256": sha256_file(EXP020_TARGET_PATH),
                },
            },
        },
        "baseline_rows_full": int(len(baseline_full)),
        "sample_rows": int(len(target)),
        "target_rate": float(target.mean()),
        "scores": {
            "baseline": raw_score(baseline, target),
            "own_endpoint_diagnostic": raw_score(own_endpoint, target),
            "hierarchical_base": raw_score(payload["valid_p0"], target),
            "candidate": raw_score(candidate, target),
        },
        "prediction": {
            "p0_mean": float(np.mean(payload["valid_p0"])),
            "candidate_mean": float(np.mean(candidate)),
            "effect_mean": float(np.mean(clipped_effect)),
            "effect_std": float(np.std(clipped_effect)),
            "effect_abs_max": float(np.max(np.abs(clipped_effect))),
            "effect_clip_rate": float(
                np.mean(np.abs(residual_prediction) > GLOBAL_EFFECT_CLIP)
            ),
        },
        "geometry": geometry,
        "optimal_blend_contributions": {
            "F": domain_contribution(
                baseline, optimal_blend, target, f_mask
            ),
            "R": domain_contribution(
                baseline, optimal_blend, target, ~f_mask
            ),
        },
        "optimal_blend_pitcher_bootstrap": pitcher_bootstrap(
            baseline,
            optimal_blend,
            target,
            pitcher,
            seed=127_000 + int(ARGS.seed),
            draws=300,
        ),
    }
    return result


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)
    for required in (
        OWN_BASE_PATH,
        V18_EFFECT_PATH,
        EXP020_OOF_PATH,
        EXP020_TARGET_PATH,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    LIVE_TXT.write_text("", encoding="utf-8")
    log(f"=== EXP-127 dynamic hierarchical residual — {MODE.upper()} DISCOVERY ===")
    row_label = (
        f"{SMOKE_ROWS}/{SMOKE_ROWS}" if IS_SMOKE else "ALL/ALL (exact fold rows)"
    )
    log(
        f"fold=train<=2022 -> valid=2023 rows={row_label} "
        f"seed={ARGS.seed} threads={ARGS.threads}"
    )
    log("primary baseline=Acur exact fixed blend; own EXP122 endpoint is diagnostic only")
    log(
        "no 2024 confirmation / no test / no EXP021 source / no F sidecar / "
        "no ZIP / no submission"
    )

    payload, cache_hit = load_or_build_cache()
    tick(
        f"prepared rows train/valid={len(payload['train_x']):,}/{len(payload['valid_x']):,} "
        f"features={len(payload['feature_names'])} cache_hit={cache_hit}"
    )
    model, model_cache_hit, contract = load_or_train_model(payload)
    tick(f"{MODE} model ready cache_hit={model_cache_hit}")
    evaluation = evaluate(payload, model)
    tick(f"{MODE} evaluation against exact Acur and blend geometry complete")

    geometry = evaluation["geometry"]
    prediction = evaluation["prediction"]
    contribution = evaluation["optimal_blend_contributions"]
    bootstrap = evaluation["optimal_blend_pitcher_bootstrap"]
    log("")
    log(
        "scores Acur/ownDiagnostic/p0/candidate="
        f"{evaluation['scores']['baseline']:.3f}/"
        f"{evaluation['scores']['own_endpoint_diagnostic']:.3f}/"
        f"{evaluation['scores']['hierarchical_base']:.3f}/"
        f"{evaluation['scores']['candidate']:.3f}"
    )
    log(
        f"geometry d={geometry['d']:+.3f} K={geometry['K']:.3f} "
        f"d+K={geometry['d_plus_K']:+.3f} "
        f"w*={geometry['optimal_weight_candidate']:.4f} "
        f"gain={geometry['analytic_gain']:+.3f} "
        f"equalMean(unclipped/clipped)="
        f"{geometry['equal_mean_unclipped_gain']:+.3f}/"
        f"{geometry['equal_mean_clipped_gain']:+.3f}"
    )
    log(
        f"blend F/R={contribution['F']:+.3f}/{contribution['R']:+.3f} "
        f"boot95=[{bootstrap['p025']:+.3f},{bootstrap['p975']:+.3f}]"
    )
    log(
        f"effect mean/std/max/clip_rate={prediction['effect_mean']:+.6f}/"
        f"{prediction['effect_std']:.6f}/{prediction['effect_abs_max']:.6f}/"
        f"{prediction['effect_clip_rate']:.6f}"
    )
    if IS_SMOKE:
        log("SMOKE ONLY: metrics are plumbing diagnostics, not a selection result.")
    else:
        log(
            "FULL 2023 DISCOVERY ONLY: do not select/tune or open 2024 confirmation "
            "until this fixed result is reviewed."
        )

    report = {
        "experiment": 127,
        "stage": 1 if IS_SMOKE else 2,
        "mode": MODE,
        "smoke_only": IS_SMOKE,
        "full_fit_authorized": not IS_SMOKE,
        "discovery_only_no_2024_confirmation": True,
        "fold": {"train_end": TRAIN_END, "valid_year": YEAR},
        "rows": {
            "train": int(len(payload["train_x"])),
            "valid": int(len(payload["valid_x"])),
        },
        "seed": int(ARGS.seed),
        "threads": int(ARGS.threads),
        "cache": {
            "path": str(CACHE_PATH),
            "hit": cache_hit,
            "schema_fingerprint": payload["schema_fingerprint"],
            "source_identity": payload["source_identity"],
        },
        "model": {
            "path": str(MODEL_PATH),
            "cache_hit": model_cache_hit,
            "contract_sha256": contract["contract_sha256"],
            "stable_params": contract["stable_params"],
        },
        "feature_count": len(payload["feature_names"]),
        "categorical_features": payload["cat_features"],
        "volatility_scale": payload["volatility_scale"],
        "diagnostics": payload["diagnostics"],
        "equal_mean_semantics": (
            "primary historical shape diagnostic shifts to Acur mean without "
            "re-clipping; clipped probability variant is also reported"
        ),
        "evaluation": evaluation,
        "elapsed_seconds": float(time.time() - STARTED),
        "prohibitions": {
            "test_read": False,
            "exp021_inference_source": False,
            "f_sidecar": False,
            "zip": False,
            "submission": False,
        },
    }
    write_json(RESULT_JSON, report)
    RESULT_TXT.write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
