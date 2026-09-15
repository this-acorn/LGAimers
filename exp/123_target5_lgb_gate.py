# -*- coding: utf-8 -*-
"""[123] One-configuration LightGBM target5 strict temporal gate.

This file tests one previously unresolved backbone question: LightGBM on the
exact current 79 columns, reconstructed five-class target, and the five
low-cardinality CAT5 fields.  It intentionally has no tuning surface.

Protocol
--------
1. Discovery: train<=2022, validate 2023, seed 42 and one fixed configuration.
2. Open 2024 exactly once, with the same configuration, only if discovery
   passes every predeclared gate.
3. Compare 2023 against the corrected exp/122 same-seed champion-space base.
   Compare 2024 against saved same-seed CAT5 probabilities after applying the
   frozen V18 gamma=.30 and deployed affine (.49, 1.06, .0066).

The exp/104 feature/label pipeline is reproduced here rather than imported.
Importing exp/104 would read the test header at module import time; this gate is
strictly train-only.  One known exp/104 typing bug is deliberately corrected:
V18 is calculated with its string keys first, then ``batter_hand`` is restored
to numeric before PFB and the 79 model features are built.  The historical
source path uses the same recent ``archive/src`` fallback as exp/104.

Fixed LightGBM configuration
----------------------------
500 trees, lr=.04, 31 leaves, min_child_samples=1000, reg_lambda=10,
feature_fraction=.8, bagging_fraction=.8, bagging_freq=1, seed=42, n_jobs=14.
There is no early stopping, parameter search, seed search, or 2024 selection.

Discovery record/open gate
--------------------------
actual raw >= 0; same-fold oracle-affine shape gain >= 15; equal-mean gain
>= 10; analytic blend gain >= 20; raw pitcher-bootstrap p2.5 > 0; and d+K > 0.
K>=250 is reported only as a diversity-moonshot diagnostic, never a hard gate.
The final moonshot flag requires consistent positive raw direction and either
2024 actual raw gain >=50 or 2024 analytic blend gain >=50.

Resumable outputs
-----------------
  lab/123_target5_lgb_y{2023,2024}_probs_seed42.npy
  lab/123_target5_lgb_y{2023,2024}_cache.json
  lab/123_target5_lgb_gate.json
  lab/123_target5_lgb_gate.txt
  lab/123_target5_lgb_gate_live.txt

Frozen baselines are ``lab/122_y2023_seed42_base_final.npy`` (already in
champion space) and ``lab/89_cat5_probs_seed42.npy`` (2024 raw five-class).

CAT5 integer mappings are fitted on training values only.  A category first
seen in validation is encoded as -1.  No test row, ZIP/bundle, submission, or
leaderboard value is read or written.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import catboost
import lightgbm
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "exp"
LAB_DIR = ROOT / "lab"
DATA_PATH = ROOT / "data" / "train.csv"
V18_PATH = EXP_DIR / "103_v18_residual.py"
S12_CANDIDATES = (
    ROOT / 'submissions/submit12_src' / "script.py",
    ROOT / "archive" / "src" / 'submissions/submit12_src' / "script.py",
)

OUT_JSON = LAB_DIR / "123_target5_lgb_gate.json"
OUT_TXT = LAB_DIR / "123_target5_lgb_gate.txt"
LIVE_TXT = LAB_DIR / "123_target5_lgb_gate_live.txt"

SEED = 42
NTHREAD = 14
N_CLASSES = 5
K_MIX = 50.0
V18_GAMMA = 0.30
AFFINE_CENTER = 0.49
AFFINE_SCALE = 1.06
AFFINE_SHIFT = 0.0066
BOOTSTRAP_DRAWS = 5000
EXPECTED_VALID_ROWS = {2023: 245_525, 2024: 253_507}
EXPECTED_TRAIN_ROWS = {2023: 976_060, 2024: 1_221_585}
EXPECTED_LABELED_TRAIN_ROWS = {2023: 975_778, 2024: 1_221_184}
EXPECTED_TARGET_RATES = {2023: 0.499957, 2024: 0.486105}
EXPECTED_BASELINE_FILE_SHA256 = {
    2024: "0122c2af82647231c3f788a8927a23958e0817bbbada5fd4f2776a52b9180258",
}
EXPECTED_EFFECT_FILE_SHA256 = {
    2023: "719392ec251d9cbb3a19e9dc01845631023bc149eb9bec90d96f877bb69f3243",
    2024: "9c6d6a55dbacc2347c44f18ab54c3aa906a8381b6fa0c3913bc46c02db2685f3",
}
CACHE_CONTRACT_VERSION = 1

LGB_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": N_CLASSES,
    "n_estimators": 500,
    "learning_rate": 0.04,
    "num_leaves": 31,
    "min_child_samples": 1000,
    "reg_lambda": 10.0,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": SEED,
    "n_jobs": NTHREAD,
    "verbosity": -1,
    # These stabilize the one declared CPU fit; they are not search arms.
    "deterministic": True,
    "force_col_wise": True,
}

DISCOVERY_GATE: dict[str, float] = {
    "raw_gain_min": 0.0,
    "oracle_shape_gain_min": 15.0,
    "equal_mean_gain_min": 10.0,
    "analytic_blend_gain_min": 20.0,
    "bootstrap_p025_min_exclusive": 0.0,
    "d_plus_K_min_exclusive": 0.0,
}

# This is the train header minus row_id and control_success.  exp/104 obtains
# the same ordered list from the test header; keeping it explicit lets exp/123
# assert the exact contract without reading any test row or header.
BASE_FEATURES = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "base_state",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
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

EXPECTED_CS_FEATURES = [
    "f_cs_p_logn",
    "f_cs_p_rate",
    "f_cs_p_delta",
    "f_cs_p_ball_d",
    "f_cs_p_strike_d",
    "f_cs_p_middle_d",
    "f_cs_p_fb_d",
    "f_cs_b_logn",
    "f_cs_b_rate",
    "f_cs_b_delta",
    "f_cs_b_middle_d",
]
EXPECTED_PT_FEATURES = ["f_mixcg_fb", "f_mixcg_brk", "f_pfb"]
CAT5 = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_team_id",
    "batter_team_id",
]
FEATURES = BASE_FEATURES + ENG18 + EXPECTED_CS_FEATURES + EXPECTED_PT_FEATURES
PLAYER_ID_COLUMNS = ["pitcher_id", "batter_id"]

BASELINE_PATHS = {
    2023: LAB_DIR / "122_y2023_seed42_base_final.npy",
    2024: LAB_DIR / "89_cat5_probs_seed42.npy",
}
EFFECT_PATHS = {
    2023: LAB_DIR / "104_v18_effect_y2023.npy",
    2024: LAB_DIR / "103_v18_effect_2024.npy",
}

STARTED = time.time()
LINES: list[str] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--discovery-only",
        action="store_true",
        help="stop after 2023 even when the fixed discovery gate passes",
    )
    return parser.parse_args()


def log(message: str = "") -> None:
    line = str(message)
    print(line, flush=True)
    LINES.append(line)
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    with LIVE_TXT.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def tick(message: str) -> None:
    log(f"  [{time.time() - STARTED:8.1f}s] {message}")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_s12_path() -> Path:
    for path in S12_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "missing submit12 feature source; checked "
        + ", ".join(str(path) for path in S12_CANDIDATES)
    )


if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))
from common import load_train, raw_score  # noqa: E402

S12_PATH = selected_s12_path()
S12 = load_module("s12_for_123", S12_PATH)
V18 = load_module("v18_for_123", V18_PATH)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def sha256_series(value: pd.Series) -> str:
    hashed = pd.util.hash_pandas_object(value, index=False, categorize=False).to_numpy(
        np.uint64
    )
    return sha256_array(hashed)


def sha256_frame(value: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(value, index=False, categorize=True).to_numpy(
        np.uint64
    )
    digest = hashlib.sha256()
    digest.update(canonical_json(list(value.columns)).encode("utf-8"))
    digest.update(canonical_json([str(dtype) for dtype in value.dtypes]).encode("utf-8"))
    digest.update(hashed.view(np.uint8))
    return digest.hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def validate_static_contract() -> dict[str, Any]:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)
    if not V18_PATH.is_file():
        raise FileNotFoundError(V18_PATH)
    if len(BASE_FEATURES) != 47 or len(ENG18) != 18 or len(FEATURES) != 79:
        raise AssertionError(
            f"79-feature partition changed: {len(BASE_FEATURES)}/{len(ENG18)}/{len(FEATURES)}"
        )
    if len(FEATURES) != len(set(FEATURES)):
        raise AssertionError("duplicate feature name")
    if list(S12.CAT) + ["pitcher_team_id", "batter_team_id"] != CAT5:
        raise AssertionError(f"CAT5 source contract changed: {S12.CAT}")
    if list(S12.CS_FEATS) != EXPECTED_CS_FEATURES:
        raise AssertionError("CS feature order changed")
    if list(S12.PT_FEATS) != EXPECTED_PT_FEATURES:
        raise AssertionError("pitch-type feature order changed")
    if any(column in CAT5 for column in PLAYER_ID_COLUMNS):
        raise AssertionError("player IDs must remain numeric, not categorical")

    header = pd.read_csv(DATA_PATH, encoding="utf-8-sig", nrows=0)
    actual = [column.replace("ï»¿", "").strip() for column in header.columns]
    expected = ["row_id"] + BASE_FEATURES + ["control_success"]
    if actual != expected:
        raise AssertionError("train header/order no longer matches exact current base features")
    return {
        "feature_count": len(FEATURES),
        "feature_order": FEATURES,
        "feature_order_sha256": sha256_bytes(canonical_json(FEATURES).encode("utf-8")),
        "categorical_features": CAT5,
        "player_ids_numeric": PLAYER_ID_COLUMNS,
        "train_header_exact": True,
        "s12_source": str(S12_PATH.relative_to(ROOT)),
        "s12_source_sha256": sha256_file(S12_PATH),
        "reads_test": False,
    }


# The next five helpers intentionally mirror exp/104 exactly.
def recover_auxiliary_labels(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["pitcher_id", "asof_pitcher_n"], kind="mergesort"
    ).reset_index()
    pitcher = ordered["pitcher_id"].to_numpy()
    count = ordered["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    next_pitch = (pitcher[1:] == pitcher[:-1]) & (
        np.abs(count[1:] - count[:-1] - 1.0) < 1e-6
    )
    for name, column in [
        ("lab_mid", "asof_pitcher_middle_rate"),
        ("lab_rev", "asof_pitcher_reverse_rate"),
        ("lab_fb", "asof_pitcher_fastball_rate"),
        ("lab_brk", "asof_pitcher_breaking_rate"),
    ]:
        cumulative = ordered[column].fillna(0).to_numpy(np.float64) * count
        label = np.full(len(ordered), np.nan)
        difference = np.round(cumulative[1:] - cumulative[:-1])
        label[:-1] = np.where(next_pitch, difference, np.nan)
        ordered[name] = np.where((label == 0.0) | (label == 1.0), label, np.nan)
    restored = ordered.set_index("index").sort_index()
    for name in ("lab_mid", "lab_rev", "lab_fb", "lab_brk"):
        frame[name] = restored[name]
    del ordered, restored, pitcher, count, next_pitch

    binary = frame["control_success"].to_numpy(np.float64)
    classes = np.full(len(frame), -1, dtype=np.int8)
    known = frame["lab_mid"].notna().to_numpy() & frame["lab_rev"].notna().to_numpy()
    middle = frame["lab_mid"].to_numpy() == 1
    reverse = frame["lab_rev"].to_numpy() == 1
    classes[known & (binary == 1)] = 0
    classes[known & (binary == 0) & middle & ~reverse] = 1
    classes[known & (binary == 0) & ~middle & reverse] = 2
    classes[known & (binary == 0) & middle & reverse] = 3
    classes[known & (binary == 0) & ~middle & ~reverse] = 4
    frame["_cls"] = classes
    balls = frame["balls_before"].to_numpy()
    strikes = frame["strikes_before"].to_numpy()
    frame["_cg"] = np.where(
        strikes > balls, 2, np.where(balls > strikes, 0, 1)
    ).astype(np.int8)
    frame["_pressure"] = V18.pressure_code(frame)
    frame["game_type"] = (
        frame["game_type"].astype("string").fillna("__MISSING__").astype(str)
    )
    frame["batter_hand"] = (
        frame["batter_hand"].astype("string").fillna("__MISSING__").astype(str)
    )
    del known, middle, reverse
    gc.collect()
    return frame


def build_const(
    source: pd.DataFrame, id_column: str, n_column: str, rates: dict[str, str]
) -> pd.DataFrame:
    last = source.sort_values(n_column).groupby(id_column).tail(1)
    output = pd.DataFrame({"id": last[id_column].to_numpy()})
    n_last = last[n_column].fillna(0).to_numpy(np.float64)
    output["N_end"] = n_last + 1.0
    for name, column in rates.items():
        rate = last[column].fillna(0).to_numpy(np.float64)
        if name == "succ":
            output[f"S_{name}"] = (
                np.round(rate * n_last)
                + last["control_success"].to_numpy(np.float64)
            )
        else:
            output[f"S_{name}"] = rate * (n_last + 1.0)
    return output


def empty_const(rates: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"id": [], "N_end": [], **{f"S_{name}": [] for name in rates}}
    )


def mix_asof_train(labeled: pd.DataFrame) -> pd.DataFrame:
    rows = labeled.dropna(subset=["lab_fb"])
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
    group_overall = overall.groupby("pitcher_id")
    for column in ("n", "fb", "brk"):
        overall[f"po_{column}"] = group_overall[column].cumsum() - overall[column]
    table = conditional.merge(
        overall[["pitcher_id", "season", "po_n", "po_fb", "po_brk"]],
        on=["pitcher_id", "season"],
    )
    over_fb = (table["po_fb"] + 0.5 * K_MIX) / (table["po_n"] + K_MIX)
    over_brk = (table["po_brk"] + 0.3 * K_MIX) / (table["po_n"] + K_MIX)
    table["mix_fb"] = np.where(
        table["po_n"] > 0,
        (table["p_fb"] + K_MIX * over_fb) / (table["p_n"] + K_MIX),
        np.nan,
    ).astype(np.float32)
    table["mix_brk"] = np.where(
        table["po_n"] > 0,
        (table["p_brk"] + K_MIX * over_brk) / (table["p_n"] + K_MIX),
        np.nan,
    ).astype(np.float32)
    return table[["pitcher_id", "season", "_cg", "mix_fb", "mix_brk"]]


def mix_career(labeled: pd.DataFrame) -> pd.DataFrame:
    rows = labeled.dropna(subset=["lab_fb"])
    conditional = (
        rows.groupby(["pitcher_id", "_cg"])
        .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
        .reset_index()
    )
    overall = (
        rows.groupby("pitcher_id")
        .agg(
            po_n=("lab_fb", "size"),
            po_fb=("lab_fb", "sum"),
            po_brk=("lab_brk", "sum"),
        )
        .reset_index()
    )
    table = conditional.merge(overall, on="pitcher_id")
    over_fb = (table["po_fb"] + 0.5 * K_MIX) / (table["po_n"] + K_MIX)
    over_brk = (table["po_brk"] + 0.3 * K_MIX) / (table["po_n"] + K_MIX)
    table["mix_fb"] = (
        (table["fb"] + K_MIX * over_fb) / (table["n"] + K_MIX)
    ).astype(np.float32)
    table["mix_brk"] = (
        (table["brk"] + K_MIX * over_brk) / (table["n"] + K_MIX)
    ).astype(np.float32)
    return table[["pitcher_id", "_cg", "mix_fb", "mix_brk"]]


def build_numeric_feature_frame(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Preserve the exact values while making non-CAT columns numeric for LGBM."""

    output = pd.DataFrame(index=data.index)
    coerced: list[str] = []
    for column in FEATURES:
        values = data[column]
        if column in CAT5:
            output[column] = values
        elif pd.api.types.is_numeric_dtype(values):
            output[column] = values
        else:
            # batter_hand must have been restored immediately after the V18
            # lookup.  Keep this conversion only as a diagnostic backstop.
            if bool(values.astype("string").eq("__MISSING__").any()):
                raise AssertionError(
                    f"non-categorical numeric-string feature has missing sentinel: {column}"
                )
            output[column] = pd.to_numeric(values, errors="raise")
            coerced.append(column)
    if list(output.columns) != FEATURES:
        raise AssertionError("feature order changed while building LGB frame")
    non_numeric = [
        column
        for column in FEATURES
        if column not in CAT5
        and not pd.api.types.is_numeric_dtype(output[column])
    ]
    if non_numeric:
        raise AssertionError(f"non-categorical non-numeric columns: {non_numeric}")
    return output, coerced


def normalize_category(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(-1).astype("int64").astype(str)
    return values.astype("string").fillna("__MISSING__").astype(str)


def encode_train_categories(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Fit CAT5 mappings on train only; validation-only values map to -1."""

    mappings: dict[str, list[str]] = {}
    diagnostics: dict[str, Any] = {}
    player_before = {
        column: {
            "train": (str(train[column].dtype), sha256_series(train[column])),
            "valid": (
                str(validation[column].dtype),
                sha256_series(validation[column]),
            ),
        }
        for column in PLAYER_ID_COLUMNS
    }
    for column in CAT5:
        train_values = normalize_category(train[column])
        valid_values = normalize_category(validation[column])
        categories = sorted(str(value) for value in pd.unique(train_values))
        if len(categories) > 255:
            raise AssertionError(
                f"{column} cardinality={len(categories)} is not low-cardinality"
            )
        mapping = {value: index for index, value in enumerate(categories)}
        train_codes = train_values.map(mapping).fillna(-1).astype(np.int32)
        valid_codes = valid_values.map(mapping).fillna(-1).astype(np.int32)
        if bool((train_codes < 0).any()):
            raise AssertionError(f"training categorical mapping produced unknown: {column}")
        train[column] = train_codes.to_numpy()
        validation[column] = valid_codes.to_numpy()
        mappings[column] = categories
        diagnostics[column] = {
            "cardinality": len(categories),
            "unknown_code": -1,
            "train_unknown": int((train_codes < 0).sum()),
            "valid_unknown": int((valid_codes < 0).sum()),
            "valid_unknown_rate": float((valid_codes < 0).mean()),
        }
    for column, split_values in player_before.items():
        for split, frame in (("train", train), ("valid", validation)):
            dtype, digest = split_values[split]
            if str(frame[column].dtype) != dtype or sha256_series(frame[column]) != digest:
                raise AssertionError(
                    f"player ID was altered by categorical encoding: {split}/{column}"
                )
            if not pd.api.types.is_numeric_dtype(frame[column]):
                raise AssertionError(f"player ID is not numeric: {split}/{column}")
    return mappings, diagnostics


def class_contract_qa(train: pd.DataFrame, mask: np.ndarray) -> dict[str, Any]:
    labeled = train.loc[mask]
    labels = labeled["_cls"].to_numpy(np.int8)
    binary = labeled["control_success"].to_numpy(np.int8)
    if set(np.unique(labels).tolist()) != set(range(N_CLASSES)):
        raise AssertionError(f"target5 classes incomplete: {np.unique(labels)}")
    if not np.array_equal(labels == 0, binary == 1):
        raise AssertionError("target5 class 0 is not exactly control_success=1")
    if not bool(np.all(binary[labels > 0] == 0)):
        raise AssertionError("a failure subclass has control_success=1")

    middle = labeled["lab_mid"].to_numpy() == 1
    reverse = labeled["lab_rev"].to_numpy() == 1
    reconstructed = np.full(len(labeled), -1, np.int8)
    reconstructed[binary == 1] = 0
    reconstructed[(binary == 0) & middle & ~reverse] = 1
    reconstructed[(binary == 0) & ~middle & reverse] = 2
    reconstructed[(binary == 0) & middle & reverse] = 3
    reconstructed[(binary == 0) & ~middle & ~reverse] = 4
    if not np.array_equal(reconstructed, labels):
        raise AssertionError("independent target5 reconstruction mismatch")
    return {
        "available_rows": int(mask.sum()),
        "unavailable_rows": int((~mask).sum()),
        "class_counts": {
            str(value): int(np.sum(labels == value)) for value in range(N_CLASSES)
        },
        "labels_sha256": sha256_array(labels),
        "class0_exact_binary_success": True,
        "reconstruction_exact": True,
    }


def effect_and_baseline_qa(
    year: int,
    history: pd.DataFrame,
    validation_rows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    baseline_path = BASELINE_PATHS[year]
    effect_path = EFFECT_PATHS[year]
    missing = [str(path) for path in (baseline_path, effect_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen comparison artifacts: {missing}")

    baseline = np.load(baseline_path, allow_pickle=False)
    baseline_file_sha256 = sha256_file(baseline_path)
    if (
        year in EXPECTED_BASELINE_FILE_SHA256
        and baseline_file_sha256 != EXPECTED_BASELINE_FILE_SHA256[year]
    ):
        raise AssertionError(
            f"frozen baseline file fingerprint changed for {year}: {baseline_file_sha256}"
        )
    if not np.isfinite(baseline).all():
        raise AssertionError("baseline contains NaN/Inf")
    if float(baseline.min()) < -1e-7 or float(baseline.max()) > 1.0 + 1e-7:
        raise AssertionError("baseline is outside probability bounds")
    baseline_final: np.ndarray | None = None
    if year == 2023:
        expected_shape = (len(validation_rows),)
        if baseline.shape != expected_shape:
            raise AssertionError(
                f"corrected exp/122 champion-space baseline shape failed: "
                f"{baseline.shape} != {expected_shape}"
            )
        baseline_final = baseline.astype(np.float64)
        baseline_space = "corrected exp/122 champion-space final"
        row_sum_error: float | None = None
    else:
        expected_shape = (len(validation_rows), N_CLASSES)
        if baseline.shape != expected_shape:
            raise AssertionError(
                f"raw CAT5 baseline row/class alignment failed "
                f"{baseline.shape} != {expected_shape}"
            )
        row_sum_error = float(np.max(np.abs(baseline.sum(axis=1) - 1.0)))
        if row_sum_error > 2e-5:
            raise AssertionError(f"baseline probability row sums failed: {row_sum_error}")
        baseline_space = "raw CAT5 seed42; pending V18+affine"

    recomputed_effect, diagnostics = V18.make_effect(history, validation_rows)
    saved_effect = np.load(effect_path, allow_pickle=False)
    effect_file_sha256 = sha256_file(effect_path)
    if effect_file_sha256 != EXPECTED_EFFECT_FILE_SHA256[year]:
        raise AssertionError(
            f"frozen V18 file fingerprint changed for {year}: {effect_file_sha256}"
        )
    if saved_effect.shape != (len(validation_rows),):
        raise AssertionError("saved V18 effect row alignment failed")
    if not np.isfinite(saved_effect).all():
        raise AssertionError("saved V18 effect contains NaN/Inf")
    maximum_difference = float(
        np.max(np.abs(saved_effect.astype(np.float64) - recomputed_effect))
    )
    if maximum_difference > 1e-7:
        raise AssertionError(f"saved/recomputed V18 effect mismatch: {maximum_difference}")
    if year == 2024:
        baseline_final = deployed_space(
            baseline[:, 0], saved_effect.astype(np.float64)
        )
        baseline_space = "saved CAT5 seed42 -> V18 gamma .30 -> deployed affine"
    if baseline_final is None:
        raise AssertionError("baseline final-space construction failed")
    baseline_qa = {
        "path": str(baseline_path.relative_to(ROOT)),
        "shape": list(baseline.shape),
        "dtype": str(baseline.dtype),
        "sha256": sha256_array(baseline),
        "file_sha256": baseline_file_sha256,
        "expected_file_sha256_exact": bool(year in EXPECTED_BASELINE_FILE_SHA256),
        "space": baseline_space,
        "already_champion_space": bool(year == 2023),
        "row_sum_max_abs_error": row_sum_error,
        "seed": SEED,
        "class0_is_success_probability": bool(year == 2024),
    }
    effect_qa = {
        "path": str(effect_path.relative_to(ROOT)),
        "shape": list(saved_effect.shape),
        "dtype": str(saved_effect.dtype),
        "sha256": sha256_array(saved_effect),
        "file_sha256": effect_file_sha256,
        "expected_file_sha256_exact": True,
        "recomputed_max_abs_difference": maximum_difference,
        "recomputed_exact_with_float32_tolerance": True,
        "diagnostics": diagnostics,
    }
    return (
        baseline_final,
        saved_effect.astype(np.float64),
        baseline_qa,
        effect_qa,
    )


def prepare_fold(year: int) -> dict[str, Any]:
    if year not in (2023, 2024):
        raise ValueError(year)
    tick(f"year={year}: load train and reconstruct exact auxiliary labels")
    frame = recover_auxiliary_labels(load_train(str(DATA_PATH)))
    if len(frame) == 0:
        raise AssertionError("empty train")
    history = frame[frame["season"] <= year - 1]
    validation_rows = frame[frame["season"] == year].reset_index(drop=True)
    if len(validation_rows) != EXPECTED_VALID_ROWS[year]:
        raise AssertionError(
            f"validation row count {len(validation_rows):,} != exact expected "
            f"{EXPECTED_VALID_ROWS[year]:,}"
        )
    expected_row_ids = validation_rows["row_id"].reset_index(drop=True)
    expected_target = validation_rows["control_success"].to_numpy(np.float64)
    if not np.isin(expected_target, [0.0, 1.0]).all():
        raise AssertionError("validation target is not complete binary data")
    if abs(float(expected_target.mean()) - EXPECTED_TARGET_RATES[year]) > 2e-6:
        raise AssertionError(
            f"validation target-rate fingerprint changed: {expected_target.mean():.9f}"
        )
    game_type = validation_rows["game_type"].to_numpy()
    if set(np.unique(game_type).tolist()) != {"F", "R"}:
        raise AssertionError(f"unexpected validation game_type values: {np.unique(game_type)}")
    pitcher = validation_rows["pitcher_id"].to_numpy()

    # Frozen artifacts are opened inside this fold only.  Thus 2024 paths are
    # not touched unless the 2023 discovery gate has already passed.
    baseline_final, effect, baseline_qa, effect_qa = effect_and_baseline_qa(
        year, history, validation_rows
    )

    # V18's categorical lookup contract uses string batter-hand keys.  The
    # model feature contract is numeric.  Restore only after the frozen effect
    # has been recomputed/checked, before PFB or s12.add_features sees a row.
    restored_batter_hand = pd.to_numeric(frame["batter_hand"], errors="coerce")
    if bool(restored_batter_hand.isna().any()):
        raise AssertionError("batter_hand could not be restored to numeric after V18")
    frame["batter_hand"] = restored_batter_hand
    history = frame[frame["season"] <= year - 1]
    validation_rows = frame[frame["season"] == year].reset_index(drop=True)
    if not validation_rows["row_id"].reset_index(drop=True).equals(expected_row_ids):
        raise AssertionError("batter_hand restoration changed validation row order")
    if not np.array_equal(
        validation_rows["control_success"].to_numpy(np.float64), expected_target
    ):
        raise AssertionError("batter_hand restoration changed validation target")
    tick(f"year={year}: V18 checked with string keys; batter_hand restored numeric")

    prior = float(history["control_success"].mean())
    pitcher_const = build_const(
        history, "pitcher_id", "asof_pitcher_n", S12.P_RATES
    )
    batter_const = build_const(
        history, "batter_id", "asof_batter_n", S12.B_RATES
    )
    pfb_rows = history.dropna(subset=["lab_fb"])
    pfb = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        verbose=False,
        thread_count=NTHREAD,
        allow_writing_files=False,
        random_seed=SEED,
    )
    pfb.fit(pfb_rows[S12.PFB_IN].fillna(-999), pfb_rows["lab_fb"].astype(int))
    del pfb_rows
    mix_train = mix_asof_train(history)
    mix_validation = mix_career(history)
    tick(f"year={year}: constant tables, pitchmix, and exact PFB ready")

    train_rows = frame[frame["season"] <= year - 1].reset_index(drop=True)
    if len(train_rows) != EXPECTED_TRAIN_ROWS[year]:
        raise AssertionError(
            f"training row count {len(train_rows):,} != exact expected "
            f"{EXPECTED_TRAIN_ROWS[year]:,}"
        )
    expected_train_row_ids = train_rows["row_id"].reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    for season in sorted(train_rows["season"].unique()):
        earlier = frame[frame["season"] <= season - 1]
        rows = train_rows[train_rows["season"] == season]
        if len(earlier) == 0:
            pitcher_season = empty_const(S12.P_RATES)
            batter_season = empty_const(S12.B_RATES)
        else:
            pitcher_season = build_const(
                earlier, "pitcher_id", "asof_pitcher_n", S12.P_RATES
            )
            batter_season = build_const(
                earlier, "batter_id", "asof_batter_n", S12.B_RATES
            )
        parts.append(S12.attach_cs(rows, pitcher_season, batter_season))
    train = S12.add_features(pd.concat(parts).sort_index(), prior)
    if not train["row_id"].reset_index(drop=True).equals(expected_train_row_ids):
        raise AssertionError("training row order changed during seasonal feature attachment")
    del parts

    key = train[["pitcher_id", "season", "_cg"]].merge(
        mix_train, on=["pitcher_id", "season", "_cg"], how="left"
    )
    train["f_mixcg_fb"] = key["mix_fb"].to_numpy(np.float32)
    train["f_mixcg_brk"] = key["mix_brk"].to_numpy(np.float32)
    train["f_pfb"] = pfb.predict_proba(train[S12.PFB_IN].fillna(-999))[:, 1].astype(
        np.float32
    )
    del key, mix_train

    validation = S12.attach_pt(
        S12.add_features(
            S12.attach_cs(validation_rows, pitcher_const, batter_const), prior
        ),
        mix_validation,
        pfb,
    )
    if not validation["row_id"].reset_index(drop=True).equals(expected_row_ids):
        raise AssertionError("validation row order changed during feature attachment")
    if not np.array_equal(
        validation["control_success"].to_numpy(np.float64), expected_target
    ):
        raise AssertionError("validation target changed during feature attachment")
    train_same_hand_rate = float(train["f_same_hand"].mean())
    valid_same_hand_rate = float(validation["f_same_hand"].mean())
    if not (0.30 <= train_same_hand_rate <= 0.75):
        raise AssertionError(f"abnormal train f_same_hand rate: {train_same_hand_rate}")
    if not (0.30 <= valid_same_hand_rate <= 0.75):
        raise AssertionError(f"abnormal valid f_same_hand rate: {valid_same_hand_rate}")
    same_hand_diagnostic = {
        "train_rate": train_same_hand_rate,
        "valid_rate": valid_same_hand_rate,
        "required_range_inclusive": [0.30, 0.75],
        "batter_hand_restored_numeric_before_model_features": True,
        "pass": True,
    }
    del pitcher_const, batter_const, mix_validation, pfb, train_rows
    gc.collect()
    tick(f"year={year}: exact 79 feature values attached")

    train_mask = train["_cls"].to_numpy() >= 0
    target5_qa = class_contract_qa(train, train_mask)
    labels = train.loc[train_mask, "_cls"].to_numpy(np.int8)
    if len(labels) != EXPECTED_LABELED_TRAIN_ROWS[year]:
        raise AssertionError(
            f"labeled target5 rows {len(labels):,} != exact expected "
            f"{EXPECTED_LABELED_TRAIN_ROWS[year]:,}"
        )
    labeled_row_ids = train.loc[train_mask, "row_id"].reset_index(drop=True)
    train_matrix, train_coerced = build_numeric_feature_frame(train.loc[train_mask])
    valid_matrix, valid_coerced = build_numeric_feature_frame(validation)
    if train_coerced != valid_coerced:
        raise AssertionError(
            f"train/valid numeric coercion mismatch: {train_coerced}/{valid_coerced}"
        )
    if train_coerced:
        raise AssertionError(
            f"model features still require numeric-string coercion: {train_coerced}"
        )
    del train, validation, history, frame
    gc.collect()

    mappings, category_qa = encode_train_categories(train_matrix, valid_matrix)
    if list(train_matrix.columns) != FEATURES or list(valid_matrix.columns) != FEATURES:
        raise AssertionError("exact 79 column order failed after categorical encoding")
    if train_matrix.shape[1] != 79 or valid_matrix.shape[1] != 79:
        raise AssertionError("feature matrix width is not 79")
    train_feature_sha = sha256_frame(train_matrix)
    valid_feature_sha = sha256_frame(valid_matrix)
    row_qa = {
        "train_rows_before_target_filter": int(len(expected_train_row_ids)),
        "expected_train_rows_before_target_filter": EXPECTED_TRAIN_ROWS[year],
        "train_rows_after_target_filter": int(len(labels)),
        "expected_train_rows_after_target_filter": EXPECTED_LABELED_TRAIN_ROWS[year],
        "valid_rows": int(len(expected_target)),
        "expected_valid_rows": EXPECTED_VALID_ROWS[year],
        "train_row_order_exact_after_features": True,
        "valid_row_order_exact_after_features": True,
        "labeled_train_row_id_sha256": sha256_series(labeled_row_ids),
        "valid_row_id_sha256": sha256_series(expected_row_ids),
    }
    target_qa = {
        "binary_exact": True,
        "rows": int(len(expected_target)),
        "mean": float(expected_target.mean()),
        "expected_mean_fingerprint": EXPECTED_TARGET_RATES[year],
        "sha256": sha256_array(expected_target),
    }
    feature_qa = {
        "count": 79,
        "order_exact": True,
        "order": FEATURES,
        "train_shape": list(train_matrix.shape),
        "valid_shape": list(valid_matrix.shape),
        "train_value_sha256": train_feature_sha,
        "valid_value_sha256": valid_feature_sha,
        "numeric_string_coercions": train_coerced,
        "corrected_same_hand_diagnostic": same_hand_diagnostic,
        "categorical_mapping_scope": "train only; validation-only values -> -1",
        "categorical_mappings": mappings,
        "categorical_diagnostics": category_qa,
        "player_ids_numeric_and_unchanged": True,
    }
    tick(
        f"year={year}: QA exact rows/features/target; "
        f"train={len(labels):,} valid={len(expected_target):,}"
    )
    return {
        "year": year,
        "train_matrix": train_matrix,
        "valid_matrix": valid_matrix,
        "labels": labels,
        "target": expected_target,
        "game_type": game_type,
        "pitcher": pitcher,
        "baseline_final": baseline_final,
        "effect": effect,
        "qa": {
            "features": feature_qa,
            "rows": row_qa,
            "binary_target": target_qa,
            "target5": target5_qa,
            "baseline": baseline_qa,
            "effect": effect_qa,
        },
    }


def prediction_path(year: int) -> Path:
    return LAB_DIR / f"123_target5_lgb_y{year}_probs_seed{SEED}.npy"


def cache_path(year: int) -> Path:
    return LAB_DIR / f"123_target5_lgb_y{year}_cache.json"


def prediction_qa(probabilities: np.ndarray, rows: int) -> dict[str, Any]:
    if probabilities.shape != (rows, N_CLASSES):
        raise AssertionError(
            f"candidate probability shape {probabilities.shape} != {(rows, N_CLASSES)}"
        )
    if not np.isfinite(probabilities).all():
        raise AssertionError("candidate probabilities contain NaN/Inf")
    minimum = float(probabilities.min())
    maximum = float(probabilities.max())
    if minimum < -1e-7 or maximum > 1.0 + 1e-7:
        raise AssertionError(f"candidate probabilities out of bounds: {minimum}/{maximum}")
    row_sum_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.0)))
    if row_sum_error > 2e-5:
        raise AssertionError(f"candidate probability row sums failed: {row_sum_error}")
    return {
        "shape": list(probabilities.shape),
        "dtype": str(probabilities.dtype),
        "min": minimum,
        "max": maximum,
        "row_sum_max_abs_error": row_sum_error,
        "sha256": sha256_array(probabilities),
    }


def cache_contract(year: int, data: dict[str, Any]) -> dict[str, Any]:
    feature_qa = data["qa"]["features"]
    row_qa = data["qa"]["rows"]
    target_qa = data["qa"]["binary_target"]
    return {
        "contract_version": CACHE_CONTRACT_VERSION,
        "experiment": 123,
        "year": year,
        "train_through": year - 1,
        "seed": SEED,
        "model_params": LGB_PARAMS,
        "features": FEATURES,
        "categorical_features": CAT5,
        "categorical_mappings": feature_qa["categorical_mappings"],
        "train_feature_sha256": feature_qa["train_value_sha256"],
        "valid_feature_sha256": feature_qa["valid_value_sha256"],
        "labeled_train_row_id_sha256": row_qa["labeled_train_row_id_sha256"],
        "valid_row_id_sha256": row_qa["valid_row_id_sha256"],
        "labels_sha256": data["qa"]["target5"]["labels_sha256"],
        "target_sha256": target_qa["sha256"],
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "s12_source_sha256": sha256_file(S12_PATH),
        "baseline_sha256": data["qa"]["baseline"]["sha256"],
        "effect_sha256": data["qa"]["effect"]["sha256"],
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "catboost": catboost.__version__,
            "lightgbm": lightgbm.__version__,
        },
    }


def load_or_train_probabilities(year: int, data: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    probabilities_path = prediction_path(year)
    metadata_path = cache_path(year)
    pending_metadata_path = metadata_path.with_name(metadata_path.name + ".pending")
    contract = cache_contract(year, data)
    signature = sha256_bytes(canonical_json(contract).encode("utf-8"))

    # Metadata is staged before the large prediction is committed.  If a
    # process dies in the one remaining two-file atomicity window, the staged
    # metadata can validate and recover the expensive prediction on restart.
    if probabilities_path.is_file() and not metadata_path.is_file():
        if pending_metadata_path.is_file():
            pending = json.loads(pending_metadata_path.read_text(encoding="utf-8"))
            pending_probability = np.load(probabilities_path, allow_pickle=False)
            pending_qa = prediction_qa(pending_probability, len(data["target"]))
            if (
                pending.get("contract_signature") == signature
                and pending.get("contract") == contract
                and pending.get("prediction_sha256") == pending_qa["sha256"]
            ):
                os.replace(pending_metadata_path, metadata_path)
                tick(f"year={year}: recovered committed prediction from staged metadata")
            else:
                raise AssertionError(
                    f"incompatible staged cache metadata: {pending_metadata_path}"
                )
        else:
            raise RuntimeError(
                f"prediction without provenance metadata: {probabilities_path}; "
                "preserve it for diagnosis before retrying"
            )
    if metadata_path.is_file() and not probabilities_path.is_file():
        raise RuntimeError(
            f"metadata without prediction: {metadata_path}; preserve it for diagnosis"
        )
    if probabilities_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("contract_signature") != signature:
            raise AssertionError(
                f"stale/incompatible cached prediction contract: {metadata_path}"
            )
        if metadata.get("contract") != contract:
            raise AssertionError(f"cached contract payload mismatch: {metadata_path}")
        probabilities = np.load(probabilities_path, allow_pickle=False)
        qa = prediction_qa(probabilities, len(data["target"]))
        if metadata.get("prediction_sha256") != qa["sha256"]:
            raise AssertionError(f"cached prediction checksum mismatch: {probabilities_path}")
        tick(f"year={year}: resumed exact cached {probabilities_path.name}")
        return probabilities.astype(np.float64), {
            "status": "resumed",
            "path": str(probabilities_path.relative_to(ROOT)),
            "metadata_path": str(metadata_path.relative_to(ROOT)),
            "contract_signature": signature,
            "prediction": qa,
        }

    tick(f"year={year}: train the single locked LightGBM configuration")
    model_started = time.time()
    model = LGBMClassifier(**LGB_PARAMS)
    model.fit(
        data["train_matrix"],
        data["labels"],
        categorical_feature=CAT5,
    )
    if not np.array_equal(np.asarray(model.classes_), np.arange(N_CLASSES)):
        raise AssertionError(f"LightGBM class order changed: {model.classes_}")
    if list(model.booster_.feature_name()) != FEATURES:
        raise AssertionError("LightGBM feature names/order mismatch")
    if int(model.n_estimators_) != int(LGB_PARAMS["n_estimators"]):
        raise AssertionError(
            f"unexpected fitted iteration count {model.n_estimators_}; early stop is forbidden"
        )
    probabilities64 = np.asarray(
        model.predict_proba(data["valid_matrix"]), dtype=np.float64
    )
    probabilities32 = probabilities64.astype(np.float32)
    qa = prediction_qa(probabilities32, len(data["target"]))
    metadata = {
        "experiment": 123,
        "year": year,
        "status": "trained",
        "contract_signature": signature,
        "contract": contract,
        "prediction_sha256": qa["sha256"],
        "prediction_shape": qa["shape"],
        "prediction_dtype": qa["dtype"],
        "fitted_iterations": int(model.n_estimators_),
        "class_order": [int(value) for value in model.classes_],
        "training_seconds": float(time.time() - model_started),
    }
    atomic_write_text(
        pending_metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_save_npy(probabilities_path, probabilities32)
    os.replace(pending_metadata_path, metadata_path)
    del model, probabilities64
    gc.collect()
    tick(
        f"year={year}: trained/saved in {metadata['training_seconds']:.1f}s "
        f"({probabilities_path.name})"
    )
    return probabilities32.astype(np.float64), {
        "status": "trained",
        "path": str(probabilities_path.relative_to(ROOT)),
        "metadata_path": str(metadata_path.relative_to(ROOT)),
        "contract_signature": signature,
        "prediction": qa,
        "training_seconds": metadata["training_seconds"],
        "fitted_iterations": metadata["fitted_iterations"],
        "class_order": metadata["class_order"],
    }


def score(probability: np.ndarray, target: np.ndarray) -> float:
    return float(raw_score(np.asarray(probability, np.float64), target))


def oracle_shape(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Same-fold free affine ceiling; it is not a deployable calibration."""

    p = np.asarray(probability, np.float64)
    y = np.asarray(target, np.float64)
    variance = float(np.var(p))
    target_variance = float(np.var(y))
    if variance <= 0.0 or target_variance <= 0.0 or not np.isfinite(variance):
        return {
            "score": float("-inf"),
            "corr": float("nan"),
            "intercept": float("nan"),
            "slope": float("nan"),
        }
    covariance = float(np.mean((p - p.mean()) * (y - y.mean())))
    slope = covariance / variance
    intercept = float(y.mean() - slope * p.mean())
    correlation = float(covariance / math.sqrt(variance * target_variance))
    return {
        "score": float(100000.0 * correlation * correlation),
        "corr": correlation,
        "intercept": intercept,
        "slope": slope,
    }


def deployed_space(probability: np.ndarray, effect: np.ndarray) -> np.ndarray:
    corrected = np.clip(
        np.asarray(probability, np.float64)
        + V18_GAMMA * np.asarray(effect, np.float64),
        0.0,
        1.0,
    )
    return np.clip(
        AFFINE_CENTER
        + AFFINE_SCALE * (corrected - AFFINE_CENTER)
        - AFFINE_SHIFT,
        0.0,
        1.0,
    )


def equal_mean_gain(base: np.ndarray, candidate: np.ndarray, target: np.ndarray) -> float:
    # Deliberately no re-clipping: this is the exact equal-mean Brier identity.
    same_mean = candidate - candidate.mean() + base.mean()
    return score(same_mean, target) - score(base, target)


def domain_contribution(
    base: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    rate = float(target.mean())
    row_gain = (base - target) ** 2 - (candidate - target) ** 2
    # Global N and global target-rate denominator make F+R exactly raw gain.
    return float(
        100000.0
        * float(np.sum(row_gain[mask]))
        / (len(target) * rate * (1.0 - rate))
    )


def blend_math(
    base: np.ndarray, candidate: np.ndarray, target: np.ndarray
) -> tuple[dict[str, float], np.ndarray]:
    base_score = score(base, target)
    candidate_score = score(candidate, target)
    d_value = candidate_score - base_score
    rate = float(target.mean())
    k_value = float(
        100000.0
        * np.mean((candidate - base) ** 2)
        / (rate * (1.0 - rate))
    )
    if k_value <= 0.0:
        weight = 0.0
    else:
        weight = float(np.clip(0.5 + d_value / (2.0 * k_value), 0.0, 1.0))
    blended = (1.0 - weight) * base + weight * candidate
    measured_gain = score(blended, target) - base_score
    formula_gain = weight * d_value + weight * (1.0 - weight) * k_value
    if not np.isclose(measured_gain, formula_gain, atol=1e-7, rtol=1e-9):
        raise AssertionError((measured_gain, formula_gain))
    return {
        "d": d_value,
        "K": k_value,
        "d_plus_K": d_value + k_value,
        "optimal_weight": weight,
        "analytic_gain": float(measured_gain),
        "formula_gain": float(formula_gain),
        "K_ge_250_diversity_moonshot_diagnostic": bool(k_value >= 250.0),
    }, blended


def summarize_bootstrap(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "p025": float(np.quantile(values, 0.025)),
        "p975": float(np.quantile(values, 0.975)),
        "prob_positive": float(np.mean(values > 0.0)),
    }


def cluster_bootstrap(
    base: np.ndarray,
    candidate: np.ndarray,
    optimal_blend: np.ndarray,
    target: np.ndarray,
    pitcher: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    if bool(pd.isna(pitcher).any()):
        raise AssertionError("pitcher bootstrap key contains missing values")
    work = pd.DataFrame(
        {
            "pitcher": pitcher,
            "n": np.ones(len(target), np.int32),
            "y": target,
            "base_se": (base - target) ** 2,
            "candidate_se": (candidate - target) ** 2,
            "blend_se": (optimal_blend - target) ** 2,
        }
    )
    grouped = work.groupby("pitcher", sort=False, observed=True).agg(
        n=("n", "sum"),
        y=("y", "sum"),
        base_se=("base_se", "sum"),
        candidate_se=("candidate_se", "sum"),
        blend_se=("blend_se", "sum"),
    )
    if int(grouped["n"].sum()) != len(target):
        raise AssertionError("pitcher grouping lost validation rows")
    values = grouped[
        ["n", "y", "base_se", "candidate_se", "blend_se"]
    ].to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    raw_gains = np.empty(BOOTSTRAP_DRAWS, np.float64)
    blend_gains = np.empty(BOOTSTRAP_DRAWS, np.float64)
    cursor = 0
    while cursor < BOOTSTRAP_DRAWS:
        width = min(200, BOOTSTRAP_DRAWS - cursor)
        indices = rng.integers(0, len(values), size=(width, len(values)))
        sampled = values[indices].sum(axis=1)
        rate = sampled[:, 1] / sampled[:, 0]
        denominator = sampled[:, 0] * rate * (1.0 - rate)
        if bool(np.any(denominator <= 0.0)):
            raise AssertionError("degenerate target rate in pitcher bootstrap")
        scale = 100000.0 / denominator
        raw_gains[cursor : cursor + width] = scale * (sampled[:, 2] - sampled[:, 3])
        blend_gains[cursor : cursor + width] = scale * (
            sampled[:, 2] - sampled[:, 4]
        )
        cursor += width
    return {
        "draws": BOOTSTRAP_DRAWS,
        "pitchers": int(len(values)),
        "sampling_unit": "pitcher_id",
        "optimal_weight_refit_inside_draw": False,
        "raw": summarize_bootstrap(raw_gains),
        "optimal_blend_fixed_weight": summarize_bootstrap(blend_gains),
    }


def evaluate_fold(
    year: int,
    baseline_final: np.ndarray,
    candidate_probs: np.ndarray,
    effect: np.ndarray,
    target: np.ndarray,
    game_type: np.ndarray,
    pitcher: np.ndarray,
) -> dict[str, Any]:
    base = np.asarray(baseline_final, np.float64)
    if base.shape != (len(target),):
        raise AssertionError(f"final-space baseline shape failed: {base.shape}")
    candidate = deployed_space(candidate_probs[:, 0], effect)
    base_shape = oracle_shape(base, target)
    candidate_shape = oracle_shape(candidate, target)
    blend, blended = blend_math(base, candidate, target)
    f_mask = game_type == "F"
    raw_gain = score(candidate, target) - score(base, target)
    f_contribution = domain_contribution(base, candidate, target, f_mask)
    r_contribution = domain_contribution(base, candidate, target, ~f_mask)
    if not np.isclose(
        f_contribution + r_contribution, raw_gain, atol=1e-7, rtol=1e-9
    ):
        raise AssertionError("F/R contributions do not sum to raw gain")
    bootstrap = cluster_bootstrap(
        base,
        candidate,
        blended,
        target,
        pitcher,
        seed=123_000 + year,
    )
    equal_mean = float(equal_mean_gain(base, candidate, target))
    result = {
        "year": year,
        "comparison_space": (
            "champion-space final; 2023 base already transformed, candidate uses "
            "clip(P0+.30*V18) then deployed affine"
            if year == 2023
            else "both use clip(P0+.30*V18) then deployed affine"
        ),
        "base": (
            "corrected exp/122 CAT5 seed42 champion-space"
            if year == 2023
            else "saved CAT5 seed42 -> V18 -> affine"
        ),
        "candidate": "fixed LightGBM seed42",
        "base_score": score(base, target),
        "candidate_score": score(candidate, target),
        "raw_gain": raw_gain,
        "base_mean": float(base.mean()),
        "candidate_mean": float(candidate.mean()),
        "mean_shift": float(candidate.mean() - base.mean()),
        "base_oracle_shape": base_shape,
        "candidate_oracle_shape": candidate_shape,
        "oracle_shape_gain": float(candidate_shape["score"] - base_shape["score"]),
        "equal_mean_gain": equal_mean,
        "center_contribution": float(raw_gain - equal_mean),
        "F_contribution": f_contribution,
        "R_contribution": r_contribution,
        "blend": blend,
        "bootstrap": bootstrap,
    }
    return result


def discovery_gate(result: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    checks = {
        "raw_gain": float(result["raw_gain"]) >= DISCOVERY_GATE["raw_gain_min"],
        "oracle_shape_gain": float(result["oracle_shape_gain"])
        >= DISCOVERY_GATE["oracle_shape_gain_min"],
        "equal_mean_gain": float(result["equal_mean_gain"])
        >= DISCOVERY_GATE["equal_mean_gain_min"],
        "analytic_blend_gain": float(result["blend"]["analytic_gain"])
        >= DISCOVERY_GATE["analytic_blend_gain_min"],
        "bootstrap_p025": float(result["bootstrap"]["raw"]["p025"])
        > DISCOVERY_GATE["bootstrap_p025_min_exclusive"],
        "d_plus_K": float(result["blend"]["d_plus_K"])
        > DISCOVERY_GATE["d_plus_K_min_exclusive"],
    }
    return bool(all(checks.values())), checks


def report_fold(result: dict[str, Any]) -> None:
    blend = result["blend"]
    raw_boot = result["bootstrap"]["raw"]
    mix_boot = result["bootstrap"]["optimal_blend_fixed_weight"]
    log(
        f"  base/candidate={result['base_score']:.2f}/{result['candidate_score']:.2f} "
        f"raw={result['raw_gain']:+.2f} oracleShape={result['oracle_shape_gain']:+.2f} "
        f"equalMean={result['equal_mean_gain']:+.2f}"
    )
    log(
        f"  d={blend['d']:+.2f} K={blend['K']:.2f} d+K={blend['d_plus_K']:+.2f} "
        f"w*={blend['optimal_weight']:.4f} blendGain={blend['analytic_gain']:+.2f} "
        f"K>=250(diag)={blend['K_ge_250_diversity_moonshot_diagnostic']}"
    )
    log(
        f"  F/R={result['F_contribution']:+.2f}/{result['R_contribution']:+.2f} "
        f"rawBoot95=[{raw_boot['p025']:+.2f},{raw_boot['p975']:+.2f}] "
        f"P>0={raw_boot['prob_positive']:.4f} "
        f"mixBoot95=[{mix_boot['p025']:+.2f},{mix_boot['p975']:+.2f}]"
    )


def save_summary(summary: dict[str, Any]) -> None:
    summary["elapsed_seconds"] = float(time.time() - STARTED)
    atomic_write_text(
        OUT_JSON,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write_text(OUT_TXT, "\n".join(LINES) + "\n")


def run_fold(year: int) -> dict[str, Any]:
    log("")
    log("=" * 100)
    log(f"YEAR {year}: train<={year - 1}, valid={year}, fixed seed={SEED}")
    log("=" * 100)
    data = prepare_fold(year)
    candidate_probs, prediction_cache = load_or_train_probabilities(year, data)

    # Matrices dominate memory and are not needed for evaluation/confirmation.
    del data["train_matrix"], data["valid_matrix"], data["labels"]
    gc.collect()
    result = evaluate_fold(
        year,
        data["baseline_final"],
        candidate_probs,
        data["effect"],
        data["target"],
        data["game_type"],
        data["pitcher"],
    )
    report_fold(result)
    return {
        "year": year,
        "train_through": year - 1,
        "qa": data["qa"],
        "prediction_cache": prediction_cache,
        "metrics": result,
    }


def main() -> None:
    args = parse_args()
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_TXT.write_text("", encoding="utf-8")
    static_qa = validate_static_contract()
    log("=== exp/123 exact target5 LightGBM strict temporal gate ===")
    log(f"source={static_qa['s12_source']} (primary path, then archive fallback)")
    log(f"fixed_config={LGB_PARAMS}")
    log(f"discovery_gate={DISCOVERY_GATE}")
    log(
        "K>=250 is diagnostic only; +20 is record/open, while +50 on 2024 "
        "raw or analytic blend is required for moonshot promotion"
    )
    log(
        "oracle shape is the same-fold unconstrained-affine ceiling; equal-mean "
        "is not re-clipped"
    )

    summary: dict[str, Any] = {
        "experiment": 123,
        "description": "exact 79-feature reconstructed-target5 CAT5 LightGBM",
        "protocol": {
            "discovery": "train<=2022, valid=2023",
            "confirmation": "train<=2023, valid=2024; opened only after discovery pass",
            "one_configuration": True,
            "early_stopping": False,
            "model_or_feature_hyperparameters_tuned_on_2024": False,
            "same_fold_analytic_blend_is_oracle_diagnostic": True,
            "analytic_blend_weight_is_not_a_deployment_weight": True,
            "target_reconstruction": "exact exp/104 full-train next-asof reconstruction",
            "categorical_mapping": "train-only values; validation-only categories encoded -1",
            "seed": SEED,
            "threads": NTHREAD,
        },
        "model_params": LGB_PARAMS,
        "postprocess": {
            "v18_gamma": V18_GAMMA,
            "affine_center": AFFINE_CENTER,
            "affine_scale": AFFINE_SCALE,
            "affine_shift": AFFINE_SHIFT,
            "order": "clip(raw + gamma*effect), then clip(center + scale*(p-center) - shift)",
            "baseline_sources": {
                "2023": "lab/122_y2023_seed42_base_final.npy (already champion-space)",
                "2024": "lab/89_cat5_probs_seed42.npy (raw five-class)",
            },
        },
        "discovery_gate": DISCOVERY_GATE,
        "record_threshold": 20.0,
        "diversity_moonshot_diagnostic_K": 250.0,
        "confirmation_moonshot_threshold": 50.0,
        "static_qa": static_qa,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "catboost": catboost.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "folds": {},
        "reads_test_zip_or_lb": False,
        "writes_bundle_or_submission": False,
    }

    discovery = run_fold(2023)
    passed, checks = discovery_gate(discovery["metrics"])
    discovery["gate_pass"] = passed
    discovery["gate_checks"] = checks
    summary["folds"]["2023"] = discovery
    summary["discovery_pass"] = passed
    summary["record_candidate"] = passed
    log(f"DISCOVERY {'PASS' if passed else 'FAIL'}: {checks}")
    if not passed:
        summary["confirmation_opened"] = False
        summary["moonshot_track"] = False
        log("2024 remains unopened; no confirmation, bundle, submission, or LB action")
        save_summary(summary)
        return

    if args.discovery_only:
        summary["confirmation_opened"] = False
        summary["moonshot_track"] = False
        log("discovery-only mode: 2024 remains unopened")
        save_summary(summary)
        return

    summary["confirmation_opened"] = True
    save_summary(summary)
    confirmation = run_fold(2024)
    summary["folds"]["2024"] = confirmation
    discovery_raw = float(discovery["metrics"]["raw_gain"])
    confirmation_raw = float(confirmation["metrics"]["raw_gain"])
    confirmation_blend = float(confirmation["metrics"]["blend"]["analytic_gain"])
    consistent_positive_raw_direction = bool(
        discovery_raw > 0.0 and confirmation_raw > 0.0
    )
    moonshot = bool(
        passed
        and consistent_positive_raw_direction
        and (confirmation_raw >= 50.0 or confirmation_blend >= 50.0)
    )
    summary["confirmation"] = {
        "consistent_positive_raw_direction": consistent_positive_raw_direction,
        "raw_gain_ge_50": bool(confirmation_raw >= 50.0),
        "analytic_blend_gain_ge_50": bool(confirmation_blend >= 50.0),
    }
    summary["moonshot_track"] = moonshot
    log(
        f"CONFIRMATION complete: same_positive_raw_direction="
        f"{consistent_positive_raw_direction}; moonshot_track={moonshot}"
    )
    log("No test, ZIP/bundle, submission, or leaderboard action was performed.")
    save_summary(summary)


if __name__ == "__main__":
    main()
