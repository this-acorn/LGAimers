# -*- coding: utf-8 -*-
"""EXP183: strict full-X176 robust-z ResMLP validation on CUDA.

This runner reconstructs the scientifically reproducible part of the public
LA9elephantmiracle ``sj_stdmlp`` endpoint from official competition data only.
It does not use that repository's arrays, lookups, predictions, checkpoints,
or submission archives.  ``run_arm.py`` and its private work directory are not
required.

Validation protocol
-------------------
1. Fit <=2022 and predict 2023 with three final-epoch MLP seeds.
2. On 2023 only, fit one global logit-affine calibration and one convex blend
   coefficient against the frozen current OOF.  Write an immutable lock file.
3. Rebuild the same endpoint <=2023 -> 2024 and apply the 2023 calibration and
   coefficient unchanged.  Save seed states, endpoints, preprocessing, and a
   strict stability report.  No test file is read.

Code provenance
---------------
Feature/preprocessing/model details are adapted from LA9elephantmiracle commit
57783aa88ccc30915a57abe887c8b8125408d483, Copyright (c) 2026
LA9elephantmiracle, under the MIT License shipped in reference/.  Preserve that
notice with redistributed copies.  Competition data are not covered by MIT.
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
REFERENCE = ROOT / "reference" / "LA9elephantmiracle"
LICENSE_PATH = REFERENCE / "LICENSE"
COMMON_PATH = REFERENCE / "cowork" / "cw" / "v17" / "src" / "common.py"
PREP_PATH = (
    REFERENCE / "performance_tracking" / "models" / "sj_stdmlp" / "prep_mlp.py"
)
DEFAULT_DATA = ROOT / "data"
DEFAULT_OUTPUT = ROOT / "lab" / "183_gpu_x176"

AUDITED_COMMIT = "57783aa88ccc30915a57abe887c8b8125408d483"
SOURCE_HASHES = {
    LICENSE_PATH: "76c97c2298b40d458f02fb30dd6a045864c3d9f6b73a3ffbea0487753192d493",
    COMMON_PATH: "91d84161d6e4a085ffa8219f1ea9e453cfd4c5409d15a2baee0dbf694671313a",
    PREP_PATH: "aadfd85573fb76ba13cc4f0c95b24886a2ea698a81399922e65ac6b520d2e8c8",
}
CURRENT_EXPECTED = {
    2023: {
        "sha256": "ee3aa1e2e777025a503790cfdc690aa5e5ed21d6b2ea97607e051cfec894a112",
        "rows": 245525,
        "row_sha256": "d24083d8772ccd6e240c130189a807fc4d7e9ace222ae6f2ea8079690449264e",
        "target_sha256": "8eb61d21dcafa7955207459d92dc9c52e30653cd178d919a439220f318ad3e55",
    },
    2024: {
        "sha256": "e78b168244675adea4bd6a3d56f021a9376cd4b78c428d536a9b53b02fcebe9b",
        "rows": 253507,
        "row_sha256": "6e5fb9e3c20ab4363de4c3be8d58246a87245646ff156eca38c7eeb9e7d70787",
        "target_sha256": "e0483e6a48d9f699ca9c4abf7b4b81aed4d679c359c7e9c3752b3f9e6e27c9d3",
    },
}

SCORE_SCALE = 100000.0
BOOTSTRAP_DRAWS = 3000
GAIN_GATE = 8.0
CALIB_SCALES = np.arange(0.20, 1.55001, 0.05, dtype=np.float64)

SEASON_FEATURE_NAMES = [
    "season_log_n_p", "season_rate_p", "season_rate_shr_p", "season_delta_p",
    "season_log_n_b", "season_rate_b", "season_rate_shr_b", "season_delta_b",
]
TRACKMAN_COLUMNS = [
    "season", "game_month", "game_dayofweek", "trackman_game_id", "pitch_no",
    "inning", "top_bottom", "pitcher_trackman_id", "pitcher_team", "batter_team",
    "pitcher_hand", "balls_before", "strikes_before", "pitch_type_group",
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension",
    "rel_height", "rel_side",
]
PHYSICAL = [
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
    "extension", "rel_height", "rel_side",
]
PITCH_GROUPS = ["fastball", "breaking", "offspeed"]
ID_COLUMNS = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]
TEAM = {
    12: "DOO_BEA", 13: "LG_TWI", 14: "KIW_HER", 15: "LOT_GIA",
    16: "KIA_TIG", 17: "HAN_EAG", 18: "SAM_LIO", 19: "NC_DIN",
    20: "KT_WIZ", 21: None,
}


def log(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray, dtype=np.float64) -> str:
    arr = np.ascontiguousarray(value, dtype=dtype)
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def sha256_row_ids(rows: pd.DataFrame) -> str:
    payload = "\n".join(rows["row_id"].astype(str).tolist()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_sources() -> dict[str, Any]:
    for path, expected in SOURCE_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing audited MIT source: {path}")
        actual = sha256_file(path)
        if actual.lower() != expected:
            raise AssertionError(f"source SHA256 mismatch: {path}\n{actual} != {expected}")
    license_text = LICENSE_PATH.read_text(encoding="utf-8")
    if "MIT License" not in license_text or "Permission is hereby granted" not in license_text:
        raise AssertionError("unexpected reference LICENSE")
    return {
        "repository": "https://github.com/whdpdms2004-bot/LA9elephantmiracle.git",
        "commit": AUDITED_COMMIT,
        "license": "MIT (code only)",
        "source_sha256": {
            str(path.relative_to(REFERENCE)): expected for path, expected in SOURCE_HASHES.items()
        },
        "forbidden_reference_artifacts_read": False,
    }


def integer_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").fillna(-1).to_numpy(np.int64)


def score(probability: np.ndarray, target: np.ndarray) -> float:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rate = float(target.mean())
    if not 0.0 < rate < 1.0:
        raise ValueError("target rate must be in (0,1)")
    return float(SCORE_SCALE * (1.0 - np.mean((probability - target) ** 2) / (rate * (1.0 - rate))))


def geometry(current: np.ndarray, endpoint: np.ndarray, target: np.ndarray) -> dict[str, float]:
    direction = endpoint - current
    ss = float(direction @ direction)
    raw_weight = float(((target - current) @ direction) / ss) if ss > 0 else 0.0
    weight = float(np.clip(raw_weight, 0.0, 1.0))
    candidate = current + weight * direction
    rate = float(target.mean())
    curvature = float(SCORE_SCALE * np.mean(direction ** 2) / (rate * (1.0 - rate)))
    return {
        "d_endpoint_score_minus_current": score(endpoint, target) - score(current, target),
        "K_direction_curvature": curvature,
        "optimal_weight_unconstrained": raw_weight,
        "optimal_weight_convex": weight,
        "current_score": score(current, target),
        "endpoint_score": score(endpoint, target),
        "optimal_blend_score": score(candidate, target),
        "optimal_blend_gain": score(candidate, target) - score(current, target),
        "correlation": float(np.corrcoef(current, endpoint)[0, 1]),
    }


def subset_gain(current: np.ndarray, candidate: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    return {
        "rows": int(mask.sum()),
        "current_score": score(current[mask], target[mask]),
        "candidate_score": score(candidate[mask], target[mask]),
        "gain": score(candidate[mask], target[mask]) - score(current[mask], target[mask]),
    }


def pitcher_bootstrap(
    current: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    pitcher: np.ndarray,
) -> dict[str, float]:
    work = pd.DataFrame({
        "pitcher": pitcher,
        "n": np.ones(len(target), dtype=np.int32),
        "y": target,
        "current_se": (current - target) ** 2,
        "candidate_se": (candidate - target) ** 2,
    })
    grouped = work.groupby("pitcher", sort=False).agg(
        n=("n", "sum"), y=("y", "sum"),
        current_se=("current_se", "sum"), candidate_se=("candidate_se", "sum"),
    ).to_numpy(np.float64)
    rng = np.random.default_rng(183)
    gains = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for start in range(0, BOOTSTRAP_DRAWS, 200):
        width = min(200, BOOTSTRAP_DRAWS - start)
        sampled = grouped[rng.integers(0, len(grouped), size=(width, len(grouped)))].sum(axis=1)
        rate = sampled[:, 1] / sampled[:, 0]
        gains[start:start + width] = (
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


def solve_intercept(logit: np.ndarray, scale: float, target_rate: float) -> float:
    lo, hi = -12.0, 12.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        mean = float((1.0 / (1.0 + np.exp(-np.clip(scale * logit + mid, -40, 40)))).mean())
        if mean < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def fit_logit_affine(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    clipped = np.clip(np.asarray(probability, np.float64), 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped))
    target_rate = float(np.mean(target))
    best: tuple[float, float, float] | None = None
    for scale in CALIB_SCALES:
        intercept = solve_intercept(logit, float(scale), target_rate)
        calibrated = 1.0 / (1.0 + np.exp(-np.clip(scale * logit + intercept, -40, 40)))
        mse = float(np.mean((calibrated - target) ** 2))
        if best is None or mse < best[0]:
            best = (mse, float(scale), float(intercept))
    assert best is not None
    return {
        "method": "scale grid 0.20:0.05:1.55; intercept solves mean(p)=mean(y)",
        "scale": best[1],
        "intercept": best[2],
        "fit_target_rate": target_rate,
        "fit_mse": best[0],
    }


def apply_logit_affine(probability: np.ndarray, calibration: dict[str, float]) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, np.float64), 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped))
    transformed = calibration["scale"] * logit + calibration["intercept"]
    return 1.0 / (1.0 + np.exp(-np.clip(transformed, -40.0, 40.0)))


# ---------------------------------------------------------------------------
# Official-only full X176 reconstruction.
# ---------------------------------------------------------------------------


def strict_platoon(frame: pd.DataFrame, target: np.ndarray, fit_year: int, common) -> np.ndarray:
    seasons = integer_array(frame, "season")
    pitcher = integer_array(frame, "pitcher_id")
    batter = integer_array(frame, "batter_id")
    pitcher_hand = integer_array(frame, "pitcher_hand")
    batter_hand = integer_array(frame, "batter_hand")
    out = np.zeros((len(frame), 4), dtype=np.float32)
    out[:, :2] = np.nan
    for season in sorted(np.unique(seasons).tolist()):
        history = (seasons < season) & (seasons <= fit_year)
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


def end_state(
    frame: pd.DataFrame,
    upto: int,
    id_column: str,
    n_column: str,
    rate_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    history = frame.loc[integer_array(frame, "season") <= upto]
    if history.empty:
        return np.array([], dtype=np.int64), np.zeros((0, 2), dtype=np.float64)
    last = history.sort_values(n_column).groupby(id_column, sort=False).tail(1)
    n = pd.to_numeric(last[n_column], errors="coerce").fillna(0).to_numpy(np.float64)
    rate = pd.to_numeric(last[rate_column], errors="coerce").fillna(0).to_numpy(np.float64)
    success = n * rate + last["control_success"].to_numpy(np.float64)
    keys = integer_array(last, id_column)
    order = np.argsort(keys)
    return keys[order], np.column_stack((n + 1.0, success))[order]


def lookup_state(ids: np.ndarray, keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(keys):
        return np.zeros(len(ids)), np.zeros(len(ids))
    pos = np.clip(np.searchsorted(keys, ids), 0, len(keys) - 1)
    hit = keys[pos] == ids
    return np.where(hit, values[pos, 0], 0.0), np.where(hit, values[pos, 1], 0.0)


def season_side_features(
    asof_n: np.ndarray,
    asof_rate: np.ndarray,
    n0: np.ndarray,
    success0: np.ndarray,
    prior: float,
) -> np.ndarray:
    n = np.nan_to_num(asof_n.astype(np.float64), nan=0.0)
    rate = asof_rate.astype(np.float64)
    cumulative = n * np.nan_to_num(rate, nan=0.0)
    season_n = np.maximum(n - n0, 0.0)
    season_success = np.clip(cumulative - success0, 0.0, None)
    career = np.where(np.isnan(rate), prior, np.nan_to_num(rate, nan=prior))
    shrunk = (season_success + 150.0 * career) / (season_n + 150.0)
    raw = np.where(season_n > 0, season_success / np.maximum(season_n, 1.0), career)
    return np.column_stack((np.log1p(season_n), raw, shrunk, shrunk - career)).astype(np.float32)


def strict_season_form(frame: pd.DataFrame, fit_year: int) -> np.ndarray:
    seasons = integer_array(frame, "season")
    fit = seasons <= fit_year
    prior = float(frame.loc[fit, "control_success"].mean())
    out = np.zeros((len(frame), 8), dtype=np.float32)
    for season in sorted(np.unique(seasons).tolist()):
        query = seasons == season
        parts = []
        for id_column, n_column, rate_column in (
            ("pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
            ("batter_id", "asof_batter_n", "asof_batter_success_rate"),
        ):
            keys, values = end_state(frame.loc[fit], season - 1, id_column, n_column, rate_column)
            ids = integer_array(frame.loc[query], id_column)
            n0, success0 = lookup_state(ids, keys, values)
            asof_n = pd.to_numeric(frame.loc[query, n_column], errors="coerce").to_numpy(np.float64)
            asof_rate = pd.to_numeric(frame.loc[query, rate_column], errors="coerce").to_numpy(np.float64)
            parts.append(season_side_features(asof_n, asof_rate, n0, success0, prior))
        out[query] = np.concatenate(parts, axis=1)
    return out


def reconstruct_games(train: pd.DataFrame) -> pd.DataFrame:
    result = train.copy()
    result["t1"] = np.minimum(result["pitcher_team_id"], result["batter_team_id"])
    result["t2"] = np.maximum(result["pitcher_team_id"], result["batter_team_id"])
    key = result[["season", "game_month", "game_dayofweek", "t1", "t2", "game_type"]]
    result["gid"] = (key != key.shift(1)).any(axis=1).cumsum()
    result["tb"] = result["top_bottom"].astype(str).str[0].str.upper()
    return result


def prepare_trackman_for_match(trackman: pd.DataFrame) -> pd.DataFrame:
    result = trackman.copy()
    pteam = result["pitcher_team"].fillna("").astype(str)
    bteam = result["batter_team"].fillna("").astype(str)
    result = result.loc[~pteam.str.startswith("MIN_") & ~bteam.str.startswith("MIN_")].copy()
    inverse = {value: key for key, value in TEAM.items() if value}
    inverse.update({"SK_WYV": 21, "SSG_LAN": 21})
    result["t1"] = np.minimum(result["pitcher_team"].map(inverse), result["batter_team"].map(inverse))
    result["t2"] = np.maximum(result["pitcher_team"].map(inverse), result["batter_team"].map(inverse))
    result["tb"] = result["top_bottom"].astype(str).str[0].str.upper()
    return result


def half_vector(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    work = frame.copy()
    work["hi"] = np.clip(pd.to_numeric(work["inning"], errors="coerce"), 1, 13) * 2 \
        + (work["tb"] == "B").astype(int)
    return work.pivot_table(index=key, columns="hi", aggfunc="size", fill_value=0)


def match_games(trackman: pd.DataFrame, train_regular: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError("scipy is required for official TrackMan Hungarian matching") from exc
    left_half = half_vector(trackman, "trackman_game_id")
    right_half = half_vector(train_regular, "gid")
    grouped = train_regular.groupby("gid")
    right_games = pd.DataFrame({
        "season": grouped.season.first(), "m": grouped.game_month.first(),
        "d": grouped.game_dayofweek.first(), "t1": grouped.t1.first(),
        "t2": grouped.t2.first(),
    }).reset_index()
    left_games = trackman.groupby("trackman_game_id").agg(
        season=("season", "first"), m=("game_month", "first"),
        d=("game_dayofweek", "first"), t1=("t1", "first"), t2=("t2", "first"),
    ).reset_index()
    pairs: list[tuple[int, Any, float]] = []
    for key, right in right_games.groupby(["season", "m", "d", "t1", "t2"], dropna=False):
        left = left_games[
            (left_games.season == key[0]) & (left_games.m == key[1])
            & (left_games.d == key[2]) & (left_games.t1 == key[3])
            & (left_games.t2 == key[4])
        ]
        if left.empty:
            continue
        columns = left_half.columns.union(right_half.columns)
        xa = left_half.reindex(index=left.trackman_game_id, columns=columns, fill_value=0).astype(float).values
        xb = right_half.reindex(index=right.gid, columns=columns, fill_value=0).astype(float).values
        xa /= np.linalg.norm(xa, axis=1, keepdims=True) + 1e-9
        xb /= np.linalg.norm(xb, axis=1, keepdims=True) + 1e-9
        similarity = xb @ xa.T
        rows, columns_index = linear_sum_assignment(-similarity)
        for i, j in zip(rows, columns_index):
            pairs.append((int(right.gid.iloc[i]), left.trackman_game_id.iloc[j], float(similarity[i, j])))
    return pd.DataFrame(pairs, columns=["gid", "tgid", "similarity"])


def match_pitchers(trackman: pd.DataFrame, train_regular: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError("scipy is required for official TrackMan Hungarian matching") from exc
    games = games.loc[games.similarity > 0.95]
    left = trackman.loc[trackman.trackman_game_id.isin(set(games.tgid))].copy()
    right = train_regular.loc[train_regular.gid.isin(set(games.gid))].copy()
    for frame in (left, right):
        frame["hi"] = np.clip(pd.to_numeric(frame["inning"], errors="coerce"), 1, 13) * 2 \
            + (frame["tb"] == "B").astype(int)
    left = left.sort_values(["trackman_game_id", "hi", "pitch_no"])
    left["k"] = left.groupby(["trackman_game_id", "hi"]).cumcount()
    right = right.merge(games[["gid", "tgid"]], on="gid")
    right["k"] = right.groupby(["gid", "hi"]).cumcount()
    n_left = left.groupby(["trackman_game_id", "hi"]).size().rename("n_left")
    n_right = right.groupby(["tgid", "hi"]).size().rename("n_right")
    equal = pd.concat((n_left, n_right), axis=1).dropna()
    equal = equal.loc[equal.n_left == equal.n_right].reset_index()
    equal.columns = ["tgid", "hi", "n_left", "n_right"]
    valid_keys = set(zip(equal.tgid, equal.hi))
    left = left.loc[[(g, h) in valid_keys for g, h in zip(left.trackman_game_id, left.hi)]]
    right = right.loc[[(g, h) in valid_keys for g, h in zip(right.tgid, right.hi)]]
    matched = right.merge(
        left,
        left_on=["tgid", "hi", "k"],
        right_on=["trackman_game_id", "hi", "k"],
        suffixes=("_train", "_tm"),
    )
    pitcher_ids = np.sort(matched.pitcher_id.unique())
    trackman_ids = np.sort(matched.pitcher_trackman_id.unique())
    if not len(pitcher_ids) or not len(trackman_ids):
        return pd.DataFrame(columns=["pitcher_id", "pitcher_trackman_id", "confidence", "evidence_pitches"])
    pindex = {value: index for index, value in enumerate(pitcher_ids)}
    tindex = {value: index for index, value in enumerate(trackman_ids)}
    votes = np.zeros((len(pitcher_ids), len(trackman_ids)), dtype=np.float64)
    np.add.at(
        votes,
        (matched.pitcher_id.map(pindex).to_numpy(), matched.pitcher_trackman_id.map(tindex).to_numpy()),
        1.0,
    )
    shares = votes / (votes.sum(axis=1, keepdims=True) + 1e-9)
    row_index, column_index = linear_sum_assignment(-shares)
    result = pd.DataFrame({
        "pitcher_id": pitcher_ids[row_index],
        "pitcher_trackman_id": trackman_ids[column_index],
        "confidence": shares[row_index, column_index],
        "evidence_pitches": votes[row_index, column_index],
    })
    return result.loc[result.evidence_pitches > 0].reset_index(drop=True)


def build_pitcher_mapping(train: pd.DataFrame, trackman: pd.DataFrame, cutoff: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.time()
    train_cut = train.loc[(integer_array(train, "season") <= cutoff) & (train.game_type == "R")].copy()
    trackman_cut = trackman.loc[integer_array(trackman, "season") <= cutoff].copy()
    reconstructed = reconstruct_games(train_cut)
    prepared = prepare_trackman_for_match(trackman_cut)
    games = match_games(prepared, reconstructed)
    mapping = match_pitchers(prepared, reconstructed, games)
    metadata = {
        "cutoff": cutoff,
        "train_regular_rows": int(len(train_cut)),
        "trackman_rows": int(len(trackman_cut)),
        "matched_games": int(len(games)),
        "matched_games_similarity_gt_095": int((games.similarity > 0.95).sum()),
        "mapped_pitchers": int(len(mapping)),
        "seconds": time.time() - started,
    }
    return mapping, metadata


def build_trackman_table(trackman: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    mapper = dict(zip(mapping.pitcher_trackman_id, mapping.pitcher_id))
    work = trackman[["pitcher_trackman_id", "season", "pitch_type_group", *PHYSICAL]].copy()
    work["pid"] = work.pitcher_trackman_id.map(mapper)
    work = work.loc[work.pid.notna() & work.pitch_type_group.isin(PITCH_GROUPS)].copy()
    work["pid"] = work.pid.astype(np.int64)
    parts = []
    for group in PITCH_GROUPS:
        subset = work.loc[work.pitch_type_group == group]
        aggregate = subset.groupby(["pid", "season"])[PHYSICAL].agg(["mean", "std"])
        aggregate.columns = [f"{group[:2]}_{column}_{stat}" for column, stat in aggregate.columns]
        aggregate[f"{group[:2]}_n"] = subset.groupby(["pid", "season"]).size()
        parts.append(aggregate)
    table = pd.concat(parts, axis=1)
    for column in ("rel_height", "rel_side", "extension", "spin_rate", "rel_speed"):
        table[f"d_fb_bk_{column}"] = table[f"fa_{column}_mean"] - table[f"br_{column}_mean"]
        table[f"d_fb_of_{column}"] = table[f"fa_{column}_mean"] - table[f"of_{column}_mean"]
    table = table.reset_index()
    names = [column for column in table.columns if column not in ("pid", "season")]
    if len(names) != 55:
        raise AssertionError(f"expected 55 TrackMan features, found {len(names)}")
    return table, names


def lag_lookup(table: pd.DataFrame, ids: np.ndarray, seasons: np.ndarray, columns: list[str]) -> np.ndarray:
    output = np.full((len(ids), len(columns)), np.nan, dtype=np.float64)
    for season in sorted(np.unique(seasons).tolist()):
        history = table.loc[table.season < season]
        if history.empty:
            continue
        grouped = history.groupby("pid")[columns].mean()
        keys = grouped.index.to_numpy(np.int64)
        values = grouped.to_numpy(np.float64)
        query = seasons == season
        pos = np.clip(np.searchsorted(keys, ids[query]), 0, max(len(keys) - 1, 0))
        hit = keys[pos] == ids[query]
        found = values[pos].copy()
        found[~hit] = np.nan
        output[query] = found
    return output


def count_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    balls = pd.to_numeric(frame.balls_before, errors="coerce").to_numpy(np.float64)
    strikes = pd.to_numeric(frame.strikes_before, errors="coerce").to_numpy(np.float64)
    ball_rate = pd.to_numeric(frame.asof_pitcher_ball_rate, errors="coerce").fillna(0).to_numpy(np.float64)
    strike_rate = pd.to_numeric(frame.asof_pitcher_strike_rate, errors="coerce").fillna(0).to_numpy(np.float64)
    tendency = ball_rate - strike_rate
    columns: list[np.ndarray] = []
    names: list[str] = []
    for ball in range(4):
        for strike in range(3):
            indicator = ((balls == ball) & (strikes == strike)).astype(np.float32)
            columns.extend((indicator, indicator * tendency))
            names.extend((f"cnt_{ball}_{strike}", f"cnt_{ball}_{strike}_xtendency"))
    advantage = (balls - strikes).astype(np.float32)
    columns.extend((advantage, advantage * tendency, ((balls == 3) & (strikes == 1)) * tendency))
    names.extend(("cnt_advantage", "cnt_advantage_xtendency", "cnt_31_xtendency"))
    return np.column_stack(columns).astype(np.float32), names


def role_features(frame: pd.DataFrame, ids: np.ndarray, seasons: np.ndarray) -> tuple[np.ndarray, list[str]]:
    work = frame[["pitcher_id", "season", "inning"]].copy()
    work["p1"] = (pd.to_numeric(work.inning, errors="coerce") <= 2).astype(np.float64)
    grouped = work.groupby(["pitcher_id", "season"]).agg(
        inn_mean=("inning", "mean"), inn_std=("inning", "std"),
        inn_min=("inning", "min"), p1_ratio=("p1", "mean"), p_season=("inning", "size"),
    ).reset_index().rename(columns={"pitcher_id": "pid"})
    names = ["inn_mean", "inn_std", "inn_min", "p1_ratio", "p_season"]
    return lag_lookup(grouped, ids, seasons, names).astype(np.float32), names


def id_frequency(frame: pd.DataFrame, fit_mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    for column in ID_COLUMNS:
        values = integer_array(frame, column)
        unique, counts = np.unique(values[fit_mask], return_counts=True)
        pos = np.clip(np.searchsorted(unique, values), 0, len(unique) - 1)
        hit = unique[pos] == values
        frequency = np.where(hit, counts[pos], 0.0).astype(np.float32)
        columns.extend((np.log1p(frequency), (frequency == 0).astype(np.float32)))
        names.extend((f"pa_{column}_logfreq", f"pa_{column}_unseen"))
    return np.column_stack(columns).astype(np.float32), names


def build_raw_x176(
    frame: pd.DataFrame,
    trackman: pd.DataFrame,
    fit_year: int,
    common,
    smoke: bool = False,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    started = time.time()
    seasons = integer_array(frame, "season")
    fit_mask = seasons <= fit_year
    target = frame.control_success.to_numpy(np.float32)
    base, names = common.build_features(frame)
    platoon = strict_platoon(frame, target, fit_year, common)
    season_form = strict_season_form(frame, fit_year)

    if smoke:
        mapping = pd.DataFrame(columns=["pitcher_id", "pitcher_trackman_id"])
        mapping_meta = {"smoke_empty_mapping": True, "seconds": 0.0}
        tm_table = pd.DataFrame(columns=["pid", "season"])
        tm_names = []
        # Keep the exact 55-column contract without running the expensive matcher.
        for group in PITCH_GROUPS:
            for column in PHYSICAL:
                tm_names.extend((f"{group[:2]}_{column}_mean", f"{group[:2]}_{column}_std"))
            tm_names.append(f"{group[:2]}_n")
        for column in ("rel_height", "rel_side", "extension", "spin_rate", "rel_speed"):
            tm_names.extend((f"d_fb_bk_{column}", f"d_fb_of_{column}"))
        trackman_features = np.full((len(frame), 55), np.nan, dtype=np.float32)
    else:
        mapping, mapping_meta = build_pitcher_mapping(frame, trackman, fit_year)
        tm_table, tm_names = build_trackman_table(trackman, mapping)
        trackman_features = lag_lookup(
            tm_table, integer_array(frame, "pitcher_id"), seasons, tm_names
        ).astype(np.float32)

    counts, count_names = count_features(frame)
    roles, role_names = role_features(frame, integer_array(frame, "pitcher_id"), seasons)
    coverage = np.isfinite(trackman_features[:, 0]).astype(np.float32)[:, None]
    domain = np.column_stack((trackman_features, counts, roles, coverage)).astype(np.float32)
    domain_names = tm_names + count_names + role_names + ["trackman_coverage"]
    frequency, frequency_names = id_frequency(frame, fit_mask)
    raw = np.ascontiguousarray(
        np.column_stack((base, platoon, season_form, domain, frequency)), dtype=np.float32
    )
    feature_names = list(names) + list(common.ENC_NAMES) + SEASON_FEATURE_NAMES \
        + domain_names + frequency_names
    if base.shape[1] != 68 or raw.shape[1] != 176 or len(feature_names) != 176:
        raise AssertionError(
            f"feature contract mismatch base={base.shape[1]} raw={raw.shape[1]} names={len(feature_names)}"
        )
    metadata = {
        "base68": 68, "platoon": 4, "season_form": 8, "trackman": 55,
        "count": 27, "role": 5, "trackman_coverage": 1, "id_frequency": 8,
        "total": 176, "mapping": mapping_meta,
        "mapped_pitcher_rows": int(len(mapping)),
        "seconds": time.time() - started,
    }
    del base, platoon, season_form, domain, frequency, counts, roles, trackman_features
    gc.collect()
    return raw, feature_names, metadata


def build_fold(
    train_all: pd.DataFrame,
    trackman_all: pd.DataFrame,
    validation_year: int,
    common,
    prep_module,
    output: Path,
) -> dict[str, Any]:
    frame = train_all.loc[integer_array(train_all, "season") <= validation_year].reset_index(drop=True)
    seasons = integer_array(frame, "season")
    fit_year = validation_year - 1
    fit_mask = seasons <= fit_year
    valid_mask = seasons == validation_year
    expected = CURRENT_EXPECTED[validation_year]
    if int(valid_mask.sum()) != expected["rows"]:
        raise AssertionError(f"unexpected {validation_year} rows: {valid_mask.sum()}")
    raw, names, feature_meta = build_raw_x176(frame, trackman_all, fit_year, common)
    train_raw = np.ascontiguousarray(raw[fit_mask], dtype=np.float32)
    valid_raw = np.ascontiguousarray(raw[valid_mask], dtype=np.float32)
    del raw
    prep = prep_module.make_prep(train_raw, "std")
    z_train = prep_module.apply_prep(train_raw, prep, True)
    z_valid = prep_module.apply_prep(valid_raw, prep, True)
    del train_raw, valid_raw
    prep_path = output / f"prep_{validation_year}.npz"
    np.savez_compressed(
        prep_path,
        med=np.asarray(prep["med"], np.float64),
        iqr=np.asarray(prep["iqr"], np.float64),
        feature_names=np.asarray(names),
    )
    rows = frame.loc[valid_mask].reset_index(drop=True)
    y_train = frame.loc[fit_mask, "control_success"].to_numpy(np.float32)
    y_valid = frame.loc[valid_mask, "control_success"].to_numpy(np.float64)
    row_hash = sha256_row_ids(rows)
    target_hash = sha256_array(y_valid)
    if row_hash != expected["row_sha256"] or target_hash != expected["target_sha256"]:
        raise AssertionError(f"{validation_year} row/target order identity mismatch")
    metadata = {
        "validation_year": validation_year,
        "fit_through": fit_year,
        "train_rows": int(fit_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "raw_features": 176,
        "network_inputs": 352,
        "row_id_sha256": row_hash,
        "target_sha256": target_hash,
        "feature_names": names,
        "feature_build": feature_meta,
        "prep": {"path": str(prep_path), "sha256": sha256_file(prep_path)},
    }
    del frame, seasons
    gc.collect()
    return {
        "z_train": z_train, "z_valid": z_valid,
        "y_train": y_train, "y_valid": y_valid,
        "rows": rows, "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# CUDA model / training.
# ---------------------------------------------------------------------------


def build_network(torch, nn, d_in: int, width: int, depth: int, dropout: float):
    class ResidualBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.ff = nn.Sequential(
                nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(width * 2, width),
            )

        def forward(self, value):
            return value + self.ff(self.norm(value))

    return nn.Sequential(
        nn.Linear(d_in, width),
        *[ResidualBlock() for _ in range(depth)],
        nn.LayerNorm(width),
        nn.Linear(width, 1),
    )


def resolve_device(torch, requested: str):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA-enabled PyTorch/device is unavailable")
    return torch.device(requested)


def autocast_context(torch, device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def make_scaler(torch, enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def predict(torch, network, matrix: np.ndarray, device, batch: int, amp: bool) -> np.ndarray:
    network.eval()
    source = torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float32))
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(source), batch):
            xb = source[start:start + batch].to(device, non_blocking=True)
            with autocast_context(torch, device, amp):
                probability = torch.sigmoid(network(xb).squeeze(-1))
            output.append(probability.float().cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def fit_seed(
    fold: dict[str, Any],
    validation_year: int,
    seed: int,
    args: argparse.Namespace,
    output: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    import torch.nn as nn

    device = resolve_device(torch, args.device)
    amp = bool(args.amp and device.type == "cuda")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    else:
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    network = build_network(
        torch, nn, fold["z_train"].shape[1], args.width, args.depth, args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x = torch.from_numpy(np.ascontiguousarray(fold["z_train"], np.float32))
    y = torch.from_numpy(np.ascontiguousarray(fold["y_train"], np.float32))
    on_gpu = False
    if device.type == "cuda":
        needed = x.numel() * x.element_size() + y.numel() * y.element_size()
        if needed < torch.cuda.mem_get_info()[0] * 0.55:
            x = x.to(device)
            y = y.to(device)
            on_gpu = True
    batches = math.ceil(len(x) / args.batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=batches * args.epochs, pct_start=0.1
    )
    scaler = make_scaler(torch, amp)
    history: list[dict[str, float]] = []
    all_started = time.time()
    for epoch in range(args.epochs):
        network.train()
        epoch_started = time.time()
        permutation = torch.randperm(len(x), device=device if on_gpu else "cpu")
        loss_sum = 0.0
        seen = 0
        for batch_index, start in enumerate(range(0, len(x), args.batch)):
            index = permutation[start:start + args.batch]
            xb, yb = x[index], y[index]
            if not on_gpu:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
            with autocast_context(torch, device, amp):
                probability = torch.sigmoid(network(xb).squeeze(-1))
                loss = torch.square(probability - yb).mean()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count = int(len(index))
            loss_sum += float(loss.detach()) * count
            seen += count
            if epoch == 0 and batch_index == 49:
                elapsed50 = time.time() - epoch_started
                eta = elapsed50 * batches * args.epochs / 50.0
                log(f"[year {validation_year} seed {seed}] 50 batches {elapsed50:.1f}s; total ETA {eta/60:.1f}m")
        record = {
            "epoch": epoch + 1,
            "seconds": time.time() - epoch_started,
            "train_brier": loss_sum / seen,
            "learning_rate_end": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        log(f"[year {validation_year} seed {seed} epoch {epoch+1}/{args.epochs}] {record}")

    # Public final builder fixes inference to fp32.  Keeping AMP training but
    # fp32 prediction also avoids batch-size-dependent fp16 accumulation noise.
    endpoint = predict(torch, network, fold["z_valid"], device, args.predict_batch, False)
    probe = min(64, len(endpoint))
    batch_prediction = predict(torch, network, fold["z_valid"][:probe], device, probe, False)
    singleton_prediction = np.concatenate([
        predict(torch, network, fold["z_valid"][index:index + 1], device, 1, False)
        for index in range(probe)
    ])
    independence = float(np.max(np.abs(batch_prediction - singleton_prediction), initial=0.0))
    state_path = output / f"state_{validation_year}_seed{seed}.pt"
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in network.state_dict().items()},
        "seed": seed,
        "fit_through": validation_year - 1,
        "input_features": 352,
        "architecture": {
            "width": args.width, "depth": args.depth, "dropout": args.dropout,
            "block": "LayerNorm -> Linear(2w) -> GELU -> Dropout -> Linear(w) + skip",
        },
        "training": {
            "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
            "weight_decay": args.weight_decay, "optimizer": "AdamW",
            "scheduler": "OneCycleLR pct_start=.1", "loss": "sigmoid Brier/MSE",
            "amp_training": amp, "fp32_inference": True, "final_epoch_only": True,
        },
        "source_commit": AUDITED_COMMIT,
        "test_read": False,
    }, state_path)
    report = {
        "seed": seed,
        "elapsed_seconds": time.time() - all_started,
        "history": history,
        "state": {"path": str(state_path), "sha256": sha256_file(state_path)},
        "row_independence_max_abs": independence,
        "on_gpu_matrix": on_gpu,
        "device": str(device),
        "amp": amp,
    }
    del network, optimizer, scheduler, scaler, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return endpoint, report


def train_fold_endpoints(
    fold: dict[str, Any],
    validation_year: int,
    args: argparse.Namespace,
    output: Path,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    endpoints: list[np.ndarray] = []
    seed_reports: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for seed in args.seeds:
        endpoint, report = fit_seed(fold, validation_year, seed, args, output)
        path = output / f"endpoint_{validation_year}_seed{seed}.npy"
        np.save(path, endpoint.astype(np.float32), allow_pickle=False)
        report["endpoint"] = {"path": str(path), "sha256": sha256_file(path)}
        endpoints.append(endpoint)
        seed_reports.append(report)
        artifacts.append(report["endpoint"])
    ensemble = np.mean(endpoints, axis=0)
    ensemble_path = output / f"endpoint_{validation_year}_ensemble_raw.npy"
    np.save(ensemble_path, ensemble.astype(np.float32), allow_pickle=False)
    artifacts.append({"path": str(ensemble_path), "sha256": sha256_file(ensemble_path)})
    return ensemble, seed_reports, artifacts


def resolve_current_path(explicit: str | None, year: int) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend((
        ROOT / "lab" / f"179_current_{year}.npy",
        ROOT.parent / "lab" / f"179_current_{year}.npy",
    ))
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"frozen current {year} OOF absent; pass --current-{year}. Tried: {candidates}"
    )


def load_current(path: Path, year: int, rows: int) -> np.ndarray:
    expected = CURRENT_EXPECTED[year]
    actual_hash = sha256_file(path)
    if actual_hash.lower() != expected["sha256"]:
        raise AssertionError(f"current {year} SHA256 mismatch: {actual_hash}")
    value = np.load(path, allow_pickle=False).astype(np.float64)
    if value.shape != (rows,) or not np.isfinite(value).all():
        raise AssertionError(f"invalid current {year} OOF shape/values: {value.shape}")
    return value


def read_official(data_dir: Path, common) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_path = data_dir / "train.csv"
    trackman_path = data_dir / "trackman_history.csv"
    if not train_path.is_file() or not trackman_path.is_file():
        raise FileNotFoundError(f"required official files: {train_path}, {trackman_path}")
    train_columns = list(dict.fromkeys([
        "row_id", "control_success", *common.NUM_COLS, *common.CAT_LEVELS.keys(),
    ]))
    train = pd.read_csv(train_path, encoding="utf-8-sig", usecols=train_columns, low_memory=False)
    train.columns = [column.replace("\ufeff", "").strip() for column in train.columns]
    train = train.loc[integer_array(train, "season") <= 2024].reset_index(drop=True)
    target = pd.to_numeric(train.control_success, errors="coerce")
    if target.isna().any() or not target.isin((0, 1)).all():
        raise ValueError("invalid official target")
    trackman = pd.read_csv(
        trackman_path, encoding="utf-8-sig", usecols=TRACKMAN_COLUMNS, low_memory=False
    )
    metadata = {
        "train": {"path": str(train_path), "sha256": sha256_file(train_path), "rows": int(len(train))},
        "trackman": {"path": str(trackman_path), "sha256": sha256_file(trackman_path), "rows": int(len(trackman))},
    }
    return train, trackman, metadata


def run_validate(args: argparse.Namespace) -> None:
    started = time.time()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "lock_2023.json"
    report_path = output / "validation_report.json"
    if lock_path.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite strict run: {output}")
    source_audit = verify_sources()
    common = load_module(COMMON_PATH, "exp183_common")
    prep_module = load_module(PREP_PATH, "exp183_prep")
    data_dir = Path(args.data_dir).resolve()
    train_all, trackman_all, data_meta = read_official(data_dir, common)
    current_paths = {
        2023: resolve_current_path(args.current_2023, 2023),
        2024: resolve_current_path(args.current_2024, 2024),
    }
    recipe = {
        "features": "full X176 -> robust median/IQR z clipped [-4,4] + finite mask = 352",
        "seeds": args.seeds, "width": args.width, "depth": args.depth,
        "dropout": args.dropout, "batch": args.batch, "epochs": args.epochs,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "amp": args.amp, "device": args.device, "threads": args.threads,
        "validation_epoch_selection": "none; final epoch only",
    }

    log("[EXP183 discovery] official <=2022 -> 2023")
    fold23 = build_fold(train_all, trackman_all, 2023, common, prep_module, output)
    raw23, seed23, artifacts23 = train_fold_endpoints(fold23, 2023, args, output)
    current23 = load_current(current_paths[2023], 2023, len(fold23["y_valid"]))
    calibration = fit_logit_affine(raw23, fold23["y_valid"])
    endpoint23 = apply_logit_affine(raw23, calibration)
    geometry23 = geometry(current23, endpoint23, fold23["y_valid"])
    locked_weight = geometry23["optimal_weight_convex"]
    candidate23 = current23 + locked_weight * (endpoint23 - current23)
    calibrated23_path = output / "endpoint_2023_ensemble_calibrated.npy"
    candidate23_path = output / "candidate_2023_locked.npy"
    np.save(calibrated23_path, endpoint23.astype(np.float32), allow_pickle=False)
    np.save(candidate23_path, candidate23.astype(np.float32), allow_pickle=False)
    lock = {
        "experiment": 183,
        "phase": "LOCKED_AFTER_2023_BEFORE_2024_BUILD",
        "source_audit": source_audit,
        "data": data_meta,
        "recipe": recipe,
        "fold_2023": fold23["metadata"],
        "seed_reports_2023": seed23,
        "calibration_from_2023": calibration,
        "blend_weight_from_2023": locked_weight,
        "geometry_2023": geometry23,
        "discovery_gate_gain_ge_8": bool(geometry23["optimal_blend_gain"] >= GAIN_GATE),
        "artifacts": artifacts23 + [
            {"path": str(calibrated23_path), "sha256": sha256_file(calibrated23_path)},
            {"path": str(candidate23_path), "sha256": sha256_file(candidate23_path)},
        ],
        "current_2023": {"path": str(current_paths[2023]), "sha256": sha256_file(current_paths[2023])},
        "test_read": False,
    }
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lock_hash = sha256_file(lock_path)
    log(
        f"[LOCKED 2023] scale={calibration['scale']:.3f} intercept={calibration['intercept']:+.6f} "
        f"w={locked_weight:.8f} gain={geometry23['optimal_blend_gain']:+.4f}"
    )
    del fold23, raw23, endpoint23, candidate23, current23
    gc.collect()

    log("[EXP183 untouched confirmation] official <=2023 -> 2024; 2023 lock unchanged")
    fold24 = build_fold(train_all, trackman_all, 2024, common, prep_module, output)
    raw24, seed24, artifacts24 = train_fold_endpoints(fold24, 2024, args, output)
    current24 = load_current(current_paths[2024], 2024, len(fold24["y_valid"]))
    endpoint24 = apply_logit_affine(raw24, calibration)
    candidate24 = current24 + locked_weight * (endpoint24 - current24)
    geometry24 = geometry(current24, endpoint24, fold24["y_valid"])
    locked_gain24 = score(candidate24, fold24["y_valid"]) - score(current24, fold24["y_valid"])
    months = integer_array(fold24["rows"], "game_month")
    stability = {
        "early": subset_gain(current24, candidate24, fold24["y_valid"], months <= 6),
        "late": subset_gain(current24, candidate24, fold24["y_valid"], months > 6),
        "pitcher_bootstrap": pitcher_bootstrap(
            current24, candidate24, fold24["y_valid"], integer_array(fold24["rows"], "pitcher_id")
        ),
    }
    calibrated24_path = output / "endpoint_2024_ensemble_calibrated.npy"
    candidate24_path = output / "candidate_2024_locked.npy"
    np.save(calibrated24_path, endpoint24.astype(np.float32), allow_pickle=False)
    np.save(candidate24_path, candidate24.astype(np.float32), allow_pickle=False)
    independence_max = max(
        [report["row_independence_max_abs"] for report in seed23 + seed24], default=0.0
    )
    passed = bool(
        geometry23["optimal_blend_gain"] >= GAIN_GATE
        and locked_gain24 >= GAIN_GATE
        and stability["early"]["gain"] > 0.0
        and stability["late"]["gain"] > 0.0
        and stability["pitcher_bootstrap"]["p025"] > 0.0
        and independence_max <= 1e-5
    )
    report = {
        "experiment": 183,
        "status": "PASS_CANDIDATE_FOR_FINAL_TRAIN" if passed else "FAIL_NO_DEPLOY",
        "protocol": "2023 calibration+blend lock transferred unchanged to untouched 2024",
        "lock": {"path": str(lock_path), "sha256": lock_hash},
        "source_audit": source_audit,
        "data": data_meta,
        "recipe": recipe,
        "fold_2023": lock["fold_2023"],
        "geometry_2023": geometry23,
        "calibration_from_2023": calibration,
        "locked_weight_from_2023": locked_weight,
        "fold_2024": fold24["metadata"],
        "seed_reports_2024": seed24,
        "geometry_2024_oracle_diagnostic_only": geometry24,
        "locked_2024": {
            "current_score": score(current24, fold24["y_valid"]),
            "endpoint_score": score(endpoint24, fold24["y_valid"]),
            "candidate_score": score(candidate24, fold24["y_valid"]),
            "gain": locked_gain24,
            "stability": stability,
        },
        "gate": {
            "discovery_gain_ge_8": bool(geometry23["optimal_blend_gain"] >= GAIN_GATE),
            "locked_2024_gain_ge_8": bool(locked_gain24 >= GAIN_GATE),
            "early_positive": bool(stability["early"]["gain"] > 0.0),
            "late_positive": bool(stability["late"]["gain"] > 0.0),
            "pitcher_p025_positive": bool(stability["pitcher_bootstrap"]["p025"] > 0.0),
            "row_independence_le_1e_5": bool(independence_max <= 1e-5),
        },
        "artifacts_2024": artifacts24 + [
            {"path": str(calibrated24_path), "sha256": sha256_file(calibrated24_path)},
            {"path": str(candidate24_path), "sha256": sha256_file(candidate24_path)},
        ],
        "current_2024": {"path": str(current_paths[2024]), "sha256": sha256_file(current_paths[2024])},
        "row_independence_max_abs": independence_max,
        "wall_seconds": time.time() - started,
        "post_2024_tuning": False,
        "test_read": False,
        "zip_created": False,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(
        f"[EXP183 FINAL] {report['status']} discovery={geometry23['optimal_blend_gain']:+.4f} "
        f"locked2024={locked_gain24:+.4f} early={stability['early']['gain']:+.4f} "
        f"late={stability['late']['gain']:+.4f} "
        f"p025={stability['pitcher_bootstrap']['p025']:+.4f}"
    )
    log(f"[REPORT] {report_path}")


def run_smoke(args: argparse.Namespace) -> None:
    source_audit = verify_sources()
    common = load_module(COMMON_PATH, "exp183_common_smoke")
    prep_module = load_module(PREP_PATH, "exp183_prep_smoke")
    data_dir = Path(args.data_dir).resolve()
    train_path = data_dir / "train.csv"
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    columns = list(dict.fromkeys([
        "row_id", "control_success", *common.NUM_COLS, *common.CAT_LEVELS.keys(),
    ]))
    frame = pd.read_csv(train_path, encoding="utf-8-sig", usecols=columns, nrows=args.smoke_rows)
    # Force two pseudo seasons so strict lookup paths are exercised without reading future rows.
    split = max(2, len(frame) // 2)
    frame.loc[frame.index[:split], "season"] = 2022
    frame.loc[frame.index[split:], "season"] = 2023
    empty_trackman = pd.DataFrame(columns=TRACKMAN_COLUMNS)
    raw, names, feature_meta = build_raw_x176(frame, empty_trackman, 2022, common, smoke=True)
    fit = integer_array(frame, "season") <= 2022
    prep = prep_module.make_prep(raw[fit], "std")
    transformed = prep_module.apply_prep(raw, prep, True)
    if transformed.shape != (len(frame), 352) or not np.isfinite(transformed).all():
        raise AssertionError(f"smoke preprocessing failure: {transformed.shape}")
    import torch
    import torch.nn as nn
    device = resolve_device(torch, args.device)
    network = build_network(torch, nn, 352, args.width, args.depth, args.dropout).to(device)
    network.eval()
    with torch.inference_mode():
        batch = torch.from_numpy(transformed[: min(8, len(transformed))]).to(device)
        output = torch.sigmoid(network(batch).squeeze(-1)).float().cpu().numpy()
    if not np.isfinite(output).all():
        raise AssertionError("smoke network output nonfinite")
    log(json.dumps({
        "status": "SMOKE_PASS_NO_TRAINING",
        "rows": len(frame), "raw_shape": list(raw.shape),
        "network_shape": list(transformed.shape), "feature_count": len(names),
        "feature_meta": feature_meta, "device": str(device),
        "source_audit": source_audit, "test_read": False,
    }, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "smoke"))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--current-2023", default=None)
    parser.add_argument("--current-2024", default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=3.0e-4)
    parser.add_argument("--predict-batch", type=int, default=16384)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--smoke-rows", type=int, default=4000)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if args.epochs <= 0 or args.batch <= 0 or args.predict_batch <= 0:
        parser.error("epochs/batch sizes must be positive")
    if args.mode == "validate":
        frozen = (args.seeds == [0, 1, 2] and args.width == 384 and args.depth == 3
                  and args.dropout == 0.30 and args.batch == 1024 and args.epochs == 8
                  and args.lr == 2.0e-3 and args.weight_decay == 3.0e-4)
        if not frozen:
            parser.error("validate recipe is frozen: seeds 0 1 2, width384/depth3/drop.3/batch1024/epochs8/lr.002/wd.0003")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        run_smoke(args)
    else:
        run_validate(args)


if __name__ == "__main__":
    main()
