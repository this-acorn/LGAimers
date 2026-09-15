# -*- coding: utf-8 -*-
"""[110] Strict temporal residual adapter on the frozen CAT5 + V18 champion.

This is an analysis-only gate.  It never trains/rebuilds CatBoost, never reads
test.csv, never creates an inference bundle, and never writes a fitted model.

Pre-registered protocol
-----------------------
1. Reconstruct the current champion-space OOF probability for each year:

       raw CAT5 seed mean
       -> clip(raw + .30 * frozen V18 effect)
       -> clip(.49 + 1.06 * (p - .49) - .0066)

2. Fit a pooled ExtraTrees *regressor* to the centred OOF residual ``y-p``.
   Raw player/team IDs and absolute season are excluded from model features.
   Only regular-season rows (game_type == "R") are fitted and corrected; F is
   a protected fallback whose correction is identically zero.
3. Discovery is exactly 2022 residual -> 2023.  Gamma is selected there once.
4. Only if discovery passes, refit on centred 2022+2023 residuals and evaluate
   the locked gamma on 2024.  The 2024 label cannot select gamma or structure.

Inputs are existing exact year-fold predictions/effects.  Outputs are reports
and correction arrays only:

  lab/110_temporal_residual_adapter.txt
  lab/110_temporal_residual_adapter.json
  lab/110_temporal_residual_adapter_effect_y2023.npy
  lab/110_temporal_residual_adapter_effect_y2024.npy  (discovery PASS only)

Run manually only after review:

  PYTHONIOENCODING=utf-8 py -3.12 -u exp/110_temporal_residual_adapter.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/train.csv"
LAB = ROOT / "lab"

OUT_TXT = LAB / "110_temporal_residual_adapter.txt"
OUT_JSON = LAB / "110_temporal_residual_adapter.json"
OUT_EFFECT = {
    2023: LAB / "110_temporal_residual_adapter_effect_y2023.npy",
    2024: LAB / "110_temporal_residual_adapter_effect_y2024.npy",
}

PROBABILITY_FILES = {
    2022: (
        LAB / "104_cat5_y2022_probs_seed42.npy",
        LAB / "104_cat5_y2022_probs_seed7.npy",
    ),
    2023: (
        LAB / "104_cat5_y2023_probs_seed42.npy",
        LAB / "104_cat5_y2023_probs_seed7.npy",
    ),
    2024: (
        LAB / "89_cat5_probs_seed42.npy",
        LAB / "89_cat5_probs_seed7.npy",
    ),
}
V18_EFFECT_FILES = {
    2022: LAB / "104_v18_effect_y2022.npy",
    2023: LAB / "104_v18_effect_y2023.npy",
    2024: LAB / "103_v18_effect_2024.npy",
}

# Current deployed champion transform.  The new adapter is measured after this
# transform, so its regression target is directly in final probability space.
V18_GAMMA = 0.30
AFFINE_CENTER = 0.49
AFFINE_SCALE = 1.06
AFFINE_SHIFT = 0.0066

# One fixed, deliberately smooth ExtraTrees setting.  This is not a parameter
# search.  Two seeds expose any residual Monte-Carlo instability.
ET_SEEDS = (42, 7)
ET_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 200,
    "max_features": 0.5,
    "criterion": "squared_error",
    "bootstrap": False,
}
EFFECT_CLIP = 0.05
GAMMAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)

# Pre-registered protected-domain structure.  Do not use 2024 to switch this to
# pooled/F-specific routing.  "pooled" below means one learner across all R
# pitchers, not separate entity experts.
CORRECTED_GAME_TYPE = "R"

DISCOVERY_GATE = {
    "raw_gain_min": 10.0,
    "shape_gain_min": 7.0,
    "seed_raw_gain_min": 0.0,
    "bootstrap_p025_min_exclusive": 0.0,
}
CONFIRMATION_GATE = {
    "raw_gain_min": 12.0,
    "shape_gain_min": 8.0,
    "seed_raw_gain_min": 0.0,
    "bootstrap_p025_min_exclusive": 0.0,
}
BOOTSTRAP_DRAWS = 5000

EXPECTED_ROWS = {2022: 247_472, 2023: 245_525, 2024: 253_507}
EXPECTED_TARGET_RATES = {2022: 0.528920, 2023: 0.499957, 2024: 0.486105}

# Explicit row-local contract.  Absolute season and every raw *_id are omitted.
# pitcher_id is loaded separately for cluster bootstrap only and is asserted not
# to enter the model matrix.
NUMERIC_COLUMNS = [
    "game_month",
    "game_dayofweek",
    "inning",
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
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
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
CATEGORICAL_COLUMNS = ["top_bottom", "base_state", "pitcher_hand", "batter_hand"]
META_COLUMNS = ["season", "game_type", "pitcher_id", "control_success"]
READ_COLUMNS = list(dict.fromkeys(META_COLUMNS + NUMERIC_COLUMNS + CATEGORICAL_COLUMNS))

DERIVED_NAMES = [
    "x_count_state",
    "x_count_diff",
    "x_is_3ball",
    "x_is_2strike",
    "x_is_full_count",
    "x_same_hand",
    "x_scoring_position",
    "x_any_runner",
    "x_log_pitcher_n",
    "x_log_batter_n",
    "x_log_pitchmix_n",
    "x_pitcher_batter_success_gap",
    "x_pitcher_strike_ball_gap",
    "x_form_dev1",
    "x_form_dev3",
    "x_form_dev5",
    "x_pitchmix_entropy",
    "x_pitchmix_dominant_share",
]
META_PREDICTION_NAMES = [
    "p5_success",
    "p5_middle_only",
    "p5_reverse_only",
    "p5_middle_reverse",
    "p5_bigmiss",
    "p5_entropy",
    "p5_success_margin",
    "p5_reverse_probability",
    "p5_success_given_not_reverse",
    "v18_effect",
    "champion_pre_affine",
    "champion_final",
]


@dataclass
class YearRecord:
    year: int
    frame: pd.DataFrame
    target: np.ndarray
    probabilities5: np.ndarray
    v18_effect: np.ndarray
    pre_affine: np.ndarray
    baseline: np.ndarray


@dataclass
class EncoderState:
    levels: dict[str, list[str]]
    feature_names: list[str]


STARTED = time.time()


def log(message: str = "") -> None:
    print(message, flush=True)


def tick(message: str) -> None:
    log(f"  [{time.time() - STARTED:7.1f}s] {message}")


def normalise_category(series: pd.Series) -> np.ndarray:
    return (
        series.astype("string")
        .fillna("__MISSING__")
        .astype(str)
        .to_numpy()
    )


def score(probability: np.ndarray, target: np.ndarray) -> float:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rate = float(target.mean())
    return 100000.0 * (
        1.0 - float(np.mean((probability - target) ** 2)) / (rate * (1.0 - rate))
    )


def shape_gain(base: np.ndarray, candidate: np.ndarray, target: np.ndarray) -> float:
    # No second clip: this is the exact equal-mean Brier decomposition used by
    # exp/103 and exp/104.
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
    return float(
        100000.0 * row_gain[np.asarray(mask, dtype=bool)].sum()
        / (len(target) * rate * (1.0 - rate))
    )


def evaluate(
    record: YearRecord,
    effect: np.ndarray,
    gamma: float,
) -> dict[str, float]:
    base = record.baseline.astype(np.float64, copy=False)
    target = record.target.astype(np.float64, copy=False)
    effect = np.asarray(effect, dtype=np.float64)
    if effect.shape != base.shape or not np.isfinite(effect).all():
        raise AssertionError(f"invalid effect shape/values for {record.year}")
    candidate = np.clip(base + float(gamma) * effect, 0.0, 1.0)
    raw = score(candidate, target) - score(base, target)
    shape = shape_gain(base, candidate, target)
    game_type = normalise_category(record.frame["game_type"])
    f_mask = game_type == "F"
    r_mask = game_type == "R"
    return {
        "base_score": float(score(base, target)),
        "candidate_score": float(score(candidate, target)),
        "raw_gain": float(raw),
        "shape_gain": float(shape),
        "center_contribution": float(raw - shape),
        "mean_shift": float(candidate.mean() - base.mean()),
        "F_contribution": domain_contribution(base, candidate, target, f_mask),
        "R_contribution": domain_contribution(base, candidate, target, r_mask),
        "effect_mean": float(effect.mean()),
        "effect_std": float(effect.std()),
        "effect_abs_mean": float(np.mean(np.abs(effect))),
        "effect_nonzero_share": float(np.mean(effect != 0.0)),
    }


def cluster_bootstrap_gain(
    record: YearRecord,
    effect: np.ndarray,
    gamma: float,
    draws: int,
    seed: int,
) -> dict[str, float]:
    base = record.baseline.astype(np.float64, copy=False)
    target = record.target.astype(np.float64, copy=False)
    candidate = np.clip(base + float(gamma) * effect, 0.0, 1.0)
    work = pd.DataFrame(
        {
            "pitcher": record.frame["pitcher_id"].to_numpy(),
            "n": np.ones(len(target), dtype=np.int32),
            "y": target,
            "base_se": (base - target) ** 2,
            "candidate_se": (candidate - target) ** 2,
        }
    )
    grouped = work.groupby("pitcher", sort=False, observed=True).agg(
        n=("n", "sum"),
        y=("y", "sum"),
        base_se=("base_se", "sum"),
        candidate_se=("candidate_se", "sum"),
    )
    values = grouped[["n", "y", "base_se", "candidate_se"]].to_numpy(float)
    if len(values) < 2:
        raise AssertionError("pitcher bootstrap needs at least two clusters")
    rng = np.random.default_rng(seed)
    gains = np.empty(draws, dtype=np.float64)
    groups = len(values)
    cursor = 0
    while cursor < draws:
        size = min(250, draws - cursor)
        indices = rng.integers(0, groups, size=(size, groups))
        sampled = values[indices].sum(axis=1)
        n = sampled[:, 0]
        rate = sampled[:, 1] / n
        gains[cursor : cursor + size] = 100000.0 * (
            sampled[:, 2] - sampled[:, 3]
        ) / (n * rate * (1.0 - rate))
        cursor += size
    return {
        "draws": int(draws),
        "pitchers": int(groups),
        "median": float(np.median(gains)),
        "p025": float(np.quantile(gains, 0.025)),
        "p975": float(np.quantile(gains, 0.975)),
        "prob_positive": float(np.mean(gains > 0.0)),
    }


def load_record(year: int) -> YearRecord:
    if year not in PROBABILITY_FILES:
        raise KeyError(year)
    required = [DATA, *PROBABILITY_FILES[year], V18_EFFECT_FILES[year]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required files for {year}: {missing}")

    frame = pd.read_csv(
        DATA,
        encoding="utf-8-sig",
        usecols=READ_COLUMNS,
        low_memory=False,
    )
    frame.columns = [column.replace("\ufeff", "").strip() for column in frame.columns]
    frame = frame.loc[frame["season"] == year].reset_index(drop=True)
    target = frame["control_success"].to_numpy(np.float64)

    seed_probabilities = [
        np.load(path, allow_pickle=False).astype(np.float64, copy=False)
        for path in PROBABILITY_FILES[year]
    ]
    effect = np.load(V18_EFFECT_FILES[year], allow_pickle=False).astype(
        np.float64, copy=False
    )
    expected = EXPECTED_ROWS[year]
    if len(frame) != expected or len(target) != expected:
        raise AssertionError(
            f"{year} train-row contract changed: {len(frame):,} != {expected:,}"
        )
    for seed_probability in seed_probabilities:
        if seed_probability.shape != (expected, 5):
            raise AssertionError(
                f"{year} probability shape {seed_probability.shape} != {(expected, 5)}"
            )
        if not np.isfinite(seed_probability).all():
            raise AssertionError(f"{year} probability contains non-finite values")
        if float(np.max(np.abs(seed_probability.sum(axis=1) - 1.0))) > 2e-5:
            raise AssertionError(f"{year} five-class probabilities do not sum to one")
    if effect.shape != (expected,) or not np.isfinite(effect).all():
        raise AssertionError(f"{year} V18 effect alignment/finite check failed")
    if not np.isin(target, (0.0, 1.0)).all():
        raise AssertionError(f"{year} target is not binary")
    if abs(float(target.mean()) - EXPECTED_TARGET_RATES[year]) > 2e-6:
        raise AssertionError(
            f"{year} target-rate fingerprint changed: {target.mean():.9f}"
        )

    probabilities5 = np.mean(seed_probabilities, axis=0)
    pre_affine = np.clip(
        probabilities5[:, 0] + V18_GAMMA * effect,
        0.0,
        1.0,
    )
    baseline = np.clip(
        AFFINE_CENTER + AFFINE_SCALE * (pre_affine - AFFINE_CENTER) - AFFINE_SHIFT,
        0.0,
        1.0,
    )
    return YearRecord(
        year=year,
        frame=frame,
        target=target,
        probabilities5=probabilities5,
        v18_effect=effect,
        pre_affine=pre_affine,
        baseline=baseline,
    )


def r_mask(record: YearRecord) -> np.ndarray:
    game_type = normalise_category(record.frame["game_type"])
    known = np.isin(game_type, ("R", "F"))
    if not known.all():
        unknown = sorted(set(game_type[~known]))
        raise AssertionError(f"unexpected game_type values in {record.year}: {unknown}")
    return game_type == CORRECTED_GAME_TYPE


def fit_encoder(records: Sequence[YearRecord]) -> EncoderState:
    levels: dict[str, list[str]] = {}
    for column in CATEGORICAL_COLUMNS:
        pieces = [
            normalise_category(record.frame.loc[r_mask(record), column])
            for record in records
        ]
        values = np.concatenate(pieces)
        levels[column] = sorted(set(values.tolist()))
    names = list(NUMERIC_COLUMNS) + list(DERIVED_NAMES) + list(META_PREDICTION_NAMES)
    for column in CATEGORICAL_COLUMNS:
        names.extend(f"{column}=={level}" for level in levels[column])
    if len(names) != len(set(names)):
        raise AssertionError("duplicate adapter feature name")
    forbidden = {"season", "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"}
    if forbidden.intersection(names) or any(name.endswith("_id") for name in names):
        raise AssertionError("raw ID/season leaked into the adapter feature contract")
    return EncoderState(levels=levels, feature_names=names)


def finite32(values: np.ndarray, missing: float = -999.0) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(output, nan=missing, posinf=999.0, neginf=-999.0)


def safe_numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return finite32(pd.to_numeric(frame[column], errors="coerce").to_numpy())


def build_matrix(
    record: YearRecord,
    mask: np.ndarray,
    state: EncoderState,
) -> np.ndarray:
    rows = record.frame.loc[mask]
    selected = np.flatnonzero(mask)
    arrays: list[np.ndarray] = [safe_numeric(rows, column) for column in NUMERIC_COLUMNS]

    balls = safe_numeric(rows, "balls_before")
    strikes = safe_numeric(rows, "strikes_before")
    runners2 = safe_numeric(rows, "runner_on_2b")
    runners3 = safe_numeric(rows, "runner_on_3b")
    runner_count = safe_numeric(rows, "num_runners_on")
    pitcher_n = np.maximum(safe_numeric(rows, "asof_pitcher_n"), 0.0)
    batter_n = np.maximum(safe_numeric(rows, "asof_batter_n"), 0.0)
    pitchmix_n = np.maximum(safe_numeric(rows, "asof_pitcher_pitchmix_n"), 0.0)
    pitcher_rate = safe_numeric(rows, "asof_pitcher_success_rate")
    batter_rate = safe_numeric(rows, "asof_batter_success_rate")
    strike_rate = safe_numeric(rows, "asof_pitcher_strike_rate")
    ball_rate = safe_numeric(rows, "asof_pitcher_ball_rate")
    prev1 = safe_numeric(rows, "asof_pitcher_prev1_game_success_rate")
    prev3 = safe_numeric(rows, "asof_pitcher_prev3_game_success_rate")
    prev5 = safe_numeric(rows, "asof_pitcher_prev5_game_success_rate")
    pitcher_hand = normalise_category(rows["pitcher_hand"])
    batter_hand = normalise_category(rows["batter_hand"])
    mix = np.column_stack(
        [
            safe_numeric(rows, "asof_pitcher_fastball_rate"),
            safe_numeric(rows, "asof_pitcher_breaking_rate"),
            safe_numeric(rows, "asof_pitcher_offspeed_rate"),
        ]
    )
    valid_mix = np.where((mix >= 0.0) & (mix <= 1.0), mix, 0.0)
    mix_sum = valid_mix.sum(axis=1, keepdims=True)
    normalised_mix = np.divide(
        valid_mix,
        mix_sum,
        out=np.zeros_like(valid_mix),
        where=mix_sum > 0.0,
    )
    mix_entropy = -np.sum(
        np.where(normalised_mix > 0.0, normalised_mix * np.log(normalised_mix + 1e-12), 0.0),
        axis=1,
    )

    derived = [
        balls * 3.0 + strikes,
        balls - strikes,
        (balls == 3.0).astype(np.float32),
        (strikes == 2.0).astype(np.float32),
        ((balls == 3.0) & (strikes == 2.0)).astype(np.float32),
        (pitcher_hand == batter_hand).astype(np.float32),
        ((runners2 == 1.0) | (runners3 == 1.0)).astype(np.float32),
        (runner_count > 0.0).astype(np.float32),
        np.log1p(pitcher_n),
        np.log1p(batter_n),
        np.log1p(pitchmix_n),
        pitcher_rate - batter_rate,
        strike_rate - ball_rate,
        prev1 - pitcher_rate,
        prev3 - pitcher_rate,
        prev5 - pitcher_rate,
        mix_entropy,
        normalised_mix.max(axis=1),
    ]
    if len(derived) != len(DERIVED_NAMES):
        raise AssertionError("derived feature contract mismatch")
    arrays.extend(finite32(values) for values in derived)

    p5 = record.probabilities5[selected]
    clipped_p5 = np.clip(p5, 1e-12, 1.0)
    entropy = -np.sum(clipped_p5 * np.log(clipped_p5), axis=1)
    reverse = p5[:, 2] + p5[:, 3]
    non_reverse = p5[:, 0] + p5[:, 1] + p5[:, 4]
    conditional_success = np.divide(
        p5[:, 0],
        non_reverse,
        out=np.zeros(len(p5), dtype=np.float64),
        where=non_reverse > 1e-12,
    )
    meta_prediction = [
        p5[:, 0],
        p5[:, 1],
        p5[:, 2],
        p5[:, 3],
        p5[:, 4],
        entropy,
        p5[:, 0] - np.max(p5[:, 1:], axis=1),
        reverse,
        conditional_success,
        record.v18_effect[selected],
        record.pre_affine[selected],
        record.baseline[selected],
    ]
    if len(meta_prediction) != len(META_PREDICTION_NAMES):
        raise AssertionError("prediction meta-feature contract mismatch")
    arrays.extend(finite32(values) for values in meta_prediction)

    for column in CATEGORICAL_COLUMNS:
        values = normalise_category(rows[column])
        for level in state.levels[column]:
            arrays.append((values == level).astype(np.float32))

    matrix = np.column_stack(arrays).astype(np.float32, copy=False)
    if matrix.shape != (int(mask.sum()), len(state.feature_names)):
        raise AssertionError(
            f"matrix contract mismatch {matrix.shape} != "
            f"{(int(mask.sum()), len(state.feature_names))}"
        )
    if not np.isfinite(matrix).all():
        raise AssertionError("adapter matrix contains non-finite values")
    return matrix


def centred_training_target(record: YearRecord, mask: np.ndarray) -> tuple[np.ndarray, float]:
    residual = record.target - record.baseline
    domain_residual = residual[mask].astype(np.float64, copy=True)
    mean = float(domain_residual.mean())
    domain_residual -= mean
    return domain_residual.astype(np.float32), mean


def fit_predict_effects(
    source_records: Sequence[YearRecord],
    destination: YearRecord,
    threads: int,
) -> tuple[dict[int, np.ndarray], np.ndarray, dict]:
    state = fit_encoder(source_records)
    train_matrices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    source_diagnostics: dict[str, dict[str, float]] = {}
    for record in source_records:
        mask = r_mask(record)
        matrix = build_matrix(record, mask, state)
        target, removed_mean = centred_training_target(record, mask)
        train_matrices.append(matrix)
        targets.append(target)
        source_diagnostics[str(record.year)] = {
            "rows": int(mask.sum()),
            "removed_R_residual_mean": float(removed_mean),
            "centered_target_std": float(target.std()),
        }
    x_train = np.concatenate(train_matrices, axis=0)
    y_train = np.concatenate(targets, axis=0)
    destination_mask = r_mask(destination)
    x_destination = build_matrix(destination, destination_mask, state)
    if abs(float(y_train.mean())) > 2e-7:
        # Each source year is centred separately.  Equal row counts are not
        # required, but the concatenated mean should still be numerical zero.
        raise AssertionError(f"centred residual mean drifted: {y_train.mean():+.3e}")

    effects: dict[int, np.ndarray] = {}
    seed_diagnostics: dict[str, dict[str, float]] = {}
    for seed in ET_SEEDS:
        tick(
            f"fit ExtraTrees seed={seed} sources={[r.year for r in source_records]} "
            f"rows={len(y_train):,} features={x_train.shape[1]}"
        )
        model = ExtraTreesRegressor(
            **ET_PARAMS,
            random_state=seed,
            n_jobs=threads,
        )
        model.fit(x_train, y_train)
        prediction = model.predict(x_destination).astype(np.float64)
        unclipped = prediction.copy()
        prediction = np.clip(prediction, -EFFECT_CLIP, EFFECT_CLIP)
        full_effect = np.zeros(len(destination.target), dtype=np.float64)
        full_effect[destination_mask] = prediction
        effects[seed] = full_effect
        seed_diagnostics[str(seed)] = {
            "prediction_mean_R": float(prediction.mean()),
            "prediction_std_R": float(prediction.std()),
            "prediction_abs_mean_R": float(np.mean(np.abs(prediction))),
            "clip_share_R": float(np.mean(np.abs(unclipped) > EFFECT_CLIP)),
        }
        del model, prediction, unclipped, full_effect
        gc.collect()

    ensemble = np.mean([effects[seed] for seed in ET_SEEDS], axis=0)
    if np.any(ensemble[~destination_mask] != 0.0):
        raise AssertionError("protected F rows received a non-zero correction")
    diagnostics = {
        "source": source_diagnostics,
        "seeds": seed_diagnostics,
        "train_rows": int(len(y_train)),
        "destination_R_rows": int(destination_mask.sum()),
        "features": int(x_train.shape[1]),
        "feature_names": state.feature_names,
        "seed_effect_correlation": float(
            np.corrcoef(
                effects[ET_SEEDS[0]][destination_mask],
                effects[ET_SEEDS[1]][destination_mask],
            )[0, 1]
        ),
    }
    del x_train, y_train, x_destination, train_matrices, targets
    gc.collect()
    return effects, ensemble, diagnostics


def gamma_curve(record: YearRecord, effect: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for gamma in GAMMAS:
        metrics = evaluate(record, effect, gamma)
        rows.append({"gamma": float(gamma), **metrics})
    return rows


def choose_gamma(curve: Sequence[dict[str, float]]) -> float:
    # One discovery fold only.  Maximise the weaker of raw and shape gain, then
    # raw gain, then prefer the smaller gamma.  Gamma zero can kill the axis.
    best = max(
        curve,
        key=lambda row: (
            min(row["raw_gain"], row["shape_gain"]),
            row["raw_gain"],
            -row["gamma"],
        ),
    )
    return float(best["gamma"])


def seed_metrics(
    record: YearRecord,
    effects: dict[int, np.ndarray],
    gamma: float,
) -> dict[str, dict[str, float]]:
    return {
        str(seed): evaluate(record, effects[seed], gamma)
        for seed in ET_SEEDS
    }


def pass_gate(
    gamma: float,
    ensemble_metrics: dict[str, float],
    per_seed: dict[str, dict[str, float]],
    bootstrap: dict[str, float],
    gate: dict[str, float],
) -> tuple[bool, dict[str, bool]]:
    checks = {
        "nonzero_gamma": bool(gamma > 0.0),
        "raw_gain": bool(ensemble_metrics["raw_gain"] >= gate["raw_gain_min"]),
        "shape_gain": bool(
            ensemble_metrics["shape_gain"] >= gate["shape_gain_min"]
        ),
        "all_seed_raw_nonnegative": bool(
            min(row["raw_gain"] for row in per_seed.values())
            >= gate["seed_raw_gain_min"]
        ),
        "pitcher_bootstrap_p025_positive": bool(
            bootstrap["p025"] > gate["bootstrap_p025_min_exclusive"]
        ),
        "R_nonnegative": bool(ensemble_metrics["R_contribution"] >= 0.0),
        "F_exact_fallback": bool(abs(ensemble_metrics["F_contribution"]) < 1e-10),
    }
    return bool(all(checks.values())), checks


def ensure_fresh_outputs() -> None:
    paths = [OUT_TXT, OUT_JSON, *OUT_EFFECT.values()]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite exp/110 artifacts; archive/remove explicitly first: "
            + ", ".join(existing)
        )


def save_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(values, dtype=np.float32), allow_pickle=False)
    os.replace(temporary, path)


def write_reports(result: dict, lines: Iterable[str]) -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).rstrip() + "\n"
    temporary_txt = OUT_TXT.with_suffix(OUT_TXT.suffix + ".tmp")
    temporary_json = OUT_JSON.with_suffix(OUT_JSON.suffix + ".tmp")
    temporary_txt.write_text(text, encoding="utf-8")
    temporary_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_txt, OUT_TXT)
    os.replace(temporary_json, OUT_JSON)


def append_fold_report(
    lines: list[str],
    title: str,
    gamma: float,
    ensemble_metrics: dict[str, float],
    per_seed: dict[str, dict[str, float]],
    bootstrap: dict[str, float],
    checks: dict[str, bool],
) -> None:
    lines.extend(
        [
            "",
            title,
            f"locked_gamma={gamma:.2f}",
            (
                "ensemble "
                f"base={ensemble_metrics['base_score']:.3f} "
                f"candidate={ensemble_metrics['candidate_score']:.3f} "
                f"raw={ensemble_metrics['raw_gain']:+.3f} "
                f"shape={ensemble_metrics['shape_gain']:+.3f} "
                f"center={ensemble_metrics['center_contribution']:+.3f} "
                f"mean_shift={ensemble_metrics['mean_shift']:+.7f}"
            ),
            (
                f"domain contribution R={ensemble_metrics['R_contribution']:+.3f} "
                f"F={ensemble_metrics['F_contribution']:+.3f}"
            ),
        ]
    )
    for seed in ET_SEEDS:
        row = per_seed[str(seed)]
        lines.append(
            f"seed{seed} raw={row['raw_gain']:+.3f} "
            f"shape={row['shape_gain']:+.3f} mean_shift={row['mean_shift']:+.7f}"
        )
    lines.append(
        f"pitcher bootstrap median={bootstrap['median']:+.3f} "
        f"95%=[{bootstrap['p025']:+.3f},{bootstrap['p975']:+.3f}] "
        f"P(>0)={bootstrap['prob_positive']:.4f} clusters={bootstrap['pitchers']}"
    )
    lines.append("gate " + " ".join(f"{key}={value}" for key, value in checks.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=14)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    ensure_fresh_outputs()
    LAB.mkdir(parents=True, exist_ok=True)

    lines = [
        "=== exp/110 strict temporal ExtraTrees residual adapter ===",
        "analysis only: no CatBoost training, test read, fitted-model save, bundle or submission",
        "structure locked: fit/correct R only; F correction exactly zero",
        "discovery 2022 residual -> 2023; confirmation 2022+2023 -> 2024",
        (
            f"ET={ET_PARAMS} seeds={ET_SEEDS} effect_clip={EFFECT_CLIP:.3f} "
            f"gammas={GAMMAS}"
        ),
    ]
    result: dict = {
        "experiment": 110,
        "description": "strict temporal protected ExtraTrees residual adapter",
        "analysis_only": True,
        "writes_model_or_submission": False,
        "backbone": {
            "cat5_seed_mean": [42, 7],
            "v18_gamma": V18_GAMMA,
            "affine_center": AFFINE_CENTER,
            "affine_scale": AFFINE_SCALE,
            "affine_shift": AFFINE_SHIFT,
            "adapter_position": "after frozen affine",
        },
        "structure": {
            "training_domain": CORRECTED_GAME_TYPE,
            "correction_domain": CORRECTED_GAME_TYPE,
            "F_fallback": "exactly zero correction",
            "raw_ids_excluded": True,
            "absolute_season_excluded": True,
            "source_residual_centering": "within each source year and R domain",
        },
        "model": {**ET_PARAMS, "seeds": list(ET_SEEDS), "effect_clip": EFFECT_CLIP},
        "gamma_grid": list(GAMMAS),
        "discovery_gate": DISCOVERY_GATE,
        "confirmation_gate": CONFIRMATION_GATE,
    }

    # Deliberately load only discovery records before gamma selection.  The
    # confirmation record is not constructed until the discovery gate passes.
    tick("load 2022/2023 exact CAT5 + V18 OOF records")
    y2022 = load_record(2022)
    y2023 = load_record(2023)
    discovery_effects, discovery_ensemble, discovery_fit = fit_predict_effects(
        [y2022], y2023, args.threads
    )
    save_npy(OUT_EFFECT[2023], discovery_ensemble)
    discovery_curve = gamma_curve(y2023, discovery_ensemble)
    locked_gamma = choose_gamma(discovery_curve)
    discovery_metrics = evaluate(y2023, discovery_ensemble, locked_gamma)
    discovery_seed_metrics = seed_metrics(y2023, discovery_effects, locked_gamma)
    discovery_bootstrap = cluster_bootstrap_gain(
        y2023,
        discovery_ensemble,
        locked_gamma,
        BOOTSTRAP_DRAWS,
        seed=110_2023,
    )
    discovery_pass, discovery_checks = pass_gate(
        locked_gamma,
        discovery_metrics,
        discovery_seed_metrics,
        discovery_bootstrap,
        DISCOVERY_GATE,
    )
    result["locked_gamma"] = locked_gamma
    result["discovery"] = {
        "source_years": [2022],
        "validation_year": 2023,
        "gamma_curve": discovery_curve,
        "ensemble": discovery_metrics,
        "seeds": discovery_seed_metrics,
        "bootstrap": discovery_bootstrap,
        "fit_diagnostics": discovery_fit,
        "gate_checks": discovery_checks,
        "gate_pass": discovery_pass,
        "effect_path": str(OUT_EFFECT[2023].relative_to(ROOT)),
    }
    append_fold_report(
        lines,
        "DISCOVERY (2022 residual -> 2023)",
        locked_gamma,
        discovery_metrics,
        discovery_seed_metrics,
        discovery_bootstrap,
        discovery_checks,
    )
    lines.append("gamma curve: " + " ".join(
        f"g={row['gamma']:.2f}:{row['raw_gain']:+.2f}/{row['shape_gain']:+.2f}"
        for row in discovery_curve
    ))

    if not discovery_pass:
        result["confirmation"] = None
        result["final_gate_pass"] = False
        result["stopped_after"] = "discovery"
        result["elapsed_seconds"] = float(time.time() - STARTED)
        lines.extend(
            [
                "",
                "DISCOVERY GATE FAIL — 2024 is not loaded; confirmation/model/deployment skipped.",
                f"elapsed={result['elapsed_seconds']:.1f}s",
            ]
        )
        write_reports(result, lines)
        return

    # Gamma and structure are now immutable.  No function below can reselect
    # them from 2024.
    lines.append("DISCOVERY GATE PASS — gamma/structure frozen before loading 2024.")
    tick("load 2024 confirmation record after gamma lock")
    y2024 = load_record(2024)
    confirmation_effects, confirmation_ensemble, confirmation_fit = fit_predict_effects(
        [y2022, y2023], y2024, args.threads
    )
    save_npy(OUT_EFFECT[2024], confirmation_ensemble)
    confirmation_metrics = evaluate(y2024, confirmation_ensemble, locked_gamma)
    confirmation_seed_metrics = seed_metrics(
        y2024, confirmation_effects, locked_gamma
    )
    confirmation_bootstrap = cluster_bootstrap_gain(
        y2024,
        confirmation_ensemble,
        locked_gamma,
        BOOTSTRAP_DRAWS,
        seed=110_2024,
    )
    confirmation_pass, confirmation_checks = pass_gate(
        locked_gamma,
        confirmation_metrics,
        confirmation_seed_metrics,
        confirmation_bootstrap,
        CONFIRMATION_GATE,
    )
    result["confirmation"] = {
        "source_years": [2022, 2023],
        "validation_year": 2024,
        "gamma_was_locked_before_load": True,
        "ensemble": confirmation_metrics,
        "seeds": confirmation_seed_metrics,
        "bootstrap": confirmation_bootstrap,
        "fit_diagnostics": confirmation_fit,
        "gate_checks": confirmation_checks,
        "gate_pass": confirmation_pass,
        "effect_path": str(OUT_EFFECT[2024].relative_to(ROOT)),
    }
    result["final_gate_pass"] = confirmation_pass
    result["stopped_after"] = "confirmation"
    result["elapsed_seconds"] = float(time.time() - STARTED)
    append_fold_report(
        lines,
        "CONFIRMATION (2022+2023 residual -> 2024; locked gamma)",
        locked_gamma,
        confirmation_metrics,
        confirmation_seed_metrics,
        confirmation_bootstrap,
        confirmation_checks,
    )
    lines.extend(
        [
            "",
            "FINAL GATE " + ("PASS" if confirmation_pass else "FAIL"),
            "PASS still authorizes analysis only; this script never builds a deployment artifact.",
            f"elapsed={result['elapsed_seconds']:.1f}s",
        ]
    )
    write_reports(result, lines)


if __name__ == "__main__":
    main()
