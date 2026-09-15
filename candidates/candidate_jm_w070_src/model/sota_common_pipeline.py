"""Common, leakage-safe feature and validation pipeline for SOTA model bake-offs.

The module deliberately separates semantic feature construction from the small
model-specific representation adapters.  Every candidate therefore receives
the same rows, time weights, hierarchical base probability, residual target,
and feature meanings.  CatBoost may consume categorical strings natively,
while the remaining estimators receive train-fitted ordinal codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ID_COL = "row_id"
TARGET_COL = "control_success"


RAW_CATEGORICAL_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "game_month",
    "game_dayofweek",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_id",
    "batter_id",
]

INTERACTION_CATEGORICAL_COLS = [
    "count_state",
    "hand_matchup",
    "count_base_context",
    "hand_count_context",
    "inning_score_context",
    "pressure_context",
    "recent_regime",
]

CATEGORICAL_COLS = RAW_CATEGORICAL_COLS + INTERACTION_CATEGORICAL_COLS


def _numeric(frame: pd.DataFrame, column: str, fallback: float = np.nan) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.fillna(fallback) if np.isfinite(fallback) else values


def _strict_previous_season_prior(data: pd.DataFrame) -> np.ndarray:
    """Return a prior formed only from seasons strictly before each row."""
    seasons = sorted(int(value) for value in data["season"].dropna().unique())
    season_summary = data.groupby("season", observed=True)[TARGET_COL].agg(["size", "sum"])
    cumulative_n = 0.0
    cumulative_y = 0.0
    global_before: dict[int, float] = {}
    for season in seasons:
        global_before[season] = cumulative_y / cumulative_n if cumulative_n else 0.5
        if season in season_summary.index:
            cumulative_n += float(season_summary.loc[season, "size"])
            cumulative_y += float(season_summary.loc[season, "sum"])

    by_type = (
        data.groupby(["season", "game_type"], observed=True)[TARGET_COL]
        .mean()
        .to_dict()
    )
    prior = np.empty(len(data), dtype=np.float64)
    season_values = data["season"].to_numpy()
    game_type_values = data["game_type"].astype(str).to_numpy()
    for index, (season, game_type) in enumerate(zip(season_values, game_type_values)):
        season = int(season)
        prior[index] = float(by_type.get((season - 1, game_type), global_before[season]))
    return np.clip(prior, 1e-4, 1.0 - 1e-4)


def finish_common_features(
    enriched: pd.DataFrame,
    prior: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Finish the shared row-local feature table from an explicit prior.

    Training obtains ``enriched`` and ``prior`` from labeled seasons.  The
    deployment path obtains them solely from the saved training artifact and
    each test row's official ``asof_*`` values.  Keeping the remaining feature
    logic in this one function prevents train/inference drift.
    """
    # Both callers pass a newly created feature frame, so in-place extension
    # avoids another multi-gigabyte copy on the 1.47M-row full training set.
    x = enriched
    prior = np.asarray(prior, dtype=np.float64)
    if len(prior) != len(x):
        raise ValueError("Prior length does not match the feature frame")

    recent_cols = [f"asof_pitcher_prev{k}_game_success_rate" for k in (1, 3, 5)]
    recent_middle_cols = [f"asof_pitcher_prev{k}_game_middle_rate" for k in (1, 3, 5)]
    recent = x[recent_cols].apply(pd.to_numeric, errors="coerce")
    recent_middle = x[recent_middle_cols].apply(pd.to_numeric, errors="coerce")
    recent_mean = recent.mean(axis=1).fillna(pd.Series(prior, index=x.index))
    recent_std = recent.std(axis=1).fillna(0.15).clip(0.0, 0.5)

    pitcher_n = _numeric(x, "asof_pitcher_n", 0.0).clip(lower=0.0)
    batter_n = _numeric(x, "asof_batter_n", 0.0).clip(lower=0.0)
    pitchmix_n = _numeric(x, "asof_pitcher_pitchmix_n", 0.0).clip(lower=0.0)
    career_rate = _numeric(x, "asof_pitcher_success_rate")
    career_rate = career_rate.fillna(pd.Series(prior, index=x.index)).clip(0.0, 1.0)
    batter_rate = _numeric(x, "asof_batter_success_rate")
    batter_rate = batter_rate.fillna(pd.Series(prior, index=x.index)).clip(0.0, 1.0)

    dynamic_strength = (55.0 + 220.0 * recent_std + 40.0 / (1.0 + np.log1p(pitcher_n))).clip(50.0, 180.0)
    career_base = (career_rate * pitcher_n + prior * dynamic_strength) / (pitcher_n + dynamic_strength)

    season_n = _numeric(x, "derived_pitcher_season_n", 0.0).clip(lower=0.0)
    season_raw = _numeric(x, "derived_pitcher_season_success_rate")
    season_raw = season_raw.fillna(pd.Series(prior, index=x.index)).clip(0.0, 1.0)
    season_estimate = (season_raw * season_n + prior * 30.0) / (season_n + 30.0)
    season_reliability = season_n / (season_n + 80.0)
    season_weight = 0.15 + 0.30 * season_reliability
    base = career_base + season_weight * (season_estimate - career_base)
    base = np.clip(base.to_numpy(dtype=np.float64), 1e-5, 1.0 - 1e-5)

    batter_season_n = _numeric(x, "derived_batter_season_n", 0.0).clip(lower=0.0)
    batter_season_raw = _numeric(x, "derived_batter_season_success_rate")
    batter_season_raw = batter_season_raw.fillna(pd.Series(prior, index=x.index)).clip(0.0, 1.0)
    batter_season_estimate = (
        batter_season_raw * batter_season_n + prior * 40.0
    ) / (batter_season_n + 40.0)

    x["strict_previous_season_prior"] = prior
    x["hierarchical_base_probability"] = base
    x["career_dynamic_base"] = career_base
    x["dynamic_smoothing_strength"] = dynamic_strength
    x["season_form_estimate"] = season_estimate
    x["season_form_reliability"] = season_reliability
    x["batter_season_success_smoothed"] = batter_season_estimate
    x["batter_season_reliability"] = batter_season_n / (batter_season_n + 80.0)

    x["pitcher_log_n"] = np.log1p(pitcher_n)
    x["batter_log_n"] = np.log1p(batter_n)
    x["pitchmix_log_n"] = np.log1p(pitchmix_n)
    x["pitcher_uncertainty"] = 1.0 / np.sqrt(pitcher_n + 1.0)
    x["batter_uncertainty"] = 1.0 / np.sqrt(batter_n + 1.0)
    x["count_pressure"] = _numeric(x, "balls_before", 0.0) - _numeric(x, "strikes_before", 0.0)
    x["late_inning"] = (_numeric(x, "inning", 0.0) >= 7).astype("int8")
    x["high_leverage"] = (_numeric(x, "li", 0.0) >= 2.0).astype("int8")
    x["scoring_position"] = ((x["runner_on_2b"] == 1) | (x["runner_on_3b"] == 1)).astype("int8")
    x["runners_x_li"] = _numeric(x, "num_runners_on", 0.0) * _numeric(x, "li", 0.0)
    x["score_abs"] = _numeric(x, "score_diff_pitcher_team", 0.0).abs()
    x["close_game"] = (x["score_abs"] <= 1.0).astype("int8")
    x["pressure_index"] = (
        (_numeric(x, "balls_before", 0.0) + 1.0)
        / (_numeric(x, "strikes_before", 0.0) + 1.0)
        * (1.0 + _numeric(x, "num_runners_on", 0.0))
        * np.log1p(_numeric(x, "li", 0.0).clip(lower=0.0))
    )
    x["recent_success_mean"] = recent_mean
    x["recent_success_std"] = recent_std
    x["recent_weighted_success"] = 0.55 * recent.iloc[:, 0] + 0.30 * recent.iloc[:, 1] + 0.15 * recent.iloc[:, 2]
    x["recent_acceleration"] = recent.iloc[:, 0] - 2.0 * recent.iloc[:, 1] + recent.iloc[:, 2]
    x["recent_vs_dynamic_base"] = recent_mean - career_base
    x["recent_middle_weighted"] = 0.55 * recent_middle.iloc[:, 0] + 0.30 * recent_middle.iloc[:, 1] + 0.15 * recent_middle.iloc[:, 2]
    x["failure_profile_sum"] = _numeric(x, "asof_pitcher_middle_rate", 0.0) + _numeric(x, "asof_pitcher_reverse_rate", 0.0)
    x["success_failure_margin"] = career_rate - x["failure_profile_sum"]
    x["batter_career_gap"] = career_rate - batter_rate
    x["batter_pitcher_form_gap"] = base - batter_season_estimate.to_numpy(dtype=np.float64)
    x["exposure_ratio"] = np.log1p(batter_n) - np.log1p(pitcher_n)

    rates = x[["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]].apply(pd.to_numeric, errors="coerce").clip(1e-7, 1.0)
    x["pitchmix_entropy"] = -(rates * np.log(rates)).sum(axis=1, min_count=1)

    balls = _numeric(x, "balls_before", 0.0).astype(int).astype(str)
    strikes = _numeric(x, "strikes_before", 0.0).astype(int).astype(str)
    x["count_state"] = balls + "-" + strikes
    x["hand_matchup"] = x["pitcher_hand"].astype(str) + "-" + x["batter_hand"].astype(str)
    x["count_base_context"] = x["count_state"] + "|" + x["base_state"].astype(str)
    x["hand_count_context"] = x["hand_matchup"] + "|" + x["count_state"]
    inning_band = pd.cut(_numeric(x, "inning", 0.0), [-np.inf, 3, 6, 9, np.inf], labels=["early", "middle", "late", "extra"]).astype(str)
    score_band = pd.cut(_numeric(x, "score_diff_pitcher_team", 0.0), [-np.inf, -3, -1, 1, 3, np.inf], labels=["far_behind", "behind", "close", "ahead", "far_ahead"]).astype(str)
    pressure_band = pd.cut(_numeric(x, "li", 0.0), [-np.inf, 0.75, 1.5, 3.0, np.inf], labels=["low", "normal", "high", "extreme"]).astype(str)
    x["inning_score_context"] = inning_band + "|" + score_band
    x["pressure_context"] = pressure_band + "|" + x["num_runners_on"].astype(str) + "|" + x["count_state"]
    x["recent_regime"] = np.select([recent_std < 0.04, recent_std < 0.10], ["stable", "normal"], default="volatile")

    feature_cols = [column for column in x.columns if column not in {ID_COL, TARGET_COL}]
    x = x[feature_cols].replace([np.inf, -np.inf], np.nan)
    for column in CATEGORICAL_COLS:
        if column in x:
            x[column] = x[column].astype("string").fillna("__MISSING__").astype(str)
    if TARGET_COL in x or ID_COL in x:
        raise AssertionError("Identifier or target leaked into the common feature table")
    return x, base


def build_common_features(data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the shared feature table and hierarchical base probability.

    `data` must be labeled historical data.  All target-derived quantities use
    prior seasons only.  All other transformations are row-local.
    """
    required = {ID_COL, TARGET_COL, "season", "game_type"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Imported lazily so the deployment package can reuse the row-local feature
    # finisher without carrying the labeled-training feature builder.
    from time_split_benchmark import add_time_safe_features

    enriched = add_time_safe_features(data)
    prior = _strict_previous_season_prior(data)
    return finish_common_features(enriched, prior)


@dataclass
class PreparedSOTASplit:
    x_train: pd.DataFrame
    x_valid: pd.DataFrame
    y_train: np.ndarray
    y_valid: np.ndarray
    residual_train: np.ndarray
    base_train: np.ndarray
    base_valid: np.ndarray
    sample_weight: np.ndarray
    valid_row_id: np.ndarray
    valid_game_type: np.ndarray
    categorical_cols: list[str]


def stratified_season_selection(
    data: pd.DataFrame,
    rows_per_season: int | None,
    seed: int,
) -> np.ndarray:
    """Select model-training rows after full-history feature construction."""
    selected = np.zeros(len(data), dtype=bool)
    if rows_per_season is None:
        selected[:] = True
        return selected
    rng = np.random.default_rng(seed)
    for _, indices in data.groupby("season", observed=True).indices.items():
        indices = np.asarray(indices)
        chosen = indices if len(indices) <= rows_per_season else rng.choice(indices, rows_per_season, replace=False)
        selected[chosen] = True
    return selected


def prepare_sota_split(
    data: pd.DataFrame,
    valid_season: int = 2024,
    train_start_season: int | None = None,
    year_decay: float = 0.55,
    rows_per_season: int | None = None,
    seed: int = 42,
) -> PreparedSOTASplit:
    if not 0.0 < year_decay <= 1.0:
        raise ValueError("year_decay must be in (0, 1]")
    x, base = build_common_features(data)
    selection = stratified_season_selection(data, rows_per_season, seed)
    train_mask = (data["season"].to_numpy() < valid_season) & selection
    valid_mask = (data["season"].to_numpy() == valid_season) & selection
    if train_start_season is not None:
        train_mask &= data["season"].to_numpy() >= train_start_season
    if not train_mask.any() or not valid_mask.any():
        raise ValueError("The requested train/validation split is empty")

    target = data[TARGET_COL].to_numpy(dtype=np.float64)
    train_seasons = data.loc[train_mask, "season"].to_numpy(dtype=np.int64)
    weights = np.power(year_decay, (valid_season - 1) - train_seasons)
    categorical = [column for column in CATEGORICAL_COLS if column in x]
    return PreparedSOTASplit(
        x_train=x.loc[train_mask].reset_index(drop=True),
        x_valid=x.loc[valid_mask].reset_index(drop=True),
        y_train=target[train_mask],
        y_valid=target[valid_mask],
        residual_train=target[train_mask] - base[train_mask],
        base_train=base[train_mask],
        base_valid=base[valid_mask],
        sample_weight=weights.astype(np.float64),
        valid_row_id=data.loc[valid_mask, ID_COL].to_numpy(),
        valid_game_type=data.loc[valid_mask, "game_type"].astype(str).to_numpy(),
        categorical_cols=categorical,
    )


@dataclass
class TrainOnlyOrdinalAdapter:
    categorical_cols: list[str]
    feature_cols: list[str] | None = None
    category_maps: dict[str, dict[str, int]] | None = None
    medians: dict[str, float] | None = None

    def fit(self, frame: pd.DataFrame) -> "TrainOnlyOrdinalAdapter":
        self.feature_cols = list(frame.columns)
        self.category_maps = {}
        self.medians = {}
        for column in self.feature_cols:
            if column in self.categorical_cols:
                values = frame[column].astype("string").fillna("__MISSING__")
                self.category_maps[column] = {
                    value: index for index, value in enumerate(sorted(values.unique()))
                }
            else:
                numeric = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
                median = float(numeric.median())
                self.medians[column] = median if np.isfinite(median) else 0.0
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.feature_cols is None or self.category_maps is None or self.medians is None:
            raise RuntimeError("Adapter is not fitted")
        if list(frame.columns) != self.feature_cols:
            raise ValueError("Feature schema or order changed")
        output: dict[str, np.ndarray] = {}
        for column in self.feature_cols:
            if column in self.categorical_cols:
                values = frame[column].astype("string").fillna("__MISSING__")
                output[column] = values.map(self.category_maps[column]).fillna(-1).to_numpy(np.int32)
            else:
                numeric = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
                output[column] = numeric.fillna(self.medians[column]).to_numpy(np.float32)
        return pd.DataFrame(output, index=frame.index)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)


def catboost_frame(frame: pd.DataFrame, categorical_cols: Iterable[str]) -> pd.DataFrame:
    """Return a CatBoost-compatible view without learning from validation rows."""
    output = frame.copy()
    categorical = set(categorical_cols)
    for column in output.columns:
        if column in categorical:
            output[column] = output[column].astype("string").fillna("__MISSING__").astype(str)
        else:
            output[column] = pd.to_numeric(output[column], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return output


def score_probabilities(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    target = np.asarray(y_true, dtype=np.float64)
    brier = float(brier_score_loss(target, probability))
    event_rate = float(target.mean())
    reference = event_rate * (1.0 - event_rate)
    bss = max(0.0, 100_000.0 * (1.0 - brier / reference))
    return {
        "brier": brier,
        "bss": bss,
        "auc": float(roc_auc_score(target, probability)),
        "event_rate": event_rate,
        "prediction_mean": float(probability.mean()),
        "prediction_std": float(probability.std()),
    }
