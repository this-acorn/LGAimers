# -*- coding: utf-8 -*-
"""
[111] Audit whether the hidden denominators of the official prev1/3/5-game
success/middle rates can be recovered from the two rounded rates alone.

This is a read-only feature pre-gate:

* no model is trained;
* no test row or submission is read or written;
* decoder tolerance, search ranges, and tie-break rule are selected with
  2020-2021 only;
* 2022, 2023, and 2024 are untouched reports;
* TrackMan game_id/game_date is used only to certify the historical game
  blocks used as denominator truth.  It is not proposed as a deploy feature.

The main table has no game_id.  We first split each pitcher's monotonically
increasing asof counter into game blocks.  A block boundary is visible because
the official previous-game rate tuple and/or the supplied calendar state
changes.  We then attach TrackMan game_id/game_date anchors using the same
strict 1:1 context join audited in exp/37 and exp/91c.  Only windows for which
the current block and every preceding block in the window have a single,
chronologically consistent TrackMan game anchor enter the primary report.

Outputs (new files only):
  lab/111_prev_window_denominator_audit.txt
  lab/111_prev_window_denominator_audit.json

Run:
  py -3.12 -u exp/111_prev_window_denominator_audit.py
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


DATA = Path("data/train.csv")
TRACKMAN = Path("data/trackman_history.csv")
ENTITY_MAP = Path("lab/entity_map_pitcher.csv")
OUT_TXT = Path("lab/111_prev_window_denominator_audit.txt")
OUT_JSON = Path("lab/111_prev_window_denominator_audit.json")

DISCOVERY_YEARS = (2020, 2021)
REPORT_YEARS = (2022, 2023, 2024)
WINDOWS = (1, 3, 5)

# Official rates are stored at about six decimal places.  The final tolerance
# is learned from 2020-2021 truth, but it is never allowed below this purely
# representational floor or above the conservative cap.
ROUNDING_FLOOR = 5.1e-7
ROUNDING_CAP = 5.0e-6

TEAM_MAP = {
    12: "DOO_BEA",
    13: "LG_TWI",
    14: "KIW_HER",
    15: "LOT_GIA",
    16: "KIA_TIG",
    17: "HAN_EAG",
    18: "SAM_LIO",
    19: "NC_DIN",
    20: "KT_WIZ",
    21: "SSG_LAN",
}
TM_TEAM_ALIAS = {"SK_WYV": "SSG_LAN"}

JOIN_KEY = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "ph",
    "bh",
    "pt",
    "bt",
]

RECENT_COLS: list[str] = []
for _window in WINDOWS:
    RECENT_COLS.extend(
        [
            f"asof_pitcher_prev{_window}_game_success_rate",
            f"asof_pitcher_prev{_window}_game_middle_rate",
        ]
    )

TRAIN_COLS = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "asof_pitcher_n",
    "asof_pitcher_middle_rate",
    "control_success",
] + RECENT_COLS

TM_COLS = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team",
    "batter_team",
    "pitcher_trackman_id",
    "trackman_game_id",
    "game_date",
]


T0 = time.time()
LINES: list[str] = []


def log(message: str = "") -> None:
    line = f"[{time.time() - T0:7.1f}s] {message}"
    print(line, flush=True)
    LINES.append(line)


def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [str(c).replace("\ufeff", "").strip() for c in frame.columns]
    return frame


def safe_float(value: object) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def make_game_segments(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Create main-table pitcher/game blocks without using a target outcome.

    The rolling official tuple is constant inside a game.  Calendar/team/side
    changes provide additional boundaries.  TrackMan anchors later certify
    the blocks independently; impure/missing anchors never enter the primary
    decoder evaluation.
    """

    ordered = train.sort_values(
        ["pitcher_id", "asof_pitcher_n", "_row"], kind="mergesort"
    ).reset_index(drop=True)

    pid = ordered["pitcher_id"].to_numpy(np.int64)
    n_asof = ordered["asof_pitcher_n"].fillna(-1).to_numpy(np.int64)
    same_pitcher = np.r_[False, pid[1:] == pid[:-1]]
    counter_contiguous = np.r_[False, n_asof[1:] == n_asof[:-1] + 1]

    # NaNs must compare equal inside the first career game.
    rate_matrix = ordered[RECENT_COLS].to_numpy(np.float64)
    rate_matrix = np.nan_to_num(rate_matrix, nan=-9.0)
    rate_same = np.r_[
        False,
        np.all(np.abs(rate_matrix[1:] - rate_matrix[:-1]) <= 1e-12, axis=1),
    ]

    stable_cols = [
        "season",
        "game_month",
        "game_dayofweek",
        "game_type",
        "top_bottom",
        "pitcher_team_id",
    ]
    stable = ordered[stable_cols].astype(str).to_numpy()
    stable_same = np.r_[False, np.all(stable[1:] == stable[:-1], axis=1)]

    continuation = same_pitcher & counter_contiguous & rate_same & stable_same
    ordered["segment_id"] = np.cumsum(~continuation).astype(np.int64) - 1

    # Put the segment id back in original train-row order for the TrackMan join.
    row_to_segment = ordered.set_index("_row")["segment_id"]
    train["segment_id"] = row_to_segment.reindex(train["_row"]).to_numpy(np.int64)

    first_cols = [
        "pitcher_id",
        "season",
        "game_month",
        "game_dayofweek",
        "game_type",
        "top_bottom",
        "pitcher_team_id",
        "asof_pitcher_n",
        "asof_pitcher_middle_rate",
    ] + RECENT_COLS
    first = ordered.groupby("segment_id", sort=False)[first_cols].first()
    agg = ordered.groupby("segment_id", sort=False).agg(
        game_n=("_row", "size"),
        success_count=("control_success", "sum"),
        first_row=("_row", "min"),
        last_row=("_row", "max"),
        n_last=("asof_pitcher_n", "max"),
    )
    segments = first.join(agg).reset_index()
    segments = segments.sort_values(
        ["pitcher_id", "asof_pitcher_n", "segment_id"], kind="mergesort"
    ).reset_index(drop=True)
    segments["pitcher_game_index"] = segments.groupby("pitcher_id").cumcount()

    # The next block's cumulative middle total minus this block's start total
    # gives the exact integer number of middle events in this game.  This is
    # used only as audit truth, never as a deploy feature.
    cumulative_middle = (
        segments["asof_pitcher_n"].to_numpy(np.float64)
        * segments["asof_pitcher_middle_rate"].fillna(0.0).to_numpy(np.float64)
    )
    next_cumulative = pd.Series(cumulative_middle).groupby(segments["pitcher_id"]).shift(-1)
    middle_delta = next_cumulative.to_numpy(np.float64) - cumulative_middle
    middle_rounded = np.rint(middle_delta)
    middle_valid = (
        np.isfinite(middle_delta)
        & (np.abs(middle_delta - middle_rounded) <= 2e-2)
        & (middle_rounded >= 0)
        & (middle_rounded <= segments["game_n"].to_numpy(np.float64))
    )
    segments["middle_count"] = np.where(middle_valid, middle_rounded, np.nan)

    counter_size = segments["n_last"].to_numpy(np.int64) - segments[
        "asof_pitcher_n"
    ].to_numpy(np.int64) + 1
    counter_ok = counter_size == segments["game_n"].to_numpy(np.int64)
    diagnostics = {
        "rows": int(len(train)),
        "segments": int(len(segments)),
        "counter_contiguous_row_fraction": float(counter_contiguous[1:].mean()),
        "segment_counter_size_agreement": float(counter_ok.mean()),
        "middle_count_coverage": float(np.isfinite(segments["middle_count"]).mean()),
        "median_game_n": float(segments["game_n"].median()),
        "p99_game_n": float(segments["game_n"].quantile(0.99)),
        "max_game_n": int(segments["game_n"].max()),
    }
    return train, segments, diagnostics


def attach_trackman_anchors(
    train: pd.DataFrame, segments: pd.DataFrame, smoke_rows: int | None = None
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Certify game blocks with strict 1:1 TrackMan context matches."""

    log("TrackMan game/date anchor loading...")
    trackman = clean_columns(
        pd.read_csv(
            TRACKMAN,
            encoding="utf-8-sig",
            usecols=TM_COLS,
            nrows=smoke_rows,
            low_memory=False,
        )
    )

    hand_map = {2: "Right", 1: "Left", 2.0: "Right", 1.0: "Left"}
    main_key = train[
        [
            "_row",
            "segment_id",
            "pitcher_id",
            "season",
            "game_month",
            "game_dayofweek",
            "inning",
            "top_bottom",
            "balls_before",
            "strikes_before",
            "outs_before",
            "pitcher_hand",
            "batter_hand",
            "pitcher_team_id",
            "batter_team_id",
        ]
    ].copy()
    main_key["ph"] = main_key["pitcher_hand"].map(hand_map)
    main_key["bh"] = main_key["batter_hand"].map(hand_map)
    main_key["pt"] = main_key["pitcher_team_id"].map(TEAM_MAP)
    main_key["bt"] = main_key["batter_team_id"].map(TEAM_MAP)
    main_key["top_bottom"] = main_key["top_bottom"].astype(str).str[0]

    trackman["pt"] = trackman["pitcher_team"].replace(TM_TEAM_ALIAS)
    trackman["bt"] = trackman["batter_team"].replace(TM_TEAM_ALIAS)
    trackman["ph"] = trackman["pitcher_hand"].astype(str)
    trackman["bh"] = trackman["batter_hand"].astype(str)
    trackman["top_bottom"] = trackman["top_bottom"].astype(str).str[0]
    numeric_key = [
        "season",
        "game_month",
        "game_dayofweek",
        "inning",
        "balls_before",
        "strikes_before",
        "outs_before",
    ]
    for column in numeric_key:
        main_key[column] = pd.to_numeric(main_key[column], errors="coerce").astype("Int64")
        trackman[column] = pd.to_numeric(trackman[column], errors="coerce").astype("Int64")

    main_eligible = main_key.dropna(subset=["pt", "bt", "ph", "bh"])
    tm_eligible = trackman[
        trackman["pt"].isin(TEAM_MAP.values())
        & trackman["bt"].isin(TEAM_MAP.values())
    ]
    main_cardinality = main_eligible.groupby(JOIN_KEY, observed=True).size()
    tm_cardinality = tm_eligible.groupby(JOIN_KEY, observed=True).size()
    keys_11 = main_cardinality[main_cardinality == 1].index.intersection(
        tm_cardinality[tm_cardinality == 1].index
    )

    tm_unique = (
        tm_eligible.set_index(JOIN_KEY)
        .loc[
            keys_11,
            ["pitcher_trackman_id", "trackman_game_id", "game_date"],
        ]
        .reset_index()
    )
    matched = main_eligible[
        JOIN_KEY + ["_row", "segment_id", "pitcher_id"]
    ].merge(tm_unique, on=JOIN_KEY, how="inner", validate="m:1")
    if not matched["_row"].is_unique:
        raise AssertionError("strict 1:1 TrackMan join produced duplicate main rows")
    matched_before_entity_guard = int(len(matched))

    # The independently saved high-purity player map is used as an additional
    # guard against context collisions.  Unmapped pitchers remain eligible;
    # mapped disagreements are rejected.
    mapped_rows = 0
    map_agreement = float("nan")
    if ENTITY_MAP.is_file():
        entity = pd.read_csv(ENTITY_MAP)
        entity = entity[(entity["purity"] >= 0.95) & (entity["votes"] >= 5)]
        pid_to_tid = dict(zip(entity["id"].astype(int), entity["tid"]))
        matched["known_tid"] = matched["pitcher_id"].map(pid_to_tid)
        comparable = matched["known_tid"].notna()
        mapped_rows = int(comparable.sum())
        if mapped_rows:
            known_numeric = pd.to_numeric(
                matched.loc[comparable, "known_tid"], errors="coerce"
            ).to_numpy(np.float64)
            observed_numeric = pd.to_numeric(
                matched.loc[comparable, "pitcher_trackman_id"], errors="coerce"
            ).to_numpy(np.float64)
            agree_comparable = (
                np.isfinite(known_numeric)
                & np.isfinite(observed_numeric)
                & (np.abs(known_numeric - observed_numeric) < 0.5)
            )
            map_agreement = float(agree_comparable.mean())
            keep = np.ones(len(matched), dtype=bool)
            keep[np.flatnonzero(comparable.to_numpy())] = agree_comparable
            matched = matched.loc[keep].copy()

    # TrackMan changes date representation across seasons (e.g. 03/29/2019
    # versus 2024-10-08).  A one-shot pandas inference locks onto the first
    # representation and silently turns later seasons into NaT, so parse both
    # documented forms explicitly.
    raw_date = matched["game_date"].astype(str)
    date_iso = pd.to_datetime(raw_date, format="%Y-%m-%d", errors="coerce")
    date_us = pd.to_datetime(raw_date, format="%m/%d/%Y", errors="coerce")
    matched["game_date_parsed"] = date_iso.fillna(date_us)
    matched_rows_by_season = {
        str(int(year)): int(count)
        for year, count in matched.groupby("season", observed=True).size().items()
    }
    anchor = matched.groupby("segment_id", sort=False).agg(
        anchor_rows=("_row", "size"),
        anchor_game_nunique=("trackman_game_id", "nunique"),
        anchor_date_nunique=("game_date_parsed", "nunique"),
        anchor_tid_nunique=("pitcher_trackman_id", "nunique"),
        anchor_game_id=("trackman_game_id", "first"),
        anchor_game_date=("game_date_parsed", "first"),
        anchor_tid=("pitcher_trackman_id", "first"),
    )
    anchor["anchor_valid"] = (
        (anchor["anchor_rows"] > 0)
        & (anchor["anchor_game_nunique"] == 1)
        & (anchor["anchor_date_nunique"] == 1)
        & (anchor["anchor_tid_nunique"] == 1)
        & anchor["anchor_game_date"].notna()
    )
    segments = segments.merge(anchor.reset_index(), on="segment_id", how="left")
    segments["anchor_valid"] = segments["anchor_valid"].fillna(False).astype(bool)
    valid_anchors_by_season = {
        str(int(year)): int(group["anchor_valid"].sum())
        for year, group in segments.groupby("season", observed=True)
    }

    diagnostics = {
        "trackman_rows": int(len(trackman)),
        "main_team_mapped_fraction": float(main_key["pt"].notna().mean()),
        "strict_11_matched_rows_before_entity_guard": matched_before_entity_guard,
        "strict_11_matched_rows": int(len(matched)),
        "strict_11_matched_row_fraction": float(len(matched) / max(len(train), 1)),
        "strict_11_matched_rows_by_season": matched_rows_by_season,
        "entity_map_comparable_rows": mapped_rows,
        "entity_map_agreement_before_reject": map_agreement,
        "segments_with_any_anchor": int(anchor.shape[0]),
        "segments_with_valid_anchor": int(anchor["anchor_valid"].sum()),
        "valid_anchor_segment_fraction": float(segments["anchor_valid"].mean()),
        "valid_anchor_segments_by_season": valid_anchors_by_season,
        "impure_anchor_segments": int((~anchor["anchor_valid"]).sum()),
    }

    del trackman, main_key, main_eligible, tm_eligible, main_cardinality
    del tm_cardinality, keys_11, tm_unique, matched, anchor
    gc.collect()
    return segments, diagnostics


def attach_true_windows(segments: pd.DataFrame) -> pd.DataFrame:
    """Add true preceding 1/3/5-game counts and anchor certification."""

    segments = segments.sort_values(
        ["pitcher_id", "asof_pitcher_n", "segment_id"], kind="mergesort"
    ).reset_index(drop=True)
    arrays: dict[tuple[int, str], np.ndarray] = {}
    for window in WINDOWS:
        for suffix in [
            "n_true",
            "success_true",
            "middle_true",
            "anchor_window",
            "date_order_ok",
        ]:
            dtype = bool if suffix in {"anchor_window", "date_order_ok"} else float
            default = False if dtype is bool else np.nan
            arrays[(window, suffix)] = np.full(len(segments), default, dtype=dtype)

    for _, loc in segments.groupby("pitcher_id", sort=False).indices.items():
        idx = np.asarray(loc, dtype=np.int64)
        game_n = segments.loc[idx, "game_n"].to_numpy(np.float64)
        success = segments.loc[idx, "success_count"].to_numpy(np.float64)
        middle = segments.loc[idx, "middle_count"].to_numpy(np.float64)
        anchored = segments.loc[idx, "anchor_valid"].to_numpy(bool)
        tids = segments.loc[idx, "anchor_tid"].astype(str).to_numpy()
        dates = pd.to_datetime(segments.loc[idx, "anchor_game_date"], errors="coerce").to_numpy()
        game_ids = segments.loc[idx, "anchor_game_id"].astype(str).to_numpy()

        cumulative_n = np.r_[0.0, np.cumsum(game_n)]
        cumulative_success = np.r_[0.0, np.cumsum(success)]
        cumulative_middle = np.r_[0.0, np.cumsum(np.nan_to_num(middle, nan=0.0))]
        cumulative_middle_missing = np.r_[0, np.cumsum(~np.isfinite(middle))]

        if len(idx) <= 1:
            continue
        positions = np.arange(1, len(idx), dtype=np.int64)
        for window in WINDOWS:
            starts = np.maximum(0, positions - window)
            outs = idx[positions]
            arrays[(window, "n_true")][outs] = cumulative_n[positions] - cumulative_n[starts]
            arrays[(window, "success_true")][outs] = (
                cumulative_success[positions] - cumulative_success[starts]
            )
            middle_sum = cumulative_middle[positions] - cumulative_middle[starts]
            middle_missing = (
                cumulative_middle_missing[positions] - cumulative_middle_missing[starts]
            )
            arrays[(window, "middle_true")][outs] = np.where(
                middle_missing == 0, middle_sum, np.nan
            )

            # Only this certification part needs a short per-game loop.  It
            # writes to NumPy arrays rather than hundreds of thousands of
            # DataFrame cells.
            for pos, start, out in zip(positions, starts, outs):
                cert = np.arange(start, pos + 1)
                anchor_ok = bool(anchored[cert].all())
                same_tid = len(set(tids[cert])) == 1 if anchor_ok else False
                distinct_games = len(set(game_ids[cert])) == len(cert) if anchor_ok else False
                date_ok = False
                if anchor_ok:
                    date_int = dates[cert].astype("datetime64[ns]").astype(np.int64)
                    date_ok = bool(np.all(np.diff(date_int) >= 0))
                arrays[(window, "date_order_ok")][out] = date_ok
                arrays[(window, "anchor_window")][out] = bool(
                    anchor_ok and same_tid and distinct_games and date_ok
                )

    for (window, suffix), values in arrays.items():
        segments[f"w{window}_{suffix}"] = values
    return segments


def integer_candidates(
    success_rate: float, middle_rate: float, max_n: int, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """All common denominators whose two rounded fractions fit the rates."""

    if not (
        math.isfinite(success_rate)
        and math.isfinite(middle_rate)
        and 0.0 <= success_rate <= 1.0
        and 0.0 <= middle_rate <= 1.0
    ):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
    n = np.arange(1, max_n + 1, dtype=np.float64)
    success_fit = np.rint(success_rate * n) / n
    middle_fit = np.rint(middle_rate * n) / n
    error = np.maximum(np.abs(success_fit - success_rate), np.abs(middle_fit - middle_rate))
    keep = error <= tolerance
    return n[keep].astype(np.int32), error[keep]


def choose_candidate(
    candidates: np.ndarray,
    rule: str,
    median_n: float,
    frequency: dict[int, int],
) -> int | None:
    if len(candidates) == 0:
        return None
    if rule == "smallest":
        return int(candidates[0])
    if rule == "largest":
        return int(candidates[-1])
    if rule == "closest_median":
        distance = np.abs(candidates.astype(float) - median_n)
        return int(candidates[np.lexsort((candidates, distance))[0]])
    if rule == "empirical_frequency":
        counts = np.asarray([frequency.get(int(n), 0) for n in candidates], dtype=np.int64)
        # Highest discovery frequency, with the smaller denominator as a
        # deterministic conservative tie-break.
        order = np.lexsort((candidates, -counts))
        return int(candidates[order[0]])
    raise ValueError(f"unknown decoder rule: {rule}")


def prepare_decoder(
    samples: pd.DataFrame, window: int
) -> tuple[dict[str, object], pd.DataFrame]:
    """Lock tolerance/range/tie-break using 2020-2021 only."""

    success_col = f"asof_pitcher_prev{window}_game_success_rate"
    middle_col = f"asof_pitcher_prev{window}_game_middle_rate"
    n_col = f"w{window}_n_true"
    s_col = f"w{window}_success_true"
    m_col = f"w{window}_middle_true"
    anchor_col = f"w{window}_anchor_window"

    discovery = samples[
        samples["season"].isin(DISCOVERY_YEARS)
        & samples[anchor_col]
        & samples[n_col].notna()
        & samples[s_col].notna()
        & samples[m_col].notna()
        & samples[success_col].notna()
        & samples[middle_col].notna()
    ].copy()
    if discovery.empty:
        raise RuntimeError(f"window {window}: no anchored 2020-2021 discovery rows")

    n_true = discovery[n_col].to_numpy(np.int64)
    success_rate = discovery[success_col].to_numpy(np.float64)
    middle_rate = discovery[middle_col].to_numpy(np.float64)
    integer_error = np.maximum(
        np.abs(np.rint(success_rate * n_true) / n_true - success_rate),
        np.abs(np.rint(middle_rate * n_true) / n_true - middle_rate),
    )
    # Learn float-rounding tolerance only from samples already consistent to
    # 1e-4 with their actual game counts; larger errors are semantic/anchor
    # mismatches, not decimal representation noise.
    clean_error = integer_error[integer_error <= 1e-4]
    learned = float(np.quantile(clean_error, 0.999)) * 1.10 if len(clean_error) else ROUNDING_FLOOR
    tolerance = float(np.clip(max(ROUNDING_FLOOR, learned), ROUNDING_FLOOR, ROUNDING_CAP))

    # Search range is locked from discovery truth with a 10% + 5 pitch pad.
    max_seen = int(np.max(n_true))
    max_n = int(math.ceil(max_seen * 1.10 + 5))
    median_n = float(np.median(n_true))
    unique_n, unique_counts = np.unique(n_true, return_counts=True)
    frequency = {int(n): int(c) for n, c in zip(unique_n, unique_counts)}

    candidate_lists: list[np.ndarray] = []
    candidate_counts: list[int] = []
    cache: dict[tuple[float, float], np.ndarray] = {}
    for sr, mr in zip(success_rate, middle_rate):
        key = (float(sr), float(mr))
        candidate = cache.get(key)
        if candidate is None:
            candidate, _ = integer_candidates(sr, mr, max_n, tolerance)
            cache[key] = candidate
        candidate_lists.append(candidate)
        candidate_counts.append(len(candidate))

    rules = ["smallest", "closest_median", "empirical_frequency", "largest"]
    rule_scores: dict[str, dict[str, float]] = {}
    decoded_by_rule: dict[str, np.ndarray] = {}
    for rule in rules:
        prediction = np.asarray(
            [
                choose_candidate(c, rule, median_n, frequency)
                if len(c)
                else -1
                for c in candidate_lists
            ],
            dtype=np.int64,
        )
        decoded = prediction > 0
        accuracy = float(np.mean(prediction[decoded] == n_true[decoded])) if decoded.any() else 0.0
        all_accuracy = float(np.mean(prediction == n_true))
        mae = float(np.mean(np.abs(prediction[decoded] - n_true[decoded]))) if decoded.any() else float("inf")
        rule_scores[rule] = {
            "decoded_accuracy": accuracy,
            "all_sample_accuracy": all_accuracy,
            "decoded_mae": mae,
        }
        decoded_by_rule[rule] = prediction

    # Prefer the simpler/smaller-denominator rule on exact ties.
    preference = {"smallest": 3, "closest_median": 2, "empirical_frequency": 1, "largest": 0}
    selected_rule = max(
        rules,
        key=lambda r: (
            rule_scores[r]["decoded_accuracy"],
            rule_scores[r]["all_sample_accuracy"],
            -rule_scores[r]["decoded_mae"],
            preference[r],
        ),
    )

    decoder = {
        "window": window,
        "discovery_years": list(DISCOVERY_YEARS),
        "discovery_samples": int(len(discovery)),
        "rounding_tolerance": tolerance,
        "rounding_floor": ROUNDING_FLOOR,
        "rounding_cap": ROUNDING_CAP,
        "max_n": max_n,
        "max_true_n_seen": max_seen,
        "median_n": median_n,
        "selected_rule": selected_rule,
        "rule_scores": rule_scores,
        "discovery_candidate_coverage": float(np.mean(np.asarray(candidate_counts) > 0)),
        "discovery_unique_coverage": float(np.mean(np.asarray(candidate_counts) == 1)),
        "discovery_true_integer_fit_q": {
            "q50": float(np.quantile(integer_error, 0.50)),
            "q90": float(np.quantile(integer_error, 0.90)),
            "q99": float(np.quantile(integer_error, 0.99)),
            "q999": float(np.quantile(integer_error, 0.999)),
            "max": float(np.max(integer_error)),
        },
        # Enough information to reproduce the empirical-frequency tie-break.
        "frequency": {str(k): v for k, v in frequency.items()},
    }
    return decoder, discovery


def evaluate_decoder(
    samples: pd.DataFrame, window: int, year: int, decoder: dict[str, object]
) -> dict[str, object]:
    success_col = f"asof_pitcher_prev{window}_game_success_rate"
    middle_col = f"asof_pitcher_prev{window}_game_middle_rate"
    n_col = f"w{window}_n_true"
    s_col = f"w{window}_success_true"
    m_col = f"w{window}_middle_true"
    anchor_col = f"w{window}_anchor_window"

    frame = samples[
        (samples["season"] == year)
        & samples[anchor_col]
        & samples[n_col].notna()
        & samples[s_col].notna()
        & samples[m_col].notna()
        & samples[success_col].notna()
        & samples[middle_col].notna()
    ].copy()
    if frame.empty:
        return {"year": year, "window": window, "samples": 0}

    n_true = frame[n_col].to_numpy(np.int64)
    s_true = frame[s_col].to_numpy(np.int64)
    m_true = frame[m_col].to_numpy(np.int64)
    sr = frame[success_col].to_numpy(np.float64)
    mr = frame[middle_col].to_numpy(np.float64)
    weights = frame["game_n"].to_numpy(np.float64)
    max_n = int(decoder["max_n"])
    tolerance = float(decoder["rounding_tolerance"])
    median_n = float(decoder["median_n"])
    rule = str(decoder["selected_rule"])
    frequency = {int(k): int(v) for k, v in dict(decoder["frequency"]).items()}

    n_hat = np.full(len(frame), -1, dtype=np.int64)
    candidate_count = np.zeros(len(frame), dtype=np.int32)
    fit_error = np.full(len(frame), np.nan, dtype=np.float64)
    cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
    for i, (success_rate, middle_rate) in enumerate(zip(sr, mr)):
        key = (float(success_rate), float(middle_rate))
        item = cache.get(key)
        if item is None:
            item = integer_candidates(success_rate, middle_rate, max_n, tolerance)
            cache[key] = item
        candidates, errors = item
        candidate_count[i] = len(candidates)
        selected = choose_candidate(candidates, rule, median_n, frequency)
        if selected is not None:
            n_hat[i] = selected
            fit_error[i] = float(errors[np.where(candidates == selected)[0][0]])

    decoded = n_hat > 0
    unique = candidate_count == 1
    exact_n = decoded & (n_hat == n_true)
    out_of_range = n_true > max_n
    s_hat = np.full(len(frame), np.nan)
    m_hat = np.full(len(frame), np.nan)
    s_hat[decoded] = np.rint(sr[decoded] * n_hat[decoded])
    m_hat[decoded] = np.rint(mr[decoded] * n_hat[decoded])

    official_success_truth_error = np.abs(sr - s_true / n_true)
    official_middle_truth_error = np.abs(mr - m_true / n_true)
    n_abs = np.where(decoded, np.abs(n_hat - n_true), np.nan)
    s_count_abs = np.where(decoded, np.abs(s_hat - s_true), np.nan)
    m_count_abs = np.where(decoded, np.abs(m_hat - m_true), np.nan)
    s_rate_truth_abs = np.full(len(frame), np.nan)
    m_rate_truth_abs = np.full(len(frame), np.nan)
    s_rate_truth_abs[decoded] = np.abs(s_hat[decoded] / n_hat[decoded] - s_true[decoded] / n_true[decoded])
    m_rate_truth_abs[decoded] = np.abs(m_hat[decoded] / n_hat[decoded] - m_true[decoded] / n_true[decoded])

    def frac(mask: np.ndarray) -> float:
        return float(mask.mean())

    def weighted_frac(mask: np.ndarray) -> float:
        return float(np.average(mask.astype(float), weights=weights))

    metrics: dict[str, object] = {
        "year": year,
        "window": window,
        "samples": int(len(frame)),
        "represented_rows": int(weights.sum()),
        "candidate_coverage": frac(decoded),
        "candidate_coverage_row_weighted": weighted_frac(decoded),
        "unique_coverage": frac(unique),
        "unique_coverage_row_weighted": weighted_frac(unique),
        "out_of_range_fraction": frac(out_of_range),
        "candidate_count_median": float(np.median(candidate_count)),
        "candidate_count_p90": float(np.quantile(candidate_count, 0.90)),
        "exact_n_accuracy_all": frac(exact_n),
        "exact_n_accuracy_decoded": float(exact_n[decoded].mean()) if decoded.any() else None,
        "exact_n_accuracy_unique": float(exact_n[unique].mean()) if unique.any() else None,
        "exact_n_accuracy_row_weighted": weighted_frac(exact_n),
        "n_mae_decoded": float(np.nanmean(n_abs)),
        "n_mae_row_weighted": weighted_mean(n_abs, weights),
        "success_count_exact_decoded": float(np.mean(s_count_abs[decoded] == 0)) if decoded.any() else None,
        "middle_count_exact_decoded": float(np.mean(m_count_abs[decoded] == 0)) if decoded.any() else None,
        "success_count_mae": float(np.nanmean(s_count_abs)),
        "middle_count_mae": float(np.nanmean(m_count_abs)),
        "selected_observed_rate_fit_mae": float(np.nanmean(fit_error)),
        "selected_observed_rate_fit_max": float(np.nanmax(fit_error)) if decoded.any() else None,
        "decoded_success_rate_vs_truth_mae": float(np.nanmean(s_rate_truth_abs)),
        "decoded_middle_rate_vs_truth_mae": float(np.nanmean(m_rate_truth_abs)),
        "official_success_rate_vs_truth_mae": float(np.mean(official_success_truth_error)),
        "official_middle_rate_vs_truth_mae": float(np.mean(official_middle_truth_error)),
        "official_both_rate_truth_within_1e-5": float(
            np.mean(np.maximum(official_success_truth_error, official_middle_truth_error) <= 1e-5)
        ),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="pipeline check only; reports are not comparable to the full audit",
    )
    args = parser.parse_args()

    log("=== exp/111 prev-window denominator audit ===")
    log("No model training, no test data, no submission. Discovery=2020-21; report=2022-24.")
    log("Loading main train columns...")
    train = clean_columns(
        pd.read_csv(DATA, encoding="utf-8-sig", usecols=TRAIN_COLS, low_memory=False)
    )
    train["_row"] = np.arange(len(train), dtype=np.int64)
    if args.smoke:
        # Keep whole pitchers so asof/game segmentation remains internally valid.
        keep_pitchers = train["pitcher_id"].drop_duplicates().head(80)
        train = train[train["pitcher_id"].isin(set(keep_pitchers))].copy()
        train["_row"] = np.arange(len(train), dtype=np.int64)
    log(f"Main rows={len(train):,}, pitchers={train['pitcher_id'].nunique():,}")

    train, segments, segment_diag = make_game_segments(train)
    log(
        "Segments="
        f"{len(segments):,}; median n={segment_diag['median_game_n']:.1f}; "
        f"counter-size agreement={segment_diag['segment_counter_size_agreement']:.4f}; "
        f"middle-count coverage={segment_diag['middle_count_coverage']:.4f}"
    )

    smoke_tm_rows = 250_000 if args.smoke else None
    segments, anchor_diag = attach_trackman_anchors(train, segments, smoke_tm_rows)
    log(
        f"Strict 1:1 matched rows={anchor_diag['strict_11_matched_rows']:,}; "
        f"valid anchored segments={anchor_diag['segments_with_valid_anchor']:,} "
        f"({anchor_diag['valid_anchor_segment_fraction']:.1%})"
    )

    segments = attach_true_windows(segments)
    audit: dict[str, object] = {
        "experiment": 111,
        "purpose": "read-only common-denominator recoverability audit",
        "smoke": bool(args.smoke),
        "discovery_years": list(DISCOVERY_YEARS),
        "untouched_report_years": list(REPORT_YEARS),
        "segment_diagnostics": segment_diag,
        "trackman_anchor_diagnostics": anchor_diag,
        "decoders": {},
        "reports": {},
    }

    log("Locking decoder rules on 2020-2021 only...")
    for window in WINDOWS:
        decoder, discovery = prepare_decoder(segments, window)
        audit["decoders"][str(window)] = decoder
        log(
            f"  prev{window}: discovery n={len(discovery):,}, tol={decoder['rounding_tolerance']:.2e}, "
            f"max_n={decoder['max_n']}, rule={decoder['selected_rule']}, "
            f"candidate={decoder['discovery_candidate_coverage']:.1%}, "
            f"unique={decoder['discovery_unique_coverage']:.1%}"
        )

    log("Untouched 2022/2023/2024 reports...")
    for year in REPORT_YEARS:
        audit["reports"][str(year)] = {}
        for window in WINDOWS:
            metrics = evaluate_decoder(
                segments, window, year, audit["decoders"][str(window)]
            )
            audit["reports"][str(year)][str(window)] = metrics
            if metrics.get("samples", 0):
                log(
                    f"  {year} prev{window}: samples={metrics['samples']:,}, "
                    f"coverage={metrics['candidate_coverage']:.1%}, "
                    f"unique={metrics['unique_coverage']:.1%}, "
                    f"exact_n(decoded)={metrics['exact_n_accuracy_decoded']:.1%}, "
                    f"n_MAE={metrics['n_mae_decoded']:.2f}, "
                    f"count exact S/M={metrics['success_count_exact_decoded']:.1%}/"
                    f"{metrics['middle_count_exact_decoded']:.1%}, "
                    f"official truth <=1e-5={metrics['official_both_rate_truth_within_1e-5']:.1%}"
                )
            else:
                log(f"  {year} prev{window}: no certified sample")

    # Conservative audit-only decision.  This does not authorize training.
    holdout_metrics = [
        audit["reports"][str(year)][str(window)]
        for year in REPORT_YEARS
        for window in WINDOWS
        if audit["reports"][str(year)][str(window)].get("samples", 0)
    ]
    if holdout_metrics:
        min_coverage = min(float(m["candidate_coverage"]) for m in holdout_metrics)
        min_exact = min(float(m["exact_n_accuracy_decoded"]) for m in holdout_metrics)
        min_truth = min(float(m["official_both_rate_truth_within_1e-5"]) for m in holdout_metrics)
    else:
        min_coverage = min_exact = min_truth = 0.0
    status = "PASS" if (min_coverage >= 0.80 and min_exact >= 0.75 and min_truth >= 0.90) else "FAIL"
    audit["gate"] = {
        "status": status,
        "criteria": {
            "min_candidate_coverage": 0.80,
            "min_exact_n_accuracy_decoded": 0.75,
            "min_official_rate_truth_agreement": 0.90,
        },
        "observed": {
            "min_candidate_coverage": min_coverage,
            "min_exact_n_accuracy_decoded": min_exact,
            "min_official_rate_truth_agreement": min_truth,
        },
        "meaning": (
            "PASS only means denominator-derived workload/reliability features deserve a separate "
            "strict temporal residual pre-gate; it does not authorize model training or deployment."
        ),
    }
    log(
        f"AUDIT GATE {status}: min coverage={min_coverage:.1%}, "
        f"min exact_n={min_exact:.1%}, min truth agreement={min_truth:.1%}"
    )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(json_ready(audit), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LINES.append("")
    LINES.append("Decoder JSON contains full per-year/per-window metrics and locked rules.")
    LINES.append(f"JSON: {OUT_JSON}")
    LINES.append(f"TXT:  {OUT_TXT}")
    OUT_TXT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    log(f"Saved {OUT_JSON} and {OUT_TXT}")


if __name__ == "__main__":
    main()
