# -*- coding: utf-8 -*-
"""[113] Strict temporal workload-denominator residual pre-gate.

IMPLEMENTED FOR REVIEW; DO NOT RUN UNTIL THE OWNER EXPLICITLY STARTS IT.

Purpose
-------
Use only the six official row-local prev1/3/5 success/middle rates to recover
the denominator information audited in exp/111.  The exp/111 decoder is
immutable here: 2020-2021 tolerance, maximum denominator, and the ``smallest``
tie-break are loaded from lab/111_prev_window_denominator_audit.json and
checked against the locked fingerprint below.

Strict protocol
---------------
* Inference features use no TrackMan data, game id, date, row order, raw player
  id, team, or absolute season.  ``pitcher_id`` is metadata for the cluster
  bootstrap only; ``season`` is used only to select temporal folds.
* The baseline is the current exact CAT5 seed mean + frozen V18 gamma=.30 +
  deployed affine transform.
* One fixed shallow HistGradientBoostingRegressor is fit on centred 2022 R
  residuals and predicts 2023 R.  F correction is exactly zero.
* Gamma is locked on 2023.  Only if raw>=8, shape>=5, and pitcher-bootstrap
  p2.5>0 (plus nonzero gamma/R/F safety checks) may the script load 2024,
  refit on centred 2022+2023 R residuals, and confirm with the same gate.
* No seed, learner, or parameter search; no fitted model, bundle, ZIP,
  deployment artifact, or test-data access.
* Existing lab/113 artifacts cause an immediate refusal; nothing is
  overwritten.

Analysis outputs if manually run after review:
  lab/113_prev_window_workload_residual.txt
  lab/113_prev_window_workload_residual.json
  lab/113_prev_window_workload_effect_y2023.npy
  lab/113_prev_window_workload_effect_y2024.npy  (discovery PASS only)
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/train.csv"
LAB = ROOT / "lab"
DECODER_AUDIT = LAB / "111_prev_window_denominator_audit.json"

OUT_TXT = LAB / "113_prev_window_workload_residual.txt"
OUT_JSON = LAB / "113_prev_window_workload_residual.json"
OUT_EFFECT = {
    2023: LAB / "113_prev_window_workload_effect_y2023.npy",
    2024: LAB / "113_prev_window_workload_effect_y2024.npy",
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

# Exact submitted champion transform.
V18_GAMMA = 0.30
AFFINE_CENTER = 0.49
AFFINE_SCALE = 1.06
AFFINE_SHIFT = 0.0066

WINDOWS = (1, 3, 5)
EXPECTED_DECODER = {
    1: {"tolerance": 5.44444444500192e-07, "max_n": 145, "rule": "smallest"},
    3: {"tolerance": 5.48208469056122e-07, "max_n": 395, "rule": "smallest"},
    5: {"tolerance": 5.49999999985285e-07, "max_n": 638, "rule": "smallest"},
}
EXPECTED_DECODER_DISCOVERY_YEARS = (2020, 2021)
EXPECTED_DECODER_SHA256 = "147DED4EB120500D7FB7A4D7237C69D63453B6CD8E4E81350ECA40C0595CF060"

# One fixed reliability and one fixed conservative learner.  These values are
# pre-registered, not selected by any validation year.
RELIABILITY_K = 50.0
HGB_PARAMS = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 120,
    "max_leaf_nodes": 7,
    "max_depth": 3,
    "min_samples_leaf": 5000,
    "l2_regularization": 50.0,
    "early_stopping": False,
    "random_state": 113,
}
EFFECT_CLIP = 0.04
GAMMAS = (0.0, 0.25, 0.50, 0.75, 1.00)
BOOTSTRAP_DRAWS = 5000
CORRECTED_GAME_TYPE = "R"

GATE = {
    "raw_gain_min": 8.0,
    "shape_gain_min": 5.0,
    "bootstrap_p025_min_exclusive": 0.0,
}

EXPECTED_ROWS = {2022: 247_472, 2023: 245_525, 2024: 253_507}
EXPECTED_TARGET_RATES = {2022: 0.528920, 2023: 0.499957, 2024: 0.486105}

SUCCESS_COLUMNS = {
    window: f"asof_pitcher_prev{window}_game_success_rate" for window in WINDOWS
}
MIDDLE_COLUMNS = {
    window: f"asof_pitcher_prev{window}_game_middle_rate" for window in WINDOWS
}
META_COLUMNS = ["season", "game_type", "pitcher_id", "control_success"]
OFFICIAL_COLUMNS = [
    "asof_pitcher_success_rate",
    "asof_pitcher_middle_rate",
    *SUCCESS_COLUMNS.values(),
    *MIDDLE_COLUMNS.values(),
]
READ_COLUMNS = list(dict.fromkeys(META_COLUMNS + OFFICIAL_COLUMNS))

# Explicit feature contract.  There is no generic pass-through of raw columns.
FEATURE_NAMES: list[str] = []
for _window in WINDOWS:
    FEATURE_NAMES.extend(
        [
            f"w{_window}_decoded",
            f"w{_window}_log_n_selected",
            f"w{_window}_unique",
            f"w{_window}_log_candidate_count",
            f"w{_window}_log_candidate_span",
            f"w{_window}_candidate_range_ratio",
            f"w{_window}_reliability_k50",
            f"w{_window}_success_dev_reliable",
            f"w{_window}_middle_dev_reliable",
        ]
    )
FEATURE_NAMES.extend(
    [
        "shell_1_to_3_log_n",
        "shell_3_to_5_log_n",
        "shell_1_to_3_nonnegative",
        "shell_3_to_5_nonnegative",
        "denominator_monotonic",
        "all_windows_decoded",
        "all_windows_unique",
        "any_window_ambiguous",
        "max_log_candidate_count",
        "max_candidate_range_ratio",
    ]
)

FORBIDDEN_MODEL_TOKENS = (
    "pitcher_id",
    "batter_id",
    "team",
    "season",
    "game_id",
    "game_date",
    "trackman",
    "row_id",
)


STARTED = time.time()


@dataclass(frozen=True)
class DecoderRule:
    window: int
    tolerance: float
    max_n: int
    rule: str


@dataclass
class DecodedWindow:
    selected_n: np.ndarray
    candidate_min: np.ndarray
    candidate_max: np.ndarray
    candidate_count: np.ndarray
    unique: np.ndarray
    decoded: np.ndarray


@dataclass
class YearRecord:
    year: int
    frame: pd.DataFrame
    target: np.ndarray
    probabilities5: np.ndarray
    v18_effect: np.ndarray
    pre_affine: np.ndarray
    baseline: np.ndarray


def log(message: str = "") -> None:
    print(message, flush=True)


def tick(message: str) -> None:
    log(f"  [{time.time() - STARTED:7.1f}s] {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def score(probability: np.ndarray, target: np.ndarray) -> float:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rate = float(target.mean())
    return 100000.0 * (
        1.0 - float(np.mean((probability - target) ** 2)) / (rate * (1.0 - rate))
    )


def shape_gain(base: np.ndarray, candidate: np.ndarray, target: np.ndarray) -> float:
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
        100000.0
        * row_gain[np.asarray(mask, dtype=bool)].sum()
        / (len(target) * rate * (1.0 - rate))
    )


def normalise_game_type(series: pd.Series) -> np.ndarray:
    values = series.astype("string").fillna("__MISSING__").astype(str).to_numpy()
    if not np.isin(values, ("R", "F")).all():
        raise AssertionError(f"unexpected game_type values: {sorted(set(values))}")
    return values


def r_mask(record: YearRecord) -> np.ndarray:
    return normalise_game_type(record.frame["game_type"]) == CORRECTED_GAME_TYPE


def load_locked_decoders() -> tuple[dict[int, DecoderRule], str]:
    if not DECODER_AUDIT.is_file():
        raise FileNotFoundError(DECODER_AUDIT)
    audit_sha = sha256(DECODER_AUDIT)
    if audit_sha != EXPECTED_DECODER_SHA256:
        raise AssertionError(
            "lab/111 decoder audit changed; refusing silent retuning: "
            f"{audit_sha} != {EXPECTED_DECODER_SHA256}"
        )
    payload = json.loads(DECODER_AUDIT.read_text(encoding="utf-8"))
    discovery = tuple(int(v) for v in payload.get("discovery_years", []))
    if discovery != EXPECTED_DECODER_DISCOVERY_YEARS:
        raise AssertionError(f"decoder discovery years changed: {discovery}")

    rules: dict[int, DecoderRule] = {}
    for window in WINDOWS:
        raw = payload["decoders"][str(window)]
        expected = EXPECTED_DECODER[window]
        tolerance = float(raw["rounding_tolerance"])
        max_n = int(raw["max_n"])
        rule = str(raw["selected_rule"])
        if not math.isclose(
            tolerance,
            float(expected["tolerance"]),
            rel_tol=0.0,
            abs_tol=1e-18,
        ):
            raise AssertionError(f"prev{window} tolerance changed: {tolerance}")
        if max_n != int(expected["max_n"]) or rule != expected["rule"]:
            raise AssertionError(
                f"prev{window} decoder changed: max_n={max_n}, rule={rule}"
            )
        if rule != "smallest":
            raise AssertionError("exp/113 implements only the locked smallest rule")
        rules[window] = DecoderRule(window, tolerance, max_n, rule)

    # Do not retain or inspect exp/111 TrackMan/game/date diagnostics.  Only
    # the locked scalar decoder contract above is allowed downstream.
    del payload
    return rules, audit_sha


def load_record(year: int) -> YearRecord:
    if year not in EXPECTED_ROWS:
        raise KeyError(year)
    required = [DATA, *PROBABILITY_FILES[year], V18_EFFECT_FILES[year]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing files for {year}: {missing}")

    frame = pd.read_csv(
        DATA,
        encoding="utf-8-sig",
        usecols=READ_COLUMNS,
        low_memory=False,
    )
    frame.columns = [str(column).replace("\ufeff", "").strip() for column in frame.columns]
    frame = frame.loc[frame["season"] == year].reset_index(drop=True)
    expected = EXPECTED_ROWS[year]
    if len(frame) != expected:
        raise AssertionError(f"{year} row contract changed: {len(frame):,} != {expected:,}")
    target = frame["control_success"].to_numpy(np.float64)
    if not np.isin(target, (0.0, 1.0)).all():
        raise AssertionError(f"{year} target is not binary")
    if abs(float(target.mean()) - EXPECTED_TARGET_RATES[year]) > 2e-6:
        raise AssertionError(f"{year} target-rate fingerprint changed: {target.mean():.9f}")

    seed_probabilities = [
        np.load(path, allow_pickle=False).astype(np.float64, copy=False)
        for path in PROBABILITY_FILES[year]
    ]
    for probability in seed_probabilities:
        if probability.shape != (expected, 5):
            raise AssertionError(f"{year} CAT5 shape changed: {probability.shape}")
        if not np.isfinite(probability).all():
            raise AssertionError(f"{year} CAT5 contains non-finite values")
        if float(np.max(np.abs(probability.sum(axis=1) - 1.0))) > 2e-5:
            raise AssertionError(f"{year} CAT5 rows do not sum to one")
    probabilities5 = np.mean(seed_probabilities, axis=0)
    effect = np.load(V18_EFFECT_FILES[year], allow_pickle=False).astype(
        np.float64, copy=False
    )
    if effect.shape != (expected,) or not np.isfinite(effect).all():
        raise AssertionError(f"{year} V18 effect contract failed")
    pre_affine = np.clip(probabilities5[:, 0] + V18_GAMMA * effect, 0.0, 1.0)
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


def decode_rate_pair(
    success_rate: float,
    middle_rate: float,
    rule: DecoderRule,
) -> tuple[int, int, int, int]:
    """Return selected/min/max/count from official rates only."""

    if not (
        math.isfinite(success_rate)
        and math.isfinite(middle_rate)
        and 0.0 <= success_rate <= 1.0
        and 0.0 <= middle_rate <= 1.0
    ):
        return 0, 0, 0, 0
    denominator = np.arange(1, rule.max_n + 1, dtype=np.float64)
    success_fit = np.rint(success_rate * denominator) / denominator
    middle_fit = np.rint(middle_rate * denominator) / denominator
    error = np.maximum(
        np.abs(success_fit - success_rate),
        np.abs(middle_fit - middle_rate),
    )
    candidates = np.flatnonzero(error <= rule.tolerance) + 1
    if len(candidates) == 0:
        return 0, 0, 0, 0
    # The rule is asserted to be "smallest" when the JSON is loaded.
    return int(candidates[0]), int(candidates[0]), int(candidates[-1]), int(len(candidates))


def decode_window(rows: pd.DataFrame, rule: DecoderRule) -> DecodedWindow:
    success = pd.to_numeric(
        rows[SUCCESS_COLUMNS[rule.window]], errors="coerce"
    ).to_numpy(np.float64)
    middle = pd.to_numeric(
        rows[MIDDLE_COLUMNS[rule.window]], errors="coerce"
    ).to_numpy(np.float64)
    pair = np.column_stack(
        [np.nan_to_num(success, nan=-9.0), np.nan_to_num(middle, nan=-9.0)]
    )
    unique_pair, inverse = np.unique(pair, axis=0, return_inverse=True)
    decoded_unique = np.asarray(
        [decode_rate_pair(float(sr), float(mr), rule) for sr, mr in unique_pair],
        dtype=np.int32,
    )
    decoded_rows = decoded_unique[inverse]
    selected = decoded_rows[:, 0]
    candidate_min = decoded_rows[:, 1]
    candidate_max = decoded_rows[:, 2]
    candidate_count = decoded_rows[:, 3]
    decoded = candidate_count > 0
    unique = candidate_count == 1
    if np.any(decoded & (selected != candidate_min)):
        raise AssertionError(f"prev{rule.window} smallest rule violated")
    if np.any(candidate_max < candidate_min):
        raise AssertionError(f"prev{rule.window} invalid candidate range")
    return DecodedWindow(
        selected_n=selected,
        candidate_min=candidate_min,
        candidate_max=candidate_max,
        candidate_count=candidate_count,
        unique=unique,
        decoded=decoded,
    )


def finite32(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def build_matrix(
    record: YearRecord,
    mask: np.ndarray,
    decoders: dict[int, DecoderRule],
) -> tuple[np.ndarray, dict[str, object]]:
    if mask.shape != (len(record.target),):
        raise AssertionError("row mask shape mismatch")
    rows = record.frame.loc[mask]
    decoded = {window: decode_window(rows, decoders[window]) for window in WINDOWS}
    career_success = pd.to_numeric(
        rows["asof_pitcher_success_rate"], errors="coerce"
    ).to_numpy(np.float64)
    career_middle = pd.to_numeric(
        rows["asof_pitcher_middle_rate"], errors="coerce"
    ).to_numpy(np.float64)

    arrays: list[np.ndarray] = []
    reliabilities: dict[int, np.ndarray] = {}
    for window in WINDOWS:
        item = decoded[window]
        n = item.selected_n.astype(np.float64)
        count = item.candidate_count.astype(np.float64)
        span = (item.candidate_max - item.candidate_min).astype(np.float64)
        range_ratio = np.divide(
            span,
            np.maximum(item.candidate_max.astype(np.float64), 1.0),
            out=np.zeros_like(span),
            where=item.decoded,
        )
        reliability = np.divide(
            n,
            n + RELIABILITY_K,
            out=np.zeros_like(n),
            where=item.decoded,
        )
        reliabilities[window] = reliability
        previous_success = pd.to_numeric(
            rows[SUCCESS_COLUMNS[window]], errors="coerce"
        ).to_numpy(np.float64)
        previous_middle = pd.to_numeric(
            rows[MIDDLE_COLUMNS[window]], errors="coerce"
        ).to_numpy(np.float64)
        success_deviation = np.where(
            np.isfinite(previous_success) & np.isfinite(career_success),
            (previous_success - career_success) * reliability,
            0.0,
        )
        middle_deviation = np.where(
            np.isfinite(previous_middle) & np.isfinite(career_middle),
            (previous_middle - career_middle) * reliability,
            0.0,
        )
        arrays.extend(
            [
                item.decoded.astype(np.float32),
                np.log1p(n),
                item.unique.astype(np.float32),
                np.log1p(count),
                np.log1p(np.maximum(span, 0.0)),
                range_ratio,
                reliability,
                success_deviation,
                middle_deviation,
            ]
        )

    n1 = decoded[1].selected_n.astype(np.float64)
    n3 = decoded[3].selected_n.astype(np.float64)
    n5 = decoded[5].selected_n.astype(np.float64)
    shell13 = n3 - n1
    shell35 = n5 - n3
    shell13_valid = decoded[1].decoded & decoded[3].decoded
    shell35_valid = decoded[3].decoded & decoded[5].decoded
    monotonic = (
        (n1 <= n3)
        & (n3 <= n5)
        & decoded[1].decoded
        & decoded[3].decoded
        & decoded[5].decoded
    )
    all_decoded = decoded[1].decoded & decoded[3].decoded & decoded[5].decoded
    all_unique = decoded[1].unique & decoded[3].unique & decoded[5].unique
    any_ambiguous = np.logical_or.reduce(
        [item.candidate_count > 1 for item in decoded.values()]
    )
    log_counts = np.column_stack(
        [np.log1p(decoded[w].candidate_count.astype(np.float64)) for w in WINDOWS]
    )
    range_ratios = np.column_stack(
        [
            np.divide(
                (decoded[w].candidate_max - decoded[w].candidate_min).astype(np.float64),
                np.maximum(decoded[w].candidate_max.astype(np.float64), 1.0),
                out=np.zeros(len(rows), dtype=np.float64),
                where=decoded[w].decoded,
            )
            for w in WINDOWS
        ]
    )
    arrays.extend(
        [
            np.log1p(np.maximum(shell13, 0.0)),
            np.log1p(np.maximum(shell35, 0.0)),
            (shell13_valid & (shell13 >= 0.0)).astype(np.float32),
            (shell35_valid & (shell35 >= 0.0)).astype(np.float32),
            monotonic.astype(np.float32),
            all_decoded.astype(np.float32),
            all_unique.astype(np.float32),
            any_ambiguous.astype(np.float32),
            log_counts.max(axis=1),
            range_ratios.max(axis=1),
        ]
    )

    matrix = np.column_stack([finite32(values) for values in arrays]).astype(
        np.float32, copy=False
    )
    if matrix.shape != (int(mask.sum()), len(FEATURE_NAMES)):
        raise AssertionError(
            f"feature contract mismatch {matrix.shape} != "
            f"{(int(mask.sum()), len(FEATURE_NAMES))}"
        )
    if not np.isfinite(matrix).all():
        raise AssertionError("workload matrix contains non-finite values")
    lowered = [name.lower() for name in FEATURE_NAMES]
    forbidden = [
        name for name in lowered if any(token in name for token in FORBIDDEN_MODEL_TOKENS)
    ]
    if forbidden:
        raise AssertionError(f"forbidden model feature names: {forbidden}")

    diagnostics: dict[str, object] = {
        "rows": int(len(rows)),
        "features": int(matrix.shape[1]),
        "feature_names": list(FEATURE_NAMES),
        "monotonic_share": float(monotonic.mean()),
        "all_decoded_share": float(all_decoded.mean()),
        "all_unique_share": float(all_unique.mean()),
        "windows": {},
    }
    for window in WINDOWS:
        item = decoded[window]
        diagnostics["windows"][str(window)] = {
            "decoded_share": float(item.decoded.mean()),
            "unique_share": float(item.unique.mean()),
            "selected_n_mean": float(item.selected_n[item.decoded].mean())
            if item.decoded.any()
            else None,
            "candidate_count_mean": float(item.candidate_count[item.decoded].mean())
            if item.decoded.any()
            else None,
            "candidate_range_mean": float(
                (item.candidate_max[item.decoded] - item.candidate_min[item.decoded]).mean()
            )
            if item.decoded.any()
            else None,
            "reliability_mean": float(reliabilities[window].mean()),
        }
    return matrix, diagnostics


def centred_target(record: YearRecord, mask: np.ndarray) -> tuple[np.ndarray, float]:
    residual = record.target - record.baseline
    selected = residual[mask].astype(np.float64, copy=True)
    removed_mean = float(selected.mean())
    selected -= removed_mean
    return selected.astype(np.float32), removed_mean


def fit_predict_effect(
    sources: Sequence[YearRecord],
    destination: YearRecord,
    decoders: dict[int, DecoderRule],
) -> tuple[np.ndarray, dict[str, object]]:
    matrices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    source_diagnostics: dict[str, object] = {}
    for record in sources:
        mask = r_mask(record)
        matrix, feature_diag = build_matrix(record, mask, decoders)
        target, removed_mean = centred_target(record, mask)
        matrices.append(matrix)
        targets.append(target)
        source_diagnostics[str(record.year)] = {
            "R_rows": int(mask.sum()),
            "removed_R_residual_mean": removed_mean,
            "centered_target_std": float(target.std()),
            "features": feature_diag,
        }
    x_train = np.concatenate(matrices, axis=0)
    y_train = np.concatenate(targets, axis=0)
    destination_mask = r_mask(destination)
    x_destination, destination_diag = build_matrix(
        destination, destination_mask, decoders
    )
    if abs(float(y_train.mean())) > 2e-7:
        raise AssertionError(f"centred residual mean drift: {y_train.mean():+.3e}")

    tick(
        f"fit fixed shallow HGB sources={[record.year for record in sources]} "
        f"rows={len(y_train):,} features={x_train.shape[1]}"
    )
    learner = HistGradientBoostingRegressor(**HGB_PARAMS)
    learner.fit(x_train, y_train)
    raw_prediction = learner.predict(x_destination).astype(np.float64)
    prediction = np.clip(raw_prediction, -EFFECT_CLIP, EFFECT_CLIP)
    effect = np.zeros(len(destination.target), dtype=np.float64)
    effect[destination_mask] = prediction
    if np.any(effect[~destination_mask] != 0.0):
        raise AssertionError("protected F rows received non-zero correction")

    diagnostics: dict[str, object] = {
        "source": source_diagnostics,
        "destination": {
            "year": destination.year,
            "R_rows": int(destination_mask.sum()),
            "features": destination_diag,
        },
        "learner": dict(HGB_PARAMS),
        "effect_clip": EFFECT_CLIP,
        "prediction_mean_R": float(prediction.mean()),
        "prediction_std_R": float(prediction.std()),
        "prediction_abs_mean_R": float(np.mean(np.abs(prediction))),
        "clip_share_R": float(np.mean(np.abs(raw_prediction) > EFFECT_CLIP)),
    }
    del learner, x_train, y_train, x_destination, matrices, targets
    gc.collect()
    return effect, diagnostics


def evaluate(record: YearRecord, effect: np.ndarray, gamma: float) -> dict[str, float]:
    base = record.baseline.astype(np.float64, copy=False)
    target = record.target.astype(np.float64, copy=False)
    effect = np.asarray(effect, dtype=np.float64)
    if effect.shape != base.shape or not np.isfinite(effect).all():
        raise AssertionError(f"{record.year} invalid effect")
    candidate = np.clip(base + float(gamma) * effect, 0.0, 1.0)
    raw_gain = score(candidate, target) - score(base, target)
    equal_mean = shape_gain(base, candidate, target)
    game_type = normalise_game_type(record.frame["game_type"])
    f_mask = game_type == "F"
    r_domain = game_type == "R"
    return {
        "base_score": float(score(base, target)),
        "candidate_score": float(score(candidate, target)),
        "raw_gain": float(raw_gain),
        "shape_gain": float(equal_mean),
        "center_contribution": float(raw_gain - equal_mean),
        "mean_shift": float(candidate.mean() - base.mean()),
        "R_contribution": domain_contribution(base, candidate, target, r_domain),
        "F_contribution": domain_contribution(base, candidate, target, f_mask),
        "effect_mean": float(effect.mean()),
        "effect_std": float(effect.std()),
        "effect_abs_mean": float(np.mean(np.abs(effect))),
        "effect_nonzero_share": float(np.mean(effect != 0.0)),
    }


def gamma_curve(record: YearRecord, effect: np.ndarray) -> list[dict[str, float]]:
    return [
        {"gamma": float(gamma), **evaluate(record, effect, gamma)}
        for gamma in GAMMAS
    ]


def choose_gamma(curve: Sequence[dict[str, float]]) -> float:
    best = max(
        curve,
        key=lambda row: (
            min(float(row["raw_gain"]), float(row["shape_gain"])),
            float(row["raw_gain"]),
            -float(row["gamma"]),
        ),
    )
    return float(best["gamma"])


def cluster_bootstrap_gain(
    record: YearRecord,
    effect: np.ndarray,
    gamma: float,
    seed: int,
) -> dict[str, float]:
    base = record.baseline.astype(np.float64, copy=False)
    target = record.target.astype(np.float64, copy=False)
    candidate = np.clip(base + float(gamma) * effect, 0.0, 1.0)
    # pitcher_id remains bootstrap metadata and never enters build_matrix().
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
    rng = np.random.default_rng(seed)
    gains = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    groups = len(values)
    cursor = 0
    while cursor < BOOTSTRAP_DRAWS:
        size = min(250, BOOTSTRAP_DRAWS - cursor)
        indices = rng.integers(0, groups, size=(size, groups))
        sampled = values[indices].sum(axis=1)
        n = sampled[:, 0]
        rate = sampled[:, 1] / n
        gains[cursor : cursor + size] = 100000.0 * (
            sampled[:, 2] - sampled[:, 3]
        ) / (n * rate * (1.0 - rate))
        cursor += size
    return {
        "draws": BOOTSTRAP_DRAWS,
        "pitchers": int(groups),
        "median": float(np.median(gains)),
        "p025": float(np.quantile(gains, 0.025)),
        "p975": float(np.quantile(gains, 0.975)),
        "prob_positive": float(np.mean(gains > 0.0)),
    }


def pass_gate(
    gamma: float,
    metrics: dict[str, float],
    bootstrap: dict[str, float],
) -> tuple[bool, dict[str, bool]]:
    checks = {
        "nonzero_gamma": bool(gamma > 0.0),
        "raw_gain": bool(metrics["raw_gain"] >= GATE["raw_gain_min"]),
        "shape_gain": bool(metrics["shape_gain"] >= GATE["shape_gain_min"]),
        "pitcher_bootstrap_p025_positive": bool(
            bootstrap["p025"] > GATE["bootstrap_p025_min_exclusive"]
        ),
        "R_nonnegative": bool(metrics["R_contribution"] >= 0.0),
        "F_exact_fallback": bool(abs(metrics["F_contribution"]) < 1e-10),
    }
    return bool(all(checks.values())), checks


def ensure_fresh_outputs() -> None:
    paths = [OUT_TXT, OUT_JSON, *OUT_EFFECT.values()]
    paths.extend(path.with_suffix(path.suffix + ".tmp") for path in list(paths))
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite lab/113 artifacts; archive/remove explicitly first: "
            + ", ".join(existing)
        )


def save_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(values, dtype=np.float32), allow_pickle=False)
    os.replace(temporary, path)


def write_reports(result: dict[str, object], lines: Iterable[str]) -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    temporary_txt = OUT_TXT.with_suffix(OUT_TXT.suffix + ".tmp")
    temporary_json = OUT_JSON.with_suffix(OUT_JSON.suffix + ".tmp")
    temporary_txt.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    temporary_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_txt, OUT_TXT)
    os.replace(temporary_json, OUT_JSON)


def append_fold_report(
    lines: list[str],
    title: str,
    gamma: float,
    metrics: dict[str, float],
    bootstrap: dict[str, float],
    checks: dict[str, bool],
) -> None:
    lines.extend(
        [
            "",
            title,
            f"locked_gamma={gamma:.2f}",
            (
                f"base={metrics['base_score']:.3f} candidate={metrics['candidate_score']:.3f} "
                f"raw={metrics['raw_gain']:+.3f} shape={metrics['shape_gain']:+.3f} "
                f"center={metrics['center_contribution']:+.3f} "
                f"mean_shift={metrics['mean_shift']:+.7f}"
            ),
            (
                f"domain contribution R={metrics['R_contribution']:+.3f} "
                f"F={metrics['F_contribution']:+.3f}"
            ),
            (
                f"pitcher bootstrap median={bootstrap['median']:+.3f} "
                f"95%=[{bootstrap['p025']:+.3f},{bootstrap['p975']:+.3f}] "
                f"P(>0)={bootstrap['prob_positive']:.4f} "
                f"clusters={bootstrap['pitchers']}"
            ),
            "gate " + " ".join(f"{key}={value}" for key, value in checks.items()),
        ]
    )


def main() -> None:
    ensure_fresh_outputs()
    LAB.mkdir(parents=True, exist_ok=True)
    if len(FEATURE_NAMES) != len(set(FEATURE_NAMES)):
        raise AssertionError("duplicate feature name")

    decoders, decoder_sha = load_locked_decoders()
    lines = [
        "=== exp/113 strict temporal prev-window workload residual ===",
        "analysis only: no test read, fitted-model save, bundle, ZIP, or submission",
        "model input: official row-local prev1/3/5 success+middle and career rates only",
        "forbidden input: TrackMan/game_id/game_date/row order/raw ID/team/season",
        "fit/correct R only; F correction exactly zero",
        "discovery 2022 residual -> 2023; confirmation 2022+2023 -> 2024 only after PASS",
        f"decoder_sha256={decoder_sha}",
        f"decoder={{{', '.join(f'{w}:tol={decoders[w].tolerance:.12g}/max={decoders[w].max_n}/smallest' for w in WINDOWS)}}}",
        f"HGB={HGB_PARAMS} effect_clip={EFFECT_CLIP:.3f} gammas={GAMMAS}",
        f"gate={GATE}",
    ]
    result: dict[str, object] = {
        "experiment": 113,
        "description": "strict temporal denominator/workload residual adapter",
        "decoder_audit": {
            "path": str(DECODER_AUDIT.relative_to(ROOT)),
            "sha256": decoder_sha,
            "discovery_years": list(EXPECTED_DECODER_DISCOVERY_YEARS),
            "rules": {
                str(window): {
                    "tolerance": decoders[window].tolerance,
                    "max_n": decoders[window].max_n,
                    "rule": decoders[window].rule,
                }
                for window in WINDOWS
            },
        },
        "feature_contract": {
            "names": list(FEATURE_NAMES),
            "reliability_k": RELIABILITY_K,
            "forbidden_tokens": list(FORBIDDEN_MODEL_TOKENS),
            "F_correction": 0.0,
        },
        "baseline": {
            "V18_gamma": V18_GAMMA,
            "affine_center": AFFINE_CENTER,
            "affine_scale": AFFINE_SCALE,
            "affine_shift": AFFINE_SHIFT,
        },
        "learner": dict(HGB_PARAMS),
        "effect_clip": EFFECT_CLIP,
        "gamma_grid": list(GAMMAS),
        "gate": dict(GATE),
    }

    # The confirmation year is deliberately not loaded before gamma/gate lock.
    tick("load 2022/2023 exact CAT5 + V18 + affine records")
    y2022 = load_record(2022)
    y2023 = load_record(2023)
    discovery_effect, discovery_fit = fit_predict_effect(
        [y2022], y2023, decoders
    )
    discovery_curve = gamma_curve(y2023, discovery_effect)
    locked_gamma = choose_gamma(discovery_curve)
    discovery_metrics = evaluate(y2023, discovery_effect, locked_gamma)
    discovery_bootstrap = cluster_bootstrap_gain(
        y2023, discovery_effect, locked_gamma, seed=113_2023
    )
    discovery_pass, discovery_checks = pass_gate(
        locked_gamma, discovery_metrics, discovery_bootstrap
    )
    save_npy(OUT_EFFECT[2023], discovery_effect)
    result["locked_gamma"] = locked_gamma
    result["discovery"] = {
        "source_years": [2022],
        "validation_year": 2023,
        "gamma_curve": discovery_curve,
        "metrics": discovery_metrics,
        "bootstrap": discovery_bootstrap,
        "fit_diagnostics": discovery_fit,
        "gate_checks": discovery_checks,
        "gate_pass": discovery_pass,
        "effect_path": str(OUT_EFFECT[2023].relative_to(ROOT)),
    }
    append_fold_report(
        lines,
        "DISCOVERY (2022 R residual -> 2023 R; F fallback)",
        locked_gamma,
        discovery_metrics,
        discovery_bootstrap,
        discovery_checks,
    )
    lines.append(
        "gamma curve: "
        + " ".join(
            f"g={row['gamma']:.2f}:{row['raw_gain']:+.2f}/{row['shape_gain']:+.2f}"
            for row in discovery_curve
        )
    )

    if not discovery_pass:
        result["confirmation"] = None
        result["final_gate_pass"] = False
        result["stopped_after"] = "discovery"
        result["elapsed_seconds"] = float(time.time() - STARTED)
        lines.extend(
            [
                "",
                "DISCOVERY GATE FAIL — 2024 was not loaded; confirmation/deployment skipped.",
                f"elapsed={result['elapsed_seconds']:.1f}s",
            ]
        )
        write_reports(result, lines)
        return

    lines.append("DISCOVERY GATE PASS — learner/gamma/features frozen before loading 2024.")
    tick("load 2024 only after discovery PASS")
    y2024 = load_record(2024)
    confirmation_effect, confirmation_fit = fit_predict_effect(
        [y2022, y2023], y2024, decoders
    )
    confirmation_metrics = evaluate(y2024, confirmation_effect, locked_gamma)
    confirmation_bootstrap = cluster_bootstrap_gain(
        y2024, confirmation_effect, locked_gamma, seed=113_2024
    )
    confirmation_pass, confirmation_checks = pass_gate(
        locked_gamma, confirmation_metrics, confirmation_bootstrap
    )
    save_npy(OUT_EFFECT[2024], confirmation_effect)
    result["confirmation"] = {
        "source_years": [2022, 2023],
        "validation_year": 2024,
        "gamma_was_locked_before_load": True,
        "metrics": confirmation_metrics,
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
        "CONFIRMATION (2022+2023 R residual -> 2024 R; locked gamma; F fallback)",
        locked_gamma,
        confirmation_metrics,
        confirmation_bootstrap,
        confirmation_checks,
    )
    lines.extend(
        [
            "",
            "FINAL GATE " + ("PASS" if confirmation_pass else "FAIL"),
            "PASS remains analysis-only; this script never builds a deployment artifact.",
            f"elapsed={result['elapsed_seconds']:.1f}s",
        ]
    )
    write_reports(result, lines)


if __name__ == "__main__":
    main()
