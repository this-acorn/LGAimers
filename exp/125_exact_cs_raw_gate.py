# -*- coding: utf-8 -*-
"""[125] Strict, zero-fit-model gate for exact current-season raw pitcher state.

This script answers the only narrow question left by the repository audit:
does a deployable pre-pitch cutoff representation of the current season add a
stable residual direction on top of frozen CAT5 + V18(gamma=.30)?

It trains no CatBoost/LightGBM model, reads no test data, and builds no bundle.
Discovery uses only 2022 residuals and 2023 labels.  Only a discovery arm that
passes all predeclared gates is allowed to open 2024 once.

Predeclared blocks
------------------
raw:
  pre-pitch-cutoff raw success/reverse/middle levels
cutoff_diff:
  raw levels minus the current champion's post/pseudo-post cutoff arithmetic
combined:
  all six terms

All coefficients are a fixed ridge (alpha=10000) on source-season champion
residuals centered within game_type.  Source feature means/scales are frozen,
low-reliability rows (since_n < 11) receive zero correction, standardized
features are clipped to +/-5, and final correction is clipped to +/-0.03.

Protocol
--------
1. Fit each block on 2022, evaluate on 2023.
2. Select only among blocks with raw>=8, shape>=5, and F/R>=0.
3. If none pass, stop without loading 2024 targets.
4. For the one selected block, confirm on 2024 in two ways:
   a. frozen22: reuse the exact 2022 coefficient/scaler unchanged;
   b. rolling23: refit the same fixed block on 2023, then apply to 2024.
5. A CatBoost seed42 arm is authorized only when both confirmations have
   raw/shape>0, rolling23 F/R>=0, and the two rolling transitions average
   raw>=12 and shape>=10.

Historical 2022/2023 CAT5 arrays came from exp/104 and have its documented
same-hand dtype caveat.  The script records that limitation explicitly; it
does not silently treat those arrays as exact current-code replicas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/train.csv"
OUT_TXT = ROOT / "lab/125_exact_cs_raw_gate.txt"
OUT_JSON = ROOT / "lab/125_exact_cs_raw_gate.json"

GAMMA = 0.30
RIDGE_ALPHA = 10000.0
MIN_SINCE_N = 11.0
RELIABILITY_K = 50.0
Z_CLIP = 5.0
EFFECT_CLIP = 0.03
EPS = 1e-6
# Static review found that the available exp/104 2022/2023 arrays contain the
# documented f_same_hand dtype bug.  Keep the implementation for repair, but
# fail closed so nobody can mistake a proxy result for an exact two-transition
# champion gate.  Remove only after wiring corrected 2023 and a regenerated
# corrected 2022 baseline with row-id fingerprints.
BLOCKED_PENDING_CORRECTED_BASELINES = True

RATE_COLUMNS = {
    "success": "asof_pitcher_success_rate",
    "reverse": "asof_pitcher_reverse_rate",
    "middle": "asof_pitcher_middle_rate",
}

BLOCKS = {
    "raw": [f"exact_{name}" for name in RATE_COLUMNS],
    "cutoff_diff": [f"cutoff_diff_{name}" for name in RATE_COLUMNS],
    "combined": [
        *[f"exact_{name}" for name in RATE_COLUMNS],
        *[f"cutoff_diff_{name}" for name in RATE_COLUMNS],
    ],
}

USECOLS = [
    "season",
    "game_type",
    "pitcher_id",
    "asof_pitcher_n",
    "control_success",
    *RATE_COLUMNS.values(),
]


@dataclass
class FrozenRidge:
    columns: list[str]
    means: np.ndarray
    scales: np.ndarray
    beta: np.ndarray
    source_year: int
    source_residual_game_means: dict[str, float]


def score(prediction: np.ndarray, target: np.ndarray) -> float:
    rate = float(np.mean(target))
    brier = float(np.mean((prediction - target) ** 2))
    return 100000.0 * (1.0 - brier / (rate * (1.0 - rate)))


def equal_mean_shape_gain(
    base: np.ndarray, candidate: np.ndarray, target: np.ndarray
) -> float:
    shifted = candidate - float(np.mean(candidate)) + float(np.mean(base))
    return score(shifted, target) - score(base, target)


def domain_contribution(
    base: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    rate = float(np.mean(target))
    improvement = (base - target) ** 2 - (candidate - target) ** 2
    return float(
        100000.0
        * np.sum(improvement[mask])
        / (len(target) * rate * (1.0 - rate))
    )


def load_cat5(year: int) -> np.ndarray:
    if year == 2024:
        paths = [
            ROOT / "lab/89_cat5_probs_seed42.npy",
            ROOT / "lab/89_cat5_probs_seed7.npy",
        ]
    else:
        paths = [
            ROOT / f"lab/104_cat5_y{year}_probs_seed42.npy",
            ROOT / f"lab/104_cat5_y{year}_probs_seed7.npy",
        ]
    arrays = [np.load(path).astype(np.float64)[:, 0] for path in paths]
    return np.mean(arrays, axis=0)


def load_v18_effect(year: int) -> np.ndarray:
    if year == 2024:
        path = ROOT / "lab/103_v18_effect_2024.npy"
    else:
        path = ROOT / f"lab/104_v18_effect_y{year}.npy"
    return np.load(path).astype(np.float64)


def champion_prediction(year: int) -> np.ndarray:
    return np.clip(load_cat5(year) + GAMMA * load_v18_effect(year), EPS, 1.0 - EPS)


def season_row_counts() -> pd.Series:
    seasons = pd.read_csv(
        DATA,
        encoding="utf-8-sig",
        usecols=["season"],
        low_memory=False,
    )["season"]
    if not seasons.is_monotonic_increasing:
        raise AssertionError("train.csv must be season-sorted for staged holdout loading")
    return seasons.value_counts(sort=False).sort_index()


def load_through_year(max_year: int, counts: pd.Series) -> pd.DataFrame:
    nrows = int(counts.loc[counts.index <= max_year].sum())
    frame = pd.read_csv(
        DATA,
        encoding="utf-8-sig",
        usecols=USECOLS,
        nrows=nrows,
        low_memory=False,
    )
    if int(frame["season"].max()) != max_year:
        raise AssertionError(f"staged load ended at {frame['season'].max()}, expected {max_year}")
    if (frame["season"] > max_year).any():
        raise AssertionError("staged load opened a later target season")
    frame["game_type"] = frame["game_type"].astype("string").fillna("__MISSING__").astype(str)
    return frame


def _last_pre_pitch_state(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(
            columns=[
                "pitcher_id",
                "cutoff_n",
                "cutoff_last_success",
                *[f"cutoff_rate_{name}" for name in RATE_COLUMNS],
            ]
        )
    ordered = history.sort_values(
        ["pitcher_id", "asof_pitcher_n"], kind="mergesort"
    )
    last = ordered.groupby("pitcher_id", sort=False).tail(1).copy()
    out = pd.DataFrame(
        {
            "pitcher_id": last["pitcher_id"].to_numpy(),
            "cutoff_n": last["asof_pitcher_n"].fillna(0.0).to_numpy(np.float64),
            "cutoff_last_success": last["control_success"].to_numpy(np.float64),
        }
    )
    for name, column in RATE_COLUMNS.items():
        out[f"cutoff_rate_{name}"] = last[column].to_numpy(np.float64)
    if out["pitcher_id"].duplicated().any():
        raise AssertionError("cutoff table must have one row per pitcher")
    return out


def build_state(frame: pd.DataFrame, year: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    history = frame.loc[frame["season"] < year]
    rows = frame.loc[frame["season"] == year].copy().reset_index(drop=True)
    rows["_row_order"] = np.arange(len(rows), dtype=np.int64)
    cutoff = _last_pre_pitch_state(history)
    work = rows.merge(
        cutoff,
        on="pitcher_id",
        how="left",
        sort=False,
        validate="many_to_one",
    ).sort_values("_row_order", kind="stable").reset_index(drop=True)

    seen = work["cutoff_n"].notna().to_numpy()
    cutoff_n = work["cutoff_n"].fillna(0.0).to_numpy(np.float64)
    current_n = work["asof_pitcher_n"].fillna(0.0).to_numpy(np.float64)
    exact_n = np.maximum(current_n - cutoff_n, 0.0)
    approx_end_n = np.where(seen, cutoff_n + 1.0, 0.0)
    approx_n = np.maximum(current_n - approx_end_n, 0.0)

    result = pd.DataFrame(
        {
            "season": year,
            "game_type": work["game_type"].to_numpy(),
            "pitcher_id": work["pitcher_id"].to_numpy(),
            "target": work["control_success"].to_numpy(np.float64),
            "since_n": exact_n,
            "approx_since_n": approx_n,
        }
    )

    invalid: dict[str, dict[str, int]] = {}
    for name, column in RATE_COLUMNS.items():
        current_rate = work[column].fillna(0.0).to_numpy(np.float64)
        cutoff_rate = work[f"cutoff_rate_{name}"].fillna(0.0).to_numpy(np.float64)
        current_num = np.rint(current_rate * current_n)
        cutoff_num = np.rint(cutoff_rate * cutoff_n)
        exact_num = current_num - cutoff_num

        below = int(np.sum((exact_n > 0.0) & (exact_num < -1e-9)))
        above = int(np.sum((exact_n > 0.0) & (exact_num > exact_n + 1e-9)))
        invalid[name] = {"below_zero": below, "above_denom": above}
        if below or above:
            raise ValueError(
                f"invalid exact numerator for {year}/{name}: below={below}, above={above}"
            )

        exact_raw = np.full(len(work), np.nan, dtype=np.float64)
        positive_exact = exact_n > 0.0
        exact_raw[positive_exact] = exact_num[positive_exact] / exact_n[positive_exact]

        if name == "success":
            cutoff_post_num = cutoff_num + np.where(
                seen,
                work["cutoff_last_success"].fillna(0.0).to_numpy(np.float64),
                0.0,
            )
        else:
            # Exact champion deployment arithmetic for unavailable last-pitch
            # component labels: pseudo-post state r*(n+1).
            cutoff_post_num = np.where(seen, cutoff_rate * (cutoff_n + 1.0), 0.0)
        approx_num = np.maximum(current_rate * current_n - cutoff_post_num, 0.0)
        approx_raw = np.full(len(work), np.nan, dtype=np.float64)
        positive_approx = approx_n > 0.0
        approx_raw[positive_approx] = approx_num[positive_approx] / approx_n[positive_approx]

        result[f"exact_{name}"] = exact_raw
        result[f"approx_{name}"] = approx_raw
        result[f"cutoff_diff_{name}"] = exact_raw - approx_raw

    diagnostics = {
        "year": year,
        "rows": int(len(result)),
        "seen_pitcher_rate": float(np.mean(seen)),
        "since_n_ge_11_rate": float(np.mean(exact_n >= MIN_SINCE_N)),
        "since_n_median": float(np.median(exact_n)),
        "invalid_numerators": invalid,
    }
    return result, diagnostics


def _design_matrix_fit(
    state: pd.DataFrame, columns: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reliable = state["since_n"].to_numpy(np.float64) >= MIN_SINCE_N
    reliability = np.sqrt(
        state["since_n"].to_numpy(np.float64)
        / (state["since_n"].to_numpy(np.float64) + RELIABILITY_K)
    )
    means = np.empty(len(columns), dtype=np.float64)
    scales = np.empty(len(columns), dtype=np.float64)
    matrix = np.zeros((len(state), len(columns)), dtype=np.float64)
    for j, column in enumerate(columns):
        values = state[column].to_numpy(np.float64)
        fit_mask = reliable & np.isfinite(values)
        if int(np.sum(fit_mask)) < 100:
            raise ValueError(f"too few reliable values for {column}")
        means[j] = float(np.mean(values[fit_mask]))
        scales[j] = float(np.std(values[fit_mask]))
        if not np.isfinite(scales[j]) or scales[j] < 1e-8:
            raise ValueError(f"degenerate source scale for {column}: {scales[j]}")
        z = (values - means[j]) / scales[j]
        z = np.where(fit_mask, np.clip(z, -Z_CLIP, Z_CLIP) * reliability, 0.0)
        matrix[:, j] = z
    return matrix, means, scales


def _design_matrix_apply(
    state: pd.DataFrame,
    columns: list[str],
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    reliable = state["since_n"].to_numpy(np.float64) >= MIN_SINCE_N
    reliability = np.sqrt(
        state["since_n"].to_numpy(np.float64)
        / (state["since_n"].to_numpy(np.float64) + RELIABILITY_K)
    )
    matrix = np.zeros((len(state), len(columns)), dtype=np.float64)
    for j, column in enumerate(columns):
        values = state[column].to_numpy(np.float64)
        valid = reliable & np.isfinite(values)
        z = (values - means[j]) / scales[j]
        matrix[:, j] = np.where(
            valid,
            np.clip(z, -Z_CLIP, Z_CLIP) * reliability,
            0.0,
        )
    return matrix


def fit_ridge(
    source: pd.DataFrame,
    source_prediction: np.ndarray,
    columns: list[str],
    source_year: int,
) -> FrozenRidge:
    if len(source) != len(source_prediction):
        raise AssertionError(f"source prediction mismatch for {source_year}")
    x, means, scales = _design_matrix_fit(source, columns)
    residual = source["target"].to_numpy(np.float64) - source_prediction
    groups = pd.Series(residual).groupby(source["game_type"], sort=False).transform("mean")
    centered = residual - groups.to_numpy(np.float64)
    gram = x.T @ x + RIDGE_ALPHA * np.eye(x.shape[1], dtype=np.float64)
    beta = np.linalg.solve(gram, x.T @ centered)
    game_means = (
        pd.DataFrame({"game_type": source["game_type"], "residual": residual})
        .groupby("game_type", sort=False)["residual"]
        .mean()
        .to_dict()
    )
    return FrozenRidge(
        columns=list(columns),
        means=means,
        scales=scales,
        beta=beta,
        source_year=source_year,
        source_residual_game_means={str(k): float(v) for k, v in game_means.items()},
    )


def apply_ridge(model: FrozenRidge, target: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    x = _design_matrix_apply(target, model.columns, model.means, model.scales)
    raw_effect = x @ model.beta
    effect = np.clip(raw_effect, -EFFECT_CLIP, EFFECT_CLIP)
    diagnostics = {
        "effect_mean": float(np.mean(effect)),
        "effect_std": float(np.std(effect)),
        "effect_abs_max": float(np.max(np.abs(effect))),
        "effect_cap_rate": float(np.mean(np.abs(raw_effect) > EFFECT_CLIP)),
    }
    return effect, diagnostics


def evaluate(
    base: np.ndarray,
    effect: np.ndarray,
    state: pd.DataFrame,
) -> dict[str, float]:
    target = state["target"].to_numpy(np.float64)
    if len(base) != len(target):
        raise AssertionError(f"target prediction mismatch for {state['season'].iloc[0]}")
    candidate = np.clip(base + effect, EPS, 1.0 - EPS)
    f_mask = state["game_type"].to_numpy() == "F"
    return {
        "base_score": float(score(base, target)),
        "candidate_score": float(score(candidate, target)),
        "raw_gain": float(score(candidate, target) - score(base, target)),
        "shape_gain": float(equal_mean_shape_gain(base, candidate, target)),
        "F_contribution": float(domain_contribution(base, candidate, target, f_mask)),
        "R_contribution": float(domain_contribution(base, candidate, target, ~f_mask)),
        "mean_shift": float(np.mean(candidate) - np.mean(base)),
    }


def serialise_model(model: FrozenRidge) -> dict[str, Any]:
    return {
        "columns": model.columns,
        "means": model.means.tolist(),
        "scales": model.scales.tolist(),
        "beta": model.beta.tolist(),
        "source_year": model.source_year,
        "source_residual_game_means": model.source_residual_game_means,
    }


def write_report(report: dict[str, Any], lines: list[str]) -> None:
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if BLOCKED_PENDING_CORRECTED_BASELINES:
        raise RuntimeError(
            "exp/125 is blocked: lab/104 2022/2023 baselines have the same-hand "
            "dtype bug. Wire corrected exact baselines before execution."
        )
    counts = season_row_counts()
    frame23 = load_through_year(2023, counts)
    state22, diag22 = build_state(frame23, 2022)
    state23, diag23 = build_state(frame23, 2023)
    base22 = champion_prediction(2022)
    base23 = champion_prediction(2023)
    for year, state, base in ((2022, state22, base22), (2023, state23, base23)):
        if len(state) != len(base):
            raise AssertionError(f"baseline row mismatch for {year}: {len(base)} != {len(state)}")

    report: dict[str, Any] = {
        "experiment": 125,
        "analysis_only": True,
        "test_opened": False,
        "submission_built": False,
        "historical_baseline_caveat": "exp/104 same-hand dtype bug on 2022/2023 arrays",
        "parameters": {
            "gamma": GAMMA,
            "ridge_alpha": RIDGE_ALPHA,
            "min_since_n": MIN_SINCE_N,
            "reliability_k": RELIABILITY_K,
            "z_clip": Z_CLIP,
            "effect_clip": EFFECT_CLIP,
        },
        "state_diagnostics": {"2022": diag22, "2023": diag23},
        "discovery": {},
    }
    lines = [
        "=== exp/125 exact current-season raw state gate ===",
        "zero model training; no test/submission; CAT5+V18 gamma=.30 baseline",
        "discovery: fit 2022 residual -> evaluate 2023",
        "historical caveat: exp/104 2022/23 baseline arrays have same-hand dtype bug",
        "",
    ]

    fitted: dict[str, FrozenRidge] = {}
    passing: list[str] = []
    for name, columns in BLOCKS.items():
        model = fit_ridge(state22, base22, columns, 2022)
        effect, effect_diag = apply_ridge(model, state23)
        result = evaluate(base23, effect, state23)
        result["effect_diagnostics"] = effect_diag
        result["model"] = serialise_model(model)
        gate = bool(
            result["raw_gain"] >= 8.0
            and result["shape_gain"] >= 5.0
            and result["F_contribution"] >= 0.0
            and result["R_contribution"] >= 0.0
        )
        result["discovery_gate_pass"] = gate
        report["discovery"][name] = result
        fitted[name] = model
        if gate:
            passing.append(name)
        lines.append(
            f"[{name}] raw={result['raw_gain']:+.3f} shape={result['shape_gain']:+.3f} "
            f"F/R={result['F_contribution']:+.3f}/{result['R_contribution']:+.3f} "
            f"mean={result['mean_shift']:+.6f} gate={'PASS' if gate else 'FAIL'}"
        )

    if not passing:
        report["selected_block"] = None
        report["confirmation_2024_opened"] = False
        report["final_gate_pass"] = False
        lines.extend(["", "DISCOVERY FAIL — 2024 targets remain unopened; no CatBoost arm."])
        write_report(report, lines)
        return

    selected = max(
        passing,
        key=lambda key: (
            min(
                float(report["discovery"][key]["raw_gain"]),
                float(report["discovery"][key]["shape_gain"]),
            ),
            float(report["discovery"][key]["raw_gain"]),
        ),
    )
    report["selected_block"] = selected
    lines.extend(["", f"selected on 2023 only: {selected}", "opening 2024 once..."])

    # Confirmation is reached only after selection is fully frozen.
    frame24 = load_through_year(2024, counts)
    state24, diag24 = build_state(frame24, 2024)
    base24 = champion_prediction(2024)
    if len(state24) != len(base24):
        raise AssertionError(f"baseline row mismatch for 2024: {len(base24)} != {len(state24)}")
    report["state_diagnostics"]["2024"] = diag24
    report["confirmation_2024_opened"] = True

    frozen_effect, frozen_effect_diag = apply_ridge(fitted[selected], state24)
    frozen_result = evaluate(base24, frozen_effect, state24)
    frozen_result["effect_diagnostics"] = frozen_effect_diag

    rolling_model = fit_ridge(state23, base23, BLOCKS[selected], 2023)
    rolling_effect, rolling_effect_diag = apply_ridge(rolling_model, state24)
    rolling_result = evaluate(base24, rolling_effect, state24)
    rolling_result["effect_diagnostics"] = rolling_effect_diag
    rolling_result["model"] = serialise_model(rolling_model)

    report["confirmation"] = {
        "frozen22_to_2024": frozen_result,
        "rolling23_to_2024": rolling_result,
    }
    discovery_selected = report["discovery"][selected]
    mean_raw = float(np.mean([discovery_selected["raw_gain"], rolling_result["raw_gain"]]))
    mean_shape = float(
        np.mean([discovery_selected["shape_gain"], rolling_result["shape_gain"]])
    )
    final_gate = bool(
        frozen_result["raw_gain"] > 0.0
        and frozen_result["shape_gain"] > 0.0
        and rolling_result["raw_gain"] > 0.0
        and rolling_result["shape_gain"] > 0.0
        and rolling_result["F_contribution"] >= 0.0
        and rolling_result["R_contribution"] >= 0.0
        and mean_raw >= 12.0
        and mean_shape >= 10.0
    )
    report["mean_rolling_transition_raw_gain"] = mean_raw
    report["mean_rolling_transition_shape_gain"] = mean_shape
    report["final_gate_pass"] = final_gate

    for label, result in (
        ("frozen22->2024", frozen_result),
        ("rolling23->2024", rolling_result),
    ):
        lines.append(
            f"[{label}] raw={result['raw_gain']:+.3f} shape={result['shape_gain']:+.3f} "
            f"F/R={result['F_contribution']:+.3f}/{result['R_contribution']:+.3f} "
            f"mean={result['mean_shift']:+.6f}"
        )
    lines.extend(
        [
            f"rolling-transition mean raw/shape={mean_raw:+.3f}/{mean_shape:+.3f}",
            "",
            (
                "FINAL PASS — authorize one seed42 CatBoost raw-state arm."
                if final_gate
                else "FINAL FAIL — do not train or submit this axis."
            ),
        ]
    )
    write_report(report, lines)


if __name__ == "__main__":
    main()
