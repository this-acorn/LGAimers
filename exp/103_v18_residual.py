# -*- coding: utf-8 -*-
"""
[103] Public V18-style hierarchical conditional residual pre-gate.

This is a reproducible, read-only validation step.  It does not train a model,
build a submission, or use test rows.  For validation year Y every lookup is
computed from train rows with season < Y.

The public repository does not contain the real V18 implementation.  This file
therefore tests only the smallest reconstructable proxy from its later public
hierarchy:

  (game_type, pitcher)
    -> (game_type, pitcher, batter_hand)
      -> (game_type, pitcher, pressure_state, batter_hand)

For each child level we form a shrunk delta from its parent and multiply it by
the child's reliability.  The final correction is the sum of those two trusted
deltas.  Gamma is selected using 2021-2023 only; 2024 is evaluated afterwards.

Outputs:
  lab/103_v18_pregate.txt
  lab/103_v18_pregate.json
  lab/103_v18_effect_2024.npy

Run:
  PYTHONIOENCODING=utf-8 py -3.12 -u exp/103_v18_residual.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


DATA = Path("data/train.csv")
OOF = Path("lab/24b_preds.npz")
CAT5_2024 = Path("lab/89_cat5.npy")
OUT_TXT = Path("lab/103_v18_pregate.txt")
OUT_JSON = Path("lab/103_v18_pregate.json")
OUT_EFFECT = Path("lab/103_v18_effect_2024.npy")

YEARS = (2021, 2022, 2023, 2024)
DISCOVERY_YEARS = (2021, 2022, 2023)
GAMMAS = np.round(np.arange(0.0, 0.701, 0.05), 2)

# Later public V25 hierarchy values.  They are not claimed to be the missing
# V18 hyperparameters; fixing them here prevents tuning them on our 2024 fold.
ROOT_PRIOR = 0.53
K_ROOT = 1200.0
K_PITCHER = 220.0
K_HAND = 110.0
K_PRESSURE_HAND = 220.0


LINES: list[str] = []


def log(message: str = "") -> None:
    print(message, flush=True)
    LINES.append(message)


def score(probability: np.ndarray, target: np.ndarray) -> float:
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    rate = float(y.mean())
    return 100000.0 * (1.0 - float(np.mean((p - y) ** 2)) / (rate * (1.0 - rate)))


def pressure_code(frame: pd.DataFrame) -> np.ndarray:
    balls = frame["balls_before"].fillna(0).to_numpy(np.int8)
    strikes = frame["strikes_before"].fillna(0).to_numpy(np.int8)
    # 2=full, 1=high (three-ball or two-strike), 0=normal.
    return np.where(
        (balls == 3) & (strikes == 2),
        2,
        np.where((balls == 3) | (strikes == 2), 1, 0),
    ).astype(np.int8)


def grouped(history: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        history.groupby(keys, sort=False, observed=True)["control_success"]
        .agg(success="sum", n="size")
    )


def lookup(table: pd.DataFrame, rows: pd.DataFrame, keys: list[str], column: str) -> np.ndarray:
    if len(keys) == 1:
        index = pd.Index(rows[keys[0]], name=keys[0])
    else:
        index = pd.MultiIndex.from_frame(rows[keys])
    return table[column].reindex(index).fillna(0.0).to_numpy(np.float64)


def make_effect(history: pd.DataFrame, validation: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    root_keys = ["game_type"]
    pitcher_keys = ["game_type", "pitcher_id"]
    hand_keys = ["game_type", "pitcher_id", "batter_hand"]
    detail_keys = ["game_type", "pitcher_id", "_pressure"]
    detail_keys.append("batter_hand")

    root = grouped(history, root_keys)
    pitcher = grouped(history, pitcher_keys)
    hand = grouped(history, hand_keys)
    detail = grouped(history, detail_keys)

    root_n = lookup(root, validation, root_keys, "n")
    root_s = lookup(root, validation, root_keys, "success")
    root_rate = (root_s + K_ROOT * ROOT_PRIOR) / (root_n + K_ROOT)

    pitcher_n = lookup(pitcher, validation, pitcher_keys, "n")
    pitcher_s = lookup(pitcher, validation, pitcher_keys, "success")
    pitcher_rate = (pitcher_s + K_PITCHER * root_rate) / (pitcher_n + K_PITCHER)

    hand_n = lookup(hand, validation, hand_keys, "n")
    hand_s = lookup(hand, validation, hand_keys, "success")
    hand_rate = (hand_s + K_HAND * pitcher_rate) / (hand_n + K_HAND)
    hand_rel = hand_n / (hand_n + K_HAND)
    hand_trusted = hand_rel * (hand_rate - pitcher_rate)

    detail_n = lookup(detail, validation, detail_keys, "n")
    detail_s = lookup(detail, validation, detail_keys, "success")
    detail_rate = (detail_s + K_PRESSURE_HAND * hand_rate) / (
        detail_n + K_PRESSURE_HAND
    )
    detail_rel = detail_n / (detail_n + K_PRESSURE_HAND)
    detail_trusted = detail_rel * (detail_rate - hand_rate)

    effect = hand_trusted + detail_trusted
    diagnostics = {
        "rows": int(len(validation)),
        "hand_coverage": float(np.mean(hand_n > 0)),
        "detail_coverage": float(np.mean(detail_n > 0)),
        "hand_n_mean": float(hand_n.mean()),
        "detail_n_mean": float(detail_n.mean()),
        "effect_mean": float(effect.mean()),
        "effect_std": float(effect.std()),
        "effect_abs_mean": float(np.mean(np.abs(effect))),
    }
    return effect, diagnostics


def equal_mean_shape_gain(base: np.ndarray, candidate: np.ndarray, y: np.ndarray) -> float:
    # Brier accepts real-valued predictions; do not clip a second time because
    # that would change the intended exact equal-mean decomposition.
    same_mean = candidate - candidate.mean() + base.mean()
    return score(same_mean, y) - score(base, y)


def domain_contribution(
    base: np.ndarray, candidate: np.ndarray, y: np.ndarray, mask: np.ndarray
) -> float:
    rate = float(y.mean())
    row_gain = (base - y) ** 2 - (candidate - y) ** 2
    return 100000.0 * float(np.sum(row_gain[mask])) / (len(y) * rate * (1.0 - rate))


def evaluate_curve(
    base: np.ndarray,
    effect: np.ndarray,
    y: np.ndarray,
    game_type: np.ndarray,
) -> dict[str, dict[str, float]]:
    base = np.asarray(base, dtype=np.float64)
    effect = np.asarray(effect, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    base_score = score(base, y)
    out: dict[str, dict[str, float]] = {}
    for gamma in GAMMAS:
        candidate = np.clip(base + float(gamma) * effect, 0.0, 1.0)
        raw = score(candidate, y) - base_score
        shape = equal_mean_shape_gain(base, candidate, y)
        f_mask = game_type == "F"
        out[f"{gamma:.2f}"] = {
            "raw_gain": float(raw),
            "shape_gain": float(shape),
            "mean_shift": float(candidate.mean() - base.mean()),
            "F_contribution": float(domain_contribution(base, candidate, y, f_mask)),
            "R_contribution": float(domain_contribution(base, candidate, y, ~f_mask)),
        }
    return out


def choose_gamma(curves: dict[int, dict[str, dict[str, float]]]) -> tuple[float, list[dict[str, float]]]:
    """Robust choice using discovery years only: maximize worst-year raw gain.

    Ties are broken by mean raw gain and then by the smaller gamma.  Gamma zero
    remains available, so this selector can reject the axis without seeing 2024.
    """
    rows: list[dict[str, float]] = []
    for gamma in GAMMAS:
        key = f"{gamma:.2f}"
        gains = [curves[year][key]["raw_gain"] for year in DISCOVERY_YEARS]
        shapes = [curves[year][key]["shape_gain"] for year in DISCOVERY_YEARS]
        rows.append(
            {
                "gamma": float(gamma),
                "min_gain": float(min(gains)),
                "mean_gain": float(np.mean(gains)),
                "min_shape": float(min(shapes)),
                "mean_shape": float(np.mean(shapes)),
            }
        )
    best = max(rows, key=lambda row: (row["min_gain"], row["mean_gain"], -row["gamma"]))
    return float(best["gamma"]), rows


def cluster_bootstrap_gain(
    base: np.ndarray,
    candidate: np.ndarray,
    y: np.ndarray,
    pitcher: np.ndarray,
    draws: int = 5000,
    seed: int = 103,
) -> dict[str, float]:
    work = pd.DataFrame(
        {
            "pitcher": pitcher,
            "n": np.ones(len(y), dtype=np.int32),
            "y": y.astype(np.float64),
            "base_se": (base - y) ** 2,
            "cand_se": (candidate - y) ** 2,
        }
    )
    agg = work.groupby("pitcher", sort=False, observed=True).agg(
        n=("n", "sum"), y=("y", "sum"), base_se=("base_se", "sum"), cand_se=("cand_se", "sum")
    )
    values = agg[["n", "y", "base_se", "cand_se"]].to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    gains = np.empty(draws, dtype=np.float64)
    groups = len(values)
    # Chunk draws to keep the temporary index array small.
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


def main() -> None:
    started = time.time()
    required = [DATA, OOF, CAT5_2024]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required files: {missing}")

    columns = [
        "season",
        "game_type",
        "pitcher_id",
        "batter_hand",
        "balls_before",
        "strikes_before",
        "control_success",
    ]
    log("=== exp/103 V18-style hierarchical residual pre-gate ===")
    log("Loading only required train columns...")
    frame = pd.read_csv(DATA, encoding="utf-8-sig", usecols=columns, low_memory=False)
    frame["game_type"] = frame["game_type"].astype("string").fillna("__MISSING__").astype(str)
    frame["batter_hand"] = frame["batter_hand"].astype("string").fillna("__MISSING__").astype(str)
    frame["_pressure"] = pressure_code(frame)
    oof = np.load(OOF)

    curves: dict[int, dict[str, dict[str, float]]] = {}
    effects: dict[int, np.ndarray] = {}
    diagnostics: dict[int, dict[str, float]] = {}

    # Compute every effect from strict past seasons.  Selection below explicitly
    # receives only DISCOVERY_YEARS; 2024 cannot influence gamma.
    for year in YEARS:
        history = frame[frame["season"] < year]
        validation = frame[frame["season"] == year]
        y = validation["control_success"].to_numpy(np.float64)
        base = oof[f"{year}_base65"].astype(np.float64)
        saved_y = oof[f"{year}_y"].astype(np.float64)
        if len(base) != len(validation) or not np.array_equal(y, saved_y):
            raise AssertionError(f"OOF alignment failed for {year}")
        effect, diag = make_effect(history, validation)
        effects[year] = effect
        diagnostics[year] = diag
        curves[year] = evaluate_curve(
            base,
            effect,
            y,
            validation["game_type"].to_numpy(),
        )
        log(
            f"{year}: rows={len(y):,} base={score(base, y):.2f} "
            f"coverage hand/detail={diag['hand_coverage']:.3f}/{diag['detail_coverage']:.3f} "
            f"effect mean/std={diag['effect_mean']:+.6f}/{diag['effect_std']:.6f}"
        )

    selected_gamma, discovery_table = choose_gamma(curves)
    selected_key = f"{selected_gamma:.2f}"
    log("")
    log("Discovery selector: maximize minimum raw gain across 2021-2023")
    log(f"LOCKED gamma={selected_gamma:.2f} before reading the 2024 result row")
    log(" gamma   min_raw  mean_raw  min_shape mean_shape")
    for row in discovery_table:
        log(
            f" {row['gamma']:5.2f} {row['min_gain']:+9.2f} {row['mean_gain']:+9.2f} "
            f"{row['min_shape']:+10.2f} {row['mean_shape']:+10.2f}"
        )

    log("")
    log("Locked-gamma year results")
    log(" year  raw_gain shape_gain  F_contrib  R_contrib mean_shift")
    for year in YEARS:
        row = curves[year][selected_key]
        log(
            f" {year} {row['raw_gain']:+9.2f} {row['shape_gain']:+10.2f} "
            f"{row['F_contribution']:+10.2f} {row['R_contribution']:+10.2f} "
            f"{row['mean_shift']:+.7f}"
        )

    # Exact cross-check on the current it500 CAT5 2024 saved validation array.
    validation_2024 = frame[frame["season"] == 2024]
    y_2024 = validation_2024["control_success"].to_numpy(np.float64)
    cat5 = np.load(CAT5_2024).astype(np.float64)
    if len(cat5) != len(y_2024):
        raise AssertionError("CAT5 2024 alignment failed")
    cat_curve = evaluate_curve(
        cat5,
        effects[2024],
        y_2024,
        validation_2024["game_type"].to_numpy(),
    )
    cat_locked = cat_curve[selected_key]
    cat_candidate = np.clip(cat5 + selected_gamma * effects[2024], 0.0, 1.0)
    bootstrap = cluster_bootstrap_gain(
        cat5,
        cat_candidate,
        y_2024,
        validation_2024["pitcher_id"].to_numpy(),
    )
    np.save(OUT_EFFECT, effects[2024].astype(np.float32))

    log("")
    log(f"Current it500 CAT5 2024 cross-check (gamma locked at {selected_gamma:.2f})")
    log(
        f"base={score(cat5, y_2024):.3f} raw={cat_locked['raw_gain']:+.3f} "
        f"shape={cat_locked['shape_gain']:+.3f} "
        f"F/R={cat_locked['F_contribution']:+.3f}/{cat_locked['R_contribution']:+.3f} "
        f"mean_shift={cat_locked['mean_shift']:+.7f}"
    )
    log(
        "pitcher bootstrap: median={median:+.3f} 95%=[{p025:+.3f},{p975:+.3f}] "
        "P(>0)={prob_positive:.4f}".format(**bootstrap)
    )

    discovery_positive = all(
        curves[year][selected_key]["raw_gain"] > 0.0
        and curves[year][selected_key]["shape_gain"] > 0.0
        for year in DISCOVERY_YEARS
    )
    hgb_confirm = curves[2024][selected_key]
    passes = bool(
        selected_gamma > 0.0
        and discovery_positive
        and hgb_confirm["raw_gain"] >= 8.0
        and hgb_confirm["shape_gain"] >= 5.0
        and cat_locked["raw_gain"] >= 15.0
        and cat_locked["shape_gain"] >= 10.0
        and cat_locked["F_contribution"] >= 0.0
        and cat_locked["R_contribution"] >= 0.0
        and bootstrap["p025"] > 0.0
    )
    log("")
    log(
        "PRE-GATE "
        + ("PASS" if passes else "FAIL")
        + ": discovery all raw/shape positive; HGB2024 raw>=8 shape>=5; "
        "CAT5 raw>=15 shape>=10 F/R>=0; bootstrap p025>0"
    )
    log("PASS only authorizes exact past-year CAT5 folds; it does not authorize an LB submission.")
    log(f"elapsed={time.time() - started:.1f}s")

    payload = {
        "experiment": 103,
        "description": "public V18-style hierarchical conditional residual proxy",
        "not_exact_v18": True,
        "history_rule": "season < validation_year",
        "constants": {
            "root_prior": ROOT_PRIOR,
            "k_root": K_ROOT,
            "k_pitcher": K_PITCHER,
            "k_hand": K_HAND,
            "k_pressure_hand": K_PRESSURE_HAND,
        },
        "discovery_years": list(DISCOVERY_YEARS),
        "gamma_grid": [float(x) for x in GAMMAS],
        "selection_rule": "maximize min raw gain over 2021-2023; tie mean then smaller gamma",
        "selected_gamma": selected_gamma,
        "discovery_table": discovery_table,
        "hgb_curves": {str(year): curves[year] for year in YEARS},
        "diagnostics": {str(year): diagnostics[year] for year in YEARS},
        "cat5_2024_curve": cat_curve,
        "cat5_2024_locked": cat_locked,
        "cat5_2024_bootstrap": bootstrap,
        "pregate_pass": passes,
        "caveat": "2024 had already been inspected manually before this script; confirmation is reproducible but not pristine untouched evidence",
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_TXT.write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
