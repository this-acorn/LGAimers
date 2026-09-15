# -*- coding: utf-8 -*-
"""EXP-154: strict past-only batter hidden-call empirical-Bayes screen.

This is a bounded, official-train-only diagnostic.  It does not fit a model.
The recovered five-class target is the already established taxonomy:

    0 success, 1 middle-only, 2 reverse-only, 3 middle/reverse overlap,
    4 big miss (failure with neither middle nor reverse).

For each validation season, a contextual parent distribution is estimated
from earlier seasons only using (both hands, exact count, runner-call state).
A batter lookup then updates only reverse/overlap/big-miss probabilities;
the remaining mass retains the parent's success:middle ratio.  Batter ID is
never exposed to a learner and validation labels never enter a lookup table.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/train.csv")
TEXT_PATH = Path("lab/154_batter_hidden_call_screen.txt")
JSON_PATH = Path("lab/154_batter_hidden_call_screen.json")

VALID_YEARS = (2022, 2023, 2024)
PARENT_K = 200.0
BATTER_KS = (50.0, 200.0, 800.0)
CAUSE_CLASSES = (2, 3, 4)
CLASS_NAMES = ("success", "middle_only", "reverse_only", "overlap", "bigmiss")
EPS = 1e-12
HARD_LIMIT_SECONDS = 570.0


def check_time(started: float, phase: str) -> None:
    elapsed = time.time() - started
    if elapsed > HARD_LIMIT_SECONDS:
        raise TimeoutError(
            f"EXP-154 hard timebox exceeded after {phase}: {elapsed:.1f}s"
        )


def recover_binary_label(
    sorted_rate: np.ndarray,
    sorted_n: np.ndarray,
    has_next: np.ndarray,
    order: np.ndarray,
) -> np.ndarray:
    cumulative = np.nan_to_num(sorted_rate, nan=0.0) * sorted_n
    delta = np.rint(cumulative[1:] - cumulative[:-1])
    recovered_sorted = np.full(len(sorted_n), -1, dtype=np.int8)
    good = has_next & ((delta == 0.0) | (delta == 1.0))
    positions = np.flatnonzero(good)
    recovered_sorted[positions] = delta[positions].astype(np.int8)
    recovered = np.full(len(sorted_n), -1, dtype=np.int8)
    recovered[order] = recovered_sorted
    return recovered


def recover_target5(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    pitcher = frame["pitcher_id"].to_numpy(np.int64)
    n = frame["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    order = np.lexsort((n, pitcher))
    pitcher_sorted = pitcher[order]
    n_sorted = n[order]
    has_next = (
        (pitcher_sorted[1:] == pitcher_sorted[:-1])
        & (np.abs(n_sorted[1:] - n_sorted[:-1] - 1.0) < 1e-6)
    )

    recovered: dict[str, np.ndarray] = {}
    for name, column in (
        ("success", "asof_pitcher_success_rate"),
        ("middle", "asof_pitcher_middle_rate"),
        ("reverse", "asof_pitcher_reverse_rate"),
    ):
        rate_sorted = frame[column].to_numpy(np.float64)[order]
        recovered[name] = recover_binary_label(
            rate_sorted, n_sorted, has_next, order
        )

    y = frame["control_success"].to_numpy(np.int8)
    success_valid = recovered["success"] >= 0
    success_mismatch = int(np.sum(recovered["success"][success_valid] != y[success_valid]))
    if success_mismatch != 0:
        raise AssertionError(f"recovered success mismatch: {success_mismatch}")

    complete = (recovered["middle"] >= 0) & (recovered["reverse"] >= 0)
    middle = recovered["middle"] == 1
    reverse = recovered["reverse"] == 1
    target5 = np.full(len(frame), -1, dtype=np.int8)
    target5[complete & (y == 1)] = 0
    target5[complete & (y == 0) & middle & ~reverse] = 1
    target5[complete & (y == 0) & ~middle & reverse] = 2
    target5[complete & (y == 0) & middle & reverse] = 3
    target5[complete & (y == 0) & ~middle & ~reverse] = 4

    if np.any((target5[complete] < 0) | (target5[complete] > 4)):
        raise AssertionError("target5 assembly failed on a complete row")
    if np.any((y == 1) & complete & (middle | reverse)):
        raise AssertionError("success overlaps recovered middle/reverse")

    diagnostics = {
        "next_row_coverage": float(success_valid.mean()),
        "target5_coverage": float((target5 >= 0).mean()),
        "success_mismatch": success_mismatch,
    }
    return target5, diagnostics


def add_static_keys(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    balls = frame["balls_before"].to_numpy(np.int16)
    strikes = frame["strikes_before"].to_numpy(np.int16)
    if not set(np.unique(balls)).issubset({0, 1, 2, 3}):
        raise ValueError("unexpected balls_before value")
    if not set(np.unique(strikes)).issubset({0, 1, 2}):
        raise ValueError("unexpected strikes_before value")
    count = balls * 3 + strikes

    r1 = frame["runner_on_1b"].to_numpy(np.int8) == 1
    r2 = frame["runner_on_2b"].to_numpy(np.int8) == 1
    r3 = frame["runner_on_3b"].to_numpy(np.int8) == 1
    steal_threat = r1 & ~r2
    scoring_position = r2 | r3
    runner_state = steal_threat.astype(np.int8) + 2 * scoring_position.astype(np.int8)

    pitcher_hand = frame["pitcher_hand"].to_numpy(np.int16)
    batter_hand = frame["batter_hand"].to_numpy(np.int16)
    if not set(np.unique(pitcher_hand)).issubset({1, 2}):
        raise ValueError("unexpected pitcher_hand value")
    if not set(np.unique(batter_hand)).issubset({1, 2}):
        raise ValueError("unexpected batter_hand value")

    # Dense integer key for exactly (pitcher_hand, batter_hand, count, runner_state).
    parent_key = (
        ((((pitcher_hand - 1) * 2 + (batter_hand - 1)) * 12 + count) * 4)
        + runner_state
    ).astype(np.int16)
    if parent_key.min() < 0 or parent_key.max() >= 192:
        raise AssertionError("parent key outside 4*12*4 states")

    batter_code, batter_levels = pd.factorize(frame["batter_id"], sort=True)
    if np.any(batter_code < 0):
        raise ValueError("missing batter_id is not supported")
    diagnostics = {
        "parent_states_observed": int(np.unique(parent_key).size),
        "batter_levels": int(len(batter_levels)),
        "runner_state_definition": {
            "steal_threat": "runner_on_1b == 1 and runner_on_2b == 0",
            "scoring_position": "runner_on_2b == 1 or runner_on_3b == 1",
            "state": "steal_threat + 2*scoring_position",
        },
    }
    return parent_key, batter_code.astype(np.int32), diagnostics


def multiclass_logloss(probability: np.ndarray, target: np.ndarray) -> float:
    selected = probability[np.arange(len(target)), target]
    return float(-np.mean(np.log(np.clip(selected, EPS, 1.0))))


def class_brier(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    observed = np.eye(5, dtype=np.float64)[target]
    return np.mean((probability - observed) ** 2, axis=0)


def build_parent(
    source_key: np.ndarray,
    source_target: np.ndarray,
    validation_key: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global_counts = np.bincount(source_target, minlength=5).astype(np.float64)
    global_probability = global_counts / global_counts.sum()
    state_n = np.bincount(source_key, minlength=192).astype(np.float64)
    state_counts = np.zeros((192, 5), dtype=np.float64)
    for class_index in range(5):
        state_counts[:, class_index] = np.bincount(
            source_key,
            weights=(source_target == class_index).astype(np.float64),
            minlength=192,
        )
    state_probability = (
        state_counts + PARENT_K * global_probability[None, :]
    ) / (state_n[:, None] + PARENT_K)
    parent = state_probability[validation_key]
    if not np.allclose(parent.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("parent probabilities do not sum to one")
    return parent, global_probability, state_n


def apply_batter_cause_eb(
    parent: np.ndarray,
    validation_batter: np.ndarray,
    batter_n: np.ndarray,
    batter_cause_counts: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    local_n = batter_n[validation_batter]
    seen = local_n > 0
    candidate = parent.copy()
    denominator = local_n + strength
    for local_column, class_index in enumerate(CAUSE_CLASSES):
        candidate[:, class_index] = (
            batter_cause_counts[validation_batter, local_column]
            + strength * parent[:, class_index]
        ) / denominator

    cause_mass = candidate[:, list(CAUSE_CLASSES)].sum(axis=1)
    remaining = np.clip(1.0 - cause_mass, 0.0, 1.0)
    parent_remaining = parent[:, 0] + parent[:, 1]
    candidate[:, 0] = remaining * parent[:, 0] / parent_remaining
    candidate[:, 1] = remaining * parent[:, 1] / parent_remaining

    # Deployment contract: an unseen batter is exactly the contextual parent.
    candidate[~seen] = parent[~seen]
    cold_max_abs_difference = (
        float(np.max(np.abs(candidate[~seen] - parent[~seen])))
        if np.any(~seen)
        else 0.0
    )
    if cold_max_abs_difference != 0.0:
        raise AssertionError("cold batter fallback is not exact")
    if not np.isfinite(candidate).all() or np.any(candidate < 0.0):
        raise AssertionError("invalid candidate probability")
    if not np.allclose(candidate.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("candidate probabilities do not sum to one")
    return candidate, seen, cold_max_abs_difference


def main() -> None:
    started = time.time()
    columns = [
        "season",
        "pitcher_id",
        "batter_id",
        "pitcher_hand",
        "batter_hand",
        "balls_before",
        "strikes_before",
        "runner_on_1b",
        "runner_on_2b",
        "runner_on_3b",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_reverse_rate",
        "control_success",
    ]
    print("EXP-154 loading official train columns...", flush=True)
    frame = pd.read_csv(DATA_PATH, encoding="utf-8-sig", usecols=columns)
    check_time(started, "data load")
    print(f"rows={len(frame):,}; recovering existing target5...", flush=True)
    target5, recovery = recover_target5(frame)
    check_time(started, "target recovery")
    parent_key, batter_code, key_diagnostics = add_static_keys(frame)
    seasons = frame["season"].to_numpy(np.int16)
    del frame
    print(
        f"target5 coverage={recovery['target5_coverage']*100:.3f}% "
        f"success mismatch={recovery['success_mismatch']}",
        flush=True,
    )

    folds: dict[str, object] = {}
    for year in VALID_YEARS:
        source_mask = (seasons < year) & (target5 >= 0)
        validation_mask = (seasons == year) & (target5 >= 0)
        source_target = target5[source_mask]
        validation_target = target5[validation_mask]
        source_key = parent_key[source_mask]
        validation_key = parent_key[validation_mask]
        source_batter = batter_code[source_mask]
        validation_batter = batter_code[validation_mask]

        parent, global_probability, parent_n = build_parent(
            source_key, source_target, validation_key
        )
        n_batters = int(batter_code.max()) + 1
        batter_n = np.bincount(source_batter, minlength=n_batters).astype(np.float64)
        batter_cause_counts = np.zeros((n_batters, 3), dtype=np.float64)
        for local_column, class_index in enumerate(CAUSE_CLASSES):
            batter_cause_counts[:, local_column] = np.bincount(
                source_batter,
                weights=(source_target == class_index).astype(np.float64),
                minlength=n_batters,
            )

        parent_logloss = multiclass_logloss(parent, validation_target)
        parent_brier = class_brier(parent, validation_target)
        candidates: dict[str, object] = {}
        for strength in BATTER_KS:
            candidate, seen, cold_diff = apply_batter_cause_eb(
                parent,
                validation_batter,
                batter_n,
                batter_cause_counts,
                strength,
            )
            candidate_logloss = multiclass_logloss(candidate, validation_target)
            candidate_brier = class_brier(candidate, validation_target)
            brier_gain = parent_brier - candidate_brier
            key = str(int(strength))
            candidates[key] = {
                "strength": strength,
                "target5_logloss": candidate_logloss,
                "target5_logloss_gain": parent_logloss - candidate_logloss,
                "one_vs_rest_brier": {
                    CLASS_NAMES[index]: float(candidate_brier[index])
                    for index in range(5)
                },
                "one_vs_rest_brier_gain": {
                    CLASS_NAMES[index]: float(brier_gain[index])
                    for index in range(5)
                },
                "coverage_seen_batter": float(seen.mean()),
                "cold_rows": int((~seen).sum()),
                "cold_fallback_max_abs_difference": cold_diff,
                "prediction_mean": {
                    CLASS_NAMES[index]: float(candidate[:, index].mean())
                    for index in range(5)
                },
            }
        folds[str(year)] = {
            "source_rows": int(source_mask.sum()),
            "validation_rows": int(validation_mask.sum()),
            "source_years": sorted(np.unique(seasons[source_mask]).astype(int).tolist()),
            "global_probability": {
                CLASS_NAMES[index]: float(global_probability[index])
                for index in range(5)
            },
            "parent_states_with_source_support": int((parent_n > 0).sum()),
            "parent_target5_logloss": parent_logloss,
            "parent_one_vs_rest_brier": {
                CLASS_NAMES[index]: float(parent_brier[index])
                for index in range(5)
            },
            "candidates": candidates,
        }
        gains = ", ".join(
            f"K={key}: {value['target5_logloss_gain']:+.6f}"
            for key, value in candidates.items()
        )
        print(
            f"year={year} source={source_mask.sum():,} valid={validation_mask.sum():,} "
            f"parent_ll={parent_logloss:.6f} gains[{gains}]",
            flush=True,
        )
        check_time(started, f"fold {year}")

    summaries: dict[str, object] = {}
    for strength in BATTER_KS:
        key = str(int(strength))
        logloss_gains = {
            year: float(folds[str(year)]["candidates"][key]["target5_logloss_gain"])
            for year in VALID_YEARS
        }
        cause_brier_gains = {
            CLASS_NAMES[class_index]: {
                year: float(
                    folds[str(year)]["candidates"][key]["one_vs_rest_brier_gain"]
                    [CLASS_NAMES[class_index]]
                )
                for year in VALID_YEARS
            }
            for class_index in CAUSE_CLASSES
        }
        coverages = {
            year: float(folds[str(year)]["candidates"][key]["coverage_seen_batter"])
            for year in VALID_YEARS
        }
        cold_exact = all(
            folds[str(year)]["candidates"][key]["cold_fallback_max_abs_difference"] == 0.0
            for year in VALID_YEARS
        )
        gate = (
            all(value > 0.0 for value in logloss_gains.values())
            and logloss_gains[2023] >= 0.003
            and logloss_gains[2024] >= 0.003
            and all(
                value > 0.0
                for class_gains in cause_brier_gains.values()
                for value in class_gains.values()
            )
            and min(coverages.values()) >= 0.70
            and cold_exact
        )
        summaries[key] = {
            "strength": strength,
            "logloss_gains": logloss_gains,
            "worst_logloss_gain": min(logloss_gains.values()),
            "mean_logloss_gain": float(np.mean(list(logloss_gains.values()))),
            "cause_brier_gains": cause_brier_gains,
            "minimum_coverage": min(coverages.values()),
            "cold_fallback_exact": cold_exact,
            "gate_pass": bool(gate),
        }

    selected_key = max(
        summaries,
        key=lambda key: (
            bool(summaries[key]["gate_pass"]),
            float(summaries[key]["worst_logloss_gain"]),
            float(summaries[key]["mean_logloss_gain"]),
        ),
    )
    overall_pass = any(bool(value["gate_pass"]) for value in summaries.values())
    elapsed = time.time() - started
    report = {
        "experiment": 154,
        "scope": "official train only; arithmetic EB screen; no learner/test/zip/LB",
        "specification": {
            "validation_years": list(VALID_YEARS),
            "parent_keys": [
                "pitcher_hand",
                "batter_hand",
                "exact_12_count",
                "runner_call_state",
            ],
            "parent_strength": PARENT_K,
            "batter_strengths": list(BATTER_KS),
            "batter_updated_classes": [CLASS_NAMES[index] for index in CAUSE_CLASSES],
            "success_middle_rule": "preserve parent ratio in remaining probability mass",
            "direct_batter_id_model": False,
            "selection": "gate first, then maximum worst-fold logloss gain",
        },
        "recovery": recovery,
        "keys": key_diagnostics,
        "folds": folds,
        "summaries": summaries,
        "selected_strength_diagnostic_only": int(selected_key),
        "overall_gate_pass": overall_pass,
        "elapsed_seconds": elapsed,
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "=== EXP-154 batter hidden-call strict EB screen ===",
        "official train only; no model/test/ZIP/LB",
        f"rows={len(seasons):,} target5_coverage={recovery['target5_coverage']:.6f} "
        f"success_mismatch={recovery['success_mismatch']}",
        f"parent=(hands, exact count, runner-call), K={PARENT_K:g}; "
        f"batter cause K={list(int(value) for value in BATTER_KS)}",
        "",
    ]
    for year in VALID_YEARS:
        fold = folds[str(year)]
        lines.append(
            f"{year}: parent logloss={fold['parent_target5_logloss']:.8f} "
            f"coverage={fold['candidates'][str(int(BATTER_KS[0]))]['coverage_seen_batter']:.3%}"
        )
        for strength in BATTER_KS:
            candidate = fold["candidates"][str(int(strength))]
            gains = candidate["one_vs_rest_brier_gain"]
            lines.append(
                f"  K={int(strength):3d} ll_gain={candidate['target5_logloss_gain']:+.8f} "
                f"R={gains['reverse_only']:+.8e} O={gains['overlap']:+.8e} "
                f"B={gains['bigmiss']:+.8e} cold_diff="
                f"{candidate['cold_fallback_max_abs_difference']:.1e}"
            )
    lines.extend(["", "Gate summary"])
    for key, summary in summaries.items():
        lines.append(
            f"  K={key}: worst_ll={summary['worst_logloss_gain']:+.8f} "
            f"mean_ll={summary['mean_logloss_gain']:+.8f} "
            f"min_coverage={summary['minimum_coverage']:.3%} PASS={summary['gate_pass']}"
        )
    lines.extend(
        [
            f"SELECTED_DIAGNOSTIC_K={selected_key}",
            f"FINAL_GATE={'PASS' if overall_pass else 'FAIL'}",
            f"elapsed={elapsed:.1f}s",
        ]
    )
    TEXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-7:]), flush=True)


if __name__ == "__main__":
    main()
