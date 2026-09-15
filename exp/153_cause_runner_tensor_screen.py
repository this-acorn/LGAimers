# -*- coding: utf-8 -*-
"""Strict-past target5 screen for a pitcher x runner-call tensor.

Official train rows and the locally reconstructed five-way target are the only
data used.  For validation year Y, every count table is fit on season < Y.
This script reads no test data, model package, third-party source, or LB value
and cannot build a model/ZIP/submission.

Parent: pitcher-only five-class EB distribution (K=220 toward global prior).
Child: pitcher x runner_call_state x count_group x batter_hand distribution,
       EB-shrunk toward that pitcher's parent over K in {110, 220, 440}.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train.csv"
OUT_JSON = ROOT / "lab" / "153_cause_runner_tensor_screen.json"
OUT_TXT = ROOT / "lab" / "153_cause_runner_tensor_screen.txt"

VALIDATION_YEARS = (2022, 2023, 2024)
N_CLASSES = 5
CLASS_NAMES = ("success", "middle", "reverse", "overlap", "bigmiss")
PARENT_K = 220.0
CHILD_K_GRID = (110.0, 220.0, 440.0)
EPS = 1e-15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore_target5(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover the established target5 contract without model/source imports."""
    ordered = frame.sort_values(
        ["pitcher_id", "asof_pitcher_n"], kind="mergesort"
    ).reset_index(names="_original_index")
    pitcher = pd.to_numeric(ordered["pitcher_id"], errors="raise").to_numpy(np.int64)
    count = pd.to_numeric(
        ordered["asof_pitcher_n"], errors="coerce"
    ).fillna(0).to_numpy(np.float64)
    consecutive = (pitcher[1:] == pitcher[:-1]) & (
        np.abs(count[1:] - count[:-1] - 1.0) < 1e-6
    )

    for name, column in (
        ("_middle", "asof_pitcher_middle_rate"),
        ("_reverse", "asof_pitcher_reverse_rate"),
    ):
        rate = pd.to_numeric(ordered[column], errors="coerce").fillna(0).to_numpy(
            np.float64
        )
        cumulative = rate * count
        recovered = np.full(len(ordered), np.nan, dtype=np.float64)
        difference = np.round(cumulative[1:] - cumulative[:-1])
        recovered[:-1] = np.where(consecutive, difference, np.nan)
        ordered[name] = np.where(
            (recovered == 0.0) | (recovered == 1.0), recovered, np.nan
        )

    restored = ordered.set_index("_original_index").sort_index()
    frame = frame.copy()
    frame["_middle"] = restored["_middle"].to_numpy()
    frame["_reverse"] = restored["_reverse"].to_numpy()

    success = pd.to_numeric(frame["control_success"], errors="raise").to_numpy(
        np.int8
    )
    known = frame["_middle"].notna().to_numpy() & frame["_reverse"].notna().to_numpy()
    middle = frame["_middle"].to_numpy() == 1.0
    reverse = frame["_reverse"].to_numpy() == 1.0
    target = np.full(len(frame), -1, dtype=np.int8)
    target[known & (success == 1)] = 0
    target[known & (success == 0) & middle & ~reverse] = 1
    target[known & (success == 0) & ~middle & reverse] = 2
    target[known & (success == 0) & middle & reverse] = 3
    target[known & (success == 0) & ~middle & ~reverse] = 4
    frame["_target5"] = target

    labeled = target >= 0
    if set(np.unique(target[labeled]).tolist()) != set(range(N_CLASSES)):
        raise AssertionError("restored target5 does not contain exactly five classes")
    if not np.array_equal(target[labeled] == 0, success[labeled] == 1):
        raise AssertionError("target5 success class contract failed")
    if not np.all(success[labeled & (target > 0)] == 0):
        raise AssertionError("target5 failure subclass contains success rows")
    return frame


def add_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    on1 = pd.to_numeric(result["runner_on_1b"], errors="raise").to_numpy(np.int8)
    on2 = pd.to_numeric(result["runner_on_2b"], errors="raise").to_numpy(np.int8)
    on3 = pd.to_numeric(result["runner_on_3b"], errors="raise").to_numpy(np.int8)
    runner_n = pd.to_numeric(result["num_runners_on"], errors="raise").to_numpy(
        np.int8
    )
    if not all(np.isin(value, [0, 1]).all() for value in (on1, on2, on3)):
        raise ValueError("runner flags are outside {0,1}")
    if not np.array_equal(on1 + on2 + on3, runner_n):
        raise AssertionError("runner flags do not exactly reconstruct num_runners_on")
    scoring = (on2 == 1) | (on3 == 1)
    # 0 none; 1 first-base steal threat only; 2 scoring position only; 3 both.
    result["_runner_call_state"] = (on1 == 1).astype(np.int8) + 2 * scoring.astype(
        np.int8
    )

    balls = pd.to_numeric(result["balls_before"], errors="raise").to_numpy(np.int8)
    strikes = pd.to_numeric(result["strikes_before"], errors="raise").to_numpy(
        np.int8
    )
    if not np.isin(balls, [0, 1, 2, 3]).all() or not np.isin(
        strikes, [0, 1, 2]
    ).all():
        raise ValueError("illegal pre-pitch count")
    # 0 batter ahead, 1 even, 2 pitcher ahead; established target5 grouping.
    result["_count_group"] = np.where(
        strikes > balls, 2, np.where(balls > strikes, 0, 1)
    ).astype(np.int8)

    hand = pd.to_numeric(result["batter_hand"], errors="raise")
    if hand.isna().any() or not hand.isin([1, 2]).all():
        raise ValueError("batter_hand outside {1,2}")
    result["batter_hand"] = hand.astype(np.int8)
    result["pitcher_id"] = pd.to_numeric(
        result["pitcher_id"], errors="raise"
    ).astype(np.int64)
    return result


def count_table(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    table = (
        frame.groupby(keys + ["_target5"], sort=False, observed=True)
        .size()
        .unstack("_target5", fill_value=0)
        .reindex(columns=range(N_CLASSES), fill_value=0)
    )
    table.columns = [f"c{index}" for index in range(N_CLASSES)]
    table["n"] = table.sum(axis=1)
    return table


def lookup_matrix(
    table: pd.DataFrame, rows: pd.DataFrame, keys: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    if len(keys) == 1:
        index: pd.Index | pd.MultiIndex = pd.Index(rows[keys[0]], name=keys[0])
    else:
        index = pd.MultiIndex.from_frame(rows[keys])
    counts = (
        table[[f"c{value}" for value in range(N_CLASSES)]]
        .reindex(index)
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    n = table["n"].reindex(index).fillna(0.0).to_numpy(np.float64)
    return counts, n


def multiclass_metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if probability.shape != (len(target), N_CLASSES):
        raise ValueError("probability/target shape mismatch")
    if not np.isfinite(probability).all() or np.max(
        np.abs(probability.sum(axis=1) - 1.0)
    ) > 1e-10:
        raise AssertionError("invalid multiclass probabilities")
    truth = np.eye(N_CLASSES, dtype=np.float64)[target]
    multi_logloss = float(-np.mean(np.log(np.clip(probability[np.arange(len(target)), target], EPS, 1.0))))
    multi_brier = float(np.mean(np.square(probability - truth).sum(axis=1)))
    per_class: dict[str, Any] = {}
    for class_index, name in enumerate(CLASS_NAMES):
        y = (target == class_index).astype(np.float64)
        p = probability[:, class_index]
        per_class[name] = {
            "prevalence": float(y.mean()),
            "prediction_mean": float(p.mean()),
            "ovr_brier": float(np.mean(np.square(p - y))),
            "ovr_logloss": float(
                -np.mean(y * np.log(np.clip(p, EPS, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - p, EPS, 1.0)))
            ),
        }
    return {
        "multiclass_logloss": multi_logloss,
        "multiclass_brier": multi_brier,
        "per_class": per_class,
    }


def compare_metrics(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    # Positive gain always means the child is better (lower loss).
    per_class: dict[str, Any] = {}
    for name in CLASS_NAMES:
        p = parent["per_class"][name]
        c = child["per_class"][name]
        brier_gain = p["ovr_brier"] - c["ovr_brier"]
        logloss_gain = p["ovr_logloss"] - c["ovr_logloss"]
        per_class[name] = {
            "ovr_brier_gain": float(brier_gain),
            "ovr_logloss_gain": float(logloss_gain),
            "prediction_mean_shift": float(c["prediction_mean"] - p["prediction_mean"]),
            "direction": (
                "BETTER_BOTH"
                if brier_gain > 0.0 and logloss_gain > 0.0
                else "WORSE_BOTH"
                if brier_gain < 0.0 and logloss_gain < 0.0
                else "MIXED"
            ),
        }
    return {
        "multiclass_logloss_gain": float(
            parent["multiclass_logloss"] - child["multiclass_logloss"]
        ),
        "multiclass_brier_gain": float(
            parent["multiclass_brier"] - child["multiclass_brier"]
        ),
        "per_class": per_class,
    }


def evaluate_fold(history: pd.DataFrame, validation: pd.DataFrame) -> dict[str, Any]:
    parent_keys = ["pitcher_id"]
    child_keys = [
        "pitcher_id",
        "_runner_call_state",
        "_count_group",
        "batter_hand",
    ]
    global_counts = np.bincount(
        history["_target5"].to_numpy(np.int8), minlength=N_CLASSES
    ).astype(np.float64)
    global_prior = global_counts / global_counts.sum()
    parent_table = count_table(history, parent_keys)
    child_table = count_table(history, child_keys)
    parent_counts, parent_n = lookup_matrix(parent_table, validation, parent_keys)
    child_counts, child_n = lookup_matrix(child_table, validation, child_keys)
    parent_probability = (
        parent_counts + PARENT_K * global_prior[None, :]
    ) / (parent_n[:, None] + PARENT_K)
    target = validation["_target5"].to_numpy(np.int8)
    parent_metrics = multiclass_metrics(parent_probability, target)

    children: dict[str, Any] = {}
    for child_k in CHILD_K_GRID:
        child_probability = (
            child_counts + child_k * parent_probability
        ) / (child_n[:, None] + child_k)
        child_metrics = multiclass_metrics(child_probability, target)
        children[f"k{int(child_k)}"] = {
            "metrics": child_metrics,
            "gain_vs_pitcher_parent": compare_metrics(parent_metrics, child_metrics),
        }

    return {
        "history_rows": int(len(history)),
        "validation_rows": int(len(validation)),
        "global_class_prior": {
            CLASS_NAMES[index]: float(global_prior[index])
            for index in range(N_CLASSES)
        },
        "coverage": {
            "seen_pitcher_rows": int(np.count_nonzero(parent_n > 0)),
            "seen_pitcher_rate": float(np.mean(parent_n > 0)),
            "seen_child_rows": int(np.count_nonzero(child_n > 0)),
            "seen_child_rate": float(np.mean(child_n > 0)),
            "parent_cells": int(len(parent_table)),
            "child_cells": int(len(child_table)),
        },
        "pitcher_parent_metrics": parent_metrics,
        "children": children,
    }


def main() -> None:
    started = time.time()
    columns = [
        "season",
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_middle_rate",
        "asof_pitcher_reverse_rate",
        "control_success",
        "runner_on_1b",
        "runner_on_2b",
        "runner_on_3b",
        "num_runners_on",
        "balls_before",
        "strikes_before",
        "batter_hand",
    ]
    frame = pd.read_csv(
        TRAIN_PATH, encoding="utf-8-sig", usecols=columns, low_memory=False
    )
    frame = add_context(restore_target5(frame))
    labeled = frame.loc[frame["_target5"] >= 0].copy()
    if not labeled["season"].isin(range(2019, 2025)).all():
        raise ValueError("unexpected season domain")

    folds: dict[str, Any] = {}
    for year in VALIDATION_YEARS:
        history = labeled.loc[labeled["season"] < year]
        validation = labeled.loc[labeled["season"] == year]
        if len(history) == 0 or len(validation) == 0:
            raise ValueError(f"empty strict-past fold {year}")
        folds[str(year)] = evaluate_fold(history, validation)

    robustness: dict[str, Any] = {}
    for child_k in CHILD_K_GRID:
        key = f"k{int(child_k)}"
        logloss_gains = [
            folds[str(year)]["children"][key]["gain_vs_pitcher_parent"][
                "multiclass_logloss_gain"
            ]
            for year in VALIDATION_YEARS
        ]
        brier_gains = [
            folds[str(year)]["children"][key]["gain_vs_pitcher_parent"][
                "multiclass_brier_gain"
            ]
            for year in VALIDATION_YEARS
        ]
        robustness[key] = {
            "all_years_logloss_better": bool(all(value > 0.0 for value in logloss_gains)),
            "all_years_brier_better": bool(all(value > 0.0 for value in brier_gains)),
            "min_logloss_gain": float(min(logloss_gains)),
            "mean_logloss_gain": float(np.mean(logloss_gains)),
            "min_brier_gain": float(min(brier_gains)),
            "mean_brier_gain": float(np.mean(brier_gains)),
        }
        robustness[key]["pass_both_all_years"] = bool(
            robustness[key]["all_years_logloss_better"]
            and robustness[key]["all_years_brier_better"]
        )

    any_robust = bool(any(row["pass_both_all_years"] for row in robustness.values()))
    verdict = (
        "SEMANTIC_AXIS_STABLE_FOR_FURTHER_WORK"
        if any_robust
        else "SEMANTIC_AXIS_UNSTABLE_NO_MODEL_WORK"
    )
    payload: dict[str, Any] = {
        "experiment": 153,
        "description": "clean-room strict-past target5 pitcher-runner-call tensor screen",
        "status": verdict,
        "analysis_only": True,
        "reads_official_train_only": True,
        "reads_test": False,
        "reads_third_party_source": False,
        "builds_model_zip_or_submission": False,
        "uses_lb": False,
        "target5_contract": {
            "class_order": list(CLASS_NAMES),
            "labeled_rows": int(len(labeled)),
            "unavailable_rows": int(len(frame) - len(labeled)),
            "coverage": float(len(labeled) / len(frame)),
            "class_counts": {
                CLASS_NAMES[index]: int(np.sum(labeled["_target5"].to_numpy() == index))
                for index in range(N_CLASSES)
            },
        },
        "context_contract": {
            "runner_call_state": {
                "0": "none",
                "1": "first_base_steal_threat_only",
                "2": "scoring_position_only",
                "3": "both_first_base_and_scoring_position",
            },
            "count_group": {
                "0": "balls_gt_strikes",
                "1": "balls_eq_strikes",
                "2": "strikes_gt_balls",
            },
            "child_keys": [
                "pitcher_id",
                "runner_call_state",
                "count_group",
                "batter_hand",
            ],
        },
        "protocol": {
            "validation_years": list(VALIDATION_YEARS),
            "history_rule": "season < validation_year",
            "parent": "pitcher-only class distribution shrunk to strict-past global class prior",
            "parent_k": PARENT_K,
            "child": "pitcher x runner_call_state x count_group x batter_hand counts shrunk to row pitcher parent",
            "child_k_grid": list(CHILD_K_GRID),
            "grid_is_diagnostic_only": True,
        },
        "folds": folds,
        "robustness": robustness,
        "input_sha256": {"data/train.csv": sha256(TRAIN_PATH)},
        "elapsed_seconds": float(time.time() - started),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "=== exp/153 cause x runner tensor strict-past screen ===",
        "DIAGNOSTIC ONLY: official train + restored target5; no test/model/ZIP/LB/submission",
        f"target5 coverage={payload['target5_contract']['coverage']:.6f} "
        f"({len(labeled):,}/{len(frame):,})",
        f"parent K={PARENT_K:.0f}; child K grid={list(CHILD_K_GRID)}",
        "",
        "year  childK coverage(parent/child)  logloss_gain   brier_gain",
    ]
    for year in VALIDATION_YEARS:
        fold = folds[str(year)]
        coverage = fold["coverage"]
        for child_k in CHILD_K_GRID:
            gain = fold["children"][f"k{int(child_k)}"]["gain_vs_pitcher_parent"]
            lines.append(
                f"{year} {child_k:7.0f} "
                f"{coverage['seen_pitcher_rate']:.4f}/{coverage['seen_child_rate']:.4f} "
                f"{gain['multiclass_logloss_gain']:+.9f} "
                f"{gain['multiclass_brier_gain']:+.9f}"
            )

    lines.extend(["", "Per-cause OVR direction at fixed child K=220 (positive gain=better)"])
    for year in VALIDATION_YEARS:
        gain = folds[str(year)]["children"]["k220"]["gain_vs_pitcher_parent"]
        lines.append(f"[{year}]")
        for name in CLASS_NAMES:
            row = gain["per_class"][name]
            lines.append(
                f"  {name:<8} brier={row['ovr_brier_gain']:+.9f} "
                f"logloss={row['ovr_logloss_gain']:+.9f} "
                f"mean_shift={row['prediction_mean_shift']:+.7f} {row['direction']}"
            )
    lines.extend(["", "Robustness across 2022/2023/2024"])
    for child_k in CHILD_K_GRID:
        row = robustness[f"k{int(child_k)}"]
        lines.append(
            f"K={child_k:.0f}: pass={row['pass_both_all_years']} "
            f"min/mean logloss gain={row['min_logloss_gain']:+.9f}/{row['mean_logloss_gain']:+.9f} "
            f"min/mean brier gain={row['min_brier_gain']:+.9f}/{row['mean_brier_gain']:+.9f}"
        )
    lines.extend(
        [
            f"FINAL: {verdict}",
            "This screen does not select a deployable K and creates no prediction/model artifact.",
            f"elapsed={payload['elapsed_seconds']:.2f}s",
        ]
    )
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
