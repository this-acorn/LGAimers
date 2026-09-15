# -*- coding: utf-8 -*-
"""Forward-only cause x runner residual gate on the current blended OOF.

The semantic tables are learned from official train seasons strictly before
each validation year.  A small linear map is fit on 2022, selected only by its
2023 transfer, then refit on 2022+2023 and evaluated once on untouched 2024.

No test row, deployable model, ZIP, leaderboard value, or third-party source
is read or written by this diagnostic.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train.csv"
OUT_JSON = ROOT / "lab" / "155_cause_runner_currentblend_gate.json"
OUT_TXT = ROOT / "lab" / "155_cause_runner_currentblend_gate.txt"

EXP021_ROOT = (
    ROOT / "lab" / "115_mkis_exp021_work" / "artifacts" / "EXP-020"
    / "low_rank_pitcher_context_eb"
)
CAT5 = {
    2022: ROOT / "lab" / "104_cat5_y2022_probs_seed42.npy",
    2023: ROOT / "lab" / "122_y2023_seed42_base_final.npy",
    2024: ROOT / "lab" / "89_cat5_probs_seed42.npy",
}
V18 = {
    2022: ROOT / "lab" / "104_v18_effect_y2022.npy",
    2023: ROOT / "lab" / "104_v18_effect_y2023.npy",
    2024: ROOT / "lab" / "103_v18_effect_2024.npy",
}
EXP021 = {
    year: EXP021_ROOT / f"predictions_lowrank_s300_r6_{year}.npy"
    for year in (2022, 2023, 2024)
}
EXP021_TARGET = {
    year: EXP021_ROOT / f"targets_{year}.npy"
    for year in (2022, 2023, 2024)
}

EXP021_WEIGHT = 0.35655716947524263
CHAMPION_WEIGHT = 1.0 - EXP021_WEIGHT
PARENT_K = 220.0
CHILD_K = 220.0
SCORE_SCALE = 100000.0
AFFINE_CENTER = 0.49
AFFINE_SCALE = 1.06
AFFINE_SHIFT = -0.0066
V18_GAMMA = 0.30
RIDGES = (0.01, 0.1, 1.0, 10.0)
STRUCTURES = {
    "success": (0,),
    "reverse_bigmiss": (2, 4),
    "middle_reverse_bigmiss": (1, 2, 4),
    "no_overlap": (0, 1, 2, 4),
    "all5": (0, 1, 2, 3, 4),
}
CLASS_NAMES = ("success", "middle", "reverse", "overlap", "bigmiss")


def load_helpers():
    path = ROOT / "exp" / "153_cause_runner_tensor_screen.py"
    spec = importlib.util.spec_from_file_location("exp153_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load exp153 helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_probability(path: Path, rows: int) -> np.ndarray:
    value = np.load(path, allow_pickle=False).astype(np.float64)
    if value.ndim == 2:
        if value.shape[1] != 5:
            raise ValueError(f"bad probability shape {path}: {value.shape}")
        value = value[:, 0]
    if value.shape != (rows,) or not np.isfinite(value).all():
        raise ValueError(f"bad vector {path}: {value.shape}")
    return value


def load_vector(path: Path, rows: int) -> np.ndarray:
    value = np.load(path, allow_pickle=False).astype(np.float64)
    if value.shape != (rows,) or not np.isfinite(value).all():
        raise ValueError(f"bad vector {path}: {value.shape}")
    return value


def score(probability: np.ndarray, target: np.ndarray) -> float:
    rate = float(target.mean())
    denominator = rate * (1.0 - rate)
    return float(
        SCORE_SCALE
        * (1.0 - float(np.mean(np.square(probability - target))) / denominator)
    )


def champion_probability(base: np.ndarray, effect: np.ndarray) -> np.ndarray:
    corrected = np.clip(base + V18_GAMMA * effect, 0.0, 1.0)
    return np.clip(
        AFFINE_CENTER + AFFINE_SCALE * (corrected - AFFINE_CENTER) + AFFINE_SHIFT,
        0.0,
        1.0,
    )


def semantic_delta(helper, history: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    parent_keys = ["pitcher_id"]
    child_keys = [
        "pitcher_id", "_runner_call_state", "_count_group", "batter_hand"
    ]
    global_counts = np.bincount(
        history["_target5"].to_numpy(np.int8), minlength=5
    ).astype(np.float64)
    global_prior = global_counts / global_counts.sum()
    parent_table = helper.count_table(history, parent_keys)
    child_table = helper.count_table(history, child_keys)
    parent_counts, parent_n = helper.lookup_matrix(parent_table, rows, parent_keys)
    child_counts, child_n = helper.lookup_matrix(child_table, rows, child_keys)
    parent = (parent_counts + PARENT_K * global_prior) / (
        parent_n[:, None] + PARENT_K
    )
    child = (child_counts + CHILD_K * parent) / (child_n[:, None] + CHILD_K)
    delta = child - parent
    # An unseen child must be an exact zero correction.
    delta[child_n == 0.0] = 0.0
    if not np.isfinite(delta).all() or np.max(np.abs(delta.sum(axis=1))) > 1e-10:
        raise AssertionError("invalid semantic delta")
    return delta


def ridge_fit(matrix: np.ndarray, residual: np.ndarray, ridge: float) -> np.ndarray:
    xtx = matrix.T @ matrix
    scale = float(np.trace(xtx) / matrix.shape[1])
    penalty = ridge * max(scale, 1e-12)
    coefficient = np.linalg.solve(
        xtx + penalty * np.eye(matrix.shape[1]), matrix.T @ residual
    )
    return np.clip(coefficient, -3.0, 3.0)


def metrics(
    baseline: np.ndarray,
    effect: np.ndarray,
    target: np.ndarray,
    game_type: np.ndarray,
) -> dict[str, float]:
    candidate = np.clip(baseline + effect, 0.0, 1.0)
    equal_mean = np.clip(
        candidate + float(baseline.mean() - candidate.mean()), 0.0, 1.0
    )
    base_score = score(baseline, target)
    raw_gain = score(candidate, target) - base_score
    denominator = float(target.mean() * (1.0 - target.mean()))
    row_gain = (
        SCORE_SCALE
        * (np.square(baseline - target) - np.square(candidate - target))
        / (denominator * len(target))
    )
    is_f = game_type == "F"
    result = {
        "raw_gain": float(raw_gain),
        "equal_mean_shape_gain": float(score(equal_mean, target) - base_score),
        "F_contribution": float(row_gain[is_f].sum()),
        "R_contribution": float(row_gain[~is_f].sum()),
        "mean_shift": float(candidate.mean() - baseline.mean()),
        "effect_std": float(effect.std()),
        "effect_mean_abs": float(np.abs(effect).mean()),
    }
    if abs(raw_gain - result["F_contribution"] - result["R_contribution"]) > 1e-7:
        raise AssertionError("F/R contribution mismatch")
    return result


def scalar_oracle(
    baseline: np.ndarray, direction: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    denominator = float(direction @ direction)
    beta = float(((target - baseline) @ direction) / denominator) if denominator else 0.0
    gain = score(np.clip(baseline + beta * direction, 0.0, 1.0), target) - score(
        baseline, target
    )
    return {"beta": beta, "gain": float(gain)}


def main() -> None:
    started = time.time()
    helper = load_helpers()
    columns = [
        "season", "pitcher_id", "asof_pitcher_n",
        "asof_pitcher_middle_rate", "asof_pitcher_reverse_rate",
        "control_success", "runner_on_1b", "runner_on_2b", "runner_on_3b",
        "num_runners_on", "balls_before", "strikes_before", "batter_hand",
        "game_type",
    ]
    frame = pd.read_csv(TRAIN, encoding="utf-8-sig", usecols=columns, low_memory=False)
    frame = helper.add_context(helper.restore_target5(frame))
    labeled = frame.loc[frame["_target5"] >= 0].copy()
    rows = {
        year: frame.loc[frame["season"].eq(year)].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    targets = {
        year: rows[year]["control_success"].to_numpy(np.float64)
        for year in rows
    }

    baseline: dict[int, np.ndarray] = {}
    deltas: dict[int, np.ndarray] = {}
    alignment: dict[str, object] = {}
    for year in (2022, 2023, 2024):
        n = len(rows[year])
        saved_target = load_vector(EXP021_TARGET[year], n)
        if not np.array_equal(saved_target, targets[year]):
            raise AssertionError(f"EXP021 target/order mismatch {year}")
        cat = load_probability(CAT5[year], n)
        v18 = load_vector(V18[year], n)
        external = load_probability(EXP021[year], n)
        champion = champion_probability(cat, v18)
        baseline[year] = CHAMPION_WEIGHT * champion + EXP021_WEIGHT * external
        deltas[year] = semantic_delta(
            helper, labeled.loc[labeled["season"] < year], rows[year]
        )
        alignment[str(year)] = {
            "rows": n,
            "target_exact": True,
            "baseline_score": score(baseline[year], targets[year]),
            "target_rate": float(targets[year].mean()),
        }

    discovery: list[dict[str, object]] = []
    for structure, columns_used in STRUCTURES.items():
        x22 = deltas[2022][:, columns_used]
        x23 = deltas[2023][:, columns_used]
        for ridge in RIDGES:
            coefficient = ridge_fit(x22, targets[2022] - baseline[2022], ridge)
            result = metrics(
                baseline[2023], x23 @ coefficient, targets[2023],
                rows[2023]["game_type"].astype(str).to_numpy(),
            )
            discovery.append({
                "structure": structure,
                "columns": [CLASS_NAMES[index] for index in columns_used],
                "ridge": ridge,
                "coefficient_fit_2022": coefficient.tolist(),
                "transfer_2023": result,
            })

    eligible = [
        row for row in discovery
        if row["transfer_2023"]["raw_gain"] > 0.0
        and row["transfer_2023"]["equal_mean_shape_gain"] > 0.0
        and row["transfer_2023"]["F_contribution"] >= 0.0
        and row["transfer_2023"]["R_contribution"] >= 0.0
    ]
    selected = max(
        eligible or discovery,
        key=lambda row: (
            min(row["transfer_2023"]["F_contribution"], row["transfer_2023"]["R_contribution"]),
            row["transfer_2023"]["raw_gain"],
        ),
    )
    selected_columns = STRUCTURES[str(selected["structure"])]
    x_fit = np.vstack([
        deltas[2022][:, selected_columns], deltas[2023][:, selected_columns]
    ])
    residual_fit = np.concatenate([
        targets[2022] - baseline[2022], targets[2023] - baseline[2023]
    ])
    coefficient = ridge_fit(x_fit, residual_fit, float(selected["ridge"]))
    effect24 = deltas[2024][:, selected_columns] @ coefficient
    confirmation = metrics(
        baseline[2024], effect24, targets[2024],
        rows[2024]["game_type"].astype(str).to_numpy(),
    )
    oracle = scalar_oracle(baseline[2024], effect24, targets[2024])
    old = np.asarray(selected["coefficient_fit_2022"], dtype=np.float64)
    stability = {
        "coefficient_refit_2022_2023": coefficient.tolist(),
        "cosine": float(
            (old @ coefficient) / (np.linalg.norm(old) * np.linalg.norm(coefficient))
        ) if np.linalg.norm(old) and np.linalg.norm(coefficient) else 0.0,
        "same_sign_fraction": float(np.mean(np.sign(old) == np.sign(coefficient))),
    }
    passed = bool(
        eligible
        and confirmation["raw_gain"] >= 10.0
        and confirmation["equal_mean_shape_gain"] >= 8.0
        and confirmation["F_contribution"] >= 0.0
        and confirmation["R_contribution"] >= 0.0
        and stability["cosine"] > 0.5
    )
    status = "PASS_FOR_BUILD" if passed else "FAIL_NO_SUBMISSION"
    payload = {
        "experiment": 155,
        "status": status,
        "analysis_only": True,
        "reads_test": False,
        "uses_lb": False,
        "protocol": {
            "semantic_history": "season < validation year",
            "parent_k": PARENT_K,
            "child_k": CHILD_K,
            "fit": "2022 residual -> select by 2023 transfer; refit 2022+2023 -> untouched 2024",
            "selection_gate": "2023 raw/shape positive and F/R nonnegative",
            "confirmation_gate": "2024 raw>=10, shape>=8, F/R nonnegative, coefficient cosine>0.5",
        },
        "alignment": alignment,
        "discovery": discovery,
        "selected": selected,
        "stability": stability,
        "confirmation_2024": confirmation,
        "same_fold_2024_scalar_oracle_diagnostic_only": oracle,
        "elapsed_seconds": float(time.time() - started),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "=== exp155 cause-runner current-blend forward gate ===",
        "DIAGNOSTIC ONLY: official train + saved OOF; no test/model/ZIP/LB",
        f"selected on 2023: {selected['structure']} ridge={selected['ridge']}",
        f"2023 transfer: {selected['transfer_2023']}",
        f"refit coefficients: {coefficient.tolist()}",
        f"coefficient stability: {stability}",
        f"untouched 2024: {confirmation}",
        f"2024 scalar oracle (diagnostic only): {oracle}",
        f"FINAL: {status}",
        f"elapsed={payload['elapsed_seconds']:.2f}s",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
