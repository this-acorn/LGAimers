# -*- coding: utf-8 -*-
"""Diagnostic-only pitcher x any-runner residual gate on the current blend.

This clean-room experiment asks one narrow label-semantic question:

    After the current CAT5+V18-affine / EXP021 blend, does a pitcher's
    residual change when any base is occupied?

Only official train rows and already-saved OOF arrays are read.  The 2023
residual is fit once and the frozen hierarchy is mapped to the untouched 2024
fold.  No test rows, third-party Python source, model package, ZIP, or LB value
is read or produced.

The fixed hierarchy is, separately within game_type:

    zero-centered game_type residual
      -> EB pitcher parent
        -> EB pitcher x any_runner child

The emitted effect is child - parent, not the parent itself.  It therefore
tests only the runner interaction left after the pitcher parent.  K=220 and
gamma=0.25 are fixed before 2024 is evaluated.  K=110/440 are descriptive
sensitivity endpoints and are never used for candidate selection.

Outputs:
  lab/152_runner_context_currentblend_gate.txt
  lab/152_runner_context_currentblend_gate.json
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
OUT_TXT = ROOT / "lab" / "152_runner_context_currentblend_gate.txt"
OUT_JSON = ROOT / "lab" / "152_runner_context_currentblend_gate.json"

# Saved OOF arrays only.  No Python source beneath these work directories is
# imported or opened by this experiment.
EXP021_OOF_ROOT = (
    ROOT
    / "lab"
    / "115_mkis_exp021_work"
    / "artifacts"
    / "EXP-020"
    / "low_rank_pitcher_context_eb"
)

CAT5_BASE = {
    2023: ROOT / "lab" / "122_y2023_seed42_base_final.npy",
    2024: ROOT / "lab" / "89_cat5_probs_seed42.npy",
}
V18_EFFECT = {
    2023: ROOT / "lab" / "104_v18_effect_y2023.npy",
    2024: ROOT / "lab" / "103_v18_effect_2024.npy",
}
EXP021_OOF = {
    year: EXP021_OOF_ROOT / f"predictions_lowrank_s300_r6_{year}.npy"
    for year in (2023, 2024)
}
EXP021_TARGET = {
    year: EXP021_OOF_ROOT / f"targets_{year}.npy" for year in (2023, 2024)
}

SOURCE_YEAR = 2023
VALIDATION_YEAR = 2024
PRIMARY_SMOOTHING = 220.0
SENSITIVITY_SMOOTHING = (110.0, 440.0)
FIXED_GAMMA = 0.25
EXP021_WEIGHT = 0.35655716947524263
CHAMPION_WEIGHT = 1.0 - EXP021_WEIGHT
V18_GAMMA = 0.30
AFFINE_CENTER = 0.49
AFFINE_SCALE = 1.06
AFFINE_SHIFT = -0.0066
SCORE_SCALE = 100000.0

# A pass only authorizes discussion/further temporal work.  It never authorizes
# a submission.  Thresholds are fixed, coarse, and not optimized on 2024.
PROMOTION_GATE = {
    "raw_gain_min": 8.0,
    "equal_mean_shape_gain_min": 5.0,
    "F_contribution_min": 0.0,
    "R_contribution_min": 0.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_array(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    value = np.load(path, allow_pickle=False).astype(np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite values in {label}")
    return value


def probability_vector(path: Path, label: str, expected_rows: int) -> np.ndarray:
    value = load_array(path, label)
    if value.ndim == 2:
        if value.shape[1] != 5:
            raise ValueError(f"unexpected class shape for {label}: {value.shape}")
        value = value[:, 0]
    if value.shape != (expected_rows,):
        raise ValueError(
            f"row mismatch for {label}: expected {(expected_rows,)}, found {value.shape}"
        )
    if not ((value >= 0.0).all() and (value <= 1.0).all()):
        raise ValueError(f"out-of-range probability in {label}")
    return value


def vector(path: Path, label: str, expected_rows: int) -> np.ndarray:
    value = load_array(path, label)
    if value.shape != (expected_rows,):
        raise ValueError(
            f"row mismatch for {label}: expected {(expected_rows,)}, found {value.shape}"
        )
    return value


def score(probability: np.ndarray, target: np.ndarray) -> float:
    rate = float(target.mean())
    denominator = rate * (1.0 - rate)
    if denominator <= 0.0:
        raise ValueError("degenerate target prevalence")
    return float(
        SCORE_SCALE
        * (1.0 - float(np.mean(np.square(probability - target))) / denominator)
    )


def champion_probability(base: np.ndarray, effect: np.ndarray) -> np.ndarray:
    corrected = np.clip(base + V18_GAMMA * effect, 0.0, 1.0)
    return np.clip(
        AFFINE_CENTER
        + AFFINE_SCALE * (corrected - AFFINE_CENTER)
        + AFFINE_SHIFT,
        0.0,
        1.0,
    )


def normalize_domains(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    games = result["game_type"].astype(str)
    if not games.isin(["F", "R"]).all():
        bad = sorted(games.loc[~games.isin(["F", "R"])].unique().tolist())
        raise ValueError(f"unexpected game_type values: {bad}")
    pitcher = pd.to_numeric(result["pitcher_id"], errors="raise")
    if pitcher.isna().any() or not np.equal(pitcher, np.floor(pitcher)).all():
        raise ValueError("pitcher_id is missing or non-integral")
    runners = pd.to_numeric(result["num_runners_on"], errors="raise")
    if runners.isna().any() or not runners.isin([0, 1, 2, 3]).all():
        bad = sorted(runners.loc[~runners.isin([0, 1, 2, 3])].unique().tolist())
        raise ValueError(f"unexpected num_runners_on values: {bad}")
    result["game_type"] = games
    result["pitcher_id"] = pitcher.astype(np.int64)
    result["_any_runner"] = (runners.to_numpy(np.int8) > 0).astype(np.int8)
    return result


def grouped_sum_count(
    frame: pd.DataFrame, keys: list[str], value_column: str
) -> pd.DataFrame:
    return frame.groupby(keys, sort=False, observed=True)[value_column].agg(
        residual_sum="sum", n="size"
    )


def lookup(
    table: pd.DataFrame,
    rows: pd.DataFrame,
    keys: list[str],
    column: str,
) -> np.ndarray:
    if len(keys) == 1:
        index: pd.Index | pd.MultiIndex = pd.Index(rows[keys[0]], name=keys[0])
    else:
        index = pd.MultiIndex.from_frame(rows[keys])
    return table[column].reindex(index).fillna(0.0).to_numpy(np.float64)


def fit_hierarchy(
    source_rows: pd.DataFrame,
    source_target: np.ndarray,
    source_baseline: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(source_rows) != len(source_target) or len(source_rows) != len(source_baseline):
        raise ValueError("source row/vector length mismatch")

    work = source_rows[["game_type", "pitcher_id", "_any_runner"]].copy()
    raw_residual = source_target - source_baseline
    game_means = (
        pd.Series(raw_residual)
        .groupby(work["game_type"], sort=False, observed=True)
        .transform("mean")
        .to_numpy(np.float64)
    )
    centered = raw_residual - game_means
    work["_residual"] = centered

    centered_means = {
        game: float(centered[work["game_type"].to_numpy() == game].mean())
        for game in ("F", "R")
    }
    if any(abs(value) > 1e-12 for value in centered_means.values()):
        raise AssertionError(f"within-game_type residual centering failed: {centered_means}")

    parent_keys = ["game_type", "pitcher_id"]
    child_keys = ["game_type", "pitcher_id", "_any_runner"]
    parent = grouped_sum_count(work, parent_keys, "_residual")
    child = grouped_sum_count(work, child_keys, "_residual")
    if int(parent["n"].sum()) != len(work) or int(child["n"].sum()) != len(work):
        raise AssertionError("source aggregation row mismatch")

    model: dict[str, Any] = {
        "parent": parent,
        "child": child,
        "parent_keys": parent_keys,
        "child_keys": child_keys,
    }
    diagnostics: dict[str, Any] = {
        "source_rows": int(len(work)),
        "raw_residual_mean": float(raw_residual.mean()),
        "raw_residual_mean_by_game_type": {
            game: float(raw_residual[work["game_type"].to_numpy() == game].mean())
            for game in ("F", "R")
        },
        "centered_residual_mean_by_game_type": centered_means,
        "parent_cells": int(len(parent)),
        "child_cells": int(len(child)),
        "source_pitchers": int(work["pitcher_id"].nunique()),
        "parent_n_median": float(parent["n"].median()),
        "child_n_median": float(child["n"].median()),
    }
    return model, diagnostics


def map_runner_differential(
    model: dict[str, Any], validation_rows: pd.DataFrame, smoothing: float
) -> tuple[np.ndarray, dict[str, Any]]:
    if smoothing <= 0.0:
        raise ValueError("smoothing must be positive")
    parent = model["parent"]
    child = model["child"]
    parent_keys = model["parent_keys"]
    child_keys = model["child_keys"]

    parent_n = lookup(parent, validation_rows, parent_keys, "n")
    parent_sum = lookup(parent, validation_rows, parent_keys, "residual_sum")
    parent_rate = parent_sum / (parent_n + smoothing)

    child_n = lookup(child, validation_rows, child_keys, "n")
    child_sum = lookup(child, validation_rows, child_keys, "residual_sum")
    child_rate = (child_sum + smoothing * parent_rate) / (child_n + smoothing)

    # The pitcher parent is used as the conditional reference, but is not added
    # to the current blend.  This is the incremental runner-only interaction.
    effect = child_rate - parent_rate
    if not np.isfinite(effect).all():
        raise AssertionError("mapped runner differential contains non-finite values")

    games = validation_rows["game_type"].to_numpy()
    diagnostics: dict[str, Any] = {
        "smoothing": float(smoothing),
        "validation_rows": int(len(validation_rows)),
        "seen_parent_rows": int(np.count_nonzero(parent_n > 0)),
        "seen_parent_rate": float(np.mean(parent_n > 0)),
        "seen_child_rows": int(np.count_nonzero(child_n > 0)),
        "seen_child_rate": float(np.mean(child_n > 0)),
        "seen_child_rate_by_game_type": {
            game: float(np.mean(child_n[games == game] > 0)) for game in ("F", "R")
        },
        "effect_mean": float(effect.mean()),
        "effect_std": float(effect.std()),
        "effect_mean_abs": float(np.abs(effect).mean()),
        "effect_min": float(effect.min()),
        "effect_max": float(effect.max()),
        "effect_mean_by_game_type": {
            game: float(effect[games == game].mean()) for game in ("F", "R")
        },
    }
    return effect, diagnostics


def similarity(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.shape != right.shape:
        raise ValueError("similarity shape mismatch")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    left_centered_norm = float(np.linalg.norm(left_centered))
    right_centered_norm = float(np.linalg.norm(right_centered))
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if min(left_centered_norm, right_centered_norm, left_norm, right_norm) == 0.0:
        raise ValueError("zero vector in similarity diagnostic")
    return {
        "pearson_correlation": float(
            np.dot(left_centered, right_centered)
            / (left_centered_norm * right_centered_norm)
        ),
        "cosine_similarity": float(np.dot(left, right) / (left_norm * right_norm)),
    }


def evaluate_fixed_gamma(
    baseline: np.ndarray,
    effect: np.ndarray,
    target: np.ndarray,
    game_type: np.ndarray,
) -> dict[str, Any]:
    candidate = np.clip(baseline + FIXED_GAMMA * effect, 0.0, 1.0)
    # Deliberately do not re-clip after exact mean restoration: Brier score is
    # defined for real-valued forecasts and this isolates shape from prevalence.
    equal_mean = candidate - candidate.mean() + baseline.mean()
    base_score = score(baseline, target)
    candidate_score = score(candidate, target)
    denominator = float(target.mean() * (1.0 - target.mean()))
    row_gain = (
        SCORE_SCALE
        * (np.square(baseline - target) - np.square(candidate - target))
        / (denominator * len(target))
    )
    is_f = game_type == "F"
    result: dict[str, Any] = {
        "gamma": FIXED_GAMMA,
        "base_score": base_score,
        "candidate_score": candidate_score,
        "raw_gain": candidate_score - base_score,
        "equal_mean_shape_gain": score(equal_mean, target) - base_score,
        "prediction_mean_shift": float(candidate.mean() - baseline.mean()),
        "equal_mean_restoration_error": float(equal_mean.mean() - baseline.mean()),
        "F_contribution": float(row_gain[is_f].sum()),
        "R_contribution": float(row_gain[~is_f].sum()),
        "clip_low_rows": int(np.count_nonzero(baseline + FIXED_GAMMA * effect < 0.0)),
        "clip_high_rows": int(np.count_nonzero(baseline + FIXED_GAMMA * effect > 1.0)),
    }
    if abs(
        result["raw_gain"]
        - result["F_contribution"]
        - result["R_contribution"]
    ) > 1e-8:
        raise AssertionError("F/R contributions do not sum to global gain")
    return result


def same_fold_oracle(
    baseline: np.ndarray, effect: np.ndarray, target: np.ndarray
) -> dict[str, Any]:
    energy = float(np.dot(effect, effect))
    beta = float(np.dot(target - baseline, effect) / energy) if energy else 0.0
    candidate = np.clip(baseline + beta * effect, 0.0, 1.0)
    return {
        "diagnostic_only": True,
        "uses_validation_2024_labels": True,
        "must_not_be_used_for_deployment": True,
        "unclipped_quadratic_beta": beta,
        "score_gain_after_probability_clip": score(candidate, target)
        - score(baseline, target),
    }


def main() -> None:
    started = time.time()
    if abs(CHAMPION_WEIGHT + EXP021_WEIGHT - 1.0) > 1e-15:
        raise AssertionError("blend weights do not sum to one")

    columns = [
        "season",
        "game_type",
        "pitcher_id",
        "num_runners_on",
        "control_success",
    ]
    train = pd.read_csv(
        TRAIN_PATH, encoding="utf-8-sig", usecols=columns, low_memory=False
    )
    rows = {
        year: normalize_domains(
            train.loc[train["season"].eq(year)].reset_index(drop=True)
        )
        for year in (SOURCE_YEAR, VALIDATION_YEAR)
    }
    targets = {
        year: rows[year]["control_success"].to_numpy(np.float64) for year in rows
    }

    alignment: dict[str, Any] = {}
    for year in (SOURCE_YEAR, VALIDATION_YEAR):
        saved_target = vector(
            EXP021_TARGET[year], f"saved EXP021 target {year}", len(rows[year])
        )
        exact = bool(np.array_equal(saved_target, targets[year]))
        if not exact:
            # Stop rather than infer or repair any row order.
            raise AssertionError(f"official target/order mismatch for {year}")
        runner_count = rows[year]["_any_runner"].to_numpy()
        alignment[str(year)] = {
            "rows": int(len(rows[year])),
            "official_vs_saved_target_exact": exact,
            "target_rate": float(targets[year].mean()),
            "any_runner_rate": float(runner_count.mean()),
            "F_rows": int(rows[year]["game_type"].eq("F").sum()),
            "R_rows": int(rows[year]["game_type"].eq("R").sum()),
        }

    cat5 = {
        year: probability_vector(
            CAT5_BASE[year], f"corrected CAT5 seed42 {year}", len(rows[year])
        )
        for year in (SOURCE_YEAR, VALIDATION_YEAR)
    }
    v18 = {
        year: vector(V18_EFFECT[year], f"V18 effect {year}", len(rows[year]))
        for year in (SOURCE_YEAR, VALIDATION_YEAR)
    }
    champion = {
        year: champion_probability(cat5[year], v18[year])
        for year in (SOURCE_YEAR, VALIDATION_YEAR)
    }
    exp021 = {
        year: probability_vector(
            EXP021_OOF[year], f"saved EXP021 OOF {year}", len(rows[year])
        )
        for year in (SOURCE_YEAR, VALIDATION_YEAR)
    }
    current_blend = {
        year: CHAMPION_WEIGHT * champion[year] + EXP021_WEIGHT * exp021[year]
        for year in (SOURCE_YEAR, VALIDATION_YEAR)
    }
    for year, probability in current_blend.items():
        if not ((probability >= 0.0).all() and (probability <= 1.0).all()):
            raise AssertionError(f"current blend out of range for {year}")

    model, fit_diagnostics = fit_hierarchy(
        rows[SOURCE_YEAR],
        targets[SOURCE_YEAR],
        current_blend[SOURCE_YEAR],
    )

    all_smoothing = (PRIMARY_SMOOTHING, *SENSITIVITY_SMOOTHING)
    variants: dict[str, Any] = {}
    effects: dict[float, np.ndarray] = {}
    games_2024 = rows[VALIDATION_YEAR]["game_type"].to_numpy()
    endpoint_direction = exp021[VALIDATION_YEAR] - champion[VALIDATION_YEAR]
    for smoothing in all_smoothing:
        effect, mapping = map_runner_differential(
            model, rows[VALIDATION_YEAR], smoothing
        )
        effects[smoothing] = effect
        variants[f"k{int(smoothing)}"] = {
            "role": (
                "fixed_primary"
                if smoothing == PRIMARY_SMOOTHING
                else "descriptive_sensitivity_only"
            ),
            "mapping": mapping,
            "fixed_gamma_metrics": evaluate_fixed_gamma(
                current_blend[VALIDATION_YEAR],
                effect,
                targets[VALIDATION_YEAR],
                games_2024,
            ),
            "same_fold_oracle": same_fold_oracle(
                current_blend[VALIDATION_YEAR], effect, targets[VALIDATION_YEAR]
            ),
            "overlap": {
                "v18_raw_effect": similarity(effect, v18[VALIDATION_YEAR]),
                "exp021_endpoint_direction": similarity(effect, endpoint_direction),
            },
        }

    primary = variants[f"k{int(PRIMARY_SMOOTHING)}"]
    primary_metrics = primary["fixed_gamma_metrics"]
    checks = {
        "raw_gain": bool(primary_metrics["raw_gain"] >= PROMOTION_GATE["raw_gain_min"]),
        "equal_mean_shape_gain": bool(
            primary_metrics["equal_mean_shape_gain"]
            >= PROMOTION_GATE["equal_mean_shape_gain_min"]
        ),
        "F_nonnegative": bool(
            primary_metrics["F_contribution"]
            >= PROMOTION_GATE["F_contribution_min"]
        ),
        "R_nonnegative": bool(
            primary_metrics["R_contribution"]
            >= PROMOTION_GATE["R_contribution_min"]
        ),
    }
    gate_pass = bool(all(checks.values()))
    verdict = (
        "PASS_FOR_FURTHER_TEMPORAL_WORK"
        if gate_pass
        else "FAIL_RUNNER_AXIS_NO_SUBMISSION"
    )

    input_paths = [
        TRAIN_PATH,
        *CAT5_BASE.values(),
        *V18_EFFECT.values(),
        *EXP021_OOF.values(),
        *EXP021_TARGET.values(),
    ]
    payload: dict[str, Any] = {
        "experiment": 152,
        "description": "clean-room pitcher x any_runner residual differential on current blend",
        "status": verdict,
        "analysis_only": True,
        "reads_official_train": True,
        "reads_test": False,
        "reads_third_party_python_source": False,
        "builds_model_or_submission": False,
        "uses_lb": False,
        "protocol": {
            "source_year": SOURCE_YEAR,
            "validation_year": VALIDATION_YEAR,
            "baseline": "seed42 corrected CAT5+V18+affine blended with saved EXP021 OOF",
            "champion_weight": CHAMPION_WEIGHT,
            "exp021_weight": EXP021_WEIGHT,
            "source_residual": "y-current_blend, centered separately within F and R",
            "hierarchy": "game_type -> pitcher parent -> pitcher x any_runner child",
            "emitted_effect": "child EB residual - pitcher parent EB residual; parent is not added",
            "primary_smoothing": PRIMARY_SMOOTHING,
            "sensitivity_smoothing": list(SENSITIVITY_SMOOTHING),
            "fixed_gamma": FIXED_GAMMA,
            "unseen_child_effect": 0.0,
            "sensitivity_is_nonselective": True,
            "same_fold_oracle_is_not_deployable": True,
        },
        "alignment": alignment,
        "baseline_scores": {
            str(year): score(current_blend[year], targets[year])
            for year in (SOURCE_YEAR, VALIDATION_YEAR)
        },
        "fit": fit_diagnostics,
        "variants": variants,
        "primary_promotion_gate": {
            "thresholds": PROMOTION_GATE,
            "checks": checks,
            "pass": gate_pass,
        },
        "input_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in input_paths
        },
        "elapsed_seconds": float(time.time() - started),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "=== exp/152 pitcher x any_runner residual gate ===",
        "DIAGNOSTIC ONLY: official train + saved OOF; no test/model/ZIP/LB/submission",
        "Third-party Python source imported/read: NO (saved OOF arrays only)",
        f"temporal mapping: fit {SOURCE_YEAR} residual -> untouched {VALIDATION_YEAR}",
        "hierarchy: within game_type, EB pitcher parent -> EB pitcher x any_runner child",
        "emitted effect: child-parent only (pitcher parent itself is NOT added)",
        f"baseline weights: champion={CHAMPION_WEIGHT:.10f} EXP021={EXP021_WEIGHT:.10f}",
        f"baseline scores: {SOURCE_YEAR}={payload['baseline_scores'][str(SOURCE_YEAR)]:.6f} "
        f"{VALIDATION_YEAR}={payload['baseline_scores'][str(VALIDATION_YEAR)]:.6f}",
        "",
        " smoothing role                         coverage(parent/child) effect_std   raw     shape    F       R       oracle_beta oracle_gain corr_V18 corr_endpoint",
    ]
    for smoothing in all_smoothing:
        row = variants[f"k{int(smoothing)}"]
        mapping = row["mapping"]
        metric = row["fixed_gamma_metrics"]
        oracle = row["same_fold_oracle"]
        overlap = row["overlap"]
        lines.append(
            f" {smoothing:9.0f} {row['role']:<28} "
            f"{mapping['seen_parent_rate']:.4f}/{mapping['seen_child_rate']:.4f} "
            f"{mapping['effect_std']:.7f} "
            f"{metric['raw_gain']:+7.3f} {metric['equal_mean_shape_gain']:+7.3f} "
            f"{metric['F_contribution']:+7.3f} {metric['R_contribution']:+7.3f} "
            f"{oracle['unclipped_quadratic_beta']:+11.5f} "
            f"{oracle['score_gain_after_probability_clip']:+10.3f} "
            f"{overlap['v18_raw_effect']['pearson_correlation']:+8.4f} "
            f"{overlap['exp021_endpoint_direction']['pearson_correlation']:+13.4f}"
        )
    lines.extend(
        [
            "",
            f"PRIMARY fixed K={PRIMARY_SMOOTHING:.0f}, gamma={FIXED_GAMMA:.2f}: checks={checks}",
            f"FINAL: {verdict}",
            "K=110/440 and the 2024 oracle are descriptive only; neither may select a deployable setting.",
            "No test data was read and no model or submission artifact was created.",
            f"elapsed={payload['elapsed_seconds']:.2f}s",
        ]
    )
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
