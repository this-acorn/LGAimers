# -*- coding: utf-8 -*-
"""Diagnostic-only low-rank residual gate on the current blended OOF.

This is a clean-room implementation of one small statistical question:

    Can a strongly-shrunk pitcher x context matrix, truncated by SVD, explain
    next-season residual left by the current CAT5+CS+V18 / EXP021 blend?

Only official train rows and already-saved OOF prediction arrays are read.
No EXP021 Python source is imported or read.  No test row, deployment model,
submission package, or leaderboard value is produced.

The sole temporal diagnostic fits a 2023 residual matrix and maps the frozen
effect to 2024.  Gamma sweeps and the 2024 optimum are explicitly descriptive:
they use 2024 labels and must never become deployment hyperparameters.

Outputs:
  lab/126_lowrank_currentblend_gate.txt
  lab/126_lowrank_currentblend_gate.json
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train.csv"
OUT_TXT = ROOT / "lab" / "126_lowrank_currentblend_gate.txt"
OUT_JSON = ROOT / "lab" / "126_lowrank_currentblend_gate.json"

EXTERNAL_OOF_ROOT = (
    ROOT
    / "lab"
    / "115_mkis_exp021_work"
    / "artifacts"
    / "EXP-020"
    / "low_rank_pitcher_context_eb"
)
TEAM_OOF_ROOT = (
    ROOT
    / "lab"
    / "115_mkis_exp021_work"
    / "artifacts"
    / "EXP-019"
    / "team_eb_ensemble"
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
    year: EXTERNAL_OOF_ROOT / f"predictions_lowrank_s300_r6_{year}.npy"
    for year in (2023, 2024)
}
EXP021_TARGET = {
    year: EXTERNAL_OOF_ROOT / f"targets_{year}.npy"
    for year in (2023, 2024)
}

EXP021_WEIGHT = 0.35655716947524263
CHAMPION_WEIGHT = 1.0 - EXP021_WEIGHT
V18_GAMMA = 0.30
AFFINE_CENTER = 0.49
AFFINE_SCALE = 1.06
AFFINE_SHIFT = -0.0066

SOURCE_YEAR = 2023
VALIDATION_YEAR = 2024
SMOOTHING = 300.0
RANK = 6
DIAGNOSTIC_GAMMAS = (0.25, 0.50, 0.75, 1.00)
PRIMARY_GAMMA = 0.25
SCORE_SCALE = 100000.0

# This is a deliberately demanding submission-promotion gate.  It is not a
# tuning objective: both candidate structures and all values are fixed here.
GATE = {
    "raw_gain_min": 10.0,
    "equal_mean_shape_gain_min": 8.0,
    "F_contribution_min": 0.0,
    "R_contribution_min": 0.0,
    "abs_corr_exp021_lowrank_max": 0.50,
    "abs_corr_endpoint_direction_max": 0.25,
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


def context_codes(frame: pd.DataFrame, include_game_type: bool) -> tuple[np.ndarray, int]:
    balls = pd.to_numeric(frame["balls_before"], errors="raise").to_numpy(np.int8)
    strikes = pd.to_numeric(frame["strikes_before"], errors="raise").to_numpy(np.int8)
    hands = pd.to_numeric(frame["batter_hand"], errors="raise").to_numpy(np.int8)
    games = frame["game_type"].astype(str).to_numpy()

    if not np.isin(balls, np.arange(4)).all():
        raise ValueError("balls_before outside 0..3")
    if not np.isin(strikes, np.arange(3)).all():
        raise ValueError("strikes_before outside 0..2")
    if not np.isin(hands, np.array([1, 2], dtype=np.int8)).all():
        raise ValueError("batter_hand outside {1,2}")
    if not np.isin(games, np.array(["F", "R"])).all():
        raise ValueError("game_type outside {F,R}")

    # Twelve legal pre-pitch counts, crossed with two batter hands.
    count_position = balls.astype(np.int16) * 3 + strikes.astype(np.int16)
    code = count_position * 2 + (hands.astype(np.int16) - 1)
    if include_game_type:
        # F occupies 0..23 and R occupies 24..47.
        code = code + 24 * (games == "R").astype(np.int16)
        return code.astype(np.int16), 48
    return code.astype(np.int16), 24


def fit_source_matrix(
    rows: pd.DataFrame,
    target: np.ndarray,
    baseline: np.ndarray,
    include_game_type: bool,
) -> dict[str, object]:
    contexts, context_count = context_codes(rows, include_game_type)
    pitcher_values = pd.to_numeric(rows["pitcher_id"], errors="raise").to_numpy(
        np.int64
    )
    pitcher_ids = np.unique(pitcher_values)
    pitcher_codes = np.searchsorted(pitcher_ids, pitcher_values)
    if not np.array_equal(pitcher_ids[pitcher_codes], pitcher_values):
        raise AssertionError("pitcher factorization mismatch")

    residual = target - baseline
    raw_residual_mean = float(residual.mean())
    residual = residual - raw_residual_mean
    if abs(float(residual.mean())) > 1e-12:
        raise AssertionError("source residual centering failed")

    sums = np.zeros((len(pitcher_ids), context_count), dtype=np.float64)
    counts = np.zeros((len(pitcher_ids), context_count), dtype=np.int64)
    np.add.at(sums, (pitcher_codes, contexts), residual)
    np.add.at(counts, (pitcher_codes, contexts), 1)
    if int(counts.sum()) != len(rows):
        raise AssertionError("source aggregation row mismatch")

    # Clean-room EB shrinkage: n/(n+s) * cell mean = sum/(n+s).
    smoothed = sums / (counts.astype(np.float64) + SMOOTHING)
    left, singular_values, right = np.linalg.svd(smoothed, full_matrices=False)
    effective_rank = min(RANK, len(singular_values))
    reconstruction = (
        left[:, :effective_rank] * singular_values[:effective_rank]
    ) @ right[:effective_rank, :]
    total_energy = float(np.square(singular_values).sum())
    retained_energy = float(np.square(singular_values[:effective_rank]).sum())

    return {
        "pitcher_ids": pitcher_ids,
        "counts": counts,
        "reconstruction": reconstruction,
        "diagnostics": {
            "source_rows": int(len(rows)),
            "source_pitchers": int(len(pitcher_ids)),
            "context_count": int(context_count),
            "observed_cells": int(np.count_nonzero(counts)),
            "matrix_cells": int(counts.size),
            "observed_density": float(np.mean(counts > 0)),
            "raw_residual_mean_before_centering": raw_residual_mean,
            "centered_residual_mean": float(residual.mean()),
            "smoothed_effect_mean_abs": float(np.abs(smoothed).mean()),
            "singular_values": [float(value) for value in singular_values],
            "retained_energy_fraction": (
                retained_energy / total_energy if total_energy > 0.0 else 0.0
            ),
        },
    }


def map_effect(
    model: dict[str, object],
    rows: pd.DataFrame,
    include_game_type: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    contexts, expected_context_count = context_codes(rows, include_game_type)
    diagnostics = model["diagnostics"]
    if expected_context_count != diagnostics["context_count"]:
        raise AssertionError("context domain drift")

    pitcher_ids = np.asarray(model["pitcher_ids"], dtype=np.int64)
    reconstruction = np.asarray(model["reconstruction"], dtype=np.float64)
    validation_pitchers = pd.to_numeric(
        rows["pitcher_id"], errors="raise"
    ).to_numpy(np.int64)
    positions = np.searchsorted(pitcher_ids, validation_pitchers)
    safe_positions = np.minimum(positions, len(pitcher_ids) - 1)
    seen = (positions < len(pitcher_ids)) & (
        pitcher_ids[safe_positions] == validation_pitchers
    )
    effect = np.zeros(len(rows), dtype=np.float64)
    effect[seen] = reconstruction[safe_positions[seen], contexts[seen]]
    if not np.isfinite(effect).all():
        raise AssertionError("mapped effect contains non-finite values")
    return effect, {
        "validation_rows": int(len(rows)),
        "seen_pitcher_rows": int(seen.sum()),
        "seen_pitcher_rate": float(seen.mean()),
        "effect_mean": float(effect.mean()),
        "effect_std": float(effect.std()),
        "effect_mean_abs": float(np.abs(effect).mean()),
        "effect_min": float(effect.min()),
        "effect_max": float(effect.max()),
    }


def similarity(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.shape != right.shape:
        raise ValueError("similarity shape mismatch")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("zero vector in similarity")
    return {
        "pearson_correlation": float(np.corrcoef(left, right)[0, 1]),
        "cosine_similarity": float(np.dot(left, right) / (left_norm * right_norm)),
    }


def candidate_metrics(
    baseline: np.ndarray,
    effect: np.ndarray,
    target: np.ndarray,
    game_type: np.ndarray,
    gamma: float,
) -> dict[str, object]:
    candidate = np.clip(baseline + gamma * effect, 0.0, 1.0)
    equal_mean = np.clip(candidate + (baseline.mean() - candidate.mean()), 0.0, 1.0)
    base_score = score(baseline, target)
    candidate_score = score(candidate, target)
    denominator = float(target.mean() * (1.0 - target.mean()))
    row_gain = (
        SCORE_SCALE
        * (np.square(baseline - target) - np.square(candidate - target))
        / (denominator * len(target))
    )
    is_f = game_type == "F"
    result = {
        "gamma": float(gamma),
        "base_score": base_score,
        "candidate_score": candidate_score,
        "raw_gain": candidate_score - base_score,
        "equal_mean_shape_gain": score(equal_mean, target) - base_score,
        "prediction_mean_shift": float(candidate.mean() - baseline.mean()),
        "F_contribution": float(row_gain[is_f].sum()),
        "R_contribution": float(row_gain[~is_f].sum()),
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
) -> dict[str, float | str | bool]:
    denominator = float(np.dot(effect, effect))
    beta = float(np.dot(target - baseline, effect) / denominator) if denominator else 0.0
    candidate = np.clip(baseline + beta * effect, 0.0, 1.0)
    return {
        "diagnostic_only": True,
        "uses_validation_2024_labels": True,
        "must_not_be_used_for_deployment": True,
        "unclipped_quadratic_beta": beta,
        "score_gain_after_probability_clip": score(candidate, target)
        - score(baseline, target),
        "note": "Descriptive ceiling only; never a selectable gamma.",
    }


def evaluate_variant(
    name: str,
    include_game_type: bool,
    source_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    source_target: np.ndarray,
    validation_target: np.ndarray,
    source_baseline: np.ndarray,
    validation_baseline: np.ndarray,
    overlap_vectors: dict[str, np.ndarray],
) -> tuple[dict[str, object], np.ndarray]:
    model = fit_source_matrix(
        source_rows,
        source_target,
        source_baseline,
        include_game_type,
    )
    effect, mapping = map_effect(model, validation_rows, include_game_type)
    overlaps = {
        overlap_name: similarity(effect, overlap_effect)
        for overlap_name, overlap_effect in overlap_vectors.items()
    }
    games = validation_rows["game_type"].astype(str).to_numpy()
    gamma_results = {
        f"{gamma:.2f}": candidate_metrics(
            validation_baseline,
            effect,
            validation_target,
            games,
            gamma,
        )
        for gamma in DIAGNOSTIC_GAMMAS
    }
    primary = gamma_results[f"{PRIMARY_GAMMA:.2f}"]
    checks = {
        "raw_gain": bool(primary["raw_gain"] >= GATE["raw_gain_min"]),
        "equal_mean_shape_gain": bool(
            primary["equal_mean_shape_gain"]
            >= GATE["equal_mean_shape_gain_min"]
        ),
        "F_nonnegative": bool(
            primary["F_contribution"] >= GATE["F_contribution_min"]
        ),
        "R_nonnegative": bool(
            primary["R_contribution"] >= GATE["R_contribution_min"]
        ),
        "exp021_lowrank_overlap": bool(
            abs(overlaps["exp021_lowrank_effect"]["pearson_correlation"])
            <= GATE["abs_corr_exp021_lowrank_max"]
        ),
        "endpoint_direction_overlap": bool(
            abs(overlaps["endpoint_direction"]["pearson_correlation"])
            <= GATE["abs_corr_endpoint_direction_max"]
        ),
    }
    return (
        {
            "name": name,
            "context_definition": (
                "pitcher x game_type(2) x legal_count(12) x batter_hand(2)"
                if include_game_type
                else "pitcher x legal_count(12) x batter_hand(2)"
            ),
            "fit": model["diagnostics"],
            "mapping": mapping,
            "overlap": overlaps,
            "diagnostic_gamma_sweep": gamma_results,
            "same_fold_oracle": same_fold_oracle(
                validation_baseline, effect, validation_target
            ),
            "promotion_gate": {
                "primary_gamma": PRIMARY_GAMMA,
                "checks": checks,
                "pass": bool(all(checks.values())),
            },
        },
        effect,
    )


def main() -> None:
    started = time.time()
    if abs(CHAMPION_WEIGHT + EXP021_WEIGHT - 1.0) > 1e-15:
        raise AssertionError("blend weights do not sum to one")

    columns = [
        "season",
        "pitcher_id",
        "balls_before",
        "strikes_before",
        "batter_hand",
        "game_type",
        "control_success",
    ]
    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=columns,
        low_memory=False,
    )
    rows = {
        year: train.loc[train["season"].eq(year)].reset_index(drop=True)
        for year in (SOURCE_YEAR, VALIDATION_YEAR)
    }
    targets = {
        year: rows[year]["control_success"].to_numpy(np.float64)
        for year in rows
    }

    alignment = {}
    for year in (SOURCE_YEAR, VALIDATION_YEAR):
        saved_target = vector(
            EXP021_TARGET[year], f"saved EXP021 target {year}", len(rows[year])
        )
        exact = bool(np.array_equal(saved_target, targets[year]))
        if not exact:
            raise AssertionError(f"official target/order mismatch for {year}")
        alignment[str(year)] = {
            "rows": int(len(rows[year])),
            "official_vs_saved_target_exact": exact,
            "official_target_rate": float(targets[year].mean()),
            "F_rows": int(rows[year]["game_type"].astype(str).eq("F").sum()),
            "R_rows": int(rows[year]["game_type"].astype(str).eq("R").sum()),
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
            EXP021_OOF[year], f"saved EXP021 rank6 OOF {year}", len(rows[year])
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

    team_2024 = probability_vector(
        TEAM_OOF_ROOT / "predictions_all_prior_s1000_2024.npy",
        "saved EXP019 team OOF 2024",
        len(rows[VALIDATION_YEAR]),
    )
    overlap_vectors = {
        "exp021_lowrank_effect": exp021[VALIDATION_YEAR] - team_2024,
        "v18_raw_effect": v18[VALIDATION_YEAR],
        "endpoint_direction": exp021[VALIDATION_YEAR]
        - champion[VALIDATION_YEAR],
    }

    variants = {}
    effects = {}
    for name, include_game_type in (("context24", False), ("context48", True)):
        result, effect = evaluate_variant(
            name,
            include_game_type,
            rows[SOURCE_YEAR],
            rows[VALIDATION_YEAR],
            targets[SOURCE_YEAR],
            targets[VALIDATION_YEAR],
            current_blend[SOURCE_YEAR],
            current_blend[VALIDATION_YEAR],
            overlap_vectors,
        )
        variants[name] = result
        effects[name] = effect

    effect_similarity = similarity(effects["context24"], effects["context48"])
    overall_pass = bool(any(row["promotion_gate"]["pass"] for row in variants.values()))
    verdict = (
        "PASS_FOR_FURTHER_EXACT_TEMPORAL_WORK"
        if overall_pass
        else "FAIL_LOW_GAIN_NO_SUBMISSION"
    )

    input_paths = [
        TRAIN_PATH,
        *CAT5_BASE.values(),
        *V18_EFFECT.values(),
        *EXP021_OOF.values(),
        *EXP021_TARGET.values(),
        TEAM_OOF_ROOT / "predictions_all_prior_s1000_2024.npy",
    ]
    payload = {
        "experiment": 126,
        "description": "clean-room low-rank pitcher-context residual on current blend",
        "status": verdict,
        "analysis_only": True,
        "reads_official_train": True,
        "reads_test": False,
        "reads_exp021_python_source": False,
        "builds_model_or_submission": False,
        "uses_public_lb_for_candidate_selection": False,
        "protocol": {
            "source_year": SOURCE_YEAR,
            "validation_year": VALIDATION_YEAR,
            "baseline": "seed42 corrected CAT5+CS+V18+affine blended with saved EXP021 OOF",
            "champion_weight": CHAMPION_WEIGHT,
            "exp021_weight": EXP021_WEIGHT,
            "effect_target": "source y-current_blend residual, centered within source season",
            "smoothing_formula": "sum(centered residual)/(cell_n+300)",
            "factorization": "deterministic full SVD truncated to rank 6",
            "unseen_pitcher_effect": 0.0,
            "primary_gamma": PRIMARY_GAMMA,
            "diagnostic_gammas_use_2024_labels": True,
            "same_fold_oracle_is_not_deployable": True,
        },
        "promotion_thresholds": GATE,
        "alignment": alignment,
        "baseline_scores": {
            str(year): score(current_blend[year], targets[year])
            for year in (SOURCE_YEAR, VALIDATION_YEAR)
        },
        "variants": variants,
        "context24_vs_context48_effect_similarity": effect_similarity,
        "input_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in input_paths
        },
        "elapsed_seconds": float(time.time() - started),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "=== exp/126 low-rank residual on current blend ===",
        "DIAGNOSTIC ONLY: official train + saved OOF; no test/model/ZIP/submission",
        "EXP021 source code imported/read: NO (saved OOF arrays only)",
        f"temporal mapping: fit {SOURCE_YEAR} residual -> evaluate {VALIDATION_YEAR}",
        f"baseline weights: champion={CHAMPION_WEIGHT:.10f} EXP021={EXP021_WEIGHT:.10f}",
        f"baseline scores: {SOURCE_YEAR}={payload['baseline_scores'][str(SOURCE_YEAR)]:.6f} "
        f"{VALIDATION_YEAR}={payload['baseline_scores'][str(VALIDATION_YEAR)]:.6f}",
        "",
    ]
    for name in ("context24", "context48"):
        row = variants[name]
        lines.append(
            f"[{name}] {row['context_definition']} | "
            f"coverage={row['mapping']['seen_pitcher_rate']:.6f} "
            f"effect_std={row['mapping']['effect_std']:.9f}"
        )
        lines.append(
            " gamma      raw     shape    F_contrib   R_contrib   mean_shift"
        )
        for gamma in DIAGNOSTIC_GAMMAS:
            metric = row["diagnostic_gamma_sweep"][f"{gamma:.2f}"]
            lines.append(
                f" {gamma:5.2f} {metric['raw_gain']:+9.3f} "
                f"{metric['equal_mean_shape_gain']:+9.3f} "
                f"{metric['F_contribution']:+11.3f} "
                f"{metric['R_contribution']:+11.3f} "
                f"{metric['prediction_mean_shift']:+.9f}"
            )
        oracle = row["same_fold_oracle"]
        lines.append(
            " 2024 SAME-FOLD ORACLE (diagnostic only; NEVER deploy): "
            f"beta={oracle['unclipped_quadratic_beta']:.6f} "
            f"gain={oracle['score_gain_after_probability_clip']:+.3f}"
        )
        for overlap_name, overlap in row["overlap"].items():
            lines.append(
                f" overlap {overlap_name}: "
                f"corr={overlap['pearson_correlation']:+.6f} "
                f"cos={overlap['cosine_similarity']:+.6f}"
            )
        checks = row["promotion_gate"]["checks"]
        lines.append(
            f" primary gamma={PRIMARY_GAMMA:.2f} gate={row['promotion_gate']['pass']} "
            f"checks={checks}"
        )
        lines.append("")
    lines.extend(
        [
            "context24/context48 effect similarity: "
            f"corr={effect_similarity['pearson_correlation']:+.6f} "
            f"cos={effect_similarity['cosine_similarity']:+.6f}",
            f"FINAL: {verdict}",
            "No 2022 rebuild is authorized by this result; no submission was created.",
            f"elapsed={payload['elapsed_seconds']:.2f}s",
        ]
    )
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
