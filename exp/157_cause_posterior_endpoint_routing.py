# -*- coding: utf-8 -*-
"""Forward screen for CAT5-cause-dependent Champion/EXP021 routing.

Fit only on 2022 residual, select a ridge/cap by 2023 transfer, refit on
2022+2023, and open 2024 once.  The 2023 posterior is an explicitly labelled
proxy from exp104 because the corrected exp122 full posterior was not cached;
therefore even a pass would require exact-posterior confirmation before build.

No model training, test data, ZIP, submission, or LB value is used.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "lab" / "157_cause_posterior_endpoint_routing.json"
OUT_TXT = ROOT / "lab" / "157_cause_posterior_endpoint_routing.txt"
POSTERIOR = {
    2022: ROOT / "lab" / "104_cat5_y2022_probs_seed42.npy",
    2023: ROOT / "lab" / "104_cat5_y2023_probs_seed42.npy",
    2024: ROOT / "lab" / "89_cat5_probs_seed42.npy",
}
RIDGES = (0.1, 1.0, 10.0, 100.0)
CAPS = (0.05, 0.10)
EPS = 1e-12


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cause_features(posterior: np.ndarray) -> np.ndarray:
    if posterior.ndim != 2 or posterior.shape[1] != 5:
        raise ValueError(f"bad CAT5 posterior shape {posterior.shape}")
    if not np.isfinite(posterior).all() or np.max(np.abs(posterior.sum(axis=1) - 1)) > 1e-5:
        raise ValueError("invalid CAT5 posterior")
    p = np.clip(posterior.astype(np.float64), EPS, 1.0)
    middle = p[:, 1] + p[:, 3]
    reverse = p[:, 2] + p[:, 3]
    bigmiss = p[:, 4]
    overlap = p[:, 3]
    entropy = -np.sum(p * np.log(p), axis=1)
    confidence = np.max(p, axis=1)
    failure = 1.0 - p[:, 0]
    # Cause shares distinguish composition from the already-strong failure level.
    middle_share = middle / np.maximum(failure, EPS)
    reverse_share = reverse / np.maximum(failure, EPS)
    bigmiss_share = bigmiss / np.maximum(failure, EPS)
    return np.column_stack([
        middle, reverse, bigmiss, overlap, entropy, confidence,
        middle_share, reverse_share, bigmiss_share,
    ])


def standardize_fit(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return (matrix - mean) / std, mean, std


def route_design(
    features: np.ndarray, direction: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return ((features - mean) / std) * direction[:, None]


def fit_ridge(design: np.ndarray, residual: np.ndarray, ridge: float) -> np.ndarray:
    xtx = design.T @ design
    scale = float(np.trace(xtx) / design.shape[1])
    return np.linalg.solve(
        xtx + ridge * max(scale, 1e-12) * np.eye(design.shape[1]),
        design.T @ residual,
    )


def routed_effect(
    features: np.ndarray,
    direction: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    coefficient: np.ndarray,
    cap: float,
) -> tuple[np.ndarray, np.ndarray]:
    weight_delta = np.clip(((features - mean) / std) @ coefficient, -cap, cap)
    return direction * weight_delta, weight_delta


def main() -> None:
    started = time.time()
    e155 = load_module(ROOT / "exp" / "155_cause_runner_currentblend_gate.py", "e155_routing")
    train = pd.read_csv(
        e155.TRAIN, encoding="utf-8-sig",
        usecols=["season", "game_type", "control_success"], low_memory=False,
    )
    rows = {
        year: train.loc[train["season"].eq(year)].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    target = {year: rows[year]["control_success"].to_numpy(float) for year in rows}
    baseline = {}
    direction = {}
    features = {}
    report = {}
    for year in (2022, 2023, 2024):
        n = len(rows[year])
        saved_target = e155.load_vector(e155.EXP021_TARGET[year], n)
        if not np.array_equal(saved_target, target[year]):
            raise AssertionError(f"target/order mismatch {year}")
        cat = e155.load_probability(e155.CAT5[year], n)
        v18 = e155.load_vector(e155.V18[year], n)
        external = e155.load_probability(e155.EXP021[year], n)
        champion = e155.champion_probability(cat, v18)
        baseline[year] = e155.CHAMPION_WEIGHT * champion + e155.EXP021_WEIGHT * external
        direction[year] = external - champion
        posterior = np.load(POSTERIOR[year], allow_pickle=False).astype(np.float64)
        features[year] = cause_features(posterior)
        report[str(year)] = {
            "rows": n,
            "baseline_score": e155.score(baseline[year], target[year]),
            "endpoint_direction_std": float(direction[year].std()),
            "posterior_source": str(POSTERIOR[year].relative_to(ROOT)).replace("\\", "/"),
            "posterior_is_corrected_current": bool(year != 2023),
        }

    x22, mean22, std22 = standardize_fit(features[2022])
    design22 = x22 * direction[2022][:, None]
    discovery = []
    for ridge in RIDGES:
        coefficient = fit_ridge(design22, target[2022] - baseline[2022], ridge)
        for cap in CAPS:
            effect, weight_delta = routed_effect(
                features[2023], direction[2023], mean22, std22, coefficient, cap
            )
            result = e155.metrics(
                baseline[2023], effect, target[2023],
                rows[2023]["game_type"].astype(str).to_numpy(),
            )
            discovery.append({
                "ridge": ridge, "cap": cap,
                "coefficient_fit_2022": coefficient.tolist(),
                "weight_delta_std": float(weight_delta.std()),
                "weight_delta_capped_rate": float(np.mean(np.abs(weight_delta) >= cap - 1e-12)),
                "transfer_2023": result,
            })
    eligible = [
        row for row in discovery
        if row["transfer_2023"]["raw_gain"] > 0
        and row["transfer_2023"]["equal_mean_shape_gain"] > 0
        and row["transfer_2023"]["F_contribution"] >= 0
        and row["transfer_2023"]["R_contribution"] >= 0
    ]
    selected = max(
        eligible or discovery,
        key=lambda row: (
            min(row["transfer_2023"]["F_contribution"], row["transfer_2023"]["R_contribution"]),
            row["transfer_2023"]["raw_gain"],
        ),
    )

    pooled_features = np.vstack([features[2022], features[2023]])
    _, pooled_mean, pooled_std = standardize_fit(pooled_features)
    pooled_direction = np.concatenate([direction[2022], direction[2023]])
    pooled_design = route_design(pooled_features, pooled_direction, pooled_mean, pooled_std)
    pooled_residual = np.concatenate([
        target[2022] - baseline[2022], target[2023] - baseline[2023]
    ])
    coefficient = fit_ridge(pooled_design, pooled_residual, float(selected["ridge"]))
    effect24, weight24 = routed_effect(
        features[2024], direction[2024], pooled_mean, pooled_std,
        coefficient, float(selected["cap"]),
    )
    confirmation = e155.metrics(
        baseline[2024], effect24, target[2024],
        rows[2024]["game_type"].astype(str).to_numpy(),
    )
    old = np.asarray(selected["coefficient_fit_2022"], dtype=float)
    stability = {
        "coefficient_refit": coefficient.tolist(),
        "cosine": float((old @ coefficient) / (np.linalg.norm(old) * np.linalg.norm(coefficient)))
        if np.linalg.norm(old) and np.linalg.norm(coefficient) else 0.0,
        "same_sign_fraction": float(np.mean(np.sign(old) == np.sign(coefficient))),
        "weight_delta_2024_std": float(weight24.std()),
        "weight_delta_2024_capped_rate": float(
            np.mean(np.abs(weight24) >= float(selected["cap"]) - 1e-12)
        ),
    }

    # Descriptive 2024 ceiling with the already selected ridge/cap.
    _, mean24, std24 = standardize_fit(features[2024])
    design24 = route_design(features[2024], direction[2024], mean24, std24)
    oracle_coefficient = fit_ridge(
        design24, target[2024] - baseline[2024], float(selected["ridge"])
    )
    oracle_effect, _ = routed_effect(
        features[2024], direction[2024], mean24, std24, oracle_coefficient,
        float(selected["cap"]),
    )
    oracle = e155.metrics(
        baseline[2024], oracle_effect, target[2024],
        rows[2024]["game_type"].astype(str).to_numpy(),
    )
    passed = bool(
        eligible
        and confirmation["raw_gain"] >= 10
        and confirmation["equal_mean_shape_gain"] >= 8
        and confirmation["F_contribution"] >= 0
        and confirmation["R_contribution"] >= 0
        and stability["cosine"] > 0.5
    )
    status = "PROXY_PASS_REQUIRES_EXACT_2023_POSTERIOR" if passed else "FAIL_NO_SUBMISSION"
    payload = {
        "experiment": 157, "status": status, "analysis_only": True,
        "reads_test": False, "trains_model": False, "uses_lb": False,
        "caveat": "2023 routing covariate is exp104 same-hand-bug proxy; no build allowed from this alone",
        "features": [
            "middle", "reverse", "bigmiss", "overlap", "entropy", "confidence",
            "middle_share", "reverse_share", "bigmiss_share",
        ],
        "folds": report, "discovery": discovery, "selected": selected,
        "stability": stability, "confirmation_2024": confirmation,
        "same_fold_2024_routing_oracle_diagnostic_only": oracle,
        "elapsed_seconds": float(time.time() - started),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "=== exp157 CAT5-cause endpoint routing forward gate ===",
        "DIAGNOSTIC ONLY: saved OOF only; no model/test/ZIP/LB",
        "CAVEAT: 2023 posterior is exp104 proxy; a pass is not build authority",
        f"selected on 2023: ridge={selected['ridge']} cap={selected['cap']}",
        f"2023 transfer: {selected['transfer_2023']}",
        f"coefficient stability: {stability}",
        f"untouched 2024: {confirmation}",
        f"2024 same-fold routing oracle (diagnostic only): {oracle}",
        f"FINAL: {status}",
        f"elapsed={payload['elapsed_seconds']:.2f}s",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
