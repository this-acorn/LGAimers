# -*- coding: utf-8 -*-
"""EXP188: transfer-gate a one-dimensional residual corrector on EXP183.

This experiment uses only official competition labels, our frozen current OOF
predictions, and the EXP183 endpoints that we trained ourselves.  It never
reads test.csv and never creates a submission package.

Protocol
--------
1. Fit three small CatBoost regressors on 2023 regular-season rows.  The sole
   input is our EXP183 endpoint probability and the target is
   ``control_success - current_anchor``.
2. Average their corrections and transfer the mapping unchanged to 2024.
3. Evaluate a scale grid that was declared in this source before inspecting
   the 2024 result.  Report overall, early/late, and pitcher-cluster bootstrap
   gains.  A later deployment script may use only a scale that passes all
   stability checks here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


SCORE_SCALE = 100000.0
SEEDS = (17, 42, 777)
SCALES = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30)
BOOTSTRAP_DRAWS = 5000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def score(probability: np.ndarray, target: np.ndarray) -> float:
    probability = np.asarray(probability, np.float64)
    target = np.asarray(target, np.float64)
    rate = float(target.mean())
    return float(
        SCORE_SCALE
        * (1.0 - np.mean((probability - target) ** 2) / (rate * (1.0 - rate)))
    )


def gain(
    current: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    if mask is None:
        mask = np.ones(len(target), dtype=bool)
    return {
        "rows": int(mask.sum()),
        "current_score": score(current[mask], target[mask]),
        "candidate_score": score(candidate[mask], target[mask]),
        "gain": score(candidate[mask], target[mask]) - score(current[mask], target[mask]),
    }


def pitcher_bootstrap(
    current: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    pitcher: np.ndarray,
) -> dict[str, float | int]:
    work = pd.DataFrame(
        {
            "pitcher": pitcher,
            "n": np.ones(len(target), dtype=np.int32),
            "y": target,
            "current_se": (current - target) ** 2,
            "candidate_se": (candidate - target) ** 2,
        }
    )
    grouped = work.groupby("pitcher", sort=False).agg(
        n=("n", "sum"),
        y=("y", "sum"),
        current_se=("current_se", "sum"),
        candidate_se=("candidate_se", "sum"),
    ).to_numpy(np.float64)
    rng = np.random.default_rng(188)
    values = np.empty(BOOTSTRAP_DRAWS, np.float64)
    for start in range(0, BOOTSTRAP_DRAWS, 200):
        width = min(200, BOOTSTRAP_DRAWS - start)
        sampled = grouped[
            rng.integers(0, len(grouped), size=(width, len(grouped)))
        ].sum(axis=1)
        rate = sampled[:, 1] / sampled[:, 0]
        values[start : start + width] = (
            SCORE_SCALE
            * (sampled[:, 2] - sampled[:, 3])
            / (sampled[:, 0] * rate * (1.0 - rate))
        )
    return {
        "draws": BOOTSTRAP_DRAWS,
        "clusters": int(len(grouped)),
        "p025": float(np.quantile(values, 0.025)),
        "median": float(np.median(values)),
        "p975": float(np.quantile(values, 0.975)),
        "prob_positive": float(np.mean(values > 0.0)),
    }


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def load_vector(path: Path, expected_rows: int) -> np.ndarray:
    value = np.load(path, allow_pickle=False).astype(np.float64)
    if value.shape != (expected_rows,) or not np.isfinite(value).all():
        raise ValueError(f"invalid vector {path}: {value.shape}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--exp183-dir", default="lab/183_gpu_x176")
    parser.add_argument("--current-2023", default="lab/179_current_2023.npy")
    parser.add_argument("--current-2024", default="lab/179_current_2024.npy")
    parser.add_argument("--output-dir", default="lab/188_exp183_residual_gate")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    started = time.time()
    data_dir = Path(args.data_dir).resolve()
    exp183_dir = Path(args.exp183_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_path = output_dir / "validation_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite: {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    columns = [
        "season", "game_type", "game_month", "pitcher_id", "control_success"
    ]
    train = pd.read_csv(
        train_path, encoding="utf-8-sig", usecols=columns, low_memory=False
    )
    train.columns = [str(column).replace("\ufeff", "").strip() for column in train.columns]
    rows = {
        year: train.loc[pd.to_numeric(train["season"], errors="coerce").eq(year)]
        .reset_index(drop=True)
        for year in (2023, 2024)
    }
    y = {
        year: pd.to_numeric(rows[year]["control_success"], errors="raise")
        .to_numpy(np.float64)
        for year in rows
    }

    endpoint_paths = {
        2023: exp183_dir / "endpoint_2023_ensemble_calibrated.npy",
        2024: exp183_dir / "endpoint_2024_ensemble_calibrated.npy",
    }
    current_paths = {
        2023: Path(args.current_2023).resolve(),
        2024: Path(args.current_2024).resolve(),
    }
    endpoint = {
        year: load_vector(endpoint_paths[year], len(rows[year])) for year in rows
    }
    current = {
        year: load_vector(current_paths[year], len(rows[year])) for year in rows
    }

    regular23 = rows[2023]["game_type"].astype(str).eq("R").to_numpy()
    regular24 = rows[2024]["game_type"].astype(str).eq("R").to_numpy()
    x23 = pd.DataFrame(
        {"auxiliary_probability": endpoint[2023][regular23].astype(np.float32)}
    )
    residual23 = y[2023][regular23] - current[2023][regular23]
    x24 = pd.DataFrame(
        {"auxiliary_probability": endpoint[2024][regular24].astype(np.float32)}
    )

    members23: list[np.ndarray] = []
    members24: list[np.ndarray] = []
    model_artifacts = []
    for seed in SEEDS:
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=150,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=20.0,
            random_seed=seed,
            random_strength=0.25,
            thread_count=args.threads,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(x23, residual23)
        members23.append(model.predict(x23).astype(np.float64))
        members24.append(model.predict(x24).astype(np.float64))
        model_path = output_dir / f"residual_seed{seed}.cbm"
        model.save_model(model_path)
        model_artifacts.append(
            {"path": str(model_path), "sha256": sha256_file(model_path)}
        )

    correction23 = np.zeros(len(rows[2023]), np.float64)
    correction24 = np.zeros(len(rows[2024]), np.float64)
    correction23[regular23] = np.mean(np.stack(members23), axis=0)
    correction24[regular24] = np.mean(np.stack(members24), axis=0)
    correction23 = np.clip(correction23, -0.25, 0.25)
    correction24 = np.clip(correction24, -0.25, 0.25)
    correction23_path = output_dir / "correction_2023.npy"
    correction24_path = output_dir / "correction_2024.npy"
    np.save(correction23_path, correction23.astype(np.float32), allow_pickle=False)
    np.save(correction24_path, correction24.astype(np.float32), allow_pickle=False)

    month24 = pd.to_numeric(rows[2024]["game_month"], errors="coerce").fillna(0).to_numpy()
    pitcher24 = pd.to_numeric(rows[2024]["pitcher_id"], errors="coerce").fillna(-1).to_numpy(np.int64)
    results = {}
    for scale in SCALES:
        candidate = np.clip(current[2024] + scale * correction24, 0.0, 1.0)
        overall = gain(current[2024], candidate, y[2024])
        early = gain(current[2024], candidate, y[2024], month24 <= 6)
        late = gain(current[2024], candidate, y[2024], month24 > 6)
        bootstrap = pitcher_bootstrap(
            current[2024], candidate, y[2024], pitcher24
        )
        results[f"{scale:.3f}"] = {
            "overall": overall,
            "early": early,
            "late": late,
            "pitcher_bootstrap": bootstrap,
            "stable": bool(
                overall["gain"] > 0.0
                and early["gain"] > 0.0
                and late["gain"] > 0.0
                and bootstrap["p025"] > 0.0
            ),
        }

    stable = [
        (float(scale), result)
        for scale, result in results.items()
        if result["stable"]
    ]
    selected = max(stable, key=lambda item: item[1]["overall"]["gain"]) if stable else None
    report = {
        "experiment": 188,
        "status": "PASS_FOR_FINAL_TRAIN" if selected else "FAIL_NO_DEPLOY",
        "protocol": "fit 2023 R residual map; transfer unchanged to 2024; predeclared scale grid",
        "data": {
            "train_path": str(train_path),
            "train_sha256": sha256_file(train_path),
            "rows_2023": len(rows[2023]),
            "rows_2024": len(rows[2024]),
            "R_rows_2023": int(regular23.sum()),
            "R_rows_2024": int(regular24.sum()),
        },
        "inputs": {
            str(year): {
                "endpoint": str(endpoint_paths[year]),
                "endpoint_sha256": sha256_file(endpoint_paths[year]),
                "current": str(current_paths[year]),
                "current_sha256": sha256_file(current_paths[year]),
            }
            for year in rows
        },
        "recipe": {
            "feature": "our EXP183 calibrated endpoint probability only",
            "target": "control_success - our frozen current anchor",
            "fit_rows": "2023 game_type R only",
            "seeds": list(SEEDS),
            "iterations": 150,
            "depth": 4,
            "learning_rate": 0.03,
            "l2_leaf_reg": 20.0,
            "scales": list(SCALES),
        },
        "fit_diagnostic_2023_in_sample": {
            "scale_0.15_gain": gain(
                current[2023],
                np.clip(current[2023] + 0.15 * correction23, 0.0, 1.0),
                y[2023],
            )["gain"],
            "correction_R_mean": float(correction23[regular23].mean()),
            "correction_R_std": float(correction23[regular23].std()),
        },
        "transfer_2024": results,
        "selected_scale": None if selected is None else selected[0],
        "selected_result": None if selected is None else selected[1],
        "artifacts": model_artifacts
        + [
            {"path": str(correction23_path), "sha256": sha256_file(correction23_path)},
            {"path": str(correction24_path), "sha256": sha256_file(correction24_path)},
        ],
        "test_read": False,
        "submission_zip_created": False,
        "wall_seconds": time.time() - started,
    }
    report_path.write_text(
        json.dumps(native(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if selected:
        result = selected[1]
        print(
            f"[EXP188 FINAL] PASS_FOR_FINAL_TRAIN scale={selected[0]:.3f} "
            f"gain={result['overall']['gain']:+.4f} "
            f"early={result['early']['gain']:+.4f} "
            f"late={result['late']['gain']:+.4f} "
            f"p025={result['pitcher_bootstrap']['p025']:+.4f}"
        )
    else:
        best = max(results.items(), key=lambda item: item[1]["overall"]["gain"])
        result = best[1]
        print(
            f"[EXP188 FINAL] FAIL_NO_DEPLOY best_scale={best[0]} "
            f"gain={result['overall']['gain']:+.4f} "
            f"early={result['early']['gain']:+.4f} "
            f"late={result['late']['gain']:+.4f} "
            f"p025={result['pitcher_bootstrap']['p025']:+.4f}"
        )
    print(f"[REPORT] {report_path}")


if __name__ == "__main__":
    main()
