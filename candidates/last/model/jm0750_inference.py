"""Dacon inference wrapper: JOA anchor plus a small R-only residual correction."""

from __future__ import annotations

import gc
import json
import os
import runpy
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
EPS = 1e-6


def add_inference_features(data: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Recreate Candidate4/time-safe derived fields from saved train history."""
    enriched = data.copy()
    for entity, id_col, n_col, rate_col in (
        ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
        ("batter", "batter_id", "asof_batter_n", "asof_batter_success_rate"),
    ):
        history = artifact["history"][entity]
        history_n = enriched[id_col].map(history["n"]).fillna(0.0).to_numpy(float)
        history_successes = (
            enriched[id_col].map(history["successes"]).fillna(0.0).to_numpy(float)
        )
        asof_n = pd.to_numeric(enriched[n_col], errors="coerce").fillna(0.0).to_numpy(float)
        asof_rate = pd.to_numeric(enriched[rate_col], errors="coerce").fillna(0.0).to_numpy(float)
        asof_successes = asof_rate * asof_n
        season_n = np.maximum(asof_n - history_n, 0.0)
        season_successes = np.clip(asof_successes - history_successes, 0.0, season_n)
        history_rate = np.divide(
            history_successes,
            history_n,
            out=np.full_like(history_successes, np.nan),
            where=history_n > 0,
        )
        season_rate = np.divide(
            season_successes,
            season_n,
            out=np.full_like(season_successes, np.nan),
            where=season_n > 0,
        )
        enriched[f"derived_{entity}_history_n"] = history_n
        enriched[f"derived_{entity}_history_success_rate"] = history_rate
        enriched[f"derived_{entity}_season_n"] = season_n
        enriched[f"derived_{entity}_season_success_rate"] = season_rate
        for strength in (10.0, 50.0, 200.0):
            fallback = np.where(np.isfinite(history_rate), history_rate, 0.5)
            enriched[f"derived_{entity}_season_success_rate_s{int(strength)}"] = (
                season_successes + strength * fallback
            ) / (season_n + strength)

    enriched["derived_prev1_minus_career_success"] = (
        enriched["asof_pitcher_prev1_game_success_rate"]
        - enriched["asof_pitcher_success_rate"]
    )
    enriched["derived_prev3_minus_career_success"] = (
        enriched["asof_pitcher_prev3_game_success_rate"]
        - enriched["asof_pitcher_success_rate"]
    )
    enriched["derived_prev5_minus_career_success"] = (
        enriched["asof_pitcher_prev5_game_success_rate"]
        - enriched["asof_pitcher_success_rate"]
    )
    enriched["derived_recent_success_slope"] = (
        enriched["asof_pitcher_prev1_game_success_rate"]
        - enriched["asof_pitcher_prev5_game_success_rate"]
    )
    enriched["derived_two_strike"] = (enriched["strikes_before"] == 2).astype("int8")
    enriched["derived_three_ball"] = (enriched["balls_before"] == 3).astype("int8")
    enriched["derived_full_count"] = (
        (enriched["balls_before"] == 3) & (enriched["strikes_before"] == 2)
    ).astype("int8")
    enriched["derived_platoon_same_hand"] = (
        enriched["pitcher_hand"] == enriched["batter_hand"]
    ).astype("int8")
    enriched["derived_futures_new_regime"] = (
        (enriched["game_type"] == "F") & (enriched["season"] >= 2023)
    ).astype("int8")
    return enriched


def beta_prediction(data: pd.DataFrame, artifact: dict) -> np.ndarray:
    beta = artifact["beta"]
    concentration = float(beta["concentration"])
    weights = np.asarray(beta["weights"], dtype=np.float64)
    prior = (
        data["game_type"]
        .map(beta["prior_by_game_type"])
        .fillna(float(beta["global_prior"]))
        .to_numpy(dtype=np.float64)
    )
    posteriors = []
    for entity in ("pitcher", "batter"):
        n = data[f"derived_{entity}_season_n"].to_numpy(dtype=np.float64)
        rate = data[f"derived_{entity}_season_success_rate"].to_numpy(dtype=np.float64)
        successes = np.rint(np.nan_to_num(rate, nan=0.0) * n)
        posteriors.append((successes + concentration * prior) / (n + concentration))
    career = data["asof_pitcher_success_rate"].to_numpy(dtype=np.float64)
    career = np.where(np.isfinite(career), career, prior)
    recent = data["asof_pitcher_prev5_game_success_rate"].to_numpy(dtype=np.float64)
    recent = np.where(np.isfinite(recent), recent, career)
    candidates = np.column_stack([posteriors[0], posteriors[1], prior, career, recent])
    return np.clip(candidates @ weights, 0.0, 1.0)


def candidate4_components(enriched: pd.DataFrame, artifact: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_cols = artifact["feature_cols"]
    missing = sorted(set(feature_cols) - set(enriched.columns))
    if missing:
        raise ValueError(f"Candidate4 inference features missing: {missing}")
    extra = artifact["extra_model"].predict_proba(enriched[feature_cols])[:, 1]
    beta = beta_prediction(enriched, artifact)
    weight = float(artifact.get("extra_weight", 0.7590660319336603))
    raw = np.clip(weight * extra + (1.0 - weight) * beta, EPS, 1.0 - EPS)
    return extra, beta, raw


def run_joa_anchor() -> pd.DataFrame:
    base_script = Path("./joa_base_script.py")
    if not base_script.exists():
        raise FileNotFoundError(base_script)
    namespace = runpy.run_path(str(base_script), run_name="__main__")
    del namespace
    gc.collect()
    output = Path("./output/submission.csv")
    if not output.exists():
        raise RuntimeError("JOA base script did not create output/submission.csv")
    return pd.read_csv(output, encoding="utf-8-sig")


def main() -> None:
    started = time.perf_counter()
    model_dir = Path("./model")
    sys.path.insert(0, str(model_dir.resolve()))
    from sota_common_pipeline import catboost_frame, finish_common_features

    manifest = json.loads((model_dir / "r_residual_manifest.json").read_text())
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig", low_memory=False)
    sample = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    if ID_COL not in test or list(sample.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError("Unexpected test/sample schema")

    print("Running frozen JOA F-Regime075 anchor...")
    base_submission = run_joa_anchor()
    base_by_id = pd.Series(
        base_submission[TARGET_COL].to_numpy(float),
        index=base_submission[ID_COL].astype(str),
    )
    anchor = test[ID_COL].astype(str).map(base_by_id)
    if anchor.isna().any():
        raise ValueError(f"JOA anchor missing rows: {int(anchor.isna().sum())}")
    anchor = anchor.to_numpy(dtype=np.float64)

    print("Building deployable Candidate4/common features...")
    c4_artifact = joblib.load(model_dir / "candidate4_ensemble.joblib")
    missing_raw = sorted(set(c4_artifact["expected_test_cols"]) - set(test.columns))
    if missing_raw:
        raise ValueError(f"Test columns missing: {missing_raw}")
    enriched = add_inference_features(test, c4_artifact)
    p_extra, p_beta, p_raw = candidate4_components(enriched, c4_artifact)
    prior = (
        test["game_type"]
        .astype(str)
        .map(manifest["prior_by_game_type"])
        .fillna(float(manifest["global_prior"]))
        .to_numpy(dtype=np.float64)
    )
    features, _ = finish_common_features(enriched, prior)
    features["joa_anchor_probability"] = anchor
    features["joa_anchor_margin"] = np.abs(anchor - 0.5)
    features["c4_extra_probability"] = p_extra
    features["c4_beta_probability"] = p_beta
    features["c4_raw_probability"] = p_raw
    features["c4_minus_joa_anchor"] = p_raw - anchor
    columns = manifest["feature_cols"]
    missing = sorted(set(columns) - set(features.columns))
    if missing:
        raise ValueError(f"Residual inference features missing: {missing}")
    categorical = [column for column in manifest["categorical_cols"] if column in columns]
    x = catboost_frame(features[columns], categorical)

    from catboost import CatBoostRegressor

    corrections = []
    for name in manifest["model_files"]:
        model = CatBoostRegressor()
        model.load_model(str(model_dir / name))
        corrections.append(model.predict(x))
        del model
    correction = np.mean(np.stack(corrections), axis=0)
    correction = np.clip(
        correction,
        -float(manifest["max_abs_correction"]),
        float(manifest["max_abs_correction"]),
    )
    prediction = anchor.copy()
    r_mask = test["game_type"].astype(str).eq("R").to_numpy()
    prediction[r_mask] = np.clip(
        prediction[r_mask] + float(manifest["scale"]) * correction[r_mask],
        EPS,
        1.0 - EPS,
    )

    prediction_by_id = pd.Series(prediction, index=test[ID_COL].astype(str))
    aligned = sample[ID_COL].astype(str).map(prediction_by_id)
    if aligned.isna().any():
        raise ValueError(f"Final prediction missing rows: {int(aligned.isna().sum())}")
    sample[TARGET_COL] = aligned.to_numpy(dtype=np.float64)
    os.makedirs("./output", exist_ok=True)
    sample.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(
        f"Saved output/submission.csv: rows={len(sample):,}, mean={sample[TARGET_COL].mean():.6f}, "
        f"R_changed={int(r_mask.sum()):,}, elapsed={time.perf_counter() - started:.2f}s"
    )


if __name__ == "__main__":
    main()

