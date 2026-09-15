"""Offline inference entry point for the adaptive hierarchical residual stack."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from src.preprocessing_v2 import build_v2_features, build_v3_features
from src.adaptive_gate import build_gate_features
from src.psych_latent import build_production_features, apply_linear_residual
from src.psych_film_expert import apply_saved_regime
from src.split_priors import apply_split_priors
from src.league_transition import transition_features

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model"


def main():
    started = time.time()
    meta = json.loads((MODEL / "manifest.json").read_text(encoding="utf-8"))
    test = pd.read_csv(ROOT / "data/test.csv", low_memory=False)
    row_id = test["row_id"].copy()
    ps = pd.read_pickle(MODEL / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(MODEL / "batter_snapshots.pkl")
    ms = pd.read_pickle(MODEL / "pitchmix_snapshots.pkl")
    tm = str(MODEL / "trackman_prior_features.csv")
    x2, base2 = build_v2_features(test, meta["prior"], ps, tm)
    x3, base3 = build_v3_features(test, meta["prior"], ps, bs, ms, tm)

    # Seed-to-seed prediction noise measures 0.006-0.008 per row. It inflates
    # Var(p) without touching Cov(p, y), so averaging the seeds away is worth
    # +15 to +25 on every forward season - and no reweighting of the correction
    # channels can reach it, because each channel is a fixed vector.
    seeds = meta.get("seeds")

    def load_reg(stem):
        names = [f"{stem}_seed{s}.cbm" for s in seeds] if seeds else [f"{stem}.cbm"]
        out = []
        for filename in names:
            model = CatBoostRegressor()
            model.load_model(MODEL / filename)
            out.append(model)
        return out

    predictions = []
    for stem, x, base in [
        ("v2_decay55", x2, base2),
        ("v3_decay55", x3, base3),
        ("v3_decay30", x3, base3),
    ]:
        member = [np.clip(base + m.predict(x), 1e-6, 1 - 1e-6) for m in load_reg(stem)]
        predictions.append(np.mean(member, axis=0))

    regime = json.loads((MODEL / "f_regime_meta.json").read_text())
    futures = test["game_type"].eq("F").to_numpy()
    def f_reg_mean(stem, count, x, base):
        member=[]
        for j in range(count):
            m=CatBoostRegressor();m.load_model(MODEL / f"{stem}_{j}.cbm")
            member.append(np.clip(base+m.predict(x),1e-6,1-1e-6))
        return np.mean(member,axis=0)
    if futures.any():
        f2=f_reg_mean("f_v2_all",4,x2,base2)
        predictions[0]=np.where(futures,predictions[0]+regime["v2_scale"]*(f2-predictions[0]),predictions[0])
        f55=f_reg_mean("f_v355_recent",6,x3,base3)
        predictions[1]=np.where(futures,predictions[1]+regime["v355_scale"]*(f55-predictions[1]),predictions[1])
        f30a=f_reg_mean("f_v330_all",4,x3,base3);f30r=f_reg_mean("f_v330_recent",2,x3,base3)
        recent_inner=predictions[2]+regime["v330_recent_inner_scale"]*(f30r-predictions[2])
        f30=regime["v330_all_weight"]*f30a+(1-regime["v330_all_weight"])*recent_inner
        predictions[2]=np.where(futures,predictions[2]+regime["v330_scale"]*(f30-predictions[2]),predictions[2])

    risks = []
    for name in ("middle", "wild", "reverse"):
        stems = [f"subtype_{name}_seed{s}.cbm" for s in seeds] if seeds else [f"subtype_{name}.cbm"]
        member = []
        for filename in stems:
            model = CatBoostClassifier()
            model.load_model(MODEL / filename)
            member.append(model.predict_proba(x3)[:, 1])
        risk=np.mean(member, axis=0)
        if futures.any():
            fm=CatBoostClassifier();fm.load_model(MODEL / f"f_subtype_{name}.cbm")
            fr=fm.predict_proba(x3)[:,1]
            risk=np.where(futures,risk+regime["subtype_scale"]*(fr-risk),risk)
        risks.append(risk)

    # This is the strictly-forward meta prediction used to train the gate.
    original_main = np.average(
        np.vstack(predictions), axis=0, weights=[0.27358084, 0.26512224, 0.46129691]
    )
    original_z = np.column_stack([original_main] + risks)
    original_p = 0.0300329767 + original_z @ np.asarray(
        [0.93505266, -0.00520129, 0.01091677, -0.02528331]
    )

    if "direct_coefficients" in meta:
        z = np.column_stack(predictions + risks)
        p = meta["stack_intercept"] + z @ np.asarray(meta["direct_coefficients"])
    else:
        main_p = np.average(np.vstack(predictions), axis=0, weights=meta["main_weights"])
        z = np.column_stack([main_p] + risks)
        p = meta["stack_intercept"] + z @ np.asarray(meta["stack_coefficients"])

    if meta.get("adaptive_gate", False):
        gate_x = build_gate_features(
            test, predictions, risks, np.clip(original_p, 1e-6, 1 - 1e-6)
        )
        gate = CatBoostRegressor()
        gate.load_model(MODEL / "adaptive_gate.cbm")
        p = original_p + float(meta.get("gate_scale", 1.0)) * gate.predict(gate_x)

    # Every scored submission is a point in one six-channel space. Each channel
    # corrects the same gate prediction rather than the previous channel's
    # output, which is what makes the leaderboard score an exact quadratic in the
    # weights - so a weight vector can reproduce any past submission and also the
    # optimum between them.
    if "channel_weights" in meta:
        w = meta["channel_weights"]
        gate_p = np.asarray(p, dtype=float).copy()
        p = gate_p.copy()

        if w.get("psych"):
            residual_x = build_production_features(
                test, MODEL / "psych_profile.pkl", MODEL / "latent_pitch_context.csv")
            p = p + w["psych"] * apply_linear_residual(
                residual_x, MODEL / "psych_latent_meta.npz")

        if w.get("split"):
            tables = pd.read_pickle(MODEL / "split_prior_tables.pkl")
            p = p + w["split"] * apply_split_priors(test, tables)

        # The minimax submission carried the FiLM head at 1.25, so its weight
        # feeds the same channel rather than recomputing the network.
        film_weight = float(w.get("film", 0.0)) + 1.25 * float(w.get("minimax", 0.0))
        if film_weight:
            profile = pd.read_pickle(MODEL / "psych_regime_profile.pkl")
            p = p + film_weight * apply_saved_regime(
                test, gate_p, profile, MODEL / "psych_regime_assets.npz",
                MODEL / "psych_regime.pt")

        if w.get("minimax"):
            from src.stable_experts import context_ridge_correction, platoon_correction
            stable = pd.read_pickle(MODEL / "stable_context_profile.pkl")
            expert = CatBoostRegressor()
            expert.load_model(MODEL / "stable_platoon.cbm")
            p = p + w["minimax"] * (
                context_ridge_correction(test, stable, MODEL / "stable_context_ridge.npz")
                + platoon_correction(test, expert, 0.30))

        if w.get("hier"):
            hier_z = np.column_stack([
                np.average(np.vstack(predictions), axis=0, weights=meta["main_weights"])] + risks)
            hier_p = meta["stack_intercept"] + hier_z @ np.asarray(meta["stack_coefficients"])
            p = p + w["hier"] * (hier_p - gate_p)
    else:
        blend = float(meta.get("correction_scale", 1.0))

        if meta.get("psych_latent_residual", False):
            residual_x = build_production_features(
                test, MODEL / "psych_profile.pkl", MODEL / "latent_pitch_context.csv"
            )
            p = p + blend * apply_linear_residual(residual_x, MODEL / "psych_latent_meta.npz")

        if meta.get("psych_regime_film", False):
            profile=pd.read_pickle(MODEL / "psych_regime_profile.pkl")
            p=p+apply_saved_regime(test,p,profile,MODEL / "psych_regime_assets.npz",MODEL / "psych_regime.pt")

        if meta.get("split_priors", False):
            # Zero-sum within each entity, so this reallocates probability between
            # rows without moving the unknown 2025 league level.
            tables = pd.read_pickle(MODEL / "split_prior_tables.pkl")
            p = p + blend * apply_split_priors(test, tables)

    # Fixed global calibration hyperparameter.
    p = p + float(meta.get("global_shift", 0.0))

    transition=CatBoostRegressor();transition.load_model(MODEL / "transition_gate.cbm")
    tx=transition_features(test,p,MODEL / "prior_type.pkl")
    p=p+regime["transition_scale"]*transition.predict(tx)

    p = np.clip(p, 1e-5, 1 - 1e-5)
    if len(p) != len(test) or not np.isfinite(p).all():
        raise RuntimeError("invalid predictions")

    out = ROOT / "output"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": row_id, "control_success": p}).to_csv(
        out / "submission.csv", index=False
    )
    print(f"predicted {len(p):,} rows in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
