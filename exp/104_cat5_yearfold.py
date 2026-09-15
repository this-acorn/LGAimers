# -*- coding: utf-8 -*-
"""
[104] Exact CAT5 past-year folds for the locked exp/103 V18-style correction.

The correction constants and gamma are loaded from lab/103_v18_pregate.json.
Nothing in this script tunes them.  It rebuilds the exact exp/89 79-feature,
five-class CatBoost for target years 2022 and 2023, applies the already locked
correction, and reports paired gains.  The saved 2024 exp/89 prediction remains
the final confirmation fold.

Outputs are resumable:
  lab/104_cat5_y{year}_probs_seed{seed}.npy
  lab/104_v18_effect_y{year}.npy
  lab/104_cat5_yearfold_summary.json

Run:
  PYTHONIOENCODING=utf-8 py -3.12 -u exp/104_cat5_yearfold.py \
      --years 2022,2023 --seeds 42,7 --threads 14
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

import sys

sys.path.insert(0, "exp")
from common import load_train, raw_score


ap = argparse.ArgumentParser()
ap.add_argument("--years", default="2022,2023")
ap.add_argument("--seeds", default="42,7")
ap.add_argument("--threads", type=int, default=14)
args = ap.parse_args()

YEARS = [int(value) for value in args.years.split(",") if value.strip()]
SEEDS = [int(value) for value in args.seeds.split(",") if value.strip()]
NTHREAD = int(args.threads)
if not YEARS or not SEEDS:
    raise ValueError("years and seeds must be non-empty")
if any(year not in (2021, 2022, 2023) for year in YEARS):
    raise ValueError("exp/104 only creates pre-2024 folds")


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S12_PATH = Path('submissions/submit12_src/script.py')
if not S12_PATH.is_file():
    # Repository cleanup moved historical source bundles under archive/src.
    # This is a path-only compatibility fallback; the fold/model contract is
    # still checked below and by exp/122 against the previously cached OOF.
    S12_PATH = Path('submissions/submit12_src/script.py')
s12 = load_module("s12_104", str(S12_PATH))
v18 = load_module("v18_103", "exp/103_v18_residual.py")

LOCK_PATH = Path("lab/103_v18_pregate.json")
SUMMARY_PATH = Path("lab/104_cat5_yearfold_summary.json")
if not LOCK_PATH.is_file():
    raise FileNotFoundError("run exp/103 first")
lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
if not lock.get("pregate_pass"):
    raise RuntimeError("exp/103 did not pass")
GAMMA = float(lock["selected_gamma"])
if not np.isclose(GAMMA, 0.30):
    raise AssertionError(f"unexpected locked gamma {GAMMA}; do not silently retune")

CB_PARAMS = dict(
    iterations=500,
    depth=6,
    learning_rate=0.08,
    l2_leaf_reg=10.0,
    verbose=False,
    thread_count=NTHREAD,
    allow_writing_files=False,
    loss_function="MultiClass",
)
K_MIX = 50.0
CAT5 = list(s12.CAT) + ["pitcher_team_id", "batter_team_id"]
ENG18 = [
    "f_count_state",
    "f_count_diff",
    "f_is_3ball",
    "f_is_2strike",
    "f_is_full",
    "f_same_hand",
    "f_scoring_pos",
    "f_any_runner",
    "f_log_pn",
    "f_log_bn",
    "f_smooth_p",
    "f_smooth_b",
    "f_form_dev1",
    "f_form_dev3",
    "f_form_trend",
    "f_pb_diff",
    "f_miss_p",
    "f_miss_prev1",
]
BASE = [
    column
    for column in pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1)
    .columns.str.replace("﻿", "")
    .str.strip()
    if column != "row_id"
]
FEATURES = BASE + ENG18 + s12.CS_FEATS + s12.PT_FEATS
if len(FEATURES) != 79 or len(CAT5) != 5:
    raise AssertionError(f"feature contract changed: {len(FEATURES)} / {len(CAT5)}")

STARTED = time.time()


def log(message: str = "") -> None:
    print(message, flush=True)


def tick(message: str) -> None:
    log(f"  [{time.time() - STARTED:7.0f}s] {message}")


def recover_auxiliary_labels(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["pitcher_id", "asof_pitcher_n"], kind="mergesort"
    ).reset_index()
    pitcher = ordered["pitcher_id"].to_numpy()
    count = ordered["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    next_pitch = (pitcher[1:] == pitcher[:-1]) & (
        np.abs(count[1:] - count[:-1] - 1.0) < 1e-6
    )
    for name, column in [
        ("lab_mid", "asof_pitcher_middle_rate"),
        ("lab_rev", "asof_pitcher_reverse_rate"),
        ("lab_fb", "asof_pitcher_fastball_rate"),
        ("lab_brk", "asof_pitcher_breaking_rate"),
    ]:
        cumulative = ordered[column].fillna(0).to_numpy(np.float64) * count
        label = np.full(len(ordered), np.nan)
        difference = np.round(cumulative[1:] - cumulative[:-1])
        label[:-1] = np.where(next_pitch, difference, np.nan)
        ordered[name] = np.where((label == 0.0) | (label == 1.0), label, np.nan)
    restored = ordered.set_index("index").sort_index()
    for name in ("lab_mid", "lab_rev", "lab_fb", "lab_brk"):
        frame[name] = restored[name]
    del ordered, restored, pitcher, count, next_pitch

    binary = frame["control_success"].to_numpy(np.float64)
    classes = np.full(len(frame), -1, dtype=np.int8)
    known = frame["lab_mid"].notna().to_numpy() & frame["lab_rev"].notna().to_numpy()
    middle = frame["lab_mid"].to_numpy() == 1
    reverse = frame["lab_rev"].to_numpy() == 1
    classes[known & (binary == 1)] = 0
    classes[known & (binary == 0) & middle & ~reverse] = 1
    classes[known & (binary == 0) & ~middle & reverse] = 2
    classes[known & (binary == 0) & middle & reverse] = 3
    classes[known & (binary == 0) & ~middle & ~reverse] = 4
    frame["_cls"] = classes
    balls = frame["balls_before"].to_numpy()
    strikes = frame["strikes_before"].to_numpy()
    frame["_cg"] = np.where(
        strikes > balls, 2, np.where(balls > strikes, 0, 1)
    ).astype(np.int8)
    frame["_pressure"] = v18.pressure_code(frame)
    frame["game_type"] = (
        frame["game_type"].astype("string").fillna("__MISSING__").astype(str)
    )
    frame["batter_hand"] = (
        frame["batter_hand"].astype("string").fillna("__MISSING__").astype(str)
    )
    del known, middle, reverse
    gc.collect()
    return frame


def build_const(
    source: pd.DataFrame, id_column: str, n_column: str, rates: dict[str, str]
) -> pd.DataFrame:
    last = source.sort_values(n_column).groupby(id_column).tail(1)
    output = pd.DataFrame({"id": last[id_column].to_numpy()})
    n_last = last[n_column].fillna(0).to_numpy(np.float64)
    output["N_end"] = n_last + 1.0
    for name, column in rates.items():
        rate = last[column].fillna(0).to_numpy(np.float64)
        if name == "succ":
            output[f"S_{name}"] = (
                np.round(rate * n_last)
                + last["control_success"].to_numpy(np.float64)
            )
        else:
            output[f"S_{name}"] = rate * (n_last + 1.0)
    return output


def empty_const(rates: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"id": [], "N_end": [], **{f"S_{name}": [] for name in rates}}
    )


def mix_asof_train(labeled: pd.DataFrame) -> pd.DataFrame:
    rows = labeled.dropna(subset=["lab_fb"])
    conditional = (
        rows.groupby(["pitcher_id", "season", "_cg"])
        .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
        .reset_index()
        .sort_values(["pitcher_id", "_cg", "season"])
    )
    group = conditional.groupby(["pitcher_id", "_cg"])
    for column in ("n", "fb", "brk"):
        conditional[f"p_{column}"] = group[column].cumsum() - conditional[column]
    overall = (
        rows.groupby(["pitcher_id", "season"])
        .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
        .reset_index()
        .sort_values(["pitcher_id", "season"])
    )
    group_overall = overall.groupby("pitcher_id")
    for column in ("n", "fb", "brk"):
        overall[f"po_{column}"] = group_overall[column].cumsum() - overall[column]
    table = conditional.merge(
        overall[["pitcher_id", "season", "po_n", "po_fb", "po_brk"]],
        on=["pitcher_id", "season"],
    )
    over_fb = (table["po_fb"] + 0.5 * K_MIX) / (table["po_n"] + K_MIX)
    over_brk = (table["po_brk"] + 0.3 * K_MIX) / (table["po_n"] + K_MIX)
    table["mix_fb"] = np.where(
        table["po_n"] > 0,
        (table["p_fb"] + K_MIX * over_fb) / (table["p_n"] + K_MIX),
        np.nan,
    ).astype(np.float32)
    table["mix_brk"] = np.where(
        table["po_n"] > 0,
        (table["p_brk"] + K_MIX * over_brk) / (table["p_n"] + K_MIX),
        np.nan,
    ).astype(np.float32)
    return table[["pitcher_id", "season", "_cg", "mix_fb", "mix_brk"]]


def mix_career(labeled: pd.DataFrame) -> pd.DataFrame:
    rows = labeled.dropna(subset=["lab_fb"])
    conditional = (
        rows.groupby(["pitcher_id", "_cg"])
        .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
        .reset_index()
    )
    overall = (
        rows.groupby("pitcher_id")
        .agg(po_n=("lab_fb", "size"), po_fb=("lab_fb", "sum"), po_brk=("lab_brk", "sum"))
        .reset_index()
    )
    table = conditional.merge(overall, on="pitcher_id")
    over_fb = (table["po_fb"] + 0.5 * K_MIX) / (table["po_n"] + K_MIX)
    over_brk = (table["po_brk"] + 0.3 * K_MIX) / (table["po_n"] + K_MIX)
    table["mix_fb"] = (
        (table["fb"] + K_MIX * over_fb) / (table["n"] + K_MIX)
    ).astype(np.float32)
    table["mix_brk"] = (
        (table["brk"] + K_MIX * over_brk) / (table["n"] + K_MIX)
    ).astype(np.float32)
    return table[["pitcher_id", "_cg", "mix_fb", "mix_brk"]]


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=data.index)
    for column in FEATURES:
        if column in CAT5:
            values = data[column]
            if pd.api.types.is_numeric_dtype(values):
                output[column] = values.fillna(-1).astype("int64").astype(str)
            else:
                output[column] = values.astype("string").fillna("__MISSING__").astype(str)
        else:
            output[column] = data[column]
    return output


def evaluate(
    probability: np.ndarray,
    effect: np.ndarray,
    target: np.ndarray,
    game_type: np.ndarray,
) -> dict[str, float]:
    base = np.asarray(probability, dtype=np.float64)
    candidate = np.clip(base + GAMMA * effect, 0.0, 1.0)
    f_mask = game_type == "F"
    return {
        "base_score": float(raw_score(base, target)),
        "candidate_score": float(raw_score(candidate, target)),
        "raw_gain": float(raw_score(candidate, target) - raw_score(base, target)),
        "shape_gain": float(v18.equal_mean_shape_gain(base, candidate, target)),
        "mean_shift": float(candidate.mean() - base.mean()),
        "F_contribution": float(v18.domain_contribution(base, candidate, target, f_mask)),
        "R_contribution": float(v18.domain_contribution(base, candidate, target, ~f_mask)),
    }


def read_summary() -> dict:
    if SUMMARY_PATH.is_file():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    return {
        "experiment": 104,
        "gamma_source": str(LOCK_PATH),
        "locked_gamma": GAMMA,
        "catboost": CB_PARAMS,
        "years": {},
    }


def save_summary(summary: dict) -> None:
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_year(year: int, summary: dict) -> None:
    log("")
    log("=" * 94)
    log(f"YEAR {year}: exact CAT5 train<= {year - 1}, locked gamma={GAMMA:.2f}")
    log("=" * 94)
    tick("loading train")
    frame = recover_auxiliary_labels(load_train())
    history = frame[frame["season"] <= year - 1]
    validation_rows = frame[frame["season"] == year].reset_index(drop=True)
    target = validation_rows["control_success"].to_numpy(np.float64)
    effect, effect_diag = v18.make_effect(history, validation_rows)
    effect_path = Path(f"lab/104_v18_effect_y{year}.npy")
    np.save(effect_path, effect.astype(np.float32))
    tick(
        f"strict correction effect ready; coverage hand/detail="
        f"{effect_diag['hand_coverage']:.3f}/{effect_diag['detail_coverage']:.3f}"
    )

    prior = float(history["control_success"].mean())
    pitcher_const = build_const(
        history, "pitcher_id", "asof_pitcher_n", s12.P_RATES
    )
    batter_const = build_const(
        history, "batter_id", "asof_batter_n", s12.B_RATES
    )
    pfb_rows = history.dropna(subset=["lab_fb"])
    pfb = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        verbose=False,
        thread_count=NTHREAD,
        allow_writing_files=False,
        random_seed=42,
    )
    pfb.fit(pfb_rows[s12.PFB_IN].fillna(-999), pfb_rows["lab_fb"].astype(int))
    del pfb_rows
    mix_train = mix_asof_train(history)
    mix_validation = mix_career(history)
    tick("constant tables, pitchmix and PFB ready")

    train_rows = frame[frame["season"] <= year - 1].reset_index(drop=True)
    parts = []
    for season in sorted(train_rows["season"].unique()):
        earlier = frame[frame["season"] <= season - 1]
        rows = train_rows[train_rows["season"] == season]
        if len(earlier) == 0:
            pitcher_season = empty_const(s12.P_RATES)
            batter_season = empty_const(s12.B_RATES)
        else:
            pitcher_season = build_const(
                earlier, "pitcher_id", "asof_pitcher_n", s12.P_RATES
            )
            batter_season = build_const(
                earlier, "batter_id", "asof_batter_n", s12.B_RATES
            )
        parts.append(s12.attach_cs(rows, pitcher_season, batter_season))
    train = s12.add_features(pd.concat(parts).sort_index(), prior)
    del parts, train_rows
    key = train[["pitcher_id", "season", "_cg"]].merge(
        mix_train, on=["pitcher_id", "season", "_cg"], how="left"
    )
    train["f_mixcg_fb"] = key["mix_fb"].to_numpy(np.float32)
    train["f_mixcg_brk"] = key["mix_brk"].to_numpy(np.float32)
    train["f_pfb"] = pfb.predict_proba(train[s12.PFB_IN].fillna(-999))[:, 1].astype(
        np.float32
    )
    del key, mix_train
    validation = s12.attach_pt(
        s12.add_features(
            s12.attach_cs(validation_rows, pitcher_const, batter_const), prior
        ),
        mix_validation,
        pfb,
    )
    game_type = validation_rows["game_type"].to_numpy()
    pitcher_ids = validation_rows["pitcher_id"].to_numpy()
    del validation_rows, history, frame, pitcher_const, batter_const, mix_validation, pfb
    gc.collect()
    tick("exact 79 features attached")

    train_mask = train["_cls"].to_numpy() >= 0
    class_target = train.loc[train_mask, "_cls"].to_numpy(np.int8)
    train_matrix = build_features(train.loc[train_mask])
    del train
    gc.collect()
    train_pool = Pool(train_matrix, class_target, cat_features=CAT5)
    del train_matrix, class_target
    validation_matrix = build_features(validation)
    del validation
    gc.collect()
    validation_pool = Pool(validation_matrix, cat_features=CAT5)
    del validation_matrix
    gc.collect()
    tick(f"Pool ready: train={train_pool.num_row():,} valid={validation_pool.num_row():,}")

    probabilities: list[np.ndarray] = []
    seed_results: dict[str, dict[str, float]] = {}
    for seed in SEEDS:
        prediction_path = Path(f"lab/104_cat5_y{year}_probs_seed{seed}.npy")
        if prediction_path.is_file():
            full_probability = np.load(prediction_path).astype(np.float64)
            if full_probability.shape != (len(target), 5):
                raise AssertionError(f"bad cached shape: {prediction_path}")
            tick(f"loaded cached {prediction_path.name}")
        else:
            model_started = time.time()
            model = CatBoostClassifier(**CB_PARAMS, random_seed=seed).fit(train_pool)
            full_probability = model.predict_proba(validation_pool)
            np.save(prediction_path, full_probability.astype(np.float32))
            del model
            gc.collect()
            tick(f"trained seed={seed} in {time.time() - model_started:.0f}s")
        success_probability = full_probability[:, 0]
        probabilities.append(success_probability)
        result = evaluate(success_probability, effect, target, game_type)
        seed_results[str(seed)] = result
        log(
            f"  seed={seed}: base={result['base_score']:.2f} "
            f"gain raw/shape={result['raw_gain']:+.2f}/{result['shape_gain']:+.2f} "
            f"F/R={result['F_contribution']:+.2f}/{result['R_contribution']:+.2f}"
        )
        del full_probability

    ensemble = np.mean(probabilities, axis=0)
    ensemble_result = evaluate(ensemble, effect, target, game_type)
    ensemble_candidate = np.clip(ensemble + GAMMA * effect, 0.0, 1.0)
    bootstrap = v18.cluster_bootstrap_gain(
        ensemble, ensemble_candidate, target, pitcher_ids, draws=5000, seed=104 + year
    )
    log(
        f"  ENSEMBLE: base={ensemble_result['base_score']:.2f} "
        f"gain raw/shape={ensemble_result['raw_gain']:+.2f}/{ensemble_result['shape_gain']:+.2f} "
        f"F/R={ensemble_result['F_contribution']:+.2f}/{ensemble_result['R_contribution']:+.2f}"
    )
    log(
        "  bootstrap median={median:+.2f} 95%=[{p025:+.2f},{p975:+.2f}] "
        "P(>0)={prob_positive:.4f}".format(**bootstrap)
    )

    year_pass = bool(
        all(result["raw_gain"] > 0.0 and result["shape_gain"] > 0.0 for result in seed_results.values())
        and ensemble_result["raw_gain"] >= 8.0
        and ensemble_result["shape_gain"] >= 5.0
        and ensemble_result["R_contribution"] >= 0.0
        and ensemble_result["F_contribution"] >= -5.0
        and bootstrap["p025"] > 0.0
    )
    log(f"  YEAR {year} GATE: {'PASS' if year_pass else 'FAIL'}")
    summary["years"][str(year)] = {
        "seeds": SEEDS,
        "seed_results": seed_results,
        "ensemble": ensemble_result,
        "bootstrap": bootstrap,
        "effect_diagnostics": effect_diag,
        "gate_pass": year_pass,
    }
    save_summary(summary)
    del train_pool, validation_pool, probabilities, ensemble, ensemble_candidate, effect
    gc.collect()


def main() -> None:
    log("=== exp/104 exact CAT5 year-fold validation ===")
    log(
        f"years={YEARS} seeds={SEEDS} threads={NTHREAD}; gamma={GAMMA:.2f} "
        "(locked by exp/103, no tuning here)"
    )
    summary = read_summary()
    for year in YEARS:
        run_year(year, summary)
    requested_pass = all(summary["years"][str(year)]["gate_pass"] for year in YEARS)
    summary["requested_years"] = YEARS
    summary["all_requested_pass"] = requested_pass
    summary["elapsed_seconds"] = time.time() - STARTED
    save_summary(summary)
    log("")
    log(f"EXP/104 FINAL: {'PASS' if requested_pass else 'FAIL'}")
    log("PASS authorizes deployment engineering only; it does not submit anything.")
    tick("finished")


if __name__ == "__main__":
    main()
