# -*- coding: utf-8 -*-
"""
[98] First-seen-pitcher specialist on top of the cat5 champion.

Train:
  - rows from each pitcher's first observed season, 2020-2023 only
  - 5-class target (M4 lost in exp/93)
  - current 79-feature basis, dropping only pitcher_id -> 78 features
  - CAT5 is unchanged, including both team IDs

Validate:
  - pitchers first observed in 2024 (50,348 rows / 81 pitchers)
  - champion elsewhere; champion/expert blends only on those rows

The full target-label reconstruction happens before the debut-row filter.
This is essential: filtering first would break the official-asof differencing.
"""

import argparse
import gc
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import log, tick, raw_score, load_train


ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true", help="20 trees, one seed, separate outputs")
args = ap.parse_args()

spec = importlib.util.spec_from_file_location("s12", 'submissions/submit12_src/script.py')
s12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s12)

SEEDS = [42] if args.smoke else [42, 7]
CB_PRM = dict(iterations=20 if args.smoke else 500, depth=6, learning_rate=0.08,
              l2_leaf_reg=10.0, verbose=False, thread_count=14,
              allow_writing_files=False, loss_function="MultiClass")
K_MIX = 50.0
OUT = "lab/98_smoke" if args.smoke else "lab/98"
WEIGHTS = (0.25, 0.50, 0.75, 1.00)


def build_const(src, id_col, n_col, rates):
    d = src.sort_values(n_col).groupby(id_col).tail(1)
    out = pd.DataFrame({"id": d[id_col].to_numpy()})
    n_last = d[n_col].fillna(0).to_numpy("float64")
    out["N_end"] = n_last + 1
    for key, col in rates.items():
        rate = d[col].fillna(0).to_numpy("float64")
        if key == "succ":
            out[f"S_{key}"] = (np.round(rate * n_last)
                                + d["control_success"].to_numpy("float64"))
        else:
            out[f"S_{key}"] = rate * (n_last + 1)
    return out


def mix_asof_train(labeled):
    t = labeled.dropna(subset=["lab_fb"])
    by_cell = (t.groupby(["pitcher_id", "season", "_cg"])
                .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"),
                     brk=("lab_brk", "sum"))
                .reset_index().sort_values(["pitcher_id", "_cg", "season"]))
    group = by_cell.groupby(["pitcher_id", "_cg"])
    for col in ["n", "fb", "brk"]:
        by_cell[f"p_{col}"] = group[col].cumsum() - by_cell[col]

    overall = (t.groupby(["pitcher_id", "season"])
                .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"),
                     brk=("lab_brk", "sum"))
                .reset_index().sort_values(["pitcher_id", "season"]))
    group_overall = overall.groupby("pitcher_id")
    for col in ["n", "fb", "brk"]:
        overall[f"po_{col}"] = group_overall[col].cumsum() - overall[col]

    table = by_cell.merge(
        overall[["pitcher_id", "season", "po_n", "po_fb", "po_brk"]],
        on=["pitcher_id", "season"])
    over_fb = (table.po_fb + 0.5 * K_MIX) / (table.po_n + K_MIX)
    over_brk = (table.po_brk + 0.3 * K_MIX) / (table.po_n + K_MIX)
    table["mix_fb"] = np.where(
        table.po_n > 0,
        (table.p_fb + K_MIX * over_fb) / (table.p_n + K_MIX),
        np.nan).astype("float32")
    table["mix_brk"] = np.where(
        table.po_n > 0,
        (table.p_brk + K_MIX * over_brk) / (table.p_n + K_MIX),
        np.nan).astype("float32")
    return table[["pitcher_id", "season", "_cg", "mix_fb", "mix_brk"]]


def mix_career(labeled):
    t = labeled.dropna(subset=["lab_fb"])
    by_cell = (t.groupby(["pitcher_id", "_cg"])
                .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"),
                     brk=("lab_brk", "sum")).reset_index())
    overall = (t.groupby("pitcher_id")
                .agg(po_n=("lab_fb", "size"), po_fb=("lab_fb", "sum"),
                     po_brk=("lab_brk", "sum")).reset_index())
    table = by_cell.merge(overall, on="pitcher_id")
    over_fb = (table.po_fb + 0.5 * K_MIX) / (table.po_n + K_MIX)
    over_brk = (table.po_brk + 0.3 * K_MIX) / (table.po_n + K_MIX)
    table["mix_fb"] = ((table.fb + K_MIX * over_fb)
                       / (table.n + K_MIX)).astype("float32")
    table["mix_brk"] = ((table.brk + K_MIX * over_brk)
                        / (table.n + K_MIX)).astype("float32")
    return table[["pitcher_id", "_cg", "mix_fb", "mix_brk"]]


def build_matrix(frame, features, cats):
    out = pd.DataFrame(index=frame.index)
    for col in features:
        if col in cats:
            values = frame[col]
            if pd.api.types.is_numeric_dtype(values):
                out[col] = values.fillna(-1).astype("int64").astype(str)
            else:
                out[col] = values.astype(str)
        else:
            out[col] = frame[col]
    return out


def route_score(champion, expert, target, rookie_mask, weight):
    pred = champion.copy()
    pred[rookie_mask] += weight * (expert[rookie_mask] - champion[rookie_mask])
    return raw_score(pred, target)


def route_terms(champion, expert, target, rookie_mask):
    rate = float(target.mean())
    den = rate * (1.0 - rate)
    scale = 100000.0 / (len(target) * den)
    d_gain = scale * float(np.sum(
        rookie_mask * ((champion - target) ** 2 - (expert - target) ** 2)))
    diversity_k = scale * float(np.sum(
        rookie_mask * (expert - champion) ** 2))
    if diversity_k <= 0:
        weight = float(d_gain > 0)
    else:
        weight = float(np.clip(0.5 + d_gain / (2.0 * diversity_k), 0.0, 1.0))
    return d_gain, diversity_k, weight


log(f"=== exp/98 rookie expert smoke={args.smoke} ===")
log("train loading...")
df = load_train()

# First observed season is computed from the complete official train history.
first_seen = df.groupby("pitcher_id")["season"].min()
df["_p_first"] = df["pitcher_id"].map(first_seen).astype("int16")

# Reconstruct labels on the full, unfiltered data.
ordered = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index()
pid = ordered.pitcher_id.to_numpy()
n_before = ordered.asof_pitcher_n.fillna(0).to_numpy("float64")
is_next = ((pid[1:] == pid[:-1])
           & (np.abs(n_before[1:] - n_before[:-1] - 1) < 1e-6))
for key, col in [("lab_mid", "asof_pitcher_middle_rate"),
                 ("lab_rev", "asof_pitcher_reverse_rate"),
                 ("lab_fb", "asof_pitcher_fastball_rate"),
                 ("lab_brk", "asof_pitcher_breaking_rate")]:
    cumulative = ordered[col].fillna(0).to_numpy("float64") * n_before
    label = np.full(len(ordered), np.nan)
    difference = np.round(cumulative[1:] - cumulative[:-1])
    label[:-1] = np.where(is_next, difference, np.nan)
    ordered[key] = np.where((label == 0) | (label == 1), label, np.nan)
restored = ordered.set_index("index").sort_index()
for key in ["lab_mid", "lab_rev", "lab_fb", "lab_brk"]:
    df[key] = restored[key]
del ordered, restored, pid, n_before, is_next
gc.collect()

y_all = df.control_success.to_numpy("float64")
classes = np.full(len(df), -1, dtype="int8")
valid_label = df.lab_mid.notna().to_numpy() & df.lab_rev.notna().to_numpy()
is_middle = df.lab_mid.to_numpy() == 1
is_reverse = df.lab_rev.to_numpy() == 1
classes[valid_label & (y_all == 1)] = 0
classes[valid_label & (y_all == 0) & is_middle & ~is_reverse] = 1
classes[valid_label & (y_all == 0) & ~is_middle & is_reverse] = 2
classes[valid_label & (y_all == 0) & is_middle & is_reverse] = 3
classes[valid_label & (y_all == 0) & ~is_middle & ~is_reverse] = 4
df["_cls"] = classes
balls = df.balls_before.to_numpy()
strikes = df.strikes_before.to_numpy()
df["_cg"] = np.where(strikes > balls, 2,
                     np.where(balls > strikes, 0, 1)).astype("int8")
del y_all, valid_label, is_middle, is_reverse, balls, strikes
gc.collect()

history = df[df.season <= 2023]
prior = float(history.control_success.mean())
constant_p = build_const(history, "pitcher_id", "asof_pitcher_n", s12.P_RATES)
constant_b = build_const(history, "batter_id", "asof_batter_n", s12.B_RATES)
pfb_rows = history.dropna(subset=["lab_fb"])
pfb_model = CatBoostClassifier(
    iterations=300, depth=6, learning_rate=0.1, verbose=False,
    thread_count=14, allow_writing_files=False, random_seed=42)
pfb_model.fit(pfb_rows[s12.PFB_IN].fillna(-999), pfb_rows["lab_fb"].astype(int))
del pfb_rows
mix_train = mix_asof_train(history)
mix_deploy = mix_career(history)
if mix_train.duplicated(["pitcher_id", "season", "_cg"]).any():
    raise ValueError("mix_train is not many-to-one on pitcher/season/count-group")
if mix_deploy.duplicated(["pitcher_id", "_cg"]).any():
    raise ValueError("mix_deploy is not many-to-one on pitcher/count-group")
tick("constants and PFB ready")

# Build features only for the requested 2020-2023 debut-season training rows.
parts = []
cohort_lines = []
for season in range(2020, 2024):
    prior_rows = df[df.season <= season - 1]
    rows = df[(df.season == season) & (df._p_first == season)]
    cp_season = build_const(prior_rows, "pitcher_id", "asof_pitcher_n", s12.P_RATES)
    cb_season = build_const(prior_rows, "batter_id", "asof_batter_n", s12.B_RATES)
    parts.append(s12.attach_cs(rows, cp_season, cb_season))
    cohort_lines.append((season, len(rows), rows.pitcher_id.nunique()))
rookie_train = s12.add_features(pd.concat(parts).sort_index(), prior)
del parts, prior_rows, cp_season, cb_season

key = rookie_train[["pitcher_id", "season", "_cg"]].merge(
    mix_train, on=["pitcher_id", "season", "_cg"], how="left",
    sort=False, validate="many_to_one")
if len(key) != len(rookie_train):
    raise ValueError("rookie-train mix merge changed row count")
rookie_train["f_mixcg_fb"] = key["mix_fb"].to_numpy("float32")
rookie_train["f_mixcg_brk"] = key["mix_brk"].to_numpy("float32")
rookie_train["f_pfb"] = pfb_model.predict_proba(
    rookie_train[s12.PFB_IN].fillna(-999))[:, 1].astype("float32")
del key, mix_train

valid_rows = df[df.season == 2024].reset_index(drop=True)
y_valid = valid_rows.control_success.to_numpy("float64")
rookie_mask = valid_rows._p_first.to_numpy() == 2024
alignment = np.load("lab/24b_preds.npz")
if not np.array_equal(y_valid.astype("int8"), alignment["2024_y"]):
    raise ValueError("2024 target order is not aligned with saved champion artifacts")
if not np.array_equal(rookie_mask, alignment["2024_meta_p_new"]):
    raise ValueError("2024 first-seen-pitcher mask differs from exp/24b")
if int(rookie_mask.sum()) != 50348:
    raise ValueError(f"expected 50,348 rookie rows, got {rookie_mask.sum():,}")
if int(valid_rows.loc[rookie_mask, "pitcher_id"].nunique()) != 81:
    raise ValueError("expected 81 first-seen pitchers in 2024")
del alignment
valid_features = s12.attach_pt(
    s12.add_features(s12.attach_cs(valid_rows, constant_p, constant_b), prior),
    mix_deploy, pfb_model)

log("training cohorts: " + " ".join(
    f"{year}:{rows:,} rows/{pitchers} pitchers"
    for year, rows, pitchers in cohort_lines))
log(f"train total={len(rookie_train):,} rows/"
    f"{rookie_train.pitcher_id.nunique()} pitchers; "
    f"valid rookie={rookie_mask.sum():,} rows/"
    f"{valid_rows.loc[rookie_mask, 'pitcher_id'].nunique()} pitchers")

base_columns = [
    col for col in pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1)
    .columns.str.replace("﻿", "").str.strip() if col != "row_id"]
engineered18 = [
    "f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
    "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
    "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
    "f_pb_diff", "f_miss_p", "f_miss_prev1"]
features79 = base_columns + engineered18 + s12.CS_FEATS + s12.PT_FEATS
features = [col for col in features79 if col != "pitcher_id"]
if len(features79) != 79 or len(features) != 78:
    raise ValueError(f"unexpected feature counts: {len(features79)} -> {len(features)}")
cats = list(s12.CAT) + ["pitcher_team_id", "batter_team_id"]

train_mask = rookie_train._cls.to_numpy() >= 0
y_class = rookie_train.loc[train_mask, "_cls"].to_numpy("int8")
log("class distribution: " + " ".join(
    f"{label}:{np.mean(y_class == label) * 100:.1f}%" for label in range(5)))
log(f"fit rows={train_mask.sum():,}/{len(rookie_train):,}; "
    f"features={len(features)} (pitcher_id removed); cats={cats}")

x_train = build_matrix(rookie_train.loc[train_mask], features, cats)
x_valid = build_matrix(valid_features.loc[rookie_mask], features, cats)
pool_train = Pool(x_train, y_class, cat_features=cats)
pool_valid = Pool(x_valid, cat_features=cats)
del x_train, x_valid, rookie_train, valid_features, history, df
gc.collect()
tick("Pools ready")

champion_ensemble = np.load("lab/89_cat5.npy").astype("float64")
if champion_ensemble.shape != y_valid.shape:
    raise ValueError("cat5 champion alignment mismatch")

expert_seed_full = []
champion_seed_full = []
for seed in SEEDS:
    start = time.time()
    model = CatBoostClassifier(**CB_PRM, random_seed=seed).fit(pool_train)
    probability = model.predict_proba(pool_valid)
    np.save(f"{OUT}_rookie_probs_seed{seed}.npy", probability.astype("float32"))
    full = champion_ensemble.copy()
    full[rookie_mask] = probability[:, 0]
    expert_seed_full.append(full)

    champion_seed = np.load(f"lab/89_cat5_probs_seed{seed}.npy")[:, 0].astype("float64")
    champion_seed_full.append(champion_seed)
    d_gain, diversity_k, optimum_weight = route_terms(
        champion_seed, full, y_valid, rookie_mask)
    gains = [route_score(champion_seed, full, y_valid, rookie_mask, weight)
             - raw_score(champion_seed, y_valid) for weight in WEIGHTS]
    tick(f"seed={seed} d={d_gain:+.2f} K={diversity_k:.2f} w*={optimum_weight:.3f} "
         + " ".join(f"w{weight:g}={gain:+.2f}"
                    for weight, gain in zip(WEIGHTS, gains))
         + f" ({time.time() - start:.0f}s)")
    del model, probability
    gc.collect()

expert_ensemble = np.mean(expert_seed_full, axis=0)
np.save(f"{OUT}_rookie_expert.npy", expert_ensemble.astype("float32"))
np.save(f"{OUT}_rookie_mask.npy", rookie_mask)

base_score = raw_score(champion_ensemble, y_valid)
d_gain, diversity_k, optimum_weight = route_terms(
    champion_ensemble, expert_ensemble, y_valid, rookie_mask)
all_weights = [*WEIGHTS, optimum_weight]

log("")
log("=" * 96)
log("rookie expert routing result (global-equivalent full-2024 points)")
log("=" * 96)
log(f"cat5 champion 2-seed={base_score:.2f}; d={d_gain:+.2f}; K={diversity_k:.2f}; "
    f"analytic w*={optimum_weight:.3f}")
for weight in all_weights:
    ensemble_score = route_score(
        champion_ensemble, expert_ensemble, y_valid, rookie_mask, weight)
    solo_scores = [route_score(champion_seed_full[i], expert_seed_full[i],
                               y_valid, rookie_mask, weight)
                   for i in range(len(SEEDS))]
    seed_mean = float(np.mean(solo_scores))
    if len(SEEDS) == 2:
        projected8 = 1.75 * ensemble_score - 0.75 * seed_mean
        projected_gain = projected8 - 889.0
        projection = f" proj8={projected8:.2f} (gain {projected_gain:+.2f})"
    else:
        projection = ""
    log(f"w={weight:.3f}: score={ensemble_score:.2f} gain={ensemble_score-base_score:+.2f} "
        f"seedmean={seed_mean:.2f}{projection}")
log(f"saved {OUT}_rookie_expert.npy and per-seed 5-class probabilities")
log("Run exp/97_rookie_route_audit.py for pitcher bootstrap and asof_n buckets.")
log("=" * 96)
