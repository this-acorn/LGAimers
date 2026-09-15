# -*- coding: utf-8 -*-
"""Hierarchical target pre-gate on the exact exp/89 CAT5 harness.

First arm (and the only implemented arm here):
    qS_new = P(success | reverse == 0)
    p_cond = (1 - qR_5class) * qS_new

This script deliberately does not create a deployment model or submission.  It
only measures whether a separate conditional-success tree is a useful partner
for the frozen exp/89 seed-matched five-class model on the 2024 fold.

Examples:
    python -u exp/codex_hier_head.py --seed 42 --smoke
    python -u exp/codex_hier_head.py --seed 42
"""

import argparse
import ctypes
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import load_train, raw_score


ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--threads", type=int, default=14)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

SEED = args.seed
NTHREAD = args.threads
SMOKE = args.smoke
K_MIX = 50.0
LAB = Path("lab")
LAB.mkdir(exist_ok=True)
SUFFIX = "_smoke" if SMOKE else ""
PREFIX = f"codex_hier_cond_seed{SEED}{SUFFIX}"
LIVE_PATH = LAB / f"{PREFIX}_live.log"
RESULT_PATH = LAB / f"{PREFIX}_result.txt"
JSON_PATH = LAB / f"{PREFIX}_result.json"
_T0 = time.time()


def emit(msg=""):
    line = str(msg)
    print(line, flush=True)
    with LIVE_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def tick(msg):
    emit(f"[{time.time() - _T0:7.1f}s] {msg}")


def set_below_normal_priority():
    """Leave Claude's High-priority training in control of CPU scheduling."""
    if os.name != "nt":
        return "not-windows"
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.SetPriorityClass.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        ok = kernel32.SetPriorityClass(handle, 0x00004000)
        return "BelowNormal" if ok else f"SetPriorityClass-failed:{ctypes.get_last_error()}"
    except Exception as exc:  # priority is courtesy, never a correctness gate
        return f"priority-error:{exc}"


def brier(p, y):
    p = np.asarray(p, dtype="float64")
    y = np.asarray(y, dtype="float64")
    return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-15, 1.0 - 1e-15)
    y = np.asarray(y, dtype="float64")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def score_parts(p, y):
    p = np.asarray(p, dtype="float64")
    y = np.asarray(y, dtype="float64")
    rate = float(y.mean())
    den = rate * (1.0 - rate)
    total = raw_score(p, y)
    penalty = 100000.0 * float((p.mean() - rate) ** 2) / den
    return {
        "score": float(total),
        "shape": float(total + penalty),
        "penalty": float(penalty),
        "mean": float(p.mean()),
    }


def mix_diagnostics(base, candidate, y):
    base = np.asarray(base, dtype="float64")
    candidate = np.asarray(candidate, dtype="float64")
    y = np.asarray(y, dtype="float64")
    rate = float(y.mean())
    den = rate * (1.0 - rate)
    d = float(raw_score(candidate, y) - raw_score(base, y))
    delta = candidate - base
    k = float((100000.0 / den) * np.mean(delta ** 2))
    if k <= 1e-15:
        w_star = 0.0
    else:
        w_star = float(np.clip((d + k) / (2.0 * k), 0.0, 1.0))
    p_star = base + w_star * delta
    gain_star = float(raw_score(p_star, y) - raw_score(base, y))
    fixed = {}
    for w in (0.1, 0.2, 0.3):
        fixed[f"{w:.1f}"] = float(raw_score(base + w * delta, y) - raw_score(base, y))
    return {
        "d": d,
        "K": k,
        "w_star": w_star,
        "gain_star": gain_star,
        "fixed": fixed,
        "p_star": p_star,
    }


def pitcher_bootstrap(base, candidate, y, pitcher_id, seed=20260830, n_boot=5000):
    """Cluster bootstrap paired score gain, kept on the global score scale."""
    base = np.asarray(base, dtype="float64")
    candidate = np.asarray(candidate, dtype="float64")
    y = np.asarray(y, dtype="float64")
    pitcher_id = np.asarray(pitcher_id)
    rate = float(y.mean())
    scale = 100000.0 / (len(y) * rate * (1.0 - rate))
    row_gain = scale * ((base - y) ** 2 - (candidate - y) ** 2)
    agg = (pd.DataFrame({"pitcher_id": pitcher_id, "gain": row_gain})
             .groupby("pitcher_id", sort=False)["gain"].sum().to_numpy("float64"))
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype="float64")
    # Batch to avoid making one unnecessarily large temporary index matrix.
    at = 0
    while at < n_boot:
        size = min(500, n_boot - at)
        idx = rng.integers(0, len(agg), size=(size, len(agg)))
        out[at:at + size] = agg[idx].sum(axis=1)
        at += size
    q = np.quantile(out, [0.025, 0.5, 0.975])
    return {
        "n_pitchers": int(len(agg)),
        "n_boot": int(n_boot),
        "q025": float(q[0]),
        "median": float(q[1]),
        "q975": float(q[2]),
        "p_gt_0": float(np.mean(out > 0)),
    }


LIVE_PATH.write_text("", encoding="utf-8")
priority = set_below_normal_priority()
emit(f"=== CODEX hierarchical conditional head seed={SEED} smoke={SMOKE} "
     f"threads={NTHREAD} priority={priority} ===")

spec = importlib.util.spec_from_file_location("s12", 'submissions/submit12_src/script.py')
s12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s12)

emit("train 로딩...")
df = load_train()

# ---- Reconstruct target5 labels exactly as exp/89. ----
dd = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index()
pid = dd.pitcher_id.to_numpy()
n = dd.asof_pitcher_n.fillna(0).to_numpy("float64")
nxt = (pid[1:] == pid[:-1]) & (np.abs(n[1:] - n[:-1] - 1) < 1e-6)
for key, col in [("lab_mid", "asof_pitcher_middle_rate"),
                 ("lab_rev", "asof_pitcher_reverse_rate"),
                 ("lab_fb", "asof_pitcher_fastball_rate"),
                 ("lab_brk", "asof_pitcher_breaking_rate")]:
    cumulative = dd[col].fillna(0).to_numpy("float64") * n
    label = np.full(len(dd), np.nan)
    diff = np.round(cumulative[1:] - cumulative[:-1])
    label[:-1] = np.where(nxt, diff, np.nan)
    dd[key] = np.where((label == 0) | (label == 1), label, np.nan)
rec = dd.set_index("index").sort_index()
for key in ["lab_mid", "lab_rev", "lab_fb", "lab_brk"]:
    df[key] = rec[key]
del dd, rec, pid, n, nxt
gc.collect()

y_bin = df.control_success.to_numpy("float64")
cls = np.full(len(df), -1, dtype="int8")
ok = df.lab_mid.notna().to_numpy() & df.lab_rev.notna().to_numpy()
mid = df.lab_mid.to_numpy() == 1
rev = df.lab_rev.to_numpy() == 1
cls[ok & (y_bin == 1)] = 0
cls[ok & (y_bin == 0) & mid & ~rev] = 1
cls[ok & (y_bin == 0) & ~mid & rev] = 2
cls[ok & (y_bin == 0) & mid & rev] = 3
cls[ok & (y_bin == 0) & ~mid & ~rev] = 4
df["_cls"] = cls
balls = df.balls_before.to_numpy()
strikes = df.strikes_before.to_numpy()
df["_cg"] = np.where(strikes > balls, 2, np.where(balls > strikes, 0, 1)).astype("int8")
emit("target5: " + " ".join(f"{c}:{np.mean(cls == c) * 100:.2f}%" for c in range(5))
     + f" missing:{np.mean(cls == -1) * 100:.3f}%")
del ok, mid, rev, balls, strikes


def build_const(src, id_col, n_col, rates):
    d = src.sort_values(n_col).groupby(id_col).tail(1)
    out = pd.DataFrame({"id": d[id_col].to_numpy()})
    n_last = d[n_col].fillna(0).to_numpy("float64")
    out["N_end"] = n_last + 1
    for key, col in rates.items():
        rate = d[col].fillna(0).to_numpy("float64")
        if key == "succ":
            out[f"S_{key}"] = np.round(rate * n_last) + d["control_success"].to_numpy("float64")
        else:
            out[f"S_{key}"] = rate * (n_last + 1)
    return out


def mix_asof_train(labeled):
    t = labeled.dropna(subset=["lab_fb"])
    count = (t.groupby(["pitcher_id", "season", "_cg"])
              .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
              .reset_index().sort_values(["pitcher_id", "_cg", "season"]))
    grouped = count.groupby(["pitcher_id", "_cg"])
    for col in ["n", "fb", "brk"]:
        count[f"p_{col}"] = grouped[col].cumsum() - count[col]
    overall = (t.groupby(["pitcher_id", "season"])
                .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
                .reset_index().sort_values(["pitcher_id", "season"]))
    grouped_overall = overall.groupby("pitcher_id")
    for col in ["n", "fb", "brk"]:
        overall[f"po_{col}"] = grouped_overall[col].cumsum() - overall[col]
    table = count.merge(overall[["pitcher_id", "season", "po_n", "po_fb", "po_brk"]],
                        on=["pitcher_id", "season"])
    over_fb = (table.po_fb + 0.5 * K_MIX) / (table.po_n + K_MIX)
    over_brk = (table.po_brk + 0.3 * K_MIX) / (table.po_n + K_MIX)
    table["mix_fb"] = np.where(table.po_n > 0,
                               (table.p_fb + K_MIX * over_fb) / (table.p_n + K_MIX),
                               np.nan).astype("float32")
    table["mix_brk"] = np.where(table.po_n > 0,
                                (table.p_brk + K_MIX * over_brk) / (table.p_n + K_MIX),
                                np.nan).astype("float32")
    return table[["pitcher_id", "season", "_cg", "mix_fb", "mix_brk"]]


def mix_career(labeled):
    t = labeled.dropna(subset=["lab_fb"])
    count = (t.groupby(["pitcher_id", "_cg"])
              .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
              .reset_index())
    overall = (t.groupby("pitcher_id")
                .agg(po_n=("lab_fb", "size"), po_fb=("lab_fb", "sum"),
                     po_brk=("lab_brk", "sum")).reset_index())
    table = count.merge(overall, on="pitcher_id")
    over_fb = (table.po_fb + 0.5 * K_MIX) / (table.po_n + K_MIX)
    over_brk = (table.po_brk + 0.3 * K_MIX) / (table.po_n + K_MIX)
    table["mix_fb"] = ((table.fb + K_MIX * over_fb) / (table.n + K_MIX)).astype("float32")
    table["mix_brk"] = ((table.brk + K_MIX * over_brk) / (table.n + K_MIX)).astype("float32")
    return table[["pitcher_id", "_cg", "mix_fb", "mix_brk"]]


# ---- Reproduce exp/89's 79 features and 2024 fold. ----
hist = df[df.season <= 2023]
prior = float(hist.control_success.mean())
cp = build_const(hist, "pitcher_id", "asof_pitcher_n", s12.P_RATES)
cb = build_const(hist, "batter_id", "asof_batter_n", s12.B_RATES)
pfb_rows = hist.dropna(subset=["lab_fb"])
pfb = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                         verbose=False, thread_count=NTHREAD,
                         allow_writing_files=False, random_seed=42)
pfb.fit(pfb_rows[s12.PFB_IN].fillna(-999), pfb_rows["lab_fb"].astype(int))
del pfb_rows
mix_train = mix_asof_train(hist)
mix_deploy = mix_career(hist)
tick("PFB 및 pitch-mix 표 준비 완료")

tr_rows = df[df.season <= 2023].reset_index(drop=True)
va_rows = df[df.season == 2024].reset_index(drop=True)
y_va = va_rows.control_success.to_numpy("float64")
cls_va = va_rows["_cls"].to_numpy("int8")
pitcher_va = va_rows.pitcher_id.to_numpy(copy=True)
parts = []
for season in sorted(tr_rows.season.unique()):
    history = df[df.season <= season - 1]
    rows = tr_rows[tr_rows.season == season]
    if len(history) == 0:
        cp_season = pd.DataFrame({"id": [], "N_end": [],
                                  **{f"S_{key}": [] for key in s12.P_RATES}})
        cb_season = pd.DataFrame({"id": [], "N_end": [],
                                  **{f"S_{key}": [] for key in s12.B_RATES}})
    else:
        cp_season = build_const(history, "pitcher_id", "asof_pitcher_n", s12.P_RATES)
        cb_season = build_const(history, "batter_id", "asof_batter_n", s12.B_RATES)
    parts.append(s12.attach_cs(rows, cp_season, cb_season))
tr = s12.add_features(pd.concat(parts).sort_index(), prior)
del parts, tr_rows, hist
gc.collect()
key = tr[["pitcher_id", "season", "_cg"]].merge(
    mix_train, on=["pitcher_id", "season", "_cg"], how="left")
tr["f_mixcg_fb"] = key.mix_fb.to_numpy("float32")
tr["f_mixcg_brk"] = key.mix_brk.to_numpy("float32")
tr["f_pfb"] = pfb.predict_proba(tr[s12.PFB_IN].fillna(-999))[:, 1].astype("float32")
del key
va = s12.attach_pt(s12.add_features(s12.attach_cs(va_rows, cp, cb), prior),
                   mix_deploy, pfb)
del va_rows, df, cp, cb, mix_train, mix_deploy, pfb
gc.collect()
tick("exp/89 피처 부착 완료")

BASE = [col for col in pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1)
        .columns.str.replace("ï»¿", "").str.strip() if col != "row_id"]
ENG18 = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
         "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
         "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
         "f_pb_diff", "f_miss_p", "f_miss_prev1"]
FEATS = BASE + ENG18 + s12.CS_FEATS + s12.PT_FEATS
CATS = list(s12.CAT) + ["pitcher_team_id", "batter_team_id"]
assert len(FEATS) == 79, f"exp/89 feature mismatch: {len(FEATS)}"
assert len(CATS) == 5, f"exp/89 categorical mismatch: {len(CATS)}"


def build(frame):
    out = pd.DataFrame(index=frame.index)
    for col in FEATS:
        if col in CATS:
            value = frame[col]
            out[col] = (value.fillna(-1).astype("int64").astype(str)
                        if pd.api.types.is_numeric_dtype(value) else value.astype(str))
        else:
            out[col] = frame[col]
    return out


# Frozen, seed-matched exp/89 decomposition.
p5_path = LAB / f"89_cat5_probs_seed{SEED}.npy"
if not p5_path.exists():
    raise SystemExit(f"missing frozen baseline: {p5_path}")
p5 = np.load(p5_path).astype("float64")
assert p5.shape == (len(va), 5), (p5.shape, len(va))
q_r5 = p5[:, 2] + p5[:, 3]
denom_s = np.clip(p5[:, 0] + p5[:, 1] + p5[:, 4], 1e-8, None)
q_s5 = p5[:, 0] / denom_s
identity_error = float(np.max(np.abs((1.0 - q_r5) * q_s5 - p5[:, 0])))
assert identity_error < 1e-7, identity_error
p_base = p5[:, 0].copy()
del p5, denom_s

cond_train_mask = np.isin(tr["_cls"].to_numpy(), [0, 1, 4])
cond_train_idx = np.flatnonzero(cond_train_mask)
y_cond_all = (tr["_cls"].to_numpy()[cond_train_idx] == 0).astype("int8")
cond_va_mask = np.isin(cls_va, [0, 1, 4])
y_cond_va = (cls_va[cond_va_mask] == 0).astype("int8")
emit(f"conditional train={len(cond_train_idx):,}/{len(tr):,} "
     f"valid={int(cond_va_mask.sum()):,}/{len(va):,} "
     f"train_rate={y_cond_all.mean():.6f} valid_rate={y_cond_va.mean():.6f}")
emit(f"frozen identity max_error={identity_error:.3e}")

if SMOKE:
    take = min(60000, len(cond_train_idx))
    pos = np.linspace(0, len(cond_train_idx) - 1, take, dtype="int64")
    selected_idx = cond_train_idx[pos]
    selected_y = y_cond_all[pos]
    iterations = 20
else:
    selected_idx = cond_train_idx
    selected_y = y_cond_all
    iterations = 500

Xtr = build(tr.iloc[selected_idx])
ptr = Pool(Xtr, selected_y, cat_features=CATS)
del Xtr, tr, cond_train_mask, cond_train_idx, y_cond_all, selected_idx, selected_y
gc.collect()
Xva = build(va)
pva = Pool(Xva, cat_features=CATS)
del Xva, va
gc.collect()
tick(f"Pool 완료; conditional rows={ptr.num_row():,}, iterations={iterations}")

params = dict(iterations=iterations, depth=6, learning_rate=0.04,
              l2_leaf_reg=10.0, verbose=False, thread_count=NTHREAD,
              allow_writing_files=False, loss_function="Logloss",
              random_seed=SEED)
t_fit = time.time()
model = CatBoostClassifier(**params).fit(ptr)
classes = list(model.classes_)
assert classes == [0, 1], f"unexpected classes: {classes}"
q_snew = model.predict_proba(pva)[:, classes.index(1)].astype("float64")
tick(f"conditional head 학습·예측 완료 ({time.time() - t_fit:.1f}s)")

p_cond = np.clip((1.0 - q_r5) * q_snew, 0.0, 1.0)
base_cond_brier = brier(q_s5[cond_va_mask], y_cond_va)
new_cond_brier = brier(q_snew[cond_va_mask], y_cond_va)
base_cond_logloss = logloss(q_s5[cond_va_mask], y_cond_va)
new_cond_logloss = logloss(q_snew[cond_va_mask], y_cond_va)
base_parts = score_parts(p_base, y_va)
candidate_parts = score_parts(p_cond, y_va)
mix = mix_diagnostics(p_base, p_cond, y_va)
shape_gain = candidate_parts["shape"] - base_parts["shape"]
center_gain = base_parts["penalty"] - candidate_parts["penalty"]
metric_gate = (new_cond_brier < base_cond_brier and
               new_cond_logloss < base_cond_logloss)
performance_gate = (shape_gain >= 5.0 and
                    max(mix["d"], mix["gain_star"]) >= 8.0)
passed = bool(metric_gate and performance_gate and not SMOKE)

bootstrap_raw = pitcher_bootstrap(p_base, p_cond, y_va, pitcher_va)
bootstrap_mix = pitcher_bootstrap(p_base, mix["p_star"], y_va, pitcher_va, seed=20260831)

q_path = LAB / f"{PREFIX}_qS.npy"
p_path = LAB / f"{PREFIX}_candidate.npy"
np.save(q_path, q_snew.astype("float32"))
np.save(p_path, p_cond.astype("float32"))

result = {
    "experiment": "hierarchical_conditional_success_head",
    "seed": SEED,
    "smoke": SMOKE,
    "threads": NTHREAD,
    "priority": priority,
    "iterations": iterations,
    "features": len(FEATS),
    "categoricals": len(CATS),
    "identity_max_abs_error": identity_error,
    "conditional_valid_n": int(cond_va_mask.sum()),
    "conditional_valid_rate": float(y_cond_va.mean()),
    "conditional_metrics": {
        "qS5_brier": base_cond_brier,
        "qSnew_brier": new_cond_brier,
        "brier_delta_new_minus_base": new_cond_brier - base_cond_brier,
        "qS5_logloss": base_cond_logloss,
        "qSnew_logloss": new_cond_logloss,
        "logloss_delta_new_minus_base": new_cond_logloss - base_cond_logloss,
    },
    "base": base_parts,
    "candidate": candidate_parts,
    "raw_gain": mix["d"],
    "center_gain": center_gain,
    "shape_gain": shape_gain,
    "K": mix["K"],
    "w_star": mix["w_star"],
    "optimal_mix_gain": mix["gain_star"],
    "fixed_mix_gain": mix["fixed"],
    "bootstrap_raw": bootstrap_raw,
    "bootstrap_optimal_mix": bootstrap_mix,
    "metric_gate": bool(metric_gate),
    "performance_gate": bool(performance_gate),
    "passed": passed,
    "verdict": ("SMOKE_ONLY" if SMOKE else
                "PASS_SEED42_RUN_SEED7" if passed else
                "FAIL_STOP_NO_SEED7"),
    "artifacts": {"qS": str(q_path), "candidate": str(p_path)},
}

lines = [
    f"CODEX hierarchical conditional head | seed={SEED} smoke={SMOKE}",
    f"identity max abs error: {identity_error:.3e}",
    (f"conditional Brier: {base_cond_brier:.9f} -> {new_cond_brier:.9f} "
     f"(new-base {new_cond_brier - base_cond_brier:+.9f})"),
    (f"conditional logloss: {base_cond_logloss:.9f} -> {new_cond_logloss:.9f} "
     f"(new-base {new_cond_logloss - base_cond_logloss:+.9f})"),
    (f"main score: {base_parts['score']:.3f} -> {candidate_parts['score']:.3f} "
     f"(raw {mix['d']:+.3f})"),
    (f"center contribution: {center_gain:+.3f} | same-mean shape gain: {shape_gain:+.3f} "
     f"| penalty {base_parts['penalty']:.3f}->{candidate_parts['penalty']:.3f}"),
    (f"mix: d={mix['d']:+.3f} K={mix['K']:.3f} w*={mix['w_star']:.4f} "
     f"gain*={mix['gain_star']:+.3f}"),
    ("fixed mix gains: " + " ".join(f"w={w}:{gain:+.3f}" for w, gain in mix["fixed"].items())),
    (f"pitcher bootstrap raw ({bootstrap_raw['n_boot']}): median={bootstrap_raw['median']:+.3f} "
     f"95%=[{bootstrap_raw['q025']:+.3f},{bootstrap_raw['q975']:+.3f}] "
     f"P(>0)={bootstrap_raw['p_gt_0']:.3f}"),
    (f"pitcher bootstrap mix ({bootstrap_mix['n_boot']}): median={bootstrap_mix['median']:+.3f} "
     f"95%=[{bootstrap_mix['q025']:+.3f},{bootstrap_mix['q975']:+.3f}] "
     f"P(>0)={bootstrap_mix['p_gt_0']:.3f}"),
    (f"gates: conditional_metrics={metric_gate} performance={performance_gate} "
     f"VERDICT={result['verdict']}"),
    f"saved: {q_path}, {p_path}, {JSON_PATH}, {RESULT_PATH}",
]
text = "\n".join(lines) + "\n"
RESULT_PATH.write_text(text, encoding="utf-8")
serializable = {key: value for key, value in result.items() if key != "p_star"}
JSON_PATH.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
emit("")
for line in lines:
    emit(line)
