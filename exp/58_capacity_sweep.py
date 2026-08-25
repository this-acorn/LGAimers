# -*- coding: utf-8 -*-
"""
[58] 학습곡선 + 깊이 스윕 — 튜닝 공백을 정확히 메운다 (2024 폴드, CS79)

설계 정정(리뷰 반영):
  · "잎당 46행"은 오류였다 — 각 트리가 전체 행을 다시 본다.
    d6=23,000행/잎, d8=5,760행/잎 → 표본 부족 아님. 따라서 +133의 답이라 단정 금지,
    현실적 기대 +10~40. 이 실험의 목적은 "학습 길이·깊이 공백을 정확히 재는 것".
  · season weight는 용량 실험이 아니므로 제외 (별도 실험으로)
  · 조기종료 대신 staged_predict_proba로 **전체 학습곡선**을 뽑는다 —
    한 번의 학습으로 250/500/750...별 점수를 전부 얻어 어디서 평평해지는지 본다
  · 판정 임계 +15 (독립 시드 확증 전제)

팔 (시드 42, 기준 exp/45 both seed42 = 777.0):
  curve6  d6 / 2000 / lr0.03    깊이 고정, 학습 길이만 (현재 500이 충분한가)
  curve8  d8 / 1000 / lr0.04    깊은 상호작용 (미검증 축)

판독:
  최고점이 400~600 → 현재 500으로 충분, 길이 축 사망
  1000 이후에도 상승 → 길이 축 생존
  d8 곡선이 d6를 위로 뛰어넘음 → 깊이 부족이 진짜 병목
  d8이 올랐다 내려감 → 깊이보다 정규화(l2/rsm) 튜닝 필요

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/58_capacity_sweep.py
"""

import importlib.util
import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import log, tick, raw_score, load_train

spec = importlib.util.spec_from_file_location("s12", "submit12_src/script.py")
s12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s12)

SEED = 42
NTHREAD = 8
REF_SEED42 = 777.0
K_MIX = 50.0
BASE_CB = dict(l2_leaf_reg=10.0, verbose=False, thread_count=NTHREAD,
               allow_writing_files=False, random_seed=SEED)
ARMS = [("curve6", dict(iterations=2000, depth=6, learning_rate=0.03), 250),
        ("curve8", dict(iterations=1000, depth=8, learning_rate=0.04), 125)]

log("train 로딩...")
df = load_train()
dd = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index()
pid = dd.pitcher_id.to_numpy()
nn = dd.asof_pitcher_n.fillna(0).to_numpy("float64")
nxt = (pid[1:] == pid[:-1]) & (np.abs(nn[1:] - nn[:-1] - 1) < 1e-6)
for k, col in [("lab_fb", "asof_pitcher_fastball_rate"),
               ("lab_brk", "asof_pitcher_breaking_rate")]:
    S = dd[col].fillna(0).to_numpy("float64") * nn
    lab = np.full(len(dd), np.nan)
    lab[:-1] = np.where(nxt, np.round(S[1:] - S[:-1]), np.nan)
    dd[k] = np.where((lab == 0) | (lab == 1), lab, np.nan)
rec = dd.set_index("index").sort_index()
df["lab_fb"], df["lab_brk"] = rec["lab_fb"], rec["lab_brk"]
b_, s_ = df.balls_before.to_numpy(), df.strikes_before.to_numpy()
df["_cg"] = np.where(s_ > b_, 2, np.where(b_ > s_, 0, 1)).astype("int8")


def build_const(src, id_col, n_col, rates):
    d = src.sort_values(n_col).groupby(id_col).tail(1)
    out = pd.DataFrame({"id": d[id_col].to_numpy()})
    n_last = d[n_col].fillna(0).to_numpy("float64")
    out["N_end"] = n_last + 1
    for k, col in rates.items():
        r = d[col].fillna(0).to_numpy("float64")
        if k == "succ":
            out[f"S_{k}"] = np.round(r * n_last) + d["control_success"].to_numpy("float64")
        else:
            out[f"S_{k}"] = r * (n_last + 1)
    return out


def mix_tables(hist):
    t = hist.dropna(subset=["lab_fb"])
    c = (t.groupby(["pitcher_id", "season", "_cg"])
          .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
          .reset_index().sort_values(["pitcher_id", "_cg", "season"]))
    g = c.groupby(["pitcher_id", "_cg"])
    for col in ["n", "fb", "brk"]:
        c[f"p_{col}"] = g[col].cumsum() - c[col]
    o = (t.groupby(["pitcher_id", "season"])
          .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
          .reset_index().sort_values(["pitcher_id", "season"]))
    go = o.groupby("pitcher_id")
    for col in ["n", "fb", "brk"]:
        o[f"po_{col}"] = go[col].cumsum() - o[col]
    tb = c.merge(o[["pitcher_id", "season", "po_n", "po_fb", "po_brk"]],
                 on=["pitcher_id", "season"])
    ofb = (tb.po_fb + 0.5 * K_MIX) / (tb.po_n + K_MIX)
    obk = (tb.po_brk + 0.3 * K_MIX) / (tb.po_n + K_MIX)
    tb["mix_fb"] = np.where(tb.po_n > 0, (tb.p_fb + K_MIX * ofb) / (tb.p_n + K_MIX),
                            np.nan).astype("float32")
    tb["mix_brk"] = np.where(tb.po_n > 0, (tb.p_brk + K_MIX * obk) / (tb.p_n + K_MIX),
                             np.nan).astype("float32")
    cc = (t.groupby(["pitcher_id", "_cg"])
           .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
           .reset_index())
    oo = (t.groupby("pitcher_id").agg(po_n=("lab_fb", "size"), po_fb=("lab_fb", "sum"),
                                      po_brk=("lab_brk", "sum")).reset_index())
    dep = cc.merge(oo, on="pitcher_id")
    o1 = (dep.po_fb + 0.5 * K_MIX) / (dep.po_n + K_MIX)
    o2 = (dep.po_brk + 0.3 * K_MIX) / (dep.po_n + K_MIX)
    dep["mix_fb"] = ((dep.fb + K_MIX * o1) / (dep.n + K_MIX)).astype("float32")
    dep["mix_brk"] = ((dep.brk + K_MIX * o2) / (dep.n + K_MIX)).astype("float32")
    return tb[["pitcher_id", "season", "_cg", "mix_fb", "mix_brk"]], \
        dep[["pitcher_id", "_cg", "mix_fb", "mix_brk"]]


hist = df[df.season <= 2023]
prior = float(hist["control_success"].mean())
cp = build_const(hist, "pitcher_id", "asof_pitcher_n", s12.P_RATES)
cb_ = build_const(hist, "batter_id", "asof_batter_n", s12.B_RATES)
pr = hist.dropna(subset=["lab_fb"])
pfb = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=False,
                         thread_count=NTHREAD, allow_writing_files=False, random_seed=42)
pfb.fit(pr[s12.PFB_IN].fillna(-999), pr["lab_fb"].astype(int))
mix_tr, mix_dep = mix_tables(hist)

tr_rows = df[df.season <= 2023].reset_index(drop=True)
va_rows = df[df.season == 2024].reset_index(drop=True)
y_tr = tr_rows["control_success"].to_numpy()
y_va = va_rows["control_success"].to_numpy()
parts = []
for S in sorted(tr_rows.season.unique()):
    h = df[df.season <= S - 1]
    rows = tr_rows[tr_rows.season == S]
    if len(h) == 0:
        cpS = pd.DataFrame({"id": [], "N_end": [], **{f"S_{k}": [] for k in s12.P_RATES}})
        cbS = pd.DataFrame({"id": [], "N_end": [], **{f"S_{k}": [] for k in s12.B_RATES}})
    else:
        cpS = build_const(h, "pitcher_id", "asof_pitcher_n", s12.P_RATES)
        cbS = build_const(h, "batter_id", "asof_batter_n", s12.B_RATES)
    parts.append(s12.attach_cs(rows, cpS, cbS))
tr = s12.add_features(pd.concat(parts).sort_index(), prior)
k_ = tr[["pitcher_id", "season", "_cg"]].merge(mix_tr, on=["pitcher_id", "season", "_cg"],
                                               how="left")
tr["f_mixcg_fb"] = k_["mix_fb"].to_numpy("float32")
tr["f_mixcg_brk"] = k_["mix_brk"].to_numpy("float32")
tr["f_pfb"] = pfb.predict_proba(tr[s12.PFB_IN].fillna(-999))[:, 1].astype("float32")
va = s12.attach_pt(s12.add_features(s12.attach_cs(va_rows, cp, cb_), prior), mix_dep, pfb)
del parts

BASE = [c for c in pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1)
        .columns.str.replace("﻿", "").str.strip() if c != "row_id"]
ENG18 = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
         "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
         "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
         "f_pb_diff", "f_miss_p", "f_miss_prev1"]
FEATS = BASE + ENG18 + s12.CS_FEATS + s12.PT_FEATS
Xtr, Xva = s12.build_matrix(tr, FEATS), s12.build_matrix(va, FEATS)
del tr, va
tick(f"준비 완료 — 학습 {len(Xtr):,} / 검증 {len(Xva):,}, 피처 {len(FEATS)}")
log(f"기준: exp/45 both seed42 = {REF_SEED42} (it500/d6/lr0.08)\n")

ptr = Pool(Xtr, y_tr, cat_features=list(s12.CAT))
pva = Pool(Xva, cat_features=list(s12.CAT))
best = {}
for name, prm, period in ARMS:
    t0 = time.time()
    m = CatBoostClassifier(**BASE_CB, **prm).fit(ptr)
    tick(f"{name} 학습 완료 {prm} ({time.time()-t0:.0f}s) — 학습곡선 산출 중")
    curve = []
    for i, pr_ in enumerate(m.staged_predict_proba(pva, eval_period=period), 1):
        it = i * period
        s = raw_score(pr_[:, 1], y_va)
        curve.append((it, s))
    bi, bs = max(curve, key=lambda x: x[1])
    best[name] = (bi, bs)
    log(f"  [{name}] 학습곡선:")
    for it, s in curve:
        mark = "  ★" if it == bi else ""
        log(f"      it={it:>5d}  {s:8.1f}  ({s-REF_SEED42:+7.1f}){mark}")
    np.save(f"lab/58_{name}_final.npy",
            m.predict_proba(pva)[:, 1].astype("float32"))
    del m

log("\n" + "=" * 78)
log("판정 (2024 폴드, 시드42 단일 · 임계 +15 & 독립시드 확증 전제)")
log("=" * 78)
log(f"  ref(현재 it500/d6)  {REF_SEED42:8.1f}")
for k, (bi, bs) in best.items():
    log(f"  {k:10s} 최고 it={bi:<5d} {bs:8.1f}  ({bs-REF_SEED42:+.1f})")
log("\n  · 최고점이 400~600 → 500으로 충분, 길이 축 사망")
log("  · 1000 이후 상승 지속 → 길이 축 생존")
log("  · curve8이 curve6를 위로 넘음 → 깊이가 병목")
log("  · curve8이 올랐다 내려감 → 깊이보다 정규화(l2/rsm) 튜닝 필요")
log("=" * 78)
