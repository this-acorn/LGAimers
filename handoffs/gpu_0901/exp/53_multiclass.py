# -*- coding: utf-8 -*-
"""
[53] 타겟 분해 멀티클래스 — 실패 유형별 분리 학습 (장부 감사 1탄의 실행)

exp/52 확정: 타겟 = {성공, 미들, 리버스, 미들∩리버스, 빅미스} 5클래스 (상호배타 분할).
  성공에서 middle/reverse는 정확히 0% — 부분집합 관계 확인됨.
가설: 실패 유형별 원인이 다르므로(미들=카운트/공격성, 리버스=의도, 빅미스=난조)
  멀티클래스 분리 학습이 이진보다 신호를 더 뽑는다. P(성공) = 클래스0 확률.
라벨: asof 차분 복원 (succ 일치율 100% 검증). 커버리지 99.9%, 미복원 행은 학습 제외.

기준: lab/45_both_ens.npy (CS79 이진, 2시드 = 800.9) — 같은 79피처, 같은 시드.
실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/53_multiclass.py  (~50분)
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

SEEDS = [42, 7]
CB_PRM = dict(iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=10.0,
              verbose=False, thread_count=14, allow_writing_files=False,
              loss_function="MultiClass")
K_MIX = 50.0

log("train 로딩...")
df = load_train()

# ---- 라벨 복원 (succ/middle/reverse + 구종) ----
dd = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index()
pid = dd.pitcher_id.to_numpy()
n = dd.asof_pitcher_n.fillna(0).to_numpy("float64")
nxt = (pid[1:] == pid[:-1]) & (np.abs(n[1:] - n[:-1] - 1) < 1e-6)
for k, col in [("lab_mid", "asof_pitcher_middle_rate"),
               ("lab_rev", "asof_pitcher_reverse_rate"),
               ("lab_fb", "asof_pitcher_fastball_rate"),
               ("lab_brk", "asof_pitcher_breaking_rate")]:
    S = dd[col].fillna(0).to_numpy("float64") * n
    lab = np.full(len(dd), np.nan)
    diff = np.round(S[1:] - S[:-1])
    lab[:-1] = np.where(nxt, diff, np.nan)
    dd[k] = np.where((lab == 0) | (lab == 1), lab, np.nan)
rec = dd.set_index("index").sort_index()
for k in ["lab_mid", "lab_rev", "lab_fb", "lab_brk"]:
    df[k] = rec[k]
y_bin = df.control_success.to_numpy("float64")
# 5클래스: 0=성공 1=미들만 2=리버스만 3=미들∩리버스 4=빅미스
cls = np.full(len(df), -1, dtype="int8")
ok = df.lab_mid.notna().to_numpy() & df.lab_rev.notna().to_numpy()
m_ = df.lab_mid.to_numpy() == 1
r_ = df.lab_rev.to_numpy() == 1
cls[ok & (y_bin == 1)] = 0
cls[ok & (y_bin == 0) & m_ & ~r_] = 1
cls[ok & (y_bin == 0) & ~m_ & r_] = 2
cls[ok & (y_bin == 0) & m_ & r_] = 3
cls[ok & (y_bin == 0) & ~m_ & ~r_] = 4
df["_cls"] = cls
log(f"클래스 분포: " + " ".join(f"{c}:{np.mean(cls==c)*100:.1f}%" for c in range(5))
    + f"  미복원 {np.mean(cls==-1)*100:.2f}%")
b_, s_ = df.balls_before.to_numpy(), df.strikes_before.to_numpy()
df["_cg"] = np.where(s_ > b_, 2, np.where(b_ > s_, 0, 1)).astype("int8")

# ---- exp/48 검증 스테이지와 동일한 피처 준비 (79피처) ----
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


def mix_asof_train(labeled):
    t = labeled.dropna(subset=["lab_fb"])
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
    tbl = c.merge(o[["pitcher_id", "season", "po_n", "po_fb", "po_brk"]],
                  on=["pitcher_id", "season"])
    over_fb = (tbl.po_fb + 0.5 * K_MIX) / (tbl.po_n + K_MIX)
    over_brk = (tbl.po_brk + 0.3 * K_MIX) / (tbl.po_n + K_MIX)
    tbl["mix_fb"] = np.where(tbl.po_n > 0,
                             (tbl.p_fb + K_MIX * over_fb) / (tbl.p_n + K_MIX),
                             np.nan).astype("float32")
    tbl["mix_brk"] = np.where(tbl.po_n > 0,
                              (tbl.p_brk + K_MIX * over_brk) / (tbl.p_n + K_MIX),
                              np.nan).astype("float32")
    return tbl[["pitcher_id", "season", "_cg", "mix_fb", "mix_brk"]]


def mix_career(labeled):
    t = labeled.dropna(subset=["lab_fb"])
    c = (t.groupby(["pitcher_id", "_cg"])
          .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
          .reset_index())
    o = (t.groupby("pitcher_id")
          .agg(po_n=("lab_fb", "size"), po_fb=("lab_fb", "sum"),
               po_brk=("lab_brk", "sum")).reset_index())
    tbl = c.merge(o, on="pitcher_id")
    over_fb = (tbl.po_fb + 0.5 * K_MIX) / (tbl.po_n + K_MIX)
    over_brk = (tbl.po_brk + 0.3 * K_MIX) / (tbl.po_n + K_MIX)
    tbl["mix_fb"] = ((tbl.fb + K_MIX * over_fb) / (tbl.n + K_MIX)).astype("float32")
    tbl["mix_brk"] = ((tbl.brk + K_MIX * over_brk) / (tbl.n + K_MIX)).astype("float32")
    return tbl[["pitcher_id", "_cg", "mix_fb", "mix_brk"]]


hist = df[df.season <= 2023]
prior = float(hist["control_success"].mean())
cp = build_const(hist, "pitcher_id", "asof_pitcher_n", s12.P_RATES)
cb_ = build_const(hist, "batter_id", "asof_batter_n", s12.B_RATES)
pfb_rows = hist.dropna(subset=["lab_fb"])
pfb = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=False,
                         thread_count=14, allow_writing_files=False, random_seed=42)
pfb.fit(pfb_rows[s12.PFB_IN].fillna(-999), pfb_rows["lab_fb"].astype(int))
mix_tr = mix_asof_train(hist)
mix_dep = mix_career(hist)
tick("테이블·PFB 준비")

tr_rows = df[df.season <= 2023].reset_index(drop=True)
va_rows = df[df.season == 2024].reset_index(drop=True)
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
key = tr[["pitcher_id", "season", "_cg"]].merge(
    mix_tr, on=["pitcher_id", "season", "_cg"], how="left")
tr["f_mixcg_fb"] = key["mix_fb"].to_numpy("float32")
tr["f_mixcg_brk"] = key["mix_brk"].to_numpy("float32")
tr["f_pfb"] = pfb.predict_proba(tr[s12.PFB_IN].fillna(-999))[:, 1].astype("float32")
va = s12.attach_pt(s12.add_features(s12.attach_cs(va_rows, cp, cb_), prior),
                   mix_dep, pfb)
del parts

BASE = [c for c in pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1)
        .columns.str.replace("﻿", "").str.strip() if c != "row_id"]
ENG18 = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
         "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
         "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
         "f_pb_diff", "f_miss_p", "f_miss_prev1"]
FEATS = BASE + ENG18 + s12.CS_FEATS + s12.PT_FEATS

# 멀티클래스 학습: 라벨 복원된 행만
m_tr = tr["_cls"].to_numpy() >= 0
log(f"멀티클래스 학습 행: {m_tr.sum():,} / {len(tr):,}")
Xtr = s12.build_matrix(tr[m_tr], FEATS)
Xva = s12.build_matrix(va, FEATS)
ptr = Pool(Xtr, tr["_cls"].to_numpy()[m_tr], cat_features=list(s12.CAT))
pva = Pool(Xva, cat_features=list(s12.CAT))
del tr, va, Xtr, Xva

p_ref = np.load("lab/45_both_ens.npy").astype("float64")
s_ref = raw_score(p_ref, y_va)
log(f"기준 (CS79 이진, 2시드): {s_ref:.1f}")

preds = []
for sd in SEEDS:
    t0 = time.time()
    m = CatBoostClassifier(**CB_PRM, random_seed=sd).fit(ptr)
    proba = m.predict_proba(pva)
    p_succ = proba[:, 0]                       # 클래스0 = 성공
    preds.append(p_succ)
    tick(f"seed={sd}  P(성공) 앙상블전 {raw_score(p_succ, y_va):8.1f}  "
         f"({time.time()-t0:.0f}s)")
ens = np.mean(preds, axis=0)
np.save("lab/53_mc_ens.npy", ens.astype("float32"))
sc = raw_score(ens, y_va)
log("\n" + "=" * 70)
log(f"판정: 멀티클래스 P(성공) = {sc:.1f}  (이진 기준 {s_ref:.1f} 대비 {sc-s_ref:+.1f})")
log("  양수면: 분리학습이 신호를 더 뽑음 → 배포/혼합 후보")
log("  음수여도 D(이진, 멀티) 크면 혼합 재료 — 저장된 npy로 즉시 확인 가능")
log("=" * 70)
