# -*- coding: utf-8 -*-
"""
[89] 챔피언 구성(target5 + team_cat = cat5, lr0.08) 2024 폴드 검증 학습 — 에러 분석용

목적: 팀 최고 기록 submit_target5_teamcat(LB 1059.05, 하민 레포)의 로컬 검증 예측이
없다. 에러 분석(19번 과제)의 1차 재료로 챔피언 구성 그대로 2024 폴드 예측을 만든다.
  - exp/87(cat_features 확장)과 동일 하네스. ARMS 중 cat5 만 실행 (cat3 은 exp/53
    817.6/812.2 + lab/53_mc_ens.npy 로 이미 존재 — 재학습 안 함)
  - thread14 · 시드 [42, 7] · it500/d6/lr0.08/l2=10 · MultiClass 5클래스
  - ★ 5클래스 확률 전체를 시드별로 저장 (에러 분석에서 혼동 구조를 봐야 하므로)

저장:
  lab/89_cat5_probs_seed{42,7}.npy   (253507, 5) float32 — 5클래스 확률
  lab/89_cat5.npy                    (253507,) float32 — 2시드 평균 P(성공)
  lab/89_result.txt                  점수 + exp/53 cat3 대비 paired 차이

메모리 주의: RAM 15.7GB 중 여유 ~4GB. del + gc 를 적극적으로 쓴다.

실행: lab/runq89.ps1 이 python.exe 직접 실행 + High 우선순위 + 하트비트로 띄운다.
"""

import gc
import importlib.util
import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import log, tick, raw_score, load_train

spec = importlib.util.spec_from_file_location("s12", 'submissions/submit12_src/script.py')
s12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s12)

SEEDS = [42, 7]
CB_PRM = dict(iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=10.0,
              verbose=False, thread_count=14, allow_writing_files=False,
              loss_function="MultiClass")
K_MIX = 50.0
# exp/53 (cat3, thread14, 같은 폴드) 실측 — paired 기준선
CAT3_SOLO = {42: 817.6, 7: 812.2}
CAT3_ENS = 837.0

log("train 로딩...")
df = load_train()

# ---- 라벨 복원 (exp/87 과 동일) ----
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
del dd, rec, pid, n, nxt
gc.collect()

y_bin = df.control_success.to_numpy("float64")
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
del pfb_rows
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
del parts, tr_rows, hist
gc.collect()
key = tr[["pitcher_id", "season", "_cg"]].merge(
    mix_tr, on=["pitcher_id", "season", "_cg"], how="left")
tr["f_mixcg_fb"] = key["mix_fb"].to_numpy("float32")
tr["f_mixcg_brk"] = key["mix_brk"].to_numpy("float32")
tr["f_pfb"] = pfb.predict_proba(tr[s12.PFB_IN].fillna(-999))[:, 1].astype("float32")
del key
va = s12.attach_pt(s12.add_features(s12.attach_cs(va_rows, cp, cb_), prior),
                   mix_dep, pfb)
del va_rows, df
gc.collect()
tick("피처 부착 완료")

BASE = [c for c in pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1)
        .columns.str.replace("﻿", "").str.strip() if c != "row_id"]
ENG18 = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
         "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
         "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
         "f_pb_diff", "f_miss_p", "f_miss_prev1"]
FEATS = BASE + ENG18 + s12.CS_FEATS + s12.PT_FEATS
m_tr = tr["_cls"].to_numpy() >= 0
y_cls = tr["_cls"].to_numpy()[m_tr]
log(f"멀티클래스 학습 행: {m_tr.sum():,} / {len(tr):,}   피처 {len(FEATS)}")

CAT5 = list(s12.CAT) + ["pitcher_team_id", "batter_team_id"]


def build(d, cats):
    out = pd.DataFrame(index=d.index)
    for c in FEATS:
        if c in cats:
            v = d[c]
            out[c] = (v.fillna(-1).astype("int64").astype(str)
                      if pd.api.types.is_numeric_dtype(v) else v.astype(str))
        else:
            out[c] = d[c]
    return out


r_va = float(y_va.mean())
DEN = r_va * (1 - r_va)

Xtr = build(tr[m_tr], CAT5)
del tr
gc.collect()
ptr = Pool(Xtr, y_cls, cat_features=CAT5)
del Xtr
gc.collect()
Xva = build(va, CAT5)
pva = Pool(Xva, cat_features=CAT5)
del Xva, va
gc.collect()
tick("Pool 구성 완료")

preds, solo = [], []
for sd in SEEDS:
    t0 = time.time()
    m = CatBoostClassifier(**CB_PRM, random_seed=sd).fit(ptr)
    prob = m.predict_proba(pva)  # (253507, 5)
    np.save(f"lab/89_cat5_probs_seed{sd}.npy", prob.astype("float32"))
    p = prob[:, 0]
    preds.append(p)
    solo.append(raw_score(p, y_va))
    d_pair = solo[-1] - CAT3_SOLO[sd]
    tick(f"cat5 seed={sd}  {solo[-1]:8.1f}  (paired vs cat3 {d_pair:+.1f}, {time.time()-t0:.0f}s)")
    del m, prob
    gc.collect()

ens = np.mean(preds, axis=0)
np.save("lab/89_cat5.npy", ens.astype("float32"))
e2 = raw_score(ens, y_va)
ms = float(np.mean(solo))
p8 = 1.75 * e2 - 0.75 * ms
pen = 100000 * (ens.mean() - r_va) ** 2 / DEN
log("")
log("=" * 80)
log(f"cat5(챔피언 구성) 2시드 {e2:7.1f}  시드평균 {ms:7.1f}  proj8 {p8:7.1f}  벌점 {pen:5.1f}")
log(f"paired vs cat3(exp/53): 시드42 {solo[0]-CAT3_SOLO[42]:+.1f}  시드7 {solo[1]-CAT3_SOLO[7]:+.1f}"
    f"  2시드 {e2-CAT3_ENS:+.1f}")
log("하민 로컬 실측은 +2.0 (lr0.08, 다른 하네스), LB 실측 +25.06 — 크기가 아니라 방향/재현만 본다")
log("저장: lab/89_cat5.npy, lab/89_cat5_probs_seed42.npy, lab/89_cat5_probs_seed7.npy")
