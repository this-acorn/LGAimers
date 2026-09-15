# -*- coding: utf-8 -*-
"""
[62] 타겟 분해 심화 — 5클래스 → 7클래스 (성공을 볼/스트라이크/기타로 세분)

근거: 이 축은 이미 +36.1(변별력 +31.1) 회수. exp/52에서 성공 내부도 분해 가능 확인:
  성공 중 ball 29.9% / strike 49.8% / 둘다아님 20.3% (ball·strike 상호배타, 둘다1은 0%)
가설: '의도적 볼로 유도한 성공'과 '잡은 스트라이크 성공'은 예측 구조가 다르다.
  분리 학습하면 softmax가 각 경로를 따로 배워 P(성공) 추정이 정밀해진다.

클래스 (상호배타):
  0 성공&볼   1 성공&스트라이크   2 성공&기타
  3 미들만    4 리버스만          5 미들∩리버스   6 빅미스
  P(성공) = P0+P1+P2

기준: exp/53 mc5 seed42 = 817.6 (같은 79피처·같은 시드). 2시드 앙상블은 837.0.
실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/62_mc7.py  (~45분)
"""

import importlib.util
import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import log, tick, raw_score, load_train

spec = importlib.util.spec_from_file_location("s12", "submissions/submit12_src/script.py")
s12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s12)

SEED = 42
NTHREAD = 8
REF_MC5_SEED42 = 817.6
REF_CS79_SEED42 = 777.0
K_MIX = 50.0
CB = dict(iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=10.0,
          verbose=False, thread_count=NTHREAD, allow_writing_files=False,
          loss_function="MultiClass", random_seed=SEED)

log("train 로딩...")
df = load_train()

# ---- 라벨 복원: 구종 + 실패유형 + 볼/스트라이크 ----
dd = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index()
pid = dd.pitcher_id.to_numpy()
nn = dd.asof_pitcher_n.fillna(0).to_numpy("float64")
nxt = (pid[1:] == pid[:-1]) & (np.abs(nn[1:] - nn[:-1] - 1) < 1e-6)
LABS = {"lab_fb": "asof_pitcher_fastball_rate", "lab_brk": "asof_pitcher_breaking_rate",
        "lab_mid": "asof_pitcher_middle_rate", "lab_rev": "asof_pitcher_reverse_rate",
        "lab_ball": "asof_pitcher_ball_rate", "lab_str": "asof_pitcher_strike_rate"}
for k, col in LABS.items():
    S = dd[col].fillna(0).to_numpy("float64") * nn
    lab = np.full(len(dd), np.nan)
    lab[:-1] = np.where(nxt, np.round(S[1:] - S[:-1]), np.nan)
    dd[k] = np.where((lab == 0) | (lab == 1), lab, np.nan)
rec = dd.set_index("index").sort_index()
for k in LABS:
    df[k] = rec[k]

y_bin = df.control_success.to_numpy("float64")
ok = np.ones(len(df), dtype=bool)
for k in ["lab_mid", "lab_rev", "lab_ball", "lab_str"]:
    ok &= df[k].notna().to_numpy()
m_ = df.lab_mid.to_numpy() == 1
r_ = df.lab_rev.to_numpy() == 1
b_l = df.lab_ball.to_numpy() == 1
s_l = df.lab_str.to_numpy() == 1
cls = np.full(len(df), -1, dtype="int8")
cls[ok & (y_bin == 1) & b_l] = 0
cls[ok & (y_bin == 1) & s_l] = 1
cls[ok & (y_bin == 1) & ~b_l & ~s_l] = 2
cls[ok & (y_bin == 0) & m_ & ~r_] = 3
cls[ok & (y_bin == 0) & ~m_ & r_] = 4
cls[ok & (y_bin == 0) & m_ & r_] = 5
cls[ok & (y_bin == 0) & ~m_ & ~r_] = 6
df["_cls7"] = cls
names = ["성공&볼", "성공&스트", "성공&기타", "미들", "리버스", "미들∩리버스", "빅미스"]
log("7클래스 분포: " + "  ".join(f"{names[c]} {np.mean(cls==c)*100:.1f}%" for c in range(7))
    + f"  | 미복원 {np.mean(cls==-1)*100:.2f}%")
# 정합성: 클래스 0~2 합 == 실제 성공률?
log(f"  정합 확인: P(cls 0~2) = {np.mean(np.isin(cls,[0,1,2]))*100:.2f}%  "
    f"vs 실제 성공률 {y_bin.mean()*100:.2f}%  (미복원 제외분만큼 차이)")
b2, s2 = df.balls_before.to_numpy(), df.strikes_before.to_numpy()
df["_cg"] = np.where(s2 > b2, 2, np.where(b2 > s2, 0, 1)).astype("int8")


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

msk = tr["_cls7"].to_numpy() >= 0
log(f"\n학습 행: {msk.sum():,} / {len(tr):,}")
Xtr = s12.build_matrix(tr[msk], FEATS)
Xva = s12.build_matrix(va, FEATS)
ptr = Pool(Xtr, tr["_cls7"].to_numpy()[msk], cat_features=list(s12.CAT))
pva = Pool(Xva, cat_features=list(s12.CAT))
del tr, va, Xtr, Xva
tick("Pool 생성")

t0 = time.time()
m = CatBoostClassifier(**CB).fit(ptr)
proba = m.predict_proba(pva)
p_succ = proba[:, 0] + proba[:, 1] + proba[:, 2]
np.save("lab/62_mc7_seed42.npy", p_succ.astype("float32"))
sc = raw_score(p_succ, y_va)
tick(f"학습 완료 ({time.time()-t0:.0f}s)")

r = y_va.mean(); DEN = r * (1 - r)
d = p_succ.mean() - r
pen = 100000 * d * d / DEN
log("\n" + "=" * 74)
log("판정 (2024 폴드, 시드42 단일)")
log("=" * 74)
log(f"  CS79 이진   {REF_CS79_SEED42:8.1f}")
log(f"  mc5         {REF_MC5_SEED42:8.1f}  (이진 대비 {REF_MC5_SEED42-REF_CS79_SEED42:+.1f})")
log(f"  mc7         {sc:8.1f}  (mc5 대비 {sc-REF_MC5_SEED42:+.1f})")
log(f"  mc7 분해: 변별력 {sc+pen:.1f} − 중심벌점 {pen:.1f}  (예측평균 {p_succ.mean():.4f})")
log("\n  +15↑ → 세분화 축 생존, 더 쪼갤 여지 탐색")
log("  ~0    → 5클래스가 이미 충분, 축 종료")
log("=" * 74)
