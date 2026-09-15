# -*- coding: utf-8 -*-
"""
[75] 7클래스 재측정 — 제구축 × 판정축, lr0.04, 스레드 정합 (exp/66과 완전 동일 조건)

★ 이건 새 축이 아니라 exp/62 재측정이다. HANDOFF §4-2 는 "7클래스 +10.9 — 임계 미달"로
  적어놨지만, 그 숫자는 **측정이 아니라 측정 인공물**이다:

    exp/62_mc7.py:34   NTHREAD = 8       <- mc7 을 스레드 8로 학습
    exp/53_multiclass.py:31 thread_count=14  <- 비교 기준 817.6 은 스레드 14로 학습
    lab/62_result.txt  판정 (2024 폴드, 시드42 **단일**)

  CatBoost는 thread_count가 바뀌면 결과가 바뀐다. 이 결함이 증류 실험에서 +37.3 짜리
  위양성을 만들었고 exp/64 가 스레드를 맞추자 +0.5 로 무너졌다(36.8점 인공물).
  거기에 시드 1개(σ=15.3)까지 겹쳐, +10.9 의 실제 오차범위는 ±29 수준이다.
  즉 이 축은 **기각된 적이 없다. 측정된 적이 없을 뿐이다.**

설계 (exp/66 과 단 두 곳만 다르다 — 나머지는 한 글자도 건드리지 않는다):
  · 라벨: 5클래스 -> 7클래스 (성공을 볼/스트라이크/기타로 세분, exp/62 와 동일 정의)
      0 성공&볼   1 성공&스트   2 성공&기타
      3 미들만    4 리버스만    5 미들∩리버스   6 빅미스
  · P(성공) = P0 + P1 + P2
  동일 유지: thread_count=14, SEEDS=[42,7], it500/d6/lr0.04/l2=10, 79피처, 같은 폴드

기준 벡터가 이미 디스크에 있다 — lab/66_mc_lr04.npy (MC04):
  시드42 842.3 / 시드7 848.9 / 시드평균 845.6 / 2시드 852.3 / K̄ 26.8 / 8시드추정 857.3

판정 (exp/74 의 교훈 반영 — 2시드 숫자만 보면 안 된다):
  ★ 시드평균 이득이 판정의 주축이다. 그것이 편향(=배포에서 살아남는 몫)이다.
    lr 축이 2시드에서 +15.3 로 보였다가 8시드에서 +3.8 로 사라진 게 그 증거다.
  proj8 = 1.75*S2 - 0.75*mean(solo)   (exp/74 에서 검증된 항등식)
  통과: 시드평균 이득 >= +10  ->  판정축은 살아있다, 배포 후보
  기각: 그 미만  ->  판정축 영구 종결. 9클래스 후속 금지.

실행: PYTHONIOENCODING=utf-8 python -u exp/75_mc7_lr04.py   (~1.3시간)
"""

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
CB_PRM = dict(iterations=500, depth=6, learning_rate=0.04, l2_leaf_reg=10.0,
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
               ("lab_brk", "asof_pitcher_breaking_rate"),
               ("lab_ball", "asof_pitcher_ball_rate"),
               ("lab_str", "asof_pitcher_strike_rate")]:
    S = dd[col].fillna(0).to_numpy("float64") * n
    lab = np.full(len(dd), np.nan)
    diff = np.round(S[1:] - S[:-1])
    lab[:-1] = np.where(nxt, diff, np.nan)
    dd[k] = np.where((lab == 0) | (lab == 1), lab, np.nan)
rec = dd.set_index("index").sort_index()
for k in ["lab_mid", "lab_rev", "lab_fb", "lab_brk", "lab_ball", "lab_str"]:
    df[k] = rec[k]
y_bin = df.control_success.to_numpy("float64")
# ★ 7클래스 (exp/62 와 동일 정의): 성공을 판정축으로 세분
#   0 성공&볼  1 성공&스트  2 성공&기타  3 미들만  4 리버스만  5 미들∩리버스  6 빅미스
cls = np.full(len(df), -1, dtype="int8")
ok = np.ones(len(df), dtype=bool)
for _k in ["lab_mid", "lab_rev", "lab_ball", "lab_str"]:
    ok &= df[_k].notna().to_numpy()
m_ = df.lab_mid.to_numpy() == 1
r_ = df.lab_rev.to_numpy() == 1
b_l = df.lab_ball.to_numpy() == 1
s_l = df.lab_str.to_numpy() == 1
cls[ok & (y_bin == 1) & b_l] = 0
cls[ok & (y_bin == 1) & s_l] = 1
cls[ok & (y_bin == 1) & ~b_l & ~s_l] = 2
cls[ok & (y_bin == 0) & m_ & ~r_] = 3
cls[ok & (y_bin == 0) & ~m_ & r_] = 4
cls[ok & (y_bin == 0) & m_ & r_] = 5
cls[ok & (y_bin == 0) & ~m_ & ~r_] = 6
df["_cls"] = cls
_names = ["성공&볼", "성공&스트", "성공&기타", "미들", "리버스", "미들∩리버스", "빅미스"]
log("7클래스 분포: " + "  ".join(f"{_names[c]} {np.mean(cls==c)*100:.1f}%"
                                for c in range(7))
    + f"  | 미복원 {np.mean(cls==-1)*100:.2f}%")
log(f"  정합 확인: P(cls 0~2) = {np.mean((cls >= 0) & (cls <= 2))*100:.2f}%"
    f"  vs 실제 성공률 {np.mean(y_bin == 1)*100:.2f}%")
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
log(f"기준 CS79 이진 2시드: {s_ref:.1f} / mc5 lr0.08 2시드: 837.0")

preds, solo = [], []
for sd in SEEDS:
    t0 = time.time()
    m = CatBoostClassifier(**CB_PRM, random_seed=sd).fit(ptr)
    proba = m.predict_proba(pva)
    p_succ = proba[:, 0] + proba[:, 1] + proba[:, 2]   # 성공 = 클래스 0,1,2
    preds.append(p_succ)
    solo.append(raw_score(p_succ, y_va))
    tick(f"seed={sd}  P(성공) 앙상블전 {solo[-1]:8.1f}  "
         f"({time.time()-t0:.0f}s)")
ens = np.mean(preds, axis=0)
np.save("lab/75_mc7_lr04.npy", ens.astype("float32"))

# ---- exp/74 판정 프레임: 편향(시드평균) / 분산(앙상블) 분리 ----
# 기준 = MC04 (lab/66_mc_lr04.npy), 완전 동일 조건에서 측정된 값
REF = dict(name="MC04(5클래스)", solo=[842.3, 848.9], e2=852.3)
REF["mean"] = sum(REF["solo"]) / 2
REF["K"] = 4 * (REF["e2"] - REF["mean"])
REF["p8"] = 1.75 * REF["e2"] - 0.75 * REF["mean"]

mean_solo = float(np.mean(solo))
e2 = raw_score(ens, y_va)
K = 4 * (e2 - mean_solo)
p8 = 1.75 * e2 - 0.75 * mean_solo
s_inf = mean_solo + K / 2
r_ = float(y_va.mean())
DEN = r_ * (1 - r_)
pen = 100000 * (ens.mean() - r_) ** 2 / DEN

log("")
log("=" * 84)
log("판정 — mc7 @ lr0.04, 스레드/시드 정합 (exp/62 의 교란 제거)")
log("=" * 84)
log(f"  {'':16s} {'시드평균':>10s} {'2시드':>9s} {'K̄':>8s} "
    f"{'8시드추정':>10s} {'S_무한':>9s}")
log(f"  {REF['name']:16s} {REF['mean']:10.1f} {REF['e2']:9.1f} {REF['K']:8.1f} "
    f"{REF['p8']:10.1f} {REF['mean']+REF['K']/2:9.1f}")
log(f"  {'mc7(7클래스)':16s} {mean_solo:10.1f} {e2:9.1f} {K:8.1f} "
    f"{p8:10.1f} {s_inf:9.1f}")
log("")
g_mean = mean_solo - REF["mean"]
g_2 = e2 - REF["e2"]
g_8 = p8 - REF["p8"]
log(f"  ★ 시드평균 이득 {g_mean:+7.1f}   <- 편향. 배포에서 살아남는 몫. 이게 판정 주축.")
log(f"    2시드 이득    {g_2:+7.1f}   <- 이 숫자만 보면 lr 축처럼 속는다")
log(f"    8시드추정 이득 {g_8:+7.1f}")
log(f"    변별력 {e2 + pen:.1f} / 중심벌점 {pen:.1f} (예측평균 {ens.mean():.4f}, r={r_:.4f})")
log("")
if g_mean >= 10:
    log(f"  ★통과 — 판정축(볼/스트라이크)에 실제 신호가 있다.")
    log(f"    다음: 8시드 배포 학습. 전이율 1.12 가정 시 LB 약 "
        f"{1033.99 + 1.12 * g_8:.0f} (8시드추정 경로)")
else:
    log(f"  기각 — 판정축 영구 종결. 9클래스 후속 금지.")
    log(f"    exp/62 의 +10.9 는 스레드 교란 인공물이었음이 확정된다.")
log("=" * 84)
