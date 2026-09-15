# -*- coding: utf-8 -*-
"""
[86] no-ID CatBoost — 혼합 파트너로서 d/K 확정 측정 (전체 데이터, 40만 프로브 아님)

왜 전체 데이터인가:
  40만 프로브는 d 에 데이터 페널티(약 -117)가 섞여 들어가 보정 논란이 남는다.
  no-ID 는 같은 CatBoost 라 exp/66 하네스를 그대로 쓸 수 있고 2시드 65분이면 끝난다.
  -> MC04(전체 2시드)와 **완전히 같은 조건**에서 d 와 K 를 잰다. 보정 불필요.

설계 (exp/66 과 단 한 곳만 다르다):
  FEATS 에서 pitcher_id, batter_id 두 개를 뺀다 (79 -> 77).
  동일 유지: MultiClass 5클래스, it500/d6/lr0.04/l2=10, thread 14, SEEDS[42,7], 같은 폴드.

왜 이게 마지막 후보인가 (08-27 프로브 결과):
    RealMLP    d=-491 K=635  로컬 +8.2  2025 +0.0
    ExtraTrees d=-332 K=446  로컬 +7.3  2025 +0.0
    FT 3종     d<-K          전부 0
    EBM        d=-413 K=315  0
  전부 d 가 너무 깊어 K 전이(0.614)를 못 견딘다. 승격 전선(2025 +15 기준):
    d=0 -> K_local 98 필요 / d=-20 -> 156 / d=-60 -> 256 / d=-330 -> 821
  no-ID 는 유일하게 d 가 얕을(-10~-30 예상) 후보다. 관건은 K 가 156 을 넘느냐.
  같은 계열 실측 K: 시드간 27 / 인접 피처셋 70 / 큰 피처셋 차이 191.
  ID 두 개는 중요도 최상위라 분할 구조가 크게 바뀔 수 있다 -> 잴 가치가 있다.

실행: PYTHONIOENCODING=utf-8 python -u exp/86_noid_full.py   (~65분)
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
FEATS_ALL = BASE + ENG18 + s12.CS_FEATS + s12.PT_FEATS
DROP_ID = ["pitcher_id", "batter_id"]
FEATS = [c for c in FEATS_ALL if c not in DROP_ID]
assert len(FEATS) == len(FEATS_ALL) - 2, "ID 컬럼 제거 실패"
log(f"no-ID: {len(FEATS_ALL)} -> {len(FEATS)} 피처 (제거 {DROP_ID})")

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
np.save("lab/86_noid_full.npy", ens.astype("float32"))
solo = [raw_score(p, y_va) for p in preds]

REF = np.load("lab/66_mc_lr04.npy").astype("float64")
S_REF = raw_score(REF, y_va)
r_ = float(y_va.mean())
DEN = r_ * (1 - r_)


def blend(d, K):
    if K <= 0:
        return 0.0, 0.0
    w = min(max(0.5 + d / (2 * K), 0.0), 1.0)
    return w, w * d + w * (1 - w) * K


s2 = raw_score(ens, y_va)
mean_solo = float(np.mean(solo))
p8 = 1.75 * s2 - 0.75 * mean_solo
pen = 100000 * (ens.mean() - r_) ** 2 / DEN
d = s2 - S_REF
D = float(np.mean((ens - REF) ** 2))
K = 100000 * D / DEN
w, g = blend(d, K)

log("")
log("=" * 84)
log("판정 — no-ID CatBoost, 전체 데이터 2시드 (MC04 와 완전 동일 조건)")
log("=" * 84)
log(f"  MC04 (79피처)   시드 842.3 / 848.9   2시드 {S_REF:.1f}   proj8 857.3")
log(f"  no-ID (77피처)  시드 {solo[0]:.1f} / {solo[1]:.1f}   2시드 {s2:.1f}   proj8 {p8:.1f}")
log(f"                  변별력 {s2 + pen:.1f} / 중심벌점 {pen:.1f}")
log("")
log(f"  d (격차)        {d:+8.1f}     <- 40만 보정 불필요, 같은 조건 직접 측정")
log(f"  K (다양성)      {K:8.1f}     (D = {D:.4e})")
log(f"  최적 가중 w*    {w:8.3f}")
log(f"  로컬 혼합 이득   {g:+8.1f}")
sens = []
for ratio in (0.4, 0.614, 0.8):
    _, gr = blend(d, K * ratio)
    sens.append(f"{ratio:g}배 {gr:+.1f}")
log("  2025 추정 이득   " + "   ".join(sens))
log("")
_, g614 = blend(d, K * 0.614)
if g614 >= 15:
    log("  ★통과 — 승격. 전체 배포 학습 후 단독 제출로 2025 전이 실측 -> 혼합 제출.")
elif g > 0:
    log("  △ 로컬은 양수지만 2025 환산에서 미달. K 전이율이 0.8 이상이어야 성립.")
    log("    이 경우 소량 가중 혼합을 1회 제출로 시험할지 결정 필요.")
else:
    log("  ✕ 기각 — 이종 파트너 축 소진. 남은 것은 submit14 확정.")
log("=" * 84)
