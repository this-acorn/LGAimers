# -*- coding: utf-8 -*-
"""
[78] MultiClassOneVsAll — 품질 후보가 아니라 **다양성 파트너**로서의 측정

왜 이걸 재는가 (exp/77 의 결론에서 직접 도출):
  우리는 사실상 단일모델이다. submit14 = CatBoost MultiClass 한 설정 × 시드 8개.
  2모델 혼합 이득에는 닫힌 해가 있다 —
      gain = (d + K)^2 / (4K),   d = S_파트너 − S_챔피언,  K = (1e5/DEN)·E[(p1−p2)^2]
  이 식이 스크리닝 기준을 바꾼다. 파트너는 **이길 필요가 없다**:
      K=117.5(계열 다름) 이면 d > −33.5 만 되어도 +15 이상
      K= 45.6(같은 계열) 이면 d > +6.7 이어야 +15
  따라서 후보마다 d 와 K 를 **함께** 재야 한다. d 만 보면 잘못 버린다.

OneVsAll 을 고른 이유:
  · 같은 5클래스·같은 79피처·같은 lr 인데 **손실함수만 다르다**
    (클래스별 독립 이진 학습 → 확률이 1로 합해지지 않는다)
  · 정보는 같고 오차 구조만 달라지므로, 품질은 비슷하고 K 는 클 가능성이 있다
    = 우리가 찾는 파트너의 정의 그 자체
  · exp/62·75(mc7), exp/51·72(CS reverse) 는 전부 '더 나은 챔피언'을 노렸다가 죽었다.
    이건 노리는 대상 자체가 다르다.

설계 (exp/66 과 단 한 곳만 다르다):
  loss_function: "MultiClass" -> "MultiClassOneVsAll"
  동일 유지: thread_count=14, SEEDS=[42,7], it500/d6/lr0.04/l2=10, 79피처, 같은 폴드
  P(성공) 은 두 가지로 낸다 — 정규화 전(raw)과 정규화 후(sum=1). OneVsAll 은 확률합이
  1이 아니므로 이 선택이 중심(calibration)을 바꾼다. 둘 다 재서 좋은 쪽을 쓴다.

기준 벡터가 디스크에 있다 — lab/66_mc_lr04.npy (MC04):
  시드42 842.3 / 시드7 848.9 / 시드평균 845.6 / 2시드 852.3 / K̄ 26.8 / 8시드추정 857.3

판정: 단독 점수가 아니라 **혼합 이득**으로 한다.
  통과: 로컬 혼합 이득 >= +15  ->  파트너 확보. 배포 학습 + 혼합 제출.
  기각: 그 미만  ->  혼합 축은 우리 손으로 만들 수 있는 범위에서 닫힌다.

실행: PYTHONIOENCODING=utf-8 python -u exp/78_ova_lr04.py   (~1.3시간)
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
              loss_function="MultiClassOneVsAll")
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
log(f"기준 CS79 이진 2시드: {s_ref:.1f} / mc5 lr0.08 2시드: 837.0")

preds, preds_n, solo, solo_n = [], [], [], []
for sd in SEEDS:
    t0 = time.time()
    m = CatBoostClassifier(**CB_PRM, random_seed=sd).fit(ptr)
    proba = m.predict_proba(pva)
    # OneVsAll 은 확률합이 1이 아니다 — 정규화 전/후를 둘 다 본다
    p_raw = proba[:, 0]
    p_nrm = proba[:, 0] / np.clip(proba.sum(axis=1), 1e-12, None)
    preds.append(p_raw)
    preds_n.append(p_nrm)
    solo.append(raw_score(p_raw, y_va))
    solo_n.append(raw_score(p_nrm, y_va))
    tick(f"seed={sd}  raw {solo[-1]:8.1f} / 정규화 {solo_n[-1]:8.1f}  "
         f"(확률합 평균 {proba.sum(axis=1).mean():.4f}, {time.time()-t0:.0f}s)")
ens_raw = np.mean(preds, axis=0)
ens_nrm = np.mean(preds_n, axis=0)
r_ = float(y_va.mean())
DEN = r_ * (1 - r_)

REF = np.load("lab/66_mc_lr04.npy").astype("float64")   # MC04 2시드
S_REF = raw_score(REF, y_va)
REF_SOLO = [842.3, 848.9]
REF_MEAN = sum(REF_SOLO) / 2
REF_K = 4 * (S_REF - REF_MEAN)
REF_P8 = 1.75 * S_REF - 0.75 * REF_MEAN
RATIO_2025 = 0.614          # K_2025 = 0.614 * K_local (exp/73, submit13 로 교정)
TAU = 1.12                  # 모델축 전이율 (submit14 실측)
LB_REF = 1033.99


def report(tag, ens, solos):
    mean_solo = float(np.mean(solos))
    e2 = raw_score(ens, y_va)
    K_self = 4 * (e2 - mean_solo)
    p8 = 1.75 * e2 - 0.75 * mean_solo
    pen = 100000 * (ens.mean() - r_) ** 2 / DEN
    D = float(np.mean((ens - REF) ** 2))
    K = 100000 * D / DEN
    d = e2 - S_REF
    g = (d + K) ** 2 / (4 * K) if K > 0 else 0.0
    w = min(max(0.5 + d / (2 * K), 0.0), 1.0) if K > 0 else 0.0
    log("")
    log(f"  [{tag}]")
    log(f"    시드 {' / '.join(f'{v:.1f}' for v in solos)}  시드평균 {mean_solo:.1f}  "
        f"2시드 {e2:.1f}  K̄(시드간) {K_self:.1f}  8시드추정 {p8:.1f}")
    log(f"    변별력 {e2 + pen:.1f} / 중심벌점 {pen:.1f} (예측평균 {ens.mean():.4f})")
    log(f"    ★ MC04 대비  d = {d:+.1f}   다양성 K_local = {K:.1f}  (D = {D:.4e})")
    log(f"      최적 가중 w*(파트너) = {w:.3f}   로컬 혼합 이득 = {g:+.1f}")
    K25 = K * RATIO_2025
    d25 = d * TAU
    g25 = (d25 + K25) ** 2 / (4 * K25) if K25 > 0 else 0.0
    log(f"      2025 환산(K×{RATIO_2025}, d×{TAU}): K={K25:.1f} d={d25:+.1f} "
        f"→ 혼합 이득 {g25:+.1f}  → LB 약 {LB_REF + g25:.0f}")
    return g, K, d


log("")
log("=" * 88)
log("판정 — MultiClassOneVsAll: 챔피언 교체가 아니라 혼합 파트너로서")
log("=" * 88)
log(f"  기준 MC04 (lab/66): 2시드 {S_REF:.1f}  시드평균 {REF_MEAN:.1f}  "
    f"K̄ {REF_K:.1f}  8시드추정 {REF_P8:.1f}")
g_raw, K_raw, d_raw = report("raw (정규화 전)", ens_raw, solo)
g_nrm, K_nrm, d_nrm = report("정규화 (합=1)", ens_nrm, solo_n)

best_tag, best_g, best_ens = (("raw", g_raw, ens_raw) if g_raw >= g_nrm
                              else ("정규화", g_nrm, ens_nrm))
np.save("lab/78_ova_lr04.npy", best_ens.astype("float32"))
log("")
log("=" * 88)
log(f"  승자: {best_tag}  로컬 혼합 이득 {best_g:+.1f}")
if best_g >= 15:
    log("  ★통과 — 혼합 파트너 확보. 배포 학습 후 가중 혼합 제출.")
    log("    ※ 파트너는 챔피언을 이길 필요가 없다. K 가 크면 져도 이득이다.")
else:
    log("  기각 — 손실함수만 바꾸는 것으로는 충분한 다양성이 안 나온다.")
    log("    혼합 축은 우리가 만들 수 있는 범위에서 닫힌다.")
    log("    (exp/77: 동급·최대다양성 파트너를 무한히 모아도 상한 +58.8)")
log("=" * 88)
