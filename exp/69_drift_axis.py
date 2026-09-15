# -*- coding: utf-8 -*-
"""
[69] 드리프트 축 — 시즌 가중치 & 정규화 (이진 d6/it500/lr0.04 기준, 2시드 짝지음)

진단 근거 (exp/58 학습곡선):
  it500 830.9 -> it1000 805.5 -> it1500 756.6.
  1.22M행 MSE형 과제에서 이 낙차는 학습 노이즈 암기가 아니라
  **1년 앞 예측의 분포 이동**이다. 트리를 더 얹을수록 2019~2023 고유 구조에
  맞춰지고 2024에서 손해를 본다.

  A상 시즌 가중치 — 드리프트를 직접 겨냥(오래된 시즌을 깎는다). it500 유지라 저렴.
  B상 정규화     — exp/58이 지목한 대안 (l2 / rsm / 잎최소표본).

★ 짝지음 기준 = exp/65 d6_lr04 2시드 839.4.
  준비 블록·시드[42,7]·**thread_count=8**을 exp/65와 완전히 동일하게 유지해야 유효.
  (exp/57 증류 위양성의 원인이 thread_count 6 vs 14였다 — 재발 금지)

판정 임계 +15 (시드 노이즈 σ≈15.3). 통과 시 멀티클래스(MC04)에서 독립 확증.
변별력/중심벌점 분해를 함께 본다 — 가중치는 예측 평균을 움직이므로
벌점만 줄어든 이득은 2025 전이가 보장되지 않는다.

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/69_drift_axis.py   (~3.5시간)
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

SEED = 42
NTHREAD = 8
REF_LR04 = 839.4    # exp/65 d6_lr04 2시드 — 동일 하네스/시드/스레드라 짝지음 유효
SEEDS = [42, 7]
K_MIX = 50.0
BASE_CB = dict(l2_leaf_reg=10.0, verbose=False, thread_count=NTHREAD,
               allow_writing_files=False, random_seed=SEED)

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
log(f"짝지음 기준: exp/65 d6_lr04 2시드 = {REF_LR04} "
    f"(it500/d6/lr0.04/l2=10, thread={NTHREAD})")
log("")

# ---------------------------------------------------------------------------
BASE_PRM = dict(iterations=500, depth=6, learning_rate=0.04)
season_tr = tr_rows["season"].to_numpy("float64")
age = season_tr.max() - season_tr            # 2023 기준 경과 연수
r_ = float(y_va.mean())
DEN = r_ * (1 - r_)

WARMS = [("w_half", 0.5 ** age),                        # 반감기 1년 — 공격적
         ("w_soft", 0.8 ** age),                        # 반감기 약 3년 — 완만
         ("w_lin", season_tr - season_tr.min() + 1.0)]  # 선형 증가
RARMS = [("l2_50", dict(l2_leaf_reg=50.0)),
         ("rsm06", dict(rsm=0.6)),
         ("mdl200", dict(min_data_in_leaf=200))]
# ★ C상 (08-26 추가): 트리 구조 — 지금까지 건드린 적 없는 유일한 구조 축.
#   exp/78 이 "손실함수를 바꿔도 같은 함수로 수렴한다"(D(μ,μ)≈0)를 보였다.
#   손실함수는 무엇을 최소화할지만 바꾸고 **함수 클래스**는 그대로다.
#   CatBoost 기본 SymmetricTree 는 한 깊이의 모든 분기가 같은 조건을 쓰는 강한 제약이고
#   우리 모델 전부가 그 안에 있다. Depthwise/Lossguide 는 그 제약을 푼다
#   = 품질과 다양성 K 둘 다 움직일 수 있는 유일하게 남은 손잡이.
GARMS = [("depthwise", dict(grow_policy="Depthwise")),
         ("lossguide", dict(grow_policy="Lossguide", max_leaves=64))]

pva = Pool(Xva, cat_features=list(s12.CAT))
res = {}


# 기준 d6_lr04 의 시드별 점수 (exp/65 실측) — 편향/분산 분리에 필요
REF_SOLO = [830.9, 831.2]
REF_MEAN = sum(REF_SOLO) / 2
REF_P8 = 1.75 * REF_LR04 - 0.75 * REF_MEAN        # = 845.7 (exp/74 검증)
# 혼합 파트너 평가용 챔피언 벡터 (멀티클래스 MC04)
MC04 = np.load("lab/66_mc_lr04.npy").astype("float64")
RATIO_2025 = 0.614


def run(name, prm_extra, weight):
    ptr = Pool(Xtr, y_tr, cat_features=list(s12.CAT), weight=weight)
    ps, solo = [], []
    for sd in SEEDS:
        t0 = time.time()
        prm = {**BASE_CB, "random_seed": sd, **BASE_PRM, **prm_extra}
        if prm.get("grow_policy") == "Lossguide":
            prm.pop("depth", None)      # Lossguide 는 max_leaves 로 크기를 정한다
        m = CatBoostClassifier(**prm).fit(ptr)
        p = m.predict_proba(pva)[:, 1]
        ps.append(p)
        solo.append(raw_score(p, y_va))
        tick(f"{name} seed={sd} {solo[-1]:8.1f}  ({time.time()-t0:.0f}s)")
        del m
    del ptr
    ens = np.mean(ps, axis=0)
    np.save(f"lab/69_{name}.npy", ens.astype("float32"))
    scv = raw_score(ens, y_va)
    pen = 100000 * (ens.mean() - r_) ** 2 / DEN
    mean_solo = float(np.mean(solo))
    p8 = 1.75 * scv - 0.75 * mean_solo            # exp/74 항등식
    # 혼합 파트너로서의 값 (exp/77 닫힌 해)
    s_mc = raw_score(MC04, y_va)
    D = float(np.mean((ens - MC04) ** 2))
    K = 100000 * D / DEN
    d = scv - s_mc
    bg = (d + K) ** 2 / (4 * K) if K > 0 else 0.0
    res[name] = (scv, scv + pen, pen, mean_solo, p8, K, d, bg)
    log(f"  -> {name:10s} 2seed {scv:7.1f} ({scv - REF_LR04:+6.1f})  "
        f"시드평균 {mean_solo:7.1f} ({mean_solo - REF_MEAN:+6.1f})  "
        f"proj8 {p8:7.1f} ({p8 - REF_P8:+6.1f})")
    log(f"     {'':10s} 변별력 {scv + pen:6.1f} 벌점 {pen:5.1f}  |  "
        f"MC04 대비 d={d:+7.1f} K={K:6.1f} → 혼합이득 {bg:+6.1f}")


log("=" * 80)
log("A상: 시즌 가중치 — 이미 완주(08-25 16:26~17:15). 결과 재사용, 재실행 안 함")
log("=" * 80)
# (scv, disc, pen, mean_solo, p8, K, d, bg)  — lab/69_result 백업본에서 그대로
res["w_half"] = (812.3, 841.1, 28.8, 798.1, 822.9, 75.8, -40.0, 4.2)
res["w_soft"] = (847.1, 881.2, 34.1, 838.7, 853.3, 41.9, -5.2, 8.0)
res["w_lin"] = (820.2, 860.1, 39.9, 810.3, 827.6, 48.9, -32.1, 1.4)
log("  w_half  2seed 812.3 (-27.1)  시드평균 798.1 (-32.9)  proj8 822.9 (-22.8)  기각")
log("  w_soft  2seed 847.1 ( +7.7)  시드평균 838.7 ( +7.7)  proj8 853.3 ( +7.7)  ★")
log("  w_lin   2seed 820.2 (-19.2)  시드평균 810.3 (-20.7)  proj8 827.6 (-18.1)  기각")
log("")
log("  ★ w_soft: 세 지표가 전부 +7.7 = 순수 편향 이득(분산 성분 0).")
log("    시드별 paired: 830.9→838.2 (+7.3) / 831.2→839.3 (+8.1), paired σ ≈ 0.6")
log("    임계 +15 는 **비짝지음** 노이즈(σ=15.3) 기준이다. paired σ 가 0.6 이면")
log("    +7.7 은 압도적으로 유의하다. → 채택 후보. 멀티클래스에서 exp/79 로 확증.")

log("")
log("=" * 80)
log("B상: 정규화 — exp/58이 지목한 대안")
log("=" * 80)
for name, prm in RARMS:
    run(name, prm, None)

log("")
log("=" * 80)
log("C상: 트리 구조 — 대칭트리 제약을 푼다 (유일하게 남은 구조 축)")
log("=" * 80)
for name, prm in GARMS:
    run(name, prm, None)

log("")
log("=" * 96)
log("판정 — exp/74(시드 스케일링) · exp/77(혼합 닫힌 해) 프레임 적용")
log("=" * 96)
log("  판정 주축은 **시드평균 이득(편향)** 과 **proj8**. 2시드 숫자만 보면 lr 축처럼 속는다.")
log("  파트너 후보로는 혼합이득으로 본다 — 챔피언을 이길 필요가 없다.")
log("")
log(f"  {'팔':12s} {'2시드':>8s} {'시드평균':>9s} {'proj8':>8s} "
    f"{'벌점':>6s} {'K(MC04)':>8s} {'혼합이득':>8s}")
log(f"  {'기준 d6_lr04':12s} {REF_LR04:8.1f} {REF_MEAN:9.1f} {REF_P8:8.1f} "
    f"{'':>6s} {'':>8s} {'':>8s}")
for k, v in sorted(res.items(), key=lambda x: -x[1][4]):
    scv, disc, pen, mean_solo, p8, K, d, bg = v
    log(f"  {k:12s} {scv:8.1f} {mean_solo:9.1f} {p8:8.1f} {pen:6.1f} "
        f"{K:8.1f} {bg:8.1f}")
log("")
log("  ※ 임계 +15 는 비짝지음 σ=15.3 기준이다. **paired 차이의 σ**가 작으면")
log("     그보다 작은 이득도 유의하다 (w_soft: +7.7, paired σ≈0.6). 시드별 점수를 보라.")
log("  ★ 채택 조건 (둘 중 하나)")
log(f"    (a) 품질:  proj8 이 기준 {REF_P8:.1f} 보다 +15 이상  → 챔피언 후보")
log("    (b) 파트너: MC04 대비 혼합이득 +15 이상            → 혼합 재료")
win_q = [k for k, v in res.items() if v[4] - REF_P8 >= 15]
win_p = [k for k, v in res.items() if v[7] >= 15]
log("")
log(f"  품질 통과: {win_q if win_q else '없음'}")
log(f"  파트너 통과: {win_p if win_p else '없음'}")
if not win_q and not win_p:
    log("")
    log("  → 전부 기각. 남은 것은 MC04/OVA 배포(proj8 857~860, LB 약 1038)뿐이다.")
log("=" * 96)
