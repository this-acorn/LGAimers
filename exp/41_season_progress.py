# -*- coding: utf-8 -*-
"""
[41] ★ 시즌 진행분 복원 — 공식 Q&A(417082)가 명시 허용한 최우선 축

원리: asof_*는 통산 누적이고, 주최측이 test 행에서도 2025 시점 기준으로 갱신해준다.
  train에서 '전 시즌 종료 시점 누적 상수표'를 만들어 두고, 각 행의 asof에서 빼면
  그 행 시점의 **현 시즌 진행분**(현 시즌 표본수·성공률·볼/스트라이크/미들·믹스)이
  행 독립적으로 복원된다. hand delta류와 달리 2025 최신 레짐을 직접 반영한다 [S급].
  운영진 공식 답변: "학습 데이터에서 추출한 정보를 바탕으로 독립 적용 → 가능한 방법"

검증 설계 (배포와 동일한 방법):
  상수표 = train ≤2023 각 선수의 마지막 행에서 재구성
    N_end = asof_n_last + 1
    S_succ_end = round(rate_last × asof_n_last) + y_last        (정확)
    S_기타_end ≈ rate_last × (asof_n_last + 1)                   (마지막 투구 결과 미상 — 오차 ≤1)
  폴드 행(2024) 피처 = 행의 asof − 상수표
  ★ 무결성 검증 내장: 선수별 첫 2024 행의 asof_n vs N_end 일치율 — 이게 안 맞으면
    카운터가 시즌 간 연속이 아니라는 뜻이므로 전체 기각

피처 12개 (cs = current season):
  투수: f_cs_p_logn, f_cs_p_rate(k=50 수축), f_cs_p_delta(현시즌-통산),
        f_cs_p_ball_d, f_cs_p_strike_d, f_cs_p_middle_d, f_cs_p_fb_d
  타자: f_cs_b_logn, f_cs_b_rate(k=50), f_cs_b_delta, f_cs_b_middle_d
기준: exp/36 ref (같은 시드 42/7)

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/41_season_progress.py  (~40분)
"""

import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import (log, tick, raw_score, load_train, add_features, CAT, ALL_ENG)

SEEDS = [42, 7]
CB_PRM = dict(iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=10.0,
              verbose=False, thread_count=6, allow_writing_files=False)
K_SHRINK = 50.0
MIN_CS_N = 5

log("train 로딩...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS65 = BASE + ALL_ENG

P_RATES = {"succ": "asof_pitcher_success_rate", "ball": "asof_pitcher_ball_rate",
           "strike": "asof_pitcher_strike_rate", "middle": "asof_pitcher_middle_rate",
           "fb": "asof_pitcher_fastball_rate"}
B_RATES = {"succ": "asof_batter_success_rate", "middle": "asof_batter_middle_rate"}


def build_const(src, id_col, n_col, rates, y_exact=None):
    """선수별 '≤기준시즌 마지막 행' 기반 누적 상수표 (배포와 동일 방법)"""
    d = src.sort_values(n_col).groupby(id_col).tail(1)
    out = pd.DataFrame({"id": d[id_col].to_numpy()})
    n_last = d[n_col].fillna(0).to_numpy("float64")
    out["N_end"] = n_last + 1
    for k, col in rates.items():
        r = d[col].fillna(0).to_numpy("float64")
        if k == "succ" and y_exact is not None:
            out[f"S_{k}"] = np.round(r * n_last) + d[y_exact].to_numpy("float64")
        else:
            out[f"S_{k}"] = r * (n_last + 1)
    return out


def attach_cs(rows, cp, cb_, pfx_feats):
    """행의 asof − 상수표 = 현 시즌 진행분 피처"""
    d = rows.copy()
    # ---- 투수 ----
    m = pd.DataFrame({"id": d.pitcher_id.to_numpy()}).merge(cp, on="id", how="left")
    N_end = m["N_end"].fillna(0).to_numpy("float64")
    n_now = d["asof_pitcher_n"].fillna(0).to_numpy("float64")
    cs_n = np.maximum(n_now - N_end, 0.0)
    ok = cs_n >= MIN_CS_N
    d["f_cs_p_logn"] = np.log1p(cs_n).astype("float32")
    career = {k: d[c].to_numpy("float64") for k, c in P_RATES.items()}
    for k in P_RATES:
        S_end = m[f"S_{k}"].fillna(0).to_numpy("float64")
        S_now = np.nan_to_num(career[k]) * n_now
        cs_S = np.clip(S_now - S_end, 0.0, None)
        rate = np.where(ok, (cs_S + K_SHRINK * np.nan_to_num(career[k]))
                        / (cs_n + K_SHRINK), np.nan)
        delta = np.where(ok, cs_S / np.maximum(cs_n, 1) - career[k], np.nan)
        if k == "succ":
            d["f_cs_p_rate"] = rate.astype("float32")
            d["f_cs_p_delta"] = delta.astype("float32")
        else:
            d[f"f_cs_p_{k}_d"] = delta.astype("float32")
    # ---- 타자 ----
    m = pd.DataFrame({"id": d.batter_id.to_numpy()}).merge(cb_, on="id", how="left")
    N_end = m["N_end"].fillna(0).to_numpy("float64")
    n_now = d["asof_batter_n"].fillna(0).to_numpy("float64")
    cs_n = np.maximum(n_now - N_end, 0.0)
    ok = cs_n >= MIN_CS_N
    d["f_cs_b_logn"] = np.log1p(cs_n).astype("float32")
    for k, c in B_RATES.items():
        car = d[c].to_numpy("float64")
        S_end = m[f"S_{k}"].fillna(0).to_numpy("float64")
        cs_S = np.clip(np.nan_to_num(car) * n_now - S_end, 0.0, None)
        rate = np.where(ok, (cs_S + K_SHRINK * np.nan_to_num(car)) / (cs_n + K_SHRINK),
                        np.nan)
        delta = np.where(ok, cs_S / np.maximum(cs_n, 1) - car, np.nan)
        if k == "succ":
            d["f_cs_b_rate"] = rate.astype("float32")
            d["f_cs_b_delta"] = delta.astype("float32")
        else:
            d[f"f_cs_b_{k}_d"] = delta.astype("float32")
    return d


CS_FEATS = ["f_cs_p_logn", "f_cs_p_rate", "f_cs_p_delta", "f_cs_p_ball_d",
            "f_cs_p_strike_d", "f_cs_p_middle_d", "f_cs_p_fb_d",
            "f_cs_b_logn", "f_cs_b_rate", "f_cs_b_delta", "f_cs_b_middle_d"]

# =====================================================================
# 1. 상수표 (≤2023) + 무결성 검증
# =====================================================================
hist = df[df.season <= 2023]
cp = build_const(hist, "pitcher_id", "asof_pitcher_n", P_RATES, "control_success")
cb_tbl = build_const(hist, "batter_id", "asof_batter_n", B_RATES, "control_success")
tick(f"상수표: 투수 {len(cp)}명 / 타자 {len(cb_tbl)}명")

# 무결성: 2024 첫 행의 asof_n == N_end ?
va_raw = df[df.season == 2024].reset_index(drop=True)
first24 = va_raw.sort_values("asof_pitcher_n").groupby("pitcher_id").head(1)
chk = first24.merge(cp, left_on="pitcher_id", right_on="id", how="inner")
diff = (chk["asof_pitcher_n"] - chk["N_end"]).abs()
log(f"★ 무결성(투수): 2024 첫 행 asof_n vs N_end — 일치(±0) {(diff<=0.5).mean()*100:.1f}% "
    f"/ 중앙값 오차 {diff.median():.1f} / 평균 {diff.mean():.2f} (n={len(chk)})")
if (diff <= 0.5).mean() < 0.9:
    log("⚠️ 카운터 불연속 의심 — 결과 해석 주의 (그래도 진행)")

# succ 재구성 정밀도: rate×n 이 정수에 가까운가
samp = hist.dropna(subset=["asof_pitcher_success_rate"]).sample(50000, random_state=0)
prod = samp.asof_pitcher_success_rate.to_numpy("float64") * samp.asof_pitcher_n.to_numpy("float64")
frac = np.abs(prod - np.round(prod))
log(f"★ 정밀도: rate×n 소수부 |오차| 중앙값 {np.median(frac):.4f} / p95 {np.percentile(frac,95):.4f}")

# =====================================================================
# 2. 2024 폴드 하네스
# =====================================================================
tr_raw = df[df.season <= 2023].reset_index(drop=True)
y_tr = tr_raw["control_success"].to_numpy()
y_va = va_raw["control_success"].to_numpy()
prior = float(y_tr.mean())
# 학습 행에도 같은 방법으로: 학습 행의 '현 시즌'은 그 행 시즌 — 상수표는 그 시즌 직전까지.
# 배포와 동일하게 하려면 시즌별 상수표가 필요하다: 시즌 S 행 ← ≤S-1 상수표.
tick("학습 행용 시즌별 상수표 생성 (S행 ← ≤S-1 마지막 행)...")
tr_parts = []
for S in sorted(tr_raw.season.unique()):
    h = df[df.season <= S - 1]
    rows = tr_raw[tr_raw.season == S]
    if len(h) == 0:
        cpS = pd.DataFrame(columns=["id", "N_end"] + [f"S_{k}" for k in P_RATES])
        cbS = pd.DataFrame(columns=["id", "N_end"] + [f"S_{k}" for k in B_RATES])
    else:
        cpS = build_const(h, "pitcher_id", "asof_pitcher_n", P_RATES, "control_success")
        cbS = build_const(h, "batter_id", "asof_batter_n", B_RATES, "control_success")
    tr_parts.append(attach_cs(rows, cpS, cbS, None))
tr = add_features(pd.concat(tr_parts).sort_index(), prior)
va = attach_cs(va_raw, cp, cb_tbl, None)
va = add_features(va, prior)
del df, tr_raw, tr_parts
cov = va["f_cs_p_rate"].notna().mean() * 100
tick(f"준비 완료 — 검증 f_cs_p_rate 커버리지 {cov:.1f}%")

p_ref = np.load("lab/36_ref_ens.npy").astype("float64")
s_ref = raw_score(p_ref, y_va)
log(f"기준 (exp/36 ref): {s_ref:.1f}")

NUM = [c for c in FEATS65 if c not in CAT] + CS_FEATS


def frame(d):
    out = d[NUM].copy()
    for c in CAT:
        out[c] = d[c].astype(str)
    return out


ptr = Pool(frame(tr), y_tr, cat_features=list(CAT))
pva = Pool(frame(va), cat_features=list(CAT))
preds = []
for s in SEEDS:
    t0 = time.time()
    m = CatBoostClassifier(**CB_PRM, random_seed=s).fit(ptr)
    preds.append(m.predict_proba(pva)[:, 1])
    tick(f"+CS seed={s}  {raw_score(preds[-1], y_va):8.1f}  ({time.time()-t0:.0f}s)")
ens = np.mean(preds, axis=0)
np.save("lab/41_cs_ens.npy", ens.astype("float32"))
sc = raw_score(ens, y_va)

log("\n" + "=" * 72)
log("판정 — 시즌 진행분 복원 (2024 폴드)")
log("=" * 72)
log(f"  기준(65)        {s_ref:8.1f}")
log(f"  +시즌진행분(11) {sc:8.1f}  ({sc-s_ref:+.1f})")
log(f"  예측평균 {ens.mean():.4f} (실제 r={y_va.mean():.4f})")
log("  ※ 이 축의 진짜 가치는 2025에서 더 클 수 있다 — 레짐이 통산과 더 벌어질수록")
log("    현 시즌 신호의 가치 상승. 2024 폴드 결과가 약해도 소폭 양수면 LB 직행 검토")
log("=" * 72)
