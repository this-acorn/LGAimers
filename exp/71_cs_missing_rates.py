# -*- coding: utf-8 -*-
"""
[71] CS 미적용 rate 감사 — reverse / breaking / offspeed 의 CS 델타가 없다.

현행 script.py:
  P_RATES = {succ, ball, strike, middle, fb}        <- 5개만 CS 복원
  누락:      asof_pitcher_reverse_rate
             asof_pitcher_breaking_rate
             asof_pitcher_offspeed_rate

reverse는 타겟 실패 유형 3번(포수 요구 반대) 그 자체이고,
CS 복원은 지금까지 최대 이득(LB +92.3)을 낸 축이다. 빠져 있을 이유가 없다.

단, 합=1 제약이 있으면 선형 종속이라 신규 정보량이 적다. 그걸 먼저 잰다:
  succ + reverse + middle + ball + strike == 1 ?
  fb + breaking + offspeed == 1 ?
종속인 항은 "트리가 직접 못 만드는 선형결합"이라 넣을 값은 있지만 기대이득은 작고,
독립인 항은 진짜 신규 정보다.

학습 없음. 필요한 컬럼만 읽는다 (~100MB) — 배포 학습과 동시 실행 안전.
실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/71_cs_missing_rates.py   (~2분)
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "exp")
from common import log

FIVE = ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate"]
THREE = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
         "asof_pitcher_offspeed_rate"]
COLS = ["season", "pitcher_id", "asof_pitcher_n", "control_success"] + FIVE + THREE

log("train.csv 해당 컬럼만 로딩...")
df = pd.read_csv("data/train.csv", encoding="utf-8-sig", usecols=COLS)
df.columns = [c.replace("﻿", "").strip() for c in df.columns]
log(f"  {len(df):,}행 · {df.memory_usage(deep=True).sum() / 1024 ** 2:.0f} MB")

log("")
log("=" * 84)
log("A. 결측률 — 애초에 쓸 수 있는 컬럼인가")
log("=" * 84)
for c in FIVE + THREE:
    v = df[c]
    log(f"  {c:34s} 결측 {v.isna().mean() * 100:6.2f}%  "
        f"범위 {v.min():.4f}~{v.max():.4f}  평균 {v.mean():.4f}")

log("")
log("=" * 84)
log("B. 합=1 제약 — 선형 종속인가")
log("=" * 84)
for tag, grp in [("결과 5종 (succ+rev+mid+ball+strike)", FIVE),
                 ("구종 3종 (fb+brk+off)", THREE)]:
    S = df[grp].sum(axis=1)
    ok = df[grp].notna().all(axis=1)
    s = S[ok]
    log(f"  {tag}")
    log(f"    유효행 {ok.sum():,}  합 평균 {s.mean():.6f}  "
        f"std {s.std():.6f}  min {s.min():.6f}  max {s.max():.6f}")
    log(f"    합이 1인 비율 {np.isclose(s, 1.0, atol=1e-6).mean() * 100:6.2f}%")

log("")
log("=" * 84)
log("C. 누락 3종의 순증 정보량 — 나머지로 예측되는가 (선형 R²)")
log("=" * 84)


def r2_of(target, givens):
    m = df[[target] + givens].dropna()
    if len(m) < 1000:
        return np.nan, 0
    A = np.column_stack([m[g].to_numpy("float64") for g in givens]
                        + [np.ones(len(m))])
    y = m[target].to_numpy("float64")
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return 1 - resid.var() / y.var(), len(m)


for tgt, giv in [("asof_pitcher_reverse_rate",
                  [c for c in FIVE if c != "asof_pitcher_reverse_rate"]),
                 ("asof_pitcher_breaking_rate", ["asof_pitcher_fastball_rate"]),
                 ("asof_pitcher_offspeed_rate",
                  ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate"])]:
    r2, n_ = r2_of(tgt, giv)
    verdict = "종속(신규정보 없음)" if r2 > 0.999 else (
        "부분중복" if r2 > 0.9 else "★독립(신규정보)")
    log(f"  {tgt:34s} R²={r2:.6f}  n={n_:,}   {verdict}")
    log(f"      given: {', '.join(g.replace('asof_pitcher_', '') for g in giv)}")

log("")
log("=" * 84)
log("D. 타겟과의 직접 상관 — reverse가 볼 값이 있는가")
log("=" * 84)
y = df.control_success.to_numpy("float64")
for c in FIVE + THREE:
    v = df[c].to_numpy("float64")
    ok = ~np.isnan(v)
    log(f"  {c:34s} corr(타겟) {np.corrcoef(v[ok], y[ok])[0, 1]:+.4f}")

log("")
log("=" * 84)
log("판독")
log("=" * 84)
log("  · C에서 R²<0.9면 그 컬럼의 CS 델타는 진짜 신규 정보 → P_RATES 확장 가치 있음")
log("  · R²>0.999면 선형 종속 — 트리가 직접 못 만드는 결합이라 넣을 값은 있으나 기대 작음")
log("  · 통과 항목만 P_RATES에 추가해 exp/72에서 2시드 짝지음 측정")
log("=" * 84)
