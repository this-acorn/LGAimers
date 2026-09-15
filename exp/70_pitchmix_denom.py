# -*- coding: utf-8 -*-
"""
[70] 구종 분모 감사 — lab_fb 복원과 CS fb가 asof_pitcher_n을 분모로 쓰고 있다.
     진짜 분모가 asof_pitcher_pitchmix_n이라면 둘 다 틀렸다.

의심 근거:
  · 대회 컬럼 그룹에서 fastball/breaking/offspeed_rate는 pitchmix_n과 한 묶음이다.
  · 그런데 exp/65·submit12 경로는 두 곳 모두 asof_pitcher_n을 곱한다:
      exp/65  S = asof_pitcher_fastball_rate * asof_pitcher_n   -> 연속행 차분 -> lab_fb
      script  attach_cs()의 P_RATES["fb"]도 n_now = asof_pitcher_n 사용
  · 분모가 틀리면 차분값이 0/1에 안 떨어져 대부분 NaN으로 버려진다.
    = 구종 라벨 표본이 조용히 잘려 있었다는 뜻 (BOTH가 +18.6에 그친 이유 후보).

이 스크립트는 학습을 하지 않는다. 7개 컬럼만 읽어 ~60MB로 끝난다
(exp/67 배포 학습과 동시에 돌려도 안전).

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/70_pitchmix_denom.py   (~2분)
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "exp")
from common import log

COLS = ["season", "pitcher_id", "asof_pitcher_n", "asof_pitcher_pitchmix_n",
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate", "asof_pitcher_success_rate"]

log("train.csv 해당 컬럼만 로딩...")
df = pd.read_csv("data/train.csv", encoding="utf-8-sig", usecols=COLS)
df.columns = [c.replace("﻿", "").strip() for c in df.columns]
log(f"  {len(df):,}행 · {df.memory_usage(deep=True).sum()/1024**2:.0f} MB")

n = df.asof_pitcher_n.fillna(0).to_numpy("float64")
pm = df.asof_pitcher_pitchmix_n.fillna(0).to_numpy("float64")

log("")
log("=" * 84)
log("A. 두 카운터는 같은 것인가")
log("=" * 84)
same = np.isclose(n, pm)
log(f"  asof_pitcher_n == asof_pitcher_pitchmix_n : {same.mean()*100:6.2f}% 행")
log(f"  n  범위 {n.min():.0f}~{n.max():.0f}   평균 {n.mean():.1f}")
log(f"  pm 범위 {pm.min():.0f}~{pm.max():.0f}   평균 {pm.mean():.1f}")
d = n - pm
log(f"  차이(n−pm) 평균 {d.mean():.2f}  중앙 {np.median(d):.1f}  "
    f"최대 {d.max():.0f}  음수비율 {(d < 0).mean()*100:.2f}%")
log(f"  pm/n 비율 (n>0) 중앙 {np.median(pm[n > 0]/n[n > 0]):.4f}")
log(f"  pitchmix_n 결측 {df.asof_pitcher_pitchmix_n.isna().mean()*100:.2f}%  "
    f"/ n 결측 {df.asof_pitcher_n.isna().mean()*100:.2f}%")

log("")
log("=" * 84)
log("B. 어느 분모가 정수 누적합을 만드는가  (S = rate × 분모 가 정수여야 한다)")
log("=" * 84)
log(f"  {'rate 컬럼':32s} {'분모=n':>12s} {'분모=pitchmix_n':>16s}")
for col in ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate", "asof_pitcher_success_rate"]:
    r = df[col].to_numpy("float64")
    ok = ~np.isnan(r)
    out = []
    for den in (n, pm):
        S = r[ok] * den[ok]
        out.append(np.isclose(S, np.round(S), atol=1e-6).mean() * 100)
    log(f"  {col:32s} {out[0]:11.2f}% {out[1]:15.2f}%")
log("  ※ 진짜 분모라면 100%에 가까워야 한다 (누적 성공횟수는 정수)")

log("")
log("=" * 84)
log("C. 연속행 차분으로 구종 라벨 복원 — 분모별 회수율")
log("=" * 84)


def recover(den_col, sort_col):
    dd = df.sort_values(["pitcher_id", sort_col], kind="mergesort").reset_index(drop=True)
    pid = dd.pitcher_id.to_numpy()
    dn = dd[den_col].fillna(0).to_numpy("float64")
    nxt = (pid[1:] == pid[:-1]) & (np.abs(dn[1:] - dn[:-1] - 1) < 1e-6)
    res = {}
    for k, col in [("fb", "asof_pitcher_fastball_rate"),
                   ("brk", "asof_pitcher_breaking_rate"),
                   ("off", "asof_pitcher_offspeed_rate")]:
        S = dd[col].fillna(0).to_numpy("float64") * dn
        lab = np.full(len(dd), np.nan)
        lab[:-1] = np.where(nxt, np.round(S[1:] - S[:-1]), np.nan)
        raw = np.full(len(dd), np.nan)
        raw[:-1] = np.where(nxt, S[1:] - S[:-1], np.nan)
        good = (lab == 0) | (lab == 1)
        near = np.isclose(raw, np.round(raw), atol=1e-6)
        res[k] = (good.mean() * 100, np.nanmean(near) * 100, np.nansum(lab == 1))
    return nxt.mean() * 100, res


for tag, den_col, sort_col in [
        ("현행  (분모=asof_pitcher_n)", "asof_pitcher_n", "asof_pitcher_n"),
        ("제안  (분모=pitchmix_n)", "asof_pitcher_pitchmix_n", "asof_pitcher_pitchmix_n")]:
    adj, res = recover(den_col, sort_col)
    log(f"\n  [{tag}]  연속행 인접비율 {adj:.2f}%")
    for k, (cov, near, pos) in res.items():
        log(f"    lab_{k:4s} 0/1 회수율 {cov:6.2f}%   차분이 정수인 비율 {near:6.2f}%   "
            f"양성 {pos:,.0f}")

log("")
log("=" * 84)
log("판독")
log("=" * 84)
log("  · B에서 pitchmix_n 쪽만 100%면 → 분모 확정, 현행 lab_fb/CS-fb 모두 오류")
log("  · C에서 회수율이 크게 오르면 → 구종 라벨 표본이 잘려 있었다")
log("    → exp/65 복원부와 script.py attach_cs P_RATES['fb'] 분모 교체 후 재측정")
log("  · 둘이 같은 카운터면(A 100%) 이 축은 무해, 즉시 종료")
log("=" * 84)
