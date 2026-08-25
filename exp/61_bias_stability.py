# -*- coding: utf-8 -*-
"""
[61] ★ 예측 편향의 연도 안정성 — 합법적 보정 상수가 가능한가 (순수 계산, 학습 없음)

배경: exp/56에서 2024 폴드 중심벌점 48.9점 확인 (예측평균 0.4972 vs 실제 0.4861).
  HANDOFF의 '드리프트 보정 사망' 판정은 **r의 추세를 외삽**하는 방식이었다
  (백테스트 -376.6/-122.6/+87.7/+31.8 → 2/4년 실패).

★ 이건 다른 질문이다:
  "train ≤Y-1 → predict Y" 라는 **같은 구조**를 반복할 때, 모델의 과대예측 폭 δ가
  연도마다 안정적인가? 안정적이면 δ는 추세 외삽이 아니라 '이 파이프라인의 고정 편향'이고,
  학습 데이터로 추정해 2025에 적용하는 것은 운영진이 명시 권장한 방식이다
  ("보정 상수는 가급적 학습 데이터 또는 별도의 검증 데이터를 근거로 설정").

측정: lab/25_preds.npz + 27_preds.npz (CB 8시드, 2021~2024 폴드)
  각 연도의 δ = mean(pred) − mean(y), 그리고 δ를 제거했을 때의 점수 이득

판독:
  δ가 연도별로 비슷(표준편차 작음) → 고정 보정 상수 정당, 기대 이득 ≈ 평균 중심벌점
  δ가 널뛰거나 부호가 섞임 → 기각 (기존 판정 유지)

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/61_bias_stability.py  (~10초)
"""

import sys
import numpy as np

sys.path.insert(0, "exp")
from common import log

z1 = np.load("lab/25_preds.npz")
z2 = np.load("lab/27_preds.npz")
YEARS = [2021, 2022, 2023, 2024]

log("=" * 86)
log("연도별 예측 편향 δ = mean(pred) − mean(y)   [train ≤Y-1 → predict Y, CB 8시드]")
log("=" * 86)
log(f"  {'연도':>6s} {'행수':>9s} {'실제 r':>8s} {'예측평균':>9s} {'δ':>9s} "
    f"{'중심벌점':>9s} {'총점':>9s} {'δ제거후':>9s}")
rows = []
for Y in YEARS:
    y = z1[f"{Y}_y"].astype("float64")
    ps = ([z1[k] for k in z1.files if k.startswith(f"{Y}_cbnum_")]
          + [z2[k] for k in z2.files if k.startswith(f"{Y}_cbnum_")])
    p = np.mean(ps, axis=0)
    r = y.mean()
    den = r * (1 - r)
    d = p.mean() - r
    pen = 100000 * d * d / den
    s = 100000 * (1 - np.mean((p - y) ** 2) / den)
    pc = np.clip(p - d, 0, 1)
    s2 = 100000 * (1 - np.mean((pc - y) ** 2) / den)
    rows.append((Y, len(y), r, p.mean(), d, pen, s, s2))
    log(f"  {Y:>6d} {len(y):>9,} {r:>8.4f} {p.mean():>9.4f} {d:>+9.4f} "
        f"{pen:>9.1f} {s:>9.1f} {s2:>9.1f}")

ds = np.array([x[4] for x in rows])
pens = np.array([x[5] for x in rows])
log("\n" + "=" * 86)
log("판정")
log("=" * 86)
log(f"  δ: 평균 {ds.mean():+.4f}  표준편차 {ds.std(ddof=1):.4f}  "
    f"범위 [{ds.min():+.4f}, {ds.max():+.4f}]  부호 일치 {np.all(ds > 0) or np.all(ds < 0)}")
log(f"  중심벌점: 평균 {pens.mean():.1f}  범위 [{pens.min():.1f}, {pens.max():.1f}]")

# 핵심 검정: 직전 연도들의 δ 평균으로 다음 연도를 보정했다면?
log("\n  [백테스트] 이전 연도들의 δ 평균을 다음 연도에 적용했다면:")
for i in range(1, len(rows)):
    Y = rows[i][0]
    d_hat = ds[:i].mean()                 # ≤Y-1 폴드들의 δ 평균
    y = z1[f"{Y}_y"].astype("float64")
    ps = ([z1[k] for k in z1.files if k.startswith(f"{Y}_cbnum_")]
          + [z2[k] for k in z2.files if k.startswith(f"{Y}_cbnum_")])
    p = np.mean(ps, axis=0)
    r = y.mean(); den = r * (1 - r)
    s0 = 100000 * (1 - np.mean((p - y) ** 2) / den)
    s1 = 100000 * (1 - np.mean((np.clip(p - d_hat, 0, 1) - y) ** 2) / den)
    log(f"    {Y}: δ̂={d_hat:+.4f} (실제 {ds[i]:+.4f}) → {s0:.1f} → {s1:.1f}  "
        f"({s1-s0:+.1f})")

log("\n  · 전 연도 이득 양수 & δ 표준편차 작음 → 합법 보정 상수 채택 후보")
log("  · 한 해라도 크게 음수 → 기각 (추세 외삽 판정과 동일 운명)")
log("=" * 86)
