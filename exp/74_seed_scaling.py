# -*- coding: utf-8 -*-
"""
[74] 시드 스케일링 감사 — "2시드에서 잰 이득이 8시드 배포에서도 남는가"

문제의식:
  모든 로컬 비교는 2시드로 했는데 실전 제출은 전부 8시드다. 그런데 앙상블 이득은
  모델마다 다르다. 시드간 다양성 D가 큰 모델일수록 앙상블로 더 많이 번다.
  즉 **저lr 모델처럼 분산이 작은 모델은 2시드 비교에서 유리하게 보이지만
  8시드에서는 상대 우위가 깎일 수 있다.**

항등식 (근사 아님):
  S(n시드 평균) = mean_i S_i + K̄ · (n-1)/(2n),    K̄ = (100000/DEN)·D̄
  n=2 → 이득 K̄/4     n=8 → 이득 0.4375·K̄
  따라서 2시드 실측에서 K̄ = 4 × (2시드앙상블 − 시드평균) 로 역산할 수 있다.

이 스크립트는 학습을 하지 않는다. 이미 로그에 있는 시드별 점수만 쓴다.
핵심 질문: 세 가지 로컬 통계(시드평균 / 2시드 / 8시드추정) 중
           **어느 것이 실측 LB를 가장 잘 예측하는가**.

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/74_seed_scaling.py   (즉시)
"""

import sys
import numpy as np

sys.path.insert(0, "exp")
from common import log

# (이름, seed42, seed7, 2시드앙상블, LB점수 or None, 출처)
ROWS = [
    ("CB65", 650.5, 633.7, 674.4, 898.61863054, "lab/36 / submit8"),
    ("CS76", 748.5, 745.9, 782.3, 990.957167531, "lab/41 / submit10"),
    ("CS79", 777.0, 760.6, 800.9, 993.6345481775, "lab/45 / submit12"),
    ("MC", 817.6, 812.2, 837.0, 1033.9866361712, "lab/53 / submit14"),
    ("BIN04", 830.9, 831.2, 839.4, None, "lab/65 (미배포)"),
    ("MC04", 842.3, 848.9, 852.3, None, "lab/66 → submit16 학습중"),
]

log("=" * 96)
log("A. 시드 분해 — 앙상블 이득에서 K̄를 역산")
log("=" * 96)
log(f"  {'모델':7s} {'seed42':>8s} {'seed7':>8s} {'시드평균':>9s} {'2시드':>8s} "
    f"{'앙상블이득':>10s} {'K̄':>8s} {'8시드추정':>10s}")
M = {}
for name, s42, s7, ens2, lb, src in ROWS:
    mean_seed = (s42 + s7) / 2
    gain2 = ens2 - mean_seed
    K = 4 * gain2                      # n=2: 이득 = K̄/4
    ens8 = mean_seed + 0.4375 * K      # n=8: 이득 = K̄·7/16
    M[name] = dict(mean=mean_seed, e2=ens2, e8=ens8, K=K, lb=lb)
    log(f"  {name:7s} {s42:8.1f} {s7:8.1f} {mean_seed:9.1f} {ens2:8.1f} "
        f"{gain2:10.1f} {K:8.1f} {ens8:10.1f}")
log("")
log("  판독: K̄가 큰 모델일수록 시드를 늘릴 때 더 많이 번다.")
log(f"    MC(lr0.08) K̄={M['MC']['K']:.1f}  vs  MC04(lr0.04) K̄={M['MC04']['K']:.1f}"
    f"  →  {M['MC']['K']/M['MC04']['K']:.1f}배 차이")
log("    저lr = 총 학습량이 작아 분산이 작다 → 앙상블로 벌 여지가 적다.")

log("")
log("=" * 96)
log("B. lr 이득의 성분 분해 — 편향(살아남음) vs 분산(앙상블에 먹힘)")
log("=" * 96)
for base, new, tag in [("MC", "MC04", "멀티클래스 lr0.08→0.04"),
                       ("CS79", "BIN04", "이진 lr0.08→0.04"),
                       ("CS79", "MC", "이진→멀티클래스 (참고: 전이 실측됨)")]:
    b, n_ = M[base], M[new]
    g_mean = n_["mean"] - b["mean"]
    g_2 = n_["e2"] - b["e2"]
    g_8 = n_["e8"] - b["e8"]
    log(f"  [{tag}]")
    log(f"    시드평균 기준 이득 {g_mean:+7.1f}   ← 편향 성분 (앙상블과 무관, 살아남음)")
    log(f"    2시드   기준 이득 {g_2:+7.1f}")
    log(f"    8시드추정 이득    {g_8:+7.1f}   ← 실제 배포에서 남는 몫")
    log(f"    앙상블이 먹은 몫  {g_mean - g_8:+7.1f}")

log("")
log("=" * 96)
log("C. ★ 어느 로컬 통계가 LB를 가장 잘 예측하는가 (LB 실측 4개)")
log("=" * 96)
have = [k for k in M if M[k]["lb"] is not None]
lb = np.array([M[k]["lb"] for k in have])
log(f"  대상: {', '.join(have)}")
log("")
log(f"  {'로컬통계':12s} {'고정오프셋 편차':>16s} {'자유기울기 기울기':>18s} "
    f"{'잔차std':>10s}")
fits = {}
for stat, label in [("mean", "시드평균"), ("e2", "2시드"), ("e8", "8시드추정")]:
    x = np.array([M[k][stat] for k in have])
    off = lb - x                                  # 기울기 1 가정 시 오프셋
    A = np.column_stack([x, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(A, lb, rcond=None)
    resid = lb - A @ coef
    fits[stat] = (coef[0], coef[1], resid.std(ddof=0), off.std(ddof=0), off.mean())
    log(f"  {label:12s} {off.std(ddof=0):16.1f} {coef[0]:18.3f} "
        f"{resid.std(ddof=0):10.1f}")
log("")
log("  · 고정오프셋 편차가 작을수록 'LB = 로컬 + 상수'가 잘 맞는다는 뜻")
log("  · 자유기울기가 1에서 멀수록 단순 덧셈 환산이 위험하다는 뜻")

log("")
log("=" * 96)
log("D. submit16(MC04) 예상 — 경로별")
log("=" * 96)
ref_lb, ref = M["MC"]["lb"], "MC"
for stat, label in [("mean", "시드평균"), ("e2", "2시드"), ("e8", "8시드추정")]:
    d_loc = M["MC04"][stat] - M[ref][stat]
    # 같은 통계로 잰 CS79→MC 전이율
    tau = (M["MC"]["lb"] - M["CS79"]["lb"]) / (M["MC"][stat] - M["CS79"][stat])
    log(f"  [{label:9s}] 로컬 격차 {d_loc:+6.1f} × τ {tau:5.2f} "
        f"= {ref_lb + tau*d_loc:8.1f}")
sl, ic, *_ = fits["e2"]
log(f"  [자유기울기·2시드] {sl*M['MC04']['e2'] + ic:8.1f}")
sl8, ic8, *_ = fits["e8"]
log(f"  [자유기울기·8시드] {sl8*M['MC04']['e8'] + ic8:8.1f}")

log("")
log("=" * 96)
log("E. 로컬 목표 재계산 — 19등 컷 1163.68")
log("=" * 96)
for stat, label in [("e2", "2시드"), ("e8", "8시드추정")]:
    tau = (M["MC"]["lb"] - M["CS79"]["lb"]) / (M["MC"][stat] - M["CS79"][stat])
    need = M["MC"][stat] + (1163.68 - M["MC"]["lb"]) / tau
    log(f"  [{label:9s}] τ={tau:.2f} → 필요 로컬 {need:7.1f} "
        f"(MC04 {M['MC04'][stat]:.1f} 대비 +{need - M['MC04'][stat]:.1f})")
log("")
log("=" * 96)
log("결론 초안")
log("=" * 96)
log("  · lr 이득의 상당 부분이 '분산 감소'라면 8시드 배포에서 상대우위가 깎인다.")
log("  · 앞으로 모든 짝지음 실험은 **시드평균 이득**과 **앙상블 이득**을 함께 보고하라.")
log("    시드평균 이득이 큰 개선 = 편향 개선 = 배포에서 살아남는다.")
log("    앙상블 이득에서만 나오는 개선 = 분산 개선 = 8시드에서 사라진다.")
log("=" * 96)
