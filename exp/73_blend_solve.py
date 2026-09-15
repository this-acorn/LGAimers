# -*- coding: utf-8 -*-
"""
[73] 혼합 최적해 — 재학습 0. 이미 배포된 번들만 섞어 오늘 제출 가능한 최고점을 찾는다.

혼합 브라이어 항등식 (근사가 아니라 항등식):
    B(blend) = sum_i w_i B_i  -  sum_{i<j} w_i w_j D_ij ,   D_ij = E[(p_i - p_j)^2]
  → S(blend) = sum_i w_i S_i  +  sum_{i<j} w_i w_j K_ij ,   K_ij = (100000/DEN) * D_ij

실측 LB 단독점수 (2025 전체 245,789행, private 없음, 완전 결정론):
    CB65  898.61863054     CS76  990.957167531
    CS79  993.6345481775   MC   1033.9866361712   <- 08-26 신규
실측 2025 혼합점 1개:
    CS79(0.5) + CB65(0.5) = 975.5048373847  ->  K2025(CB65,CS79) = 117.5

K2025를 직접 못 재는 쌍은 로컬 D로부터 환산한다. 환산율은 위 한 쌍에서 고정:
    ratio = D_local(CB65,CS79) / D_2025(CB65,CS79)
이 스크립트는 그 ratio를 **재계산**해서 HANDOFF의 0.614가 맞는지 먼저 검증한다.

★ 규칙: 제출 간 모델 선택(가중치 고르기)은 공식 Q&A(08-12) 명시 허용.
  금지된 것은 LB 점수로 2025 r을 역산해 예측을 이동시키는 것 — 그건 하지 않는다.

학습 없음. 캐시된 .npy만 읽는다 (~수십 MB). 배포 학습과 동시 실행 안전.
실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/73_blend_solve.py   (~2분)
"""

import itertools
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "exp")
from common import log

# 배포 번들이 실제로 존재하는 모델만 후보로 둔다 (오늘 재학습 없이 제출 가능해야 함)
DEPLOYED = {
    "CB65": ("lab/36_ref_ens.npy", 898.61863054, "submit8.zip"),
    "CS76": ("lab/41_cs_ens.npy", 990.957167531, 'submissions/submit10_src'),
    "CS79": ("lab/45_both_ens.npy", 993.6345481775, 'submissions/submit12_src'),
    "MC": ("lab/53_mc_ens.npy", 1033.9866361712, 'submissions/submit14_src'),
}
LB_PAIR = ("CB65", "CS79", 0.5, 975.5048373847)   # 실측 혼합점

va = pd.read_csv("data/train.csv", encoding="utf-8-sig",
                 usecols=["season", "control_success"])
va.columns = [c.replace("﻿", "").strip() for c in va.columns]
y = va.loc[va.season == 2024, "control_success"].to_numpy("float64")
r = y.mean()
DEN = r * (1 - r)
log(f"2024 폴드 {len(y):,}행, r={r:.6f}, DEN={DEN:.6f}")

P, S_lb = {}, {}
for k, (path, lb, src) in DEPLOYED.items():
    if not os.path.exists(path):
        log(f"  !! {k}: {path} 없음 — 후보에서 제외")
        continue
    p = np.load(path).astype("float64")
    if len(p) != len(y):
        log(f"  !! {k}: 길이 {len(p)} != {len(y)} — 제외")
        continue
    P[k], S_lb[k] = p, lb
    log(f"  {k:5s} 로컬 {100000*(1-np.mean((p-y)**2)/DEN):8.1f}   LB {lb:12.4f}   ({src})")

KEYS = [k for k in DEPLOYED if k in P]

log("")
log("=" * 86)
log("A. 로컬 다양성 D와 K, 그리고 2025 환산율 검증")
log("=" * 86)
D_loc, K_loc = {}, {}
for a, b in itertools.combinations(KEYS, 2):
    d = float(np.mean((P[a] - P[b]) ** 2))
    D_loc[(a, b)] = d
    K_loc[(a, b)] = 100000 * d / DEN
    log(f"  D_local({a:5s},{b:5s}) = {d:.6e}   K_local = {K_loc[(a,b)]:7.1f}")

a0, b0, w0, s_mix = LB_PAIR
key0 = (a0, b0) if (a0, b0) in K_loc else (b0, a0)
K25_meas = (s_mix - w0 * S_lb[a0] - (1 - w0) * S_lb[b0]) / (w0 * (1 - w0))
ratio = K_loc[key0] / K25_meas
log("")
log(f"  실측 2025 혼합점: {a0}({w0}) + {b0}({1-w0}) = {s_mix}")
log(f"    -> K2025({a0},{b0}) = {K25_meas:.2f}")
log(f"    -> 환산율 K_local/K_2025 = {ratio:.4f}   (HANDOFF 기록값 0.614)")
if abs(ratio - 0.614) > 0.03:
    log(f"    ⚠ 기록값과 {abs(ratio-0.614):.3f} 어긋남 — 아래 추정치는 그만큼 불확실")

K25 = {pair: K_loc[pair] / ratio for pair in K_loc}
log("")
log("  환산된 K2025 (실측 쌍 제외 전부 추정치):")
for pair, v in K25.items():
    tag = " ← 실측" if pair == key0 else ""
    log(f"    K2025({pair[0]:5s},{pair[1]:5s}) = {v:7.1f}{tag}")


def blend_score(ws):
    """S(blend) = sum w_i S_i + sum_{i<j} w_i w_j K_ij  (2025 추정)"""
    s = sum(ws[k] * S_lb[k] for k in ws)
    for a, b in itertools.combinations(ws, 2):
        pair = (a, b) if (a, b) in K25 else (b, a)
        s += ws[a] * ws[b] * K25[pair]
    return s


log("")
log("=" * 86)
log("B. 2모델 최적 (해석해)  w* = 0.5 + (S1-S0)/(2K)")
log("=" * 86)
log(f"  {'쌍':16s} {'w*(첫번째)':>11s} {'예상 LB':>10s} {'최고단독 대비':>12s}")
best2 = None
for a, b in itertools.combinations(KEYS, 2):
    pair = (a, b)
    K = K25[pair]
    w = 0.5 + (S_lb[a] - S_lb[b]) / (2 * K)
    w = min(max(w, 0.0), 1.0)
    s = blend_score({a: w, b: 1 - w})
    solo = max(S_lb[a], S_lb[b])
    log(f"  {a+'+'+b:16s} {w:11.3f} {s:10.2f} {s-solo:+12.2f}")
    if best2 is None or s > best2[0]:
        best2 = (s, {a: w, b: 1 - w})

log("")
log("=" * 86)
log("C. 다모델 최적 (격자 탐색, 0.02 간격)")
log("=" * 86)
best = None
grid = np.arange(0, 1.0001, 0.02)
for n in range(2, len(KEYS) + 1):
    for combo in itertools.combinations(KEYS, n):
        if n == 2:
            cands = [(w, 1 - w) for w in grid]
        elif n == 3:
            cands = [(w1, w2, 1 - w1 - w2) for w1 in grid for w2 in grid
                     if w1 + w2 <= 1.0001]
        else:
            cands = [(w1, w2, w3, 1 - w1 - w2 - w3)
                     for w1 in grid for w2 in grid for w3 in grid
                     if w1 + w2 + w3 <= 1.0001]
        for ws in cands:
            d = {k: float(w) for k, w in zip(combo, ws) if w > 1e-9}
            if len(d) < 2:
                continue
            s = blend_score(d)
            if best is None or s > best[0]:
                best = (s, d)
    if best:
        log(f"  {n}모델까지: 최고 {best[0]:.2f}  "
            + " / ".join(f"{k} {v:.2f}" for k, v in sorted(best[1].items())))

log("")
log("=" * 86)
log("판정")
log("=" * 86)
solo_best = max(S_lb.values())
solo_name = max(S_lb, key=S_lb.get)
log(f"  현재 최고 단독:  {solo_name} {solo_best:.2f}")
log(f"  최적 혼합 예상:  {best[0]:.2f}  ({best[0]-solo_best:+.2f})")
log("    가중치: " + " / ".join(f"{k} {v:.3f}" for k, v in sorted(best[1].items())))
log("")
log("  ※ K2025는 실측 쌍 1개를 빼면 전부 로컬에서 환산한 추정치다.")
log("    2025 혼합 실측점이 늘어날수록 정확해진다 (submit13이 그 첫 점이었다).")
log("  ※ 재학습 0 — 위 번들들을 병합해 오늘 바로 제출 가능하다.")
log("=" * 86)
