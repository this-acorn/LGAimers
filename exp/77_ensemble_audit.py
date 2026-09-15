# -*- coding: utf-8 -*-
"""
[77] 앙상블 감사 — "우리는 단일모델인가" + "8시드가 진짜 8개인가" + 혼합 필요조건

A. 배포 번들(submit14 = 현 LB 1033.99)이 실제로 몇 개의 서로 다른 모델인가.
   리뷰 항목 12 "seed별 모델 파일이 실제로 모두 다른지" 감사.
   만약 버그로 동일하다면 시드앙상블 이득(~+25)을 통째로 잃고 있는 것이다.

B. 혼합 필요조건의 닫힌 해. 2모델 혼합의 최적 이득은 정확히
       gain = (d + K)^2 / (4K),    d = S_partner - S_champion,  K = (1e5/DEN)*E[(p1-p2)^2]
   (S(w) = (1-w)S0 + wS1 + w(1-w)K 를 w로 최적화한 결과. 근사 아님.)
   → "+124를 혼합으로 얻으려면 K가 얼마여야 하는가"를 역산한다.
   실측 K 값과 비교하면 혼합 축의 천장이 즉시 나온다.

학습 없음. 번들 로드 + 5행 추론 + 산술. exp/72와 동시 실행 안전.
실행: PYTHONIOENCODING=utf-8 python -u exp/77_ensemble_audit.py   (~2분)
"""

import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, "exp")
from common import log

log("=" * 88)
log("A. submit14 배포 번들 감사 — 8시드가 진짜 8개인가")
log("=" * 88)
b = joblib.load('submissions/submit14_src/model/model.pkl')
log(f"  번들 키: {sorted(b.keys())}")
key = "cb_models" if "cb_models" in b else [k for k in b if "model" in k][0]
models = b[key]
log(f"  모델 개수: {len(models)}  (키='{key}')")

# 각 모델의 학습 시드와 트리 구조 지문
seeds, hashes = [], []
for i, m in enumerate(models):
    try:
        sd = m.get_all_params().get("random_seed", "?")
    except Exception:
        sd = "?"
    try:
        lv = np.asarray(m.get_leaf_values(), dtype="float64")
        h = (float(lv.sum()), float(np.abs(lv).sum()), int(lv.size))
    except Exception as e:
        h = ("no_leaf_values", str(e)[:30], 0)
    seeds.append(sd)
    hashes.append(h)
    log(f"    [{i}] random_seed={sd}  잎값 개수={h[2]:,}  합={h[0]:+.6e}")

log(f"\n  서로 다른 시드 {len(set(map(str, seeds)))}/{len(models)}")
uniq = len(set(hashes))
log(f"  서로 다른 트리 지문 {uniq}/{len(models)}")
if uniq == len(models):
    log("  ★ 8개 모델 전부 다르다 — 시드 앙상블 정상 작동")
else:
    log("  🚨 중복 모델 발견 — 시드 앙상블 이득을 잃고 있다. 즉시 조사")

# 실제 예측이 다른지 (진짜 테스트, 5행)
try:
    t5 = pd.read_csv("data/test.csv", encoding="utf-8-sig")
    t5.columns = [c.replace("﻿", "").strip() for c in t5.columns]
    import importlib.util
    spec = importlib.util.spec_from_file_location("s14", 'submissions/submit14_src/script.py')
    s14 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s14)
    ft = s14.attach_pt(s14.attach_cs(s14.add_features(t5, b["prior"]),
                                     b["cs_const_p"], b["cs_const_b"]),
                       b["mix_tbl"], b["pfb_model"])
    X = s14.build_matrix(ft, b["feats"])
    P = np.array([m.predict_proba(X)[:, 0] for m in models])
    log(f"\n  5행 예측 (클래스0=성공), 시드별:")
    for i in range(len(models)):
        log(f"    [{i}] " + "  ".join(f"{v:.6f}" for v in P[i]))
    spread = P.max(axis=0) - P.min(axis=0)
    log(f"  행별 시드간 최대-최소 폭: " + "  ".join(f"{v:.6f}" for v in spread))
    log(f"  ★ 예측이 시드마다 다름: {bool(spread.max() > 1e-9)}")
except Exception as e:
    log(f"  (5행 추론 생략: {type(e).__name__}: {e})")

# =====================================================================
log("")
log("=" * 88)
log("B. 혼합 필요조건 — +124를 혼합으로 얻으려면 K가 얼마여야 하나")
log("=" * 88)
S0 = 1033.99          # 현 챔피언 (submit14, 2025 실측)
TARGET = 1175.0

log("  닫힌 해:  gain = (d + K)^2 / (4K),   d = S_partner - S_champion")
log("            최적 가중 w* = 0.5 + d/(2K)   (0..1로 클립)")
log("")
log("  실측 K (2025 환산):")
log("    CB65 ↔ CS79   117.5   ← 유일한 2025 직접 실측. 피처셋이 달랐다(65 vs 79)")
log("    CS79 ↔ MC      45.6   | CS76 ↔ MC  49.8 | CS76 ↔ CS79  42.7   ← 같은 계열")
log("    같은 모델 시드간(MC04)  26.8 (로컬)")
log("")


def gain(d, K):
    return (d + K) ** 2 / (4 * K) if K > 0 else 0.0


def w_star(d, K):
    return min(max(0.5 + d / (2 * K), 0.0), 1.0)


log("  [B-1] 파트너가 챔피언과 동급(d=0)일 때, 목표 이득에 필요한 K")
for g in [20, 30, 50, 80, 124]:
    log(f"    이득 +{g:5.0f} 필요  →  K = {4*g:6.1f}"
        f"   (실측 최대 117.5의 {4*g/117.5:.1f}배)")

log("")
log("  [B-2] 현실적 파트너들의 혼합 이득 (2025 기준)")
log(f"  {'파트너 성격':28s} {'d':>7s} {'K':>7s} {'w*':>6s} {'혼합이득':>9s} {'결과':>9s}")
CASES = [
    ("챔피언 동급·다른 계열", 0.0, 117.5),
    ("챔피언 동급·같은 계열", 0.0, 45.6),
    ("30 열세·다른 계열", -30.0, 117.5),
    ("60 열세·다른 계열", -60.0, 117.5),
    ("100 열세·다른 계열", -100.0, 117.5),
    ("CB65 실측 (135 열세)", 898.61863054 - S0, 117.5),
    ("HGB79 추정 (전이벌 70)", -100.8, 92.0),
]
for name, d, K in CASES:
    g = gain(d, K)
    w = w_star(d, K)
    log(f"  {name:28s} {d:+7.1f} {K:7.1f} {w:6.3f} {g:+9.2f} {S0+g:9.2f}")

log("")
log("  [B-3] 파트너 3개를 동시에 쓰면? (서로도 다양해야 함)")
log("    동급 파트너 2개 + 챔피언, 모든 쌍 K=117.5 → 균등가중 이득 = K/3 = "
    f"{117.5/3:.1f}")
log("    동급 파트너 3개 + 챔피언, 모든 쌍 K=117.5 → 이득 = 3K/8 = "
    f"{3*117.5/8:.1f}")
log("    (n개 균등혼합 이득 = K*(n-1)/(2n) — 시드앙상블과 같은 항등식)")

log("")
log("=" * 88)
log("판정")
log("=" * 88)
log(f"  · 목표 {TARGET:.0f}까지 +{TARGET-S0:.0f}. 혼합만으로 채우려면 K≈{4*(TARGET-S0):.0f}가")
log(f"    필요한데 이는 우리가 관측한 최대 K(117.5)의 {4*(TARGET-S0)/117.5:.1f}배다.")
log("  · 챔피언과 동급이면서 최대 다양성을 가진 파트너를 구해도 상한은 "
    f"+{117.5/4:.1f}.")
log("    무한히 많은 동급·최대다양성 파트너를 모아도 상한은 K/2 = "
    f"+{117.5/2:.1f}.")
log("  · 따라서 혼합은 **진짜 축이지만 +30~60짜리 축**이지 +124짜리가 아니다.")
log("    그리고 그 +30조차 '챔피언과 동급이면서 계열이 다른 모델'을 요구한다 —")
log("    현재 그런 모델은 없다(CB65는 계열은 다르나 135점 열세).")
log("=" * 88)
