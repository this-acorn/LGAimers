"""
[09] 하이퍼파라미터 튜닝 — HistGradientBoosting

실행:  PYTHONIOENCODING=utf-8 py -3.12 -u exp/09_tune.py

배경:
  현재 설정은 실험 속도 때문에 일부러 약하게 잡은 값이다.
    max_iter=120         (원래 300에서 낮춤)
    learning_rate=0.08   (그 보상으로 올림)
    max_leaf_nodes=31    (기본값, 한 번도 안 건드림)
  GBDT는 '나무를 많이 + 한 그루당 조금씩'이 보통 더 정확하다. 지금은 그 반대다.

★ 모든 조합을 4시드 앙상블로 잰다.
  단일 학습의 노이즈가 σ=15.3이라 30점 미만 차이를 분간할 수 없기 때문.
  4시드 평균이면 측정 편차가 대략 절반으로 줄어 15점 차이도 읽힌다.

단계:
  A. 용량(capacity)  나무를 늘리면 좋아지나?   3조합
  B. 나무 모양       A의 승자에서 잎 개수 변경   2조합
  → 총 5조합 × 4시드 = 20회 학습 (약 45분)
"""

import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "exp")
from common import (log, tick, decompose, load_train, add_features,
                    fit_encoder, to_matrix, ALL_ENG, LB_BASELINE, LB_TOP)

SEEDS = [42, 7, 123, 2024]        # 4시드 (8시드는 시간이 2배라 튜닝엔 4개로)
NOISE_SD = 15.3                   # 06번에서 측정한 단일학습 노이즈
CUR_LOCAL = 708.7                 # 현재 제출 모델의 8시드 로컬 점수
CUR_LB = 830.32                   # 그 모델의 실제 리더보드 점수
TRANSFER = 0.958                  # 로컬 → 리더보드 전이율

BASE_PARAMS = dict(max_iter=120, learning_rate=0.08, max_leaf_nodes=31,
                   min_samples_leaf=200, l2_regularization=10.0,
                   early_stopping=False)

log("train.csv 로딩 중...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS = BASE + ALL_ENG

tr = df[df["season"] <= 2023].reset_index(drop=True)
va = df[df["season"] == 2024].reset_index(drop=True)
y_tr, y_va = tr["control_success"].to_numpy(), va["control_success"].to_numpy()
PRIOR = float(y_tr.mean())
tr, va = add_features(tr, PRIOR), add_features(va, PRIOR)
enc = fit_encoder(tr, FEATS)
Xtr, Xva = to_matrix(tr, FEATS, enc), to_matrix(va, FEATS, enc)
tick(f"준비 완료 — {len(FEATS)}피처, 학습 {len(tr):,} / 검증 {len(va):,}")
log(f"\n기준: 현재 제출 모델 로컬 {CUR_LOCAL} → 리더보드 {CUR_LB}")
log(f"      노이즈 σ={NOISE_SD} (단일학습). 4시드 앙상블로 측정한다.\n")


def evaluate(name, **override):
    """주어진 설정으로 4시드 학습 → 앙상블 예측 → 점수"""
    prm = dict(BASE_PARAMS, **override)
    t0 = time.time()
    acc = np.zeros(len(va))
    singles = []
    for s in SEEDS:
        m = HistGradientBoostingClassifier(**prm, random_state=s).fit(Xtr, y_tr)
        p = m.predict_proba(Xva)[:, 1]
        acc += p
        singles.append(decompose(p, y_va)["변별력"])
    z = decompose(acc / len(SEEDS), y_va)
    took = time.time() - t0
    log(f"  {name:26s} 총점 {z['총점']:7.1f}  변별력 {z['변별력']:7.1f}  "
        f"(단일 {min(singles):.0f}~{max(singles):.0f})  [{took:.0f}s]")
    return z


# =====================================================================
log("=" * 92)
log("A단계. 용량 — 나무를 늘리고 learning_rate를 낮추면?")
log("=" * 92)

CAPACITY = [
    ("현재 (120, lr .08)", dict(max_iter=120, learning_rate=0.08)),
    ("확장 (400, lr .03)", dict(max_iter=400, learning_rate=0.03)),
    ("최대 (800, lr .02)", dict(max_iter=800, learning_rate=0.02)),
]

capA = {}
for i, (name, ov) in enumerate(CAPACITY, 1):
    tick(f"A{i}/{len(CAPACITY)} 학습 중 — {name}")
    capA[name] = (evaluate(name, **ov), ov)

bestA = max(capA, key=lambda k: capA[k][0]["변별력"])
base_disc = capA["현재 (120, lr .08)"][0]["변별력"]
gain = capA[bestA][0]["변별력"] - base_disc
log(f"\n  ▶ A단계 승자: {bestA}")
log(f"     현재 대비 변별력 {gain:+.1f}점  "
    f"({'유의미' if abs(gain) >= 2*NOISE_SD/2 else '노이즈 범위'})")


# =====================================================================
log("\n" + "=" * 92)
log(f"B단계. 나무 모양 — 승자 설정({bestA})에서 잎 개수 변경")
log("=" * 92)

win_ov = capA[bestA][1]
SHAPE = [
    ("잎 15개 (단순)", dict(win_ov, max_leaf_nodes=15)),
    ("잎 63개 (복잡)", dict(win_ov, max_leaf_nodes=63)),
]

allres = {bestA + " [잎31]": capA[bestA][0]}
for i, (name, ov) in enumerate(SHAPE, 1):
    tick(f"B{i}/{len(SHAPE)} 학습 중 — {name}")
    allres[name] = evaluate(name, **ov)


# =====================================================================
log("\n" + "=" * 92)
log("종합")
log("=" * 92)

everything = {k: v[0] for k, v in capA.items()}
everything.update(allres)
log(f"  {'설정':30s} {'총점':>8s} {'변별력':>8s} {'현재대비':>9s}")
log("  " + "-" * 62)
for name, z in sorted(everything.items(), key=lambda x: -x[1]["변별력"]):
    log(f"  {name:30s} {z['총점']:8.1f} {z['변별력']:8.1f} "
        f"{z['변별력']-base_disc:+9.1f}")

best = max(everything, key=lambda k: everything[k]["변별력"])
bz = everything[best]
delta = bz["변별력"] - base_disc
log(f"\n  ▶ 최종 승자: {best}")
log(f"     변별력 {delta:+.1f}점 (현재 설정 대비)")

if delta >= 2 * NOISE_SD / 2:      # 4시드 앙상블 기준 유의 임계
    proj_local = CUR_LOCAL + delta
    proj_lb = CUR_LB + delta * TRANSFER
    log(f"\n     예상 로컬(8시드)  {CUR_LOCAL:.1f} → {proj_local:.1f}")
    log(f"     예상 리더보드     {CUR_LB:.1f} → {proj_lb:.1f}  (전이율 {TRANSFER})")
    log(f"     1위 {LB_TOP:.1f} 까지 남는 격차 {LB_TOP-proj_lb:.1f}")
else:
    log(f"     ⚠️ 노이즈 범위 — 튜닝 효과 없음으로 판단. 다른 축을 파야 함")
log("=" * 92)
