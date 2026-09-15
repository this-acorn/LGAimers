"""
[06] 노이즈 바닥 측정 — 우리가 보는 ±10~20점 차이는 진짜인가?

실행:  PYTHONIOENCODING=utf-8 py -3.12 -u exp/06_noise_floor.py

왜 필요한가:
  04 ablation과 05 피처실험에서 ±10~20점 크기의 차이들이 잔뜩 나왔다.
    G1 볼카운트  -13.1 / -12.5
    G4 최근폼    +20.2 / -10.6
    G5 투타차이  +20.7 /  -4.2
    v1→v2         -9.4
    v2→v3        -13.0
  이게 '진짜 피처 효과'인지 '모델이 매번 다른 곳에 안착해서 생긴 흔들림'인지
  구분하지 못하면, 앞으로 모든 실험 결과가 무의미해진다.

방법:
  ① 완전히 동일한 피처 세트를 seed만 바꿔 여러 번 학습 → 점수 분포
  ② 그 표준편차가 '노이즈 바닥'
  ③ 노이즈 바닥보다 작은 차이는 앞으로 무시한다

  HistGradientBoosting의 random_state는 (n>10k일 때) 히스토그램 구간을 정하는
  서브샘플링에 쓰인다. 시드가 바뀌면 구간 경계가 미세하게 달라지고,
  탐욕적 분기 선택이 다른 경로로 흘러 최종 트리가 달라진다.

  덤: 시드 앙상블(여러 시드 예측의 평균)이 단일 시드보다 나은지도 같이 잰다.
      노이즈가 크다면 평균내기만으로 점수가 오른다.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "exp")
from common import (log, tick, decompose, load_train, add_features,
                    fit_encoder, to_matrix, ALL_ENG, HGB_FAST,
                    LB_BASELINE, LB_TOP)

SEEDS = [42, 7, 123, 2024, 99, 555, 31337, 1]

log("train.csv 로딩 중...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]

tr = df[df["season"] <= 2023].reset_index(drop=True)
va = df[df["season"] == 2024].reset_index(drop=True)
y_tr = tr["control_success"].to_numpy()
y_va = va["control_success"].to_numpy()
PRIOR = float(y_tr.mean())
tr, va = add_features(tr, PRIOR), add_features(va, PRIOR)

FEATS = BASE + ALL_ENG           # v1 (65피처) 고정 — 오직 seed만 바꾼다
enc = fit_encoder(tr, FEATS)
Xtr, Xva = to_matrix(tr, FEATS, enc), to_matrix(va, FEATS, enc)
tick(f"준비 완료 — {len(FEATS)}피처 고정, seed {len(SEEDS)}종만 변경")
log(f"\n학습 {len(tr):,}행 / 검증 {len(va):,}행\n")


# =====================================================================
log("=" * 84)
log("동일 조건 · 시드만 변경")
log("=" * 84)

preds, scores, discr = [], [], []
for i, s in enumerate(SEEDS, 1):
    tick(f"seed={s} 학습 중 ({i}/{len(SEEDS)})...")
    prm = dict(HGB_FAST)
    prm["random_state"] = s
    m = HistGradientBoostingClassifier(**prm).fit(Xtr, y_tr)
    p = m.predict_proba(Xva)[:, 1]
    z = decompose(p, y_va)
    preds.append(p)
    scores.append(z["총점"])
    discr.append(z["변별력"])
    log(f"  seed {s:6d}   총점 {z['총점']:8.1f}   변별력 {z['변별력']:7.1f}   "
        f"d={z['d']:+.4f}")

sc, di = np.array(scores), np.array(discr)


# =====================================================================
log("\n" + "=" * 84)
log("노이즈 바닥")
log("=" * 84)
log(f"  총점    평균 {sc.mean():8.1f}   표준편차 {sc.std(ddof=1):6.1f}   "
    f"최소~최대 {sc.min():.1f} ~ {sc.max():.1f}  (폭 {sc.max()-sc.min():.1f})")
log(f"  변별력  평균 {di.mean():8.1f}   표준편차 {di.std(ddof=1):6.1f}   "
    f"최소~최대 {di.min():.1f} ~ {di.max():.1f}  (폭 {di.max()-di.min():.1f})")

sd = di.std(ddof=1)
log(f"\n  ▶ 두 실험 결과가 '다르다'고 말하려면 최소 {2*sd:.1f}점 (2σ) 차이가 필요")
log(f"  ▶ 신뢰구간 밖에서 판단하려면 {3*sd:.1f}점 (3σ) 이상 권장")

log("\n  지금까지 관측한 차이들을 이 잣대로 재검토:")
past = [("G2 매치업 LOO", 107.0), ("G2 매치업 ADD", 63.4),
        ("G5 투타차이 LOO", 20.7), ("G4 최근폼 LOO", 20.2),
        ("G1 볼카운트 LOO", -13.1), ("v2→v3 차이피처", -13.0),
        ("v1→v2 가지치기", -9.4), ("G3 asof ADD", 4.8)]
for nm, v in past:
    mark = "진짜" if abs(v) >= 2 * sd else ("애매" if abs(v) >= sd else "노이즈")
    log(f"    {nm:20s} {v:+7.1f}   → {mark}")


# =====================================================================
log("\n" + "=" * 84)
log("덤: 시드 앙상블 (여러 시드 예측을 평균내면?)")
log("=" * 84)

for k in [2, 4, len(SEEDS)]:
    p_avg = np.mean(preds[:k], axis=0)
    z = decompose(p_avg, y_va)
    log(f"  시드 {k}개 평균   총점 {z['총점']:8.1f}   변별력 {z['변별력']:7.1f}   "
        f"(단일 평균 대비 {z['변별력']-di.mean():+.1f})")

log(f"\n  ▶ 앙상블이 단일 평균보다 확실히 높으면, 노이즈가 크다는 증거이자")
log(f"     '시드 앙상블'만으로 공짜 점수를 얻을 수 있다는 뜻이다.")
log(f"  참고: 리더보드 베이스라인 {LB_BASELINE:.1f}, 1위 {LB_TOP:.1f}")
log("=" * 84)
