"""
[13] 보정기 전이 검증 — '보정기를 적합한 모델'과 '적용받는 모델'이 달라도 통하는가

실행:  PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 py -3.12 -u exp/13_calib_transfer.py

배경:
  exp/10 후속에서 누수 없는 Isotonic 보정이 극단 설정에서 -1882 → +159 (+2040)를
  회수했다. 그런데 그 검증은 '같은 모델'(2019~2022 학습) 안에서만 이뤄졌다:
      A(≤2022) → 2023 예측으로 적합 → A의 2024 예측에 적용   ✅ 이미 검증됨
  실제 배포(HANDOFF 7-2)는 보정기를 적합한 모델과 적용받는 모델이 다르다:
      A'(≤2023) → 2024 예측으로 적합 → 최종모델(≤2024)의 2025 예측에 적용
  보정기가 '다른 모델의 예측 분포'에도 통하는지는 검증된 적 없다.
  이 실험은 그 배포 상황을 1년 앞당겨 그대로 모사한다:
      A(≤2022) 학습 → 2023 예측으로 Isotonic/Platt 적합
      B(≤2023) 학습 → B의 2024 예측에 그 보정기 적용 → 2024 정답으로 채점

합법성 (HANDOFF §6):
  보정기는 train.csv 안의 연도로만 적합하고, 각 행에 독립적으로
  (고정된 1차원 함수로) 적용한다. test 분포를 이용한 사후 보정이 아니다.

판정:
  ★ B+보정(전이) − B원본 ≥ +15  →  제출 파이프라인에 보정 도입
  ※ 두 값은 '같은 예측'에 보정만 얹은 짝지은 비교라 시드 노이즈가 대부분
    상쇄된다. 임계 15점은 보수적으로 그대로 쓴다.
  ※ 반칙 천장(2024 정답으로 직접 적합)과의 격차 = 전이에서 잃는 몫.
"""

import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "exp")
from common import (log, tick, decompose, load_train, add_features,
                    fit_encoder, to_matrix, ALL_ENG, HGB_FAST, LB_TOP)

SEEDS = [42, 7, 123, 2024]
THR = 15.0

log("train.csv 로딩 중...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS = BASE + ALL_ENG

raw_A = df[df["season"] <= 2022].reset_index(drop=True)   # 모델 A 학습
raw_23 = df[df["season"] == 2023].reset_index(drop=True)  # 보정기 적합용
raw_B = df[df["season"] <= 2023].reset_index(drop=True)   # 모델 B 학습
raw_24 = df[df["season"] == 2024].reset_index(drop=True)  # 채점용
y_A = raw_A["control_success"].to_numpy()
y_23 = raw_23["control_success"].to_numpy()
y_B = raw_B["control_success"].to_numpy()
y_24 = raw_24["control_success"].to_numpy()

# 각 모델은 자기 학습 데이터로 prior/인코더를 만든다 (실제 파이프라인과 동일)
PRIOR_A, PRIOR_B = float(y_A.mean()), float(y_B.mean())

fa = add_features(raw_A, PRIOR_A)
enc_A = fit_encoder(fa, FEATS)
X_A = to_matrix(fa, FEATS, enc_A)
X23_A = to_matrix(add_features(raw_23, PRIOR_A), FEATS, enc_A)
X24_A = to_matrix(add_features(raw_24, PRIOR_A), FEATS, enc_A)

fb = add_features(raw_B, PRIOR_B)
enc_B = fit_encoder(fb, FEATS)
X_B = to_matrix(fb, FEATS, enc_B)
X24_B = to_matrix(add_features(raw_24, PRIOR_B), FEATS, enc_B)

tick(f"준비 완료 — A학습 {len(raw_A):,} / 보정 {len(raw_23):,} / "
     f"B학습 {len(raw_B):,} / 채점 {len(raw_24):,}")
log(f"  r:  A학습 {PRIOR_A:.4f}  2023 {y_23.mean():.4f}  "
    f"B학습 {PRIOR_B:.4f}  2024 {y_24.mean():.4f}\n")


def fit_seeds(Xa, ya, name):
    models = []
    for i, s in enumerate(SEEDS, 1):
        tick(f"{name} seed={s} 학습 중 ({i}/{len(SEEDS)})...")
        m = HistGradientBoostingClassifier(**{**HGB_FAST, "random_state": s})
        models.append(m.fit(Xa, ya))
    return models


def predict(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


models_A = fit_seeds(X_A, y_A, "A(≤2022)")
p_A23 = predict(models_A, X23_A)      # 보정기 적합 재료
p_A24 = predict(models_A, X24_A)      # 참고: exp/10 재현용
models_B = fit_seeds(X_B, y_B, "B(≤2023)")
p_B24 = predict(models_B, X24_B)      # ★ 주인공
tick("학습·예측 완료\n")

# ---------------------------------------------------------------------
# 보정기 — 전부 A(≤2022)의 2023 예측으로만 적합 (train.csv 내부 연도, 합법)
# ---------------------------------------------------------------------
iso_A = IsotonicRegression(out_of_bounds="clip").fit(p_A23, y_23)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


lr_A = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_A23).reshape(-1, 1), y_23)


def platt_A(p):
    return lr_A.predict_proba(logit(p).reshape(-1, 1))[:, 1]


# 반칙 천장 — 2024 정답으로 직접 적합 (비교 기준일 뿐, 절대 제출 금지)
iso_cheat = IsotonicRegression(out_of_bounds="clip").fit(p_B24, y_24)


# ---------------------------------------------------------------------
# 신뢰도 곡선 — B원본 vs B+전이보정
# ---------------------------------------------------------------------
def reliab(p, y, title):
    log(f"  [{title}]")
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    ece = 0.0
    for b in range(int(q.max()) + 1):
        m = q == b
        pm, ym = p[m].mean(), y[m].mean()
        ece += m.sum() * abs(pm - ym) / len(p)
        flag = "  ←" if abs(pm - ym) > 0.02 else ""
        log(f"    {b:4d} {m.sum():9,d} {pm:10.4f} {ym:10.4f} {pm-ym:+9.4f}{flag}")
    log(f"    → ECE = {ece:.5f}\n")


log("=" * 88)
log("A. 신뢰도 곡선 (구간 / 행수 / 예측평균 / 실제평균 / 차이)")
log("=" * 88)
reliab(p_B24, y_24, "B(≤2023) 원본 → 2024   ※ exp/11 A-2와 교차검증용")
reliab(np.clip(iso_A.predict(p_B24), 0, 1), y_24, "B + Isotonic(A로 적합) → 2024")

# ---------------------------------------------------------------------
# B. 점수 비교
# ---------------------------------------------------------------------
log("=" * 88)
log("B. 점수 비교 (전부 2024 채점)")
log("=" * 88)

rows = [
    ("A(≤2022) 원본            [참고]", p_A24),
    ("A + Isotonic(A적합)      [exp/10 재현]", np.clip(iso_A.predict(p_A24), 0, 1)),
    ("B(≤2023) 원본            ← 현재 파이프라인", p_B24),
    ("★ B + Isotonic(A적합)    ← 배포 모사", np.clip(iso_A.predict(p_B24), 0, 1)),
    ("B + Platt(A적합)", platt_A(p_B24)),
    ("B + Isotonic(2024적합)   ← 반칙 천장", np.clip(iso_cheat.predict(p_B24), 0, 1)),
]
z_all = {}
log(f"  {'설정':44s} {'총점':>8s} {'변별력':>8s} {'벌점':>7s}")
log("  " + "-" * 72)
for nm, p in rows:
    z = decompose(p, y_24)
    z_all[nm] = z
    log(f"  {nm:44s} {z['총점']:8.1f} {z['변별력']:8.1f} {z['중심벌점']:7.1f}")

# ---------------------------------------------------------------------
# 판정
# ---------------------------------------------------------------------
log("\n" + "=" * 88)
log("판정")
log("=" * 88)
base = z_all["B(≤2023) 원본            ← 현재 파이프라인"]["총점"]
transfer = z_all["★ B + Isotonic(A적합)    ← 배포 모사"]["총점"]
ceil = z_all["B + Isotonic(2024적합)   ← 반칙 천장"]["총점"]
gain = transfer - base
log(f"  전이 보정 이득     {gain:+8.1f}   (임계 {THR:.0f})")
log(f"  반칙 천장 이득     {ceil - base:+8.1f}")
log(f"  전이에서 잃는 몫   {ceil - transfer:8.1f}")
if gain >= THR:
    log(f"\n  [채택] 전이가 통한다 — 제출 파이프라인에 Isotonic 보정 도입할 것")
    log(f"     실전 절차: 2019~2023 학습 → 2024 예측으로 적합 → 최종모델(≤2024)의 2025 예측에 적용")
elif gain <= -THR:
    log(f"\n  [기각] 전이가 해롭다 — 보정기는 같은 모델에서만 유효. 도입하지 말 것")
else:
    log(f"\n  [보류] 노이즈 범위 — 실전 설정에서는 보정 여지 자체가 작다는 뜻일 가능성")
    log(f"     (극단 설정의 -1882는 학습이 옛날일수록 과신이 커져 생긴 것)")
log(f"  참고: 1위 {LB_TOP:.1f} / 현재 제출 리더보드 830.32")
log("=" * 88)
