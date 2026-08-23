"""
[10] 확률 보정(calibration) — 지금까지의 사각지대

실행:  PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 py -3.12 -u exp/10_calibration.py
       (튜닝이 동시에 돌고 있으면 스레드를 제한해서 방해하지 않게 한다)

왜 하는가:
  Brier = 불확실성 − 해상도 + 신뢰도오차
                     └피처·모델┘  └calibration┘

  신뢰도오차는 두 층이다.
    ① 전역: 예측 평균 == 실제 평균인가   → "중심벌점"으로 계속 재왔음. 소거 완료
    ② 국소: 0.6이라 한 행들이 진짜 60% 성공하는가 → ★ 한 번도 안 재봤다

  ②는 우리가 "변별력"이라 부르던 숫자 안에 숨어 있다 (변별력 = 해상도 − 국소오차).
  드리프트가 심한 데이터에서는 평균은 맞아도 '퍼짐'이 틀어지는 일이 흔하다.
  그건 평균 이동으로 절대 고쳐지지 않는다.

누수 방지 설계:
  학습    2019~2022
  보정기  2023 에 적합      ← 2024로 적합하면 미래를 보는 것이라 반칙
  채점    2024
  (실전에서는 학습 2019~2023 / 보정 2024 / 채점 2025 로 한 칸씩 밀면 된다)
"""

import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, "exp")
from common import (log, tick, decompose, load_train, add_features,
                    fit_encoder, to_matrix, ALL_ENG, HGB_FAST, LB_TOP)

SEEDS = [42, 7, 123, 2024]
NOISE_SD = 15.3
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


log("train.csv 로딩 중...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS = BASE + ALL_ENG

tr = df[df["season"] <= 2022].reset_index(drop=True)   # 학습
cal = df[df["season"] == 2023].reset_index(drop=True)  # 보정기 적합
te = df[df["season"] == 2024].reset_index(drop=True)   # 채점

y_tr = tr["control_success"].to_numpy()
y_cal = cal["control_success"].to_numpy()
y_te = te["control_success"].to_numpy()
PRIOR = float(y_tr.mean())

tr, cal, te = (add_features(x, PRIOR) for x in (tr, cal, te))
enc = fit_encoder(tr, FEATS)
Xtr, Xcal, Xte = (to_matrix(x, FEATS, enc) for x in (tr, cal, te))
tick(f"준비 완료 — 학습 {len(tr):,} / 보정 {len(cal):,} / 채점 {len(te):,}")
log(f"  r:  학습 {PRIOR:.6f}  보정 {y_cal.mean():.6f}  채점 {y_te.mean():.6f}\n")

# ---- 4시드 앙상블 학습 ----
acc_cal = np.zeros(len(cal))
acc_te = np.zeros(len(te))
for i, s in enumerate(SEEDS, 1):
    tick(f"seed={s} 학습 중 ({i}/{len(SEEDS)})...")
    m = HistGradientBoostingClassifier(**dict(HGB_FAST, random_state=s)).fit(Xtr, y_tr)
    acc_cal += m.predict_proba(Xcal)[:, 1]
    acc_te += m.predict_proba(Xte)[:, 1]
p_cal = acc_cal / len(SEEDS)
p_te = acc_te / len(SEEDS)
tick("학습 완료")


# =====================================================================
# A. 국소 신뢰도 진단 — 어디가 얼마나 틀어져 있나
# =====================================================================
log("\n" + "=" * 88)
log("A. 신뢰도 곡선 — '0.6이라 한 행들이 진짜 60% 성공하는가'")
log("=" * 88)

def reliability(p, y, label, nbins=10):
    q = pd.qcut(p, nbins, labels=False, duplicates="drop")
    log(f"\n  [{label}]")
    log(f"    {'구간':>4s} {'행수':>9s} {'예측평균':>10s} {'실제평균':>10s} {'차이':>9s}")
    log("    " + "-" * 46)
    for b in range(q.max() + 1):
        m = q == b
        pm, ym = p[m].mean(), y[m].mean()
        flag = "  ←" if abs(pm - ym) > 0.02 else ""
        log(f"    {b:4d} {m.sum():9,d} {pm:10.4f} {ym:10.4f} {pm-ym:+9.4f}{flag}")
    # 가중 평균 절대오차 (ECE)
    ece = sum((q == b).sum() * abs(p[q == b].mean() - y[q == b].mean())
              for b in range(q.max() + 1)) / len(p)
    log(f"    → ECE(가중평균 절대오차) = {ece:.5f}")
    return ece

ece_cal = reliability(p_cal, y_cal, "2023 (보정용, 모델이 안 본 해)")
ece_te = reliability(p_te, y_te, "2024 (채점용)")


# =====================================================================
# B. 보정 방법 비교
# =====================================================================
log("\n" + "=" * 88)
log("B. 보정 방법 — 2023으로 적합해서 2024에 적용")
log("=" * 88)

results = {}
z0 = decompose(p_te, y_te)
results["원본 (보정 없음)"] = z0
log(f"  {'원본 (보정 없음)':28s} 총점 {z0['총점']:7.1f}  변별력 {z0['변별력']:7.1f}  "
    f"벌점 {z0['중심벌점']:5.1f}")

# --- ① shrink: p' = r + w(p - r).  w는 2023에서 최적값을 찾는다 ---
r_cal = float(p_cal.mean())
ws = np.arange(0.5, 1.51, 0.05)
briers = [np.mean((r_cal + w * (p_cal - r_cal) - y_cal) ** 2) for w in ws]
w_best = float(ws[int(np.argmin(briers))])
p_shrink = np.clip(p_te.mean() + w_best * (p_te - p_te.mean()), EPS, 1 - EPS)
z = decompose(p_shrink, y_te)
results[f"① shrink (w={w_best:.2f})"] = z
log(f"  {'① shrink (w=%.2f)' % w_best:28s} 총점 {z['총점']:7.1f}  변별력 {z['변별력']:7.1f}  "
    f"벌점 {z['중심벌점']:5.1f}")

# --- ② Platt: 로그오즈에 로지스틱 회귀 ---
pl = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_cal).reshape(-1, 1), y_cal)
p_platt = pl.predict_proba(logit(p_te).reshape(-1, 1))[:, 1]
z = decompose(p_platt, y_te)
results["② Platt scaling"] = z
log(f"  {'② Platt scaling':28s} 총점 {z['총점']:7.1f}  변별력 {z['변별력']:7.1f}  "
    f"벌점 {z['중심벌점']:5.1f}   (a={pl.coef_[0][0]:.4f}, b={pl.intercept_[0]:.4f})")

# --- ③ Isotonic: 단조증가 함수로 자유롭게 재사상 ---
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_cal, y_cal)
p_iso = iso.predict(p_te)
z = decompose(p_iso, y_te)
results["③ Isotonic"] = z
log(f"  {'③ Isotonic':28s} 총점 {z['총점']:7.1f}  변별력 {z['변별력']:7.1f}  "
    f"벌점 {z['중심벌점']:5.1f}")

# --- 참고: 2024 정답으로 적합 (반칙. 보정의 천장) ---
iso_cheat = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_te, y_te)
z_cheat = decompose(iso_cheat.predict(p_te), y_te)
log(f"  {'★ 반칙: 2024로 적합':28s} 총점 {z_cheat['총점']:7.1f}  "
    f"변별력 {z_cheat['변별력']:7.1f}   ← 보정으로 딸 수 있는 절대 천장")


# =====================================================================
log("\n" + "=" * 88)
log("판정")
log("=" * 88)

best = max(results, key=lambda k: results[k]["총점"])
gain = results[best]["총점"] - z0["총점"]
log(f"  {'방법':28s} {'총점':>8s} {'원본대비':>9s}")
log("  " + "-" * 48)
for k, z in sorted(results.items(), key=lambda x: -x[1]["총점"]):
    log(f"  {k:28s} {z['총점']:8.1f} {z['총점']-z0['총점']:+9.1f}")

log(f"\n  ▶ 최고: {best}  ({gain:+.1f}점)")
log(f"  ▶ 반칙 천장 대비: {z_cheat['총점']-z0['총점']:+.1f}점이 이론상 최대")

thr = 2 * NOISE_SD / 2      # 4시드 앙상블 기준 유의 임계 (약 15점)
if gain >= thr:
    log(f"\n  ✅ 유의미 (임계 {thr:.0f}점 초과) — 제출 모델에 보정 적용할 것")
    log(f"     실전 적용법: 학습 2019~2023 / 보정기 2024에 적합 / 2025에 적용")
else:
    log(f"\n  ❌ 노이즈 범위 (임계 {thr:.0f}점 미만) — 모델이 이미 잘 보정돼 있음")
    log(f"     GBDT는 log loss로 학습하므로 확률이 원래 정직한 편이다")
log("=" * 88)
