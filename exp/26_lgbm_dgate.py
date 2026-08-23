# -*- coding: utf-8 -*-
"""
[26] LightGBM D-게이트 프로브 — 하네스에 태울 가치가 있는지 5분에 거른다

배경:
  HistGradientBoosting은 sklearn이 LightGBM을 본떠 만든 구현이다 (히스토그램 binning,
  leaf-wise 성장). 앙상블 다양성 관점에서 LGBM↔HGB 거리는 XGB↔HGB보다 가깝고,
  XGB 혼합은 이미 이득 0.0으로 사망했다. 반면 CatBoost(cb_num)는 구조가 진짜 달라서
  2023 프로브 +98.4 / D=2.935e-04 였다.

게이트 설계 (자체 보정):
  D(lgbm, hgb앙상블) 을 두 참조값 사이에서 판정한다.
    바닥: D(hgb 시드i, 시드j) 평균  ← "그냥 시드 하나 더"의 거리 (exp/25 npz에서 공짜)
    천장: cb_num의 D=2.935e-04     ← "구조가 다른 모델"의 거리
  판정:
    D < 바닥×2      → LGBM은 9번째 시드일 뿐. 축 폐기.
    D ≥ 천장×0.5    → 예상 밖 다양성. 4개 연도 하네스로 승격.
    그 사이         → 혼합곡선 이득 크기로 판단 (2023 한 해 참고치).

전제: exp/25가 lab/25_preds.npz 를 저장한 뒤 실행 (2023 hgb 시드별 예측 재사용,
      HGB 재학습 0회 — LGBM 2시드만 새로 학습).

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/26_lgbm_dgate.py
"""

import sys
import time
import numpy as np
import lightgbm as lgb

sys.path.insert(0, "exp")
from common import (log, tick, raw_score, load_train, add_features,
                    fit_encoder, to_matrix, ALL_ENG)

YEAR = 2023
SEEDS = [42, 7]
CB_D_REF = 2.935e-04          # exp/23 실측: cb_num↔hgb 불일치 (구조 다른 모델의 참조)
NPZ = "lab/25_preds.npz"
# HGB_FAST를 최대한 미러링 — 하이퍼파라미터 차이가 아니라 '구현 다양성'만 재기 위해
LGB_PARAMS = dict(n_estimators=120, learning_rate=0.08, num_leaves=31,
                  min_child_samples=200, reg_lambda=10.0, max_bin=255,
                  verbose=-1, n_jobs=6)

# ---- exp/25 npz에서 2023 HGB 예측 로드 ----
z = np.load(NPZ)
hgb_seeds = [z[k] for k in z.files if k.startswith(f"{YEAR}_hgb_")]
y_va = z[f"{YEAR}_y"].astype(float)
assert len(hgb_seeds) >= 2, f"npz에 {YEAR} hgb 예측 없음: {z.files}"
p_hgb = np.mean(hgb_seeds, axis=0)
log(f"npz 로드 — {YEAR} hgb {len(hgb_seeds)}시드, 검증 {len(y_va):,}행")

# 바닥 참조: HGB 시드끼리의 평균 불일치
floor_ds = [float(np.mean((hgb_seeds[i] - hgb_seeds[j]) ** 2))
            for i in range(len(hgb_seeds)) for j in range(i + 1, len(hgb_seeds))]
D_FLOOR = float(np.mean(floor_ds))
log(f"바닥 D (hgb 시드간 평균)      = {D_FLOOR:.3e}")
log(f"천장 D (cb_num, exp/23 실측)  = {CB_D_REF:.3e}   비율 {CB_D_REF/D_FLOOR:.1f}배")

# ---- LGBM 학습 (2023 검증 = 2022까지 학습) ----
log("\ntrain.csv 로딩 중...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS65 = BASE + ALL_ENG
tr_raw = df[df["season"] <= YEAR - 1].reset_index(drop=True)
va_raw = df[df["season"] == YEAR].reset_index(drop=True)
y_tr = tr_raw["control_success"].to_numpy()
prior = float(y_tr.mean())
tr, va = add_features(tr_raw, prior), add_features(va_raw, prior)
enc = fit_encoder(tr, FEATS65)
Xtr, Xva = to_matrix(tr, FEATS65, enc), to_matrix(va, FEATS65, enc)
del df, tr, va, tr_raw, va_raw
tick(f"준비 완료 — 학습 {len(y_tr):,}")

preds = []
for s in SEEDS:
    t0 = time.time()
    m = lgb.LGBMClassifier(**LGB_PARAMS, random_state=s).fit(Xtr, y_tr)
    p = m.predict_proba(Xva)[:, 1]
    preds.append(p)
    tick(f"lgbm seed={s}  {raw_score(p, y_va):8.1f}  ({time.time()-t0:.0f}s)")
p_lgb = np.mean(preds, axis=0)
s_hgb, s_lgb = raw_score(p_hgb, y_va), raw_score(p_lgb, y_va)

# ---- D 게이트 ----
D = float(np.mean((p_hgb - p_lgb) ** 2))
log(f"\n{'='*70}")
log(f"D 게이트  (검증 {YEAR})")
log(f"{'='*70}")
log(f"  hgb 앙상블 {s_hgb:8.1f}   lgbm 앙상블 {s_lgb:8.1f}  (차이 {s_lgb-s_hgb:+.1f})")
log(f"  D(lgbm, hgb) = {D:.3e}")
log(f"    바닥(시드간) 대비 {D/D_FLOOR:5.1f}배   천장(cb_num) 대비 {D/CB_D_REF:5.2f}배")
log(f"\n  {'w':>5s} {'혼합점수':>10s} {'기준대비':>10s}")
for w in [0.0, 0.1, 0.2, 0.3, 0.5]:
    sc = raw_score((1 - w) * p_hgb + w * p_lgb, y_va)
    log(f"  {w:>5.1f} {sc:>10.1f} {sc-s_hgb:>+10.1f}")

log(f"\n판정:")
if D < 2 * D_FLOOR:
    log(f"  [폐기] D가 시드간 거리의 2배 미만 — LGBM은 사실상 추가 시드. "
        f"하네스 불필요.")
elif D >= 0.5 * CB_D_REF:
    log(f"  [승격] 예상 밖 다양성 — 4개 연도 하네스로 올릴 것.")
else:
    log(f"  [회색] 혼합 이득 크기로 판단할 것 (위 표, 2023 한 해 참고치).")
log("=" * 70)
