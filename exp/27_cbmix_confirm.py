# -*- coding: utf-8 -*-
"""
[27] HGB×CatBoost 혼합 — 독립 시드 확증 (회색지대/채택 전 표준 절차)

exp/25 결과 (선택 시드 [42,7,123,2024]):
  · cb_num 단독: 2021 -49.7 / 2022 -38.8 / 2023 +116.6 / 2024 -4.1 → 단독 탈락
  · 혼합 w=0.5: +9.0 / +7.3 / +81.4 / +23.7 → 사전 등록 기준 3개 전부 통과
  · 교훈: 프로브의 +98.4는 2023이라는 '운 좋은 해'였다. 단독은 지고 혼합만 이긴다
    (다양성 항 w(1-w)D 가 이득의 원천 — 순수 실력 차가 아님).

이 스크립트: 선택에 쓰지 않은 독립 시드 [99, 555, 31337, 1]로 같은 절차를 반복한다.
  w는 재선택하지 않는다 — exp/25가 고른 w=0.5 를 그대로 고정 적용해서
  "그 결정이 새 시드에서도 유지되는가"만 본다 (재선택하면 확증이 아니라 재선택이다).
  참고로 w=0.3 (보수 팔)도 함께 찍는다 — 판정에는 쓰지 않고 기록만.

확증 통과 기준 (사전 등록):
  · w=0.5 고정 혼합이 4년 중 3년 이상 양수
  · 2024 이득 > 0
  · 4년 평균 ≥ +15 (exp/25의 +30.4 대비 절반 수준까지 허용 — 시드 노이즈 감안)

예측은 전부 lab/27_preds.npz 에 저장 (후속 tm_cov 하네스가 base65 팔 재사용).

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/27_cbmix_confirm.py
소요 예상: CatBoost 16회 + HGB 16회 ≈ 80~90분
"""

import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import (log, tick, raw_score, load_train, add_features,
                    fit_encoder, to_matrix, ALL_ENG, HGB_FAST, CAT)

YEARS = [2021, 2022, 2023, 2024]
SEEDS = [99, 555, 31337, 1]          # ★ 독립 시드 — exp/23/25 선택에 미사용
W_FIXED = 0.5                         # exp/25가 선택한 값. 여기서 재선택 금지
W_REF = 0.3                           # 보수 참고 팔 (판정 미사용)
CB_PARAMS = dict(iterations=500, learning_rate=0.08, depth=6,
                 l2_leaf_reg=10.0, verbose=False, thread_count=6,
                 allow_writing_files=False)
NPZ = "lab/27_preds.npz"

log("train.csv 로딩 중...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS65 = BASE + ALL_ENG
NUM_FEATS = [c for c in FEATS65 if c not in CAT]
tick(f"로드 {df.shape}")

dump, results = {}, {}

for Y in YEARS:
    tr_raw = df[df["season"] <= Y - 1].reset_index(drop=True)
    va_raw = df[df["season"] == Y].reset_index(drop=True)
    y_tr = tr_raw["control_success"].to_numpy()
    y_va = va_raw["control_success"].to_numpy()
    prior = float(y_tr.mean())
    tr, va = add_features(tr_raw, prior), add_features(va_raw, prior)
    log(f"\n{'='*88}")
    log(f"검증연도 {Y}  학습 {len(tr):,} / 검증 {len(va):,}")
    log("=" * 88)

    enc = fit_encoder(tr, FEATS65)
    Xtr, Xva = to_matrix(tr, FEATS65, enc), to_matrix(va, FEATS65, enc)
    hgb_p = []
    for s in SEEDS:
        t0 = time.time()
        m = HistGradientBoostingClassifier(**{**HGB_FAST, "random_state": s}).fit(Xtr, y_tr)
        p = m.predict_proba(Xva)[:, 1]
        hgb_p.append(p)
        dump[f"{Y}_hgb_{s}"] = p.astype("float32")
        tick(f"{Y} hgb seed={s}  {raw_score(p, y_va):8.1f}  ({time.time()-t0:.0f}s)")
    del Xtr, Xva

    def frame(d):
        out = d[NUM_FEATS].copy()
        for c in CAT:
            out[c] = d[c].astype(str)
        return out

    ptr = Pool(frame(tr), y_tr, cat_features=list(CAT))
    pva = Pool(frame(va), cat_features=list(CAT))
    cb_p = []
    for s in SEEDS:
        t0 = time.time()
        m = CatBoostClassifier(**CB_PARAMS, random_seed=s).fit(ptr)
        p = m.predict_proba(pva)[:, 1]
        cb_p.append(p)
        dump[f"{Y}_cbnum_{s}"] = p.astype("float32")
        tick(f"{Y} cb_num seed={s}  {raw_score(p, y_va):8.1f}  ({time.time()-t0:.0f}s)")
    del ptr, pva

    dump[f"{Y}_y"] = y_va.astype("int8")
    h, c = np.mean(hgb_p, axis=0), np.mean(cb_p, axis=0)
    s_h = raw_score(h, y_va)
    g05 = raw_score((1 - W_FIXED) * h + W_FIXED * c, y_va) - s_h
    g03 = raw_score((1 - W_REF) * h + W_REF * c, y_va) - s_h
    g10 = raw_score(c, y_va) - s_h
    results[Y] = (g05, g03, g10)
    log(f"    → hgb {s_h:8.1f}   혼합w0.5 {g05:+7.1f}   (참고 w0.3 {g03:+7.1f} / "
        f"cb단독 {g10:+7.1f})")
    np.savez_compressed(NPZ, **dump)      # 연도마다 즉시 저장
    log(f"    [저장] {Y}까지 {NPZ}")
    del tr, va, tr_raw, va_raw

# =====================================================================
log("\n" + "=" * 88)
log("독립 시드 확증 판정  (w=0.5 고정 — exp/25 선택값)")
log("=" * 88)
log(f"  {'연도':>6s} {'w0.5 이득':>10s} {'(w0.3)':>9s} {'(cb단독)':>9s}")
g = []
for Y in YEARS:
    g05, g03, g10 = results[Y]
    g.append(g05)
    log(f"  {Y:>6d} {g05:>+10.1f} {g03:>+9.1f} {g10:>+9.1f}")
a = np.array(g)
log(f"  {'─'*40}")
log(f"  평균 {a.mean():+.1f} / 최악 {a.min():+.1f} / 양수 {(a>0).sum()}/4")
log(f"\n  exp/25 (선택 시드): +9.0 / +7.3 / +81.4 / +23.7  평균 +30.4")
c1 = (a > 0).sum() >= 3
c2 = results[2024][0] > 0
c3 = a.mean() >= 15.0
log(f"\n  [{'O' if c1 else 'X'}] 4년 중 3년 이상 양수: {(a>0).sum()}/4")
log(f"  [{'O' if c2 else 'X'}] 2024 양수: {results[2024][0]:+.1f}")
log(f"  [{'O' if c3 else 'X'}] 평균 ≥ +15: {a.mean():+.1f}")
if c1 and c2 and c3:
    log(f"\n  ★ 확증 통과 — HGB+CatBoost w=0.5 혼합을 제출 아키텍처 후보로 승격.")
    log(f"    다음: 전체 데이터(≤2024) 학습 + venv311 저장 + script.py 개편 + zip 검증")
else:
    log(f"\n  ✗ 확증 실패 — 선택 시드 결과는 시드 노이즈였을 가능성. 혼합 채택 보류.")
log("=" * 88)
