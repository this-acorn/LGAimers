# -*- coding: utf-8 -*-
"""
[34] CatBoost 파라미터 프로브 — 2023 폴드 (2025와 가장 닮은 CB 친화 해)

배경: 현재 CB 파라미터(iter500/depth6/lr0.08/l2=10)는 HGB 미러링일 뿐 튜닝 이력 0.
  2025 실측으로 CB 단독이 최적임이 확정된 지금, 이 축의 기대값이 가장 크다.

주의(정직하게): 2023 한 해로 고르는 것은 단일연도 선택이다. 로컬 4년 중 3년은
  CB 적대적이라 다연도 평균은 오히려 오도한다 — 2025가 2023형임이 '실측'됐으므로
  2023을 프록시로 쓰되, 최종 확인은 LB 제출로 한다 (private 없음 = LB가 곧 최종).

팔 (각 2시드 [42, 7], paired 비교):
  ref      iter500  d6 lr0.08 l2=10   (현재)
  it1000   iter1000 d6 lr0.05 l2=10   (수렴 연장)
  it2000   iter2000 d6 lr0.03 l2=10   (수렴 연장 강)
  d8       iter700  d8 lr0.06 l2=10   (용량 확장)
판정: ref 대비 paired 이득. 2023 폴드라 임계는 참고용 — 상위 1~2개만 2024 폴드로
  교차 확인 후 전체 학습 → LB.

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/34_cb_tune_probe.py   (~2.5시간)
"""

import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import (log, tick, raw_score, load_train, add_features, CAT, ALL_ENG)

YEAR = 2023
SEEDS = [42, 7]
BASE_PRM = dict(l2_leaf_reg=10.0, verbose=False, thread_count=6,
                allow_writing_files=False)
ARMS = [
    ("ref", dict(iterations=500, depth=6, learning_rate=0.08)),
    ("it1000", dict(iterations=1000, depth=6, learning_rate=0.05)),
    ("it2000", dict(iterations=2000, depth=6, learning_rate=0.03)),
    ("d8", dict(iterations=700, depth=8, learning_rate=0.06)),
]

log("train.csv 로딩...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS65 = BASE + ALL_ENG
NUM_FEATS = [c for c in FEATS65 if c not in CAT]
tr_raw = df[df["season"] <= YEAR - 1].reset_index(drop=True)
va_raw = df[df["season"] == YEAR].reset_index(drop=True)
y_tr = tr_raw["control_success"].to_numpy()
y_va = va_raw["control_success"].to_numpy()
prior = float(y_tr.mean())
tr, va = add_features(tr_raw, prior), add_features(va_raw, prior)
del df, tr_raw, va_raw


def frame(d):
    out = d[NUM_FEATS].copy()
    for c in CAT:
        out[c] = d[c].astype(str)
    return out


ptr = Pool(frame(tr), y_tr, cat_features=list(CAT))
pva = Pool(frame(va), cat_features=list(CAT))
del tr, va
tick(f"준비 완료 — 검증 {YEAR}: 학습 {len(y_tr):,} / 검증 {len(y_va):,}")

results = {}
for name, prm in ARMS:
    scores, preds = [], []
    for s in SEEDS:
        t0 = time.time()
        m = CatBoostClassifier(**BASE_PRM, **prm, random_seed=s).fit(ptr)
        p = m.predict_proba(pva)[:, 1]
        preds.append(p)
        scores.append(raw_score(p, y_va))
        tick(f"{name} seed={s}  {scores[-1]:8.1f}  ({time.time()-t0:.0f}s)")
    ens = raw_score(np.mean(preds, axis=0), y_va)
    results[name] = {"seed": scores, "ens": ens}
    log(f"  → {name:8s} 앙상블 {ens:8.1f}")

log("\n" + "=" * 76)
log(f"판정 (2023 폴드, ref 대비)")
log("=" * 76)
ref = results["ref"]
log(f"  {'팔':>8s} {'seed42':>9s} {'seed7':>9s} {'앙상블':>9s} {'이득':>8s}")
for name, _ in ARMS:
    r = results[name]
    g = r["ens"] - ref["ens"]
    log(f"  {name:>8s} {r['seed'][0]:>9.1f} {r['seed'][1]:>9.1f} "
        f"{r['ens']:>9.1f} {g:>+8.1f}")
log("\n  다음: 최고 팔이 +15 이상이면 2024 폴드 교차 확인 → 전체 학습 → LB 검증")
log("=" * 76)
