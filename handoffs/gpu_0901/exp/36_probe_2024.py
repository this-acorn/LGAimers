# -*- coding: utf-8 -*-
"""
[36] 2024 폴드 재설계 프로브 — 리더보드 정찰이 촉발한 재심 3건 + 수렴 연장

배경 (왜 exp/34를 중단하고 이걸로 교체했나):
  · 리더보드 상위 10% = 117팀이 1119~1225에 밀집. 우리는 898 — 표준 레시피 하나를
    빠뜨리고 있다는 강한 신호. 코드공유엔 RF 베이스라인뿐 → 공유 노트북이 아니라
    다수 팀이 아는 표준 기법이다.
  · §2-5 확정 이후 재심 필요: cb_cat(선수 ID 범주형, 정형대회의 표준 재료)의 -160.9는
    '오염된 2023 폴드' 단독 판정이었다. 캘리브레이션 아티팩트였을 가능성.
  · 드리프트 축(최근 연도만 학습)도 HGB 시절 판정 — CB에서 미측정.

팔 (전부 2024 폴드: train ≤2023 → validate 2024, 2시드 [42,7]):
  ref       CB iter500/d6/lr0.08, 학습 2019~2023            (기준)
  it1000    iter1000/lr0.05                                  (수렴 연장)
  cb_cat    ref + pitcher_id/batter_id를 범주형으로 추가      (★재심 — 표준 레시피)
  win2123   ref, 학습 2021~2023만                            (★재심 — 최근성)
  win2223   ref, 학습 2022~2023만                            (★재심 — 최근성 강)

판정: ref 대비. 2024는 정보 비교의 표준 폴드(§2-5)라 이 순위는 신뢰 가능.
  승자는 LB로 확인 (private 없음 = LB가 곧 최종).

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/36_probe_2024.py   (~2시간)
"""

import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import (log, tick, raw_score, load_train, add_features, CAT, ALL_ENG)

SEEDS = [42, 7]
BASE_PRM = dict(l2_leaf_reg=10.0, verbose=False, thread_count=6,
                allow_writing_files=False)

log("train.csv 로딩...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS65 = BASE + ALL_ENG
NUM_FEATS = [c for c in FEATS65 if c not in CAT]

va_raw = df[df["season"] == 2024].reset_index(drop=True)
y_va = va_raw["control_success"].to_numpy()


def make_pools(tr_from, with_id_cat):
    tr_raw = df[(df["season"] >= tr_from) & (df["season"] <= 2023)].reset_index(drop=True)
    y_tr = tr_raw["control_success"].to_numpy()
    prior = float(y_tr.mean())
    tr = add_features(tr_raw, prior)
    va = add_features(va_raw, prior)
    cats = list(CAT) + (["pid_cat", "bid_cat"] if with_id_cat else [])

    def frame(d):
        out = d[NUM_FEATS].copy()
        for c in CAT:
            out[c] = d[c].astype(str)
        if with_id_cat:
            out["pid_cat"] = d["pitcher_id"].astype(str)
            out["bid_cat"] = d["batter_id"].astype(str)
        return out

    ptr = Pool(frame(tr), y_tr, cat_features=cats)
    pva = Pool(frame(va), cat_features=cats)
    return ptr, pva, len(tr_raw)


ARMS = [
    ("ref", 2019, False, dict(iterations=500, depth=6, learning_rate=0.08)),
    ("it1000", 2019, False, dict(iterations=1000, depth=6, learning_rate=0.05)),
    ("cb_cat", 2019, True, dict(iterations=500, depth=6, learning_rate=0.08)),
    ("win2123", 2021, False, dict(iterations=500, depth=6, learning_rate=0.08)),
    ("win2223", 2022, False, dict(iterations=500, depth=6, learning_rate=0.08)),
]

results = {}
for name, tr_from, idcat, prm in ARMS:
    ptr, pva, ntr = make_pools(tr_from, idcat)
    tick(f"{name}: 학습 {ntr:,}행 ({tr_from}~2023{', +ID범주' if idcat else ''})")
    scores, preds = [], []
    for s in SEEDS:
        t0 = time.time()
        m = CatBoostClassifier(**BASE_PRM, **prm, random_seed=s).fit(ptr)
        p = m.predict_proba(pva)[:, 1]
        preds.append(p)
        scores.append(raw_score(p, y_va))
        tick(f"  {name} seed={s}  {scores[-1]:8.1f}  ({time.time()-t0:.0f}s)")
    ens = np.mean(preds, axis=0)
    results[name] = {"seed": scores, "ens": raw_score(ens, y_va),
                     "mean": float(ens.mean()), "std": float(ens.std())}
    np.save(f"lab/36_{name}_ens.npy", ens.astype("float32"))
    log(f"  → {name:8s} 앙상블 {results[name]['ens']:8.1f}  "
        f"(예측평균 {results[name]['mean']:.4f})")
    del ptr, pva

log("\n" + "=" * 78)
log("판정 (2024 폴드 = 정보 비교 표준, ref 대비)")
log("=" * 78)
ref = results["ref"]["ens"]
log(f"  {'팔':>8s} {'seed42':>9s} {'seed7':>9s} {'앙상블':>9s} {'이득':>8s} {'예측평균':>9s}")
for name, *_ in ARMS:
    r = results[name]
    log(f"  {name:>8s} {r['seed'][0]:>9.1f} {r['seed'][1]:>9.1f} "
        f"{r['ens']:>9.1f} {r['ens']-ref:>+8.1f} {r['mean']:>9.4f}")
log(f"\n  (2024 실제 r = {y_va.mean():.4f})")
log("  cb_cat 양수면 → 2023의 -160.9는 오염폴드 아티팩트였음 → 표준 레시피 복권")
log("  win 팔 양수면 → 드리프트 축 부활 (CB 한정) → 2025 중심 개선 후보")
log("=" * 78)
