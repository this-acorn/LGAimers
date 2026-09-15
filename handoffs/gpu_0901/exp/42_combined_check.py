# -*- coding: utf-8 -*-
"""
[42] 적층 확인 — 시즌 진행분(+107.9) + 물리 프로필(+32.2) 결합, 2024 폴드

배포 피처셋 결정용: 결합 > CS 단독(782.3)이면 88피처 배포, 아니면 76피처(CS만).
기준 (같은 시드 42/7): ref 674.4 / +물리 706.6 / +CS 782.3

구현: exp/39 §1(트랙맨 프로필·attach)과 exp/41 앞부분(상수표·attach_cs 정의+train 로드)을
  마커 기준으로 exec 재사용 — 로직 복제 없이 동일 코드 실행 보장.

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/42_combined_check.py  (~20분)
"""

import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import log, tick, raw_score, add_features, CAT, ALL_ENG

MARK = "# =====================================================================\n# "

# ---- exp/39 §1 실행: ptbl/btbl/attach/P_FEATS/B_FEATS 확보 (트랙맨 로드 포함) ----
src39 = open("exp/39_tm_physical_v2.py", encoding="utf-8").read()
sec39 = src39.split(MARK + "2.")[0]
exec(compile(sec39, "exp39_sec1", "exec"))
assert "attach" in dir() and len(ptbl) > 0

# ---- exp/41 앞부분 실행: train 로드 + build_const/attach_cs/CS_FEATS/P_RATES 확보 ----
src41 = open("exp/41_season_progress.py", encoding="utf-8").read()
head41 = src41.split(MARK + "1.")[0]
exec(compile(head41, "exp41_head", "exec"))
assert "build_const" in dir() and "attach_cs" in dir() and "df" in dir()

SEEDS = [42, 7]
CB_PRM = dict(iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=10.0,
              verbose=False, thread_count=14, allow_writing_files=False)
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS65 = BASE + ALL_ENG

hist = df[df.season <= 2023]
cp = build_const(hist, "pitcher_id", "asof_pitcher_n", P_RATES, "control_success")
cb_tbl = build_const(hist, "batter_id", "asof_batter_n", B_RATES, "control_success")
tick("상수표 완료")

tr_raw = df[df.season <= 2023].reset_index(drop=True)
va_raw = df[df.season == 2024].reset_index(drop=True)
y_tr = tr_raw["control_success"].to_numpy()
y_va = va_raw["control_success"].to_numpy()
prior = float(y_tr.mean())

tick("학습 행 시즌별 CS 부착...")
parts = []
for S in sorted(tr_raw.season.unique()):
    h = df[df.season <= S - 1]
    rows = tr_raw[tr_raw.season == S]
    if len(h) == 0:
        cpS, cbS = cp.iloc[0:0], cb_tbl.iloc[0:0]
    else:
        cpS = build_const(h, "pitcher_id", "asof_pitcher_n", P_RATES, "control_success")
        cbS = build_const(h, "batter_id", "asof_batter_n", B_RATES, "control_success")
    parts.append(attach_cs(rows, cpS, cbS, None))
tr = attach(add_features(pd.concat(parts).sort_index(), prior))
va = attach(add_features(attach_cs(va_raw, cp, cb_tbl, None), prior))
del df, tr_raw, va_raw, parts
tick("피처 부착 완료 (CS 11 + 물리 12)")

p_ref = np.load("lab/36_ref_ens.npy").astype("float64")
s_ref = raw_score(p_ref, y_va)
COMBO = CS_FEATS + P_FEATS + B_FEATS
NUM = [c for c in FEATS65 if c not in CAT] + COMBO


def frame(d):
    out = d[NUM].copy()
    for c in CAT:
        out[c] = d[c].astype(str)
    return out


ptr = Pool(frame(tr), y_tr, cat_features=list(CAT))
pva = Pool(frame(va), cat_features=list(CAT))
preds = []
for sd in SEEDS:
    t0 = time.time()
    m = CatBoostClassifier(**CB_PRM, random_seed=sd).fit(ptr)
    preds.append(m.predict_proba(pva)[:, 1])
    tick(f"combo seed={sd}  {raw_score(preds[-1], y_va):8.1f}  ({time.time()-t0:.0f}s)")
ens = np.mean(preds, axis=0)
np.save("lab/42_combo_ens.npy", ens.astype("float32"))
sc = raw_score(ens, y_va)

log("\n" + "=" * 68)
log("적층 판정 (2024 폴드)")
log("=" * 68)
log(f"  기준(65)              674.4")
log(f"  +물리(77)             706.6  (+32.2)")
log(f"  +CS(76)               782.3  (+107.9)")
log(f"  +CS+물리(88)        {sc:8.1f}  ({sc-s_ref:+.1f})")
log(f"  → 배포 피처셋: {'88 (결합)' if sc > 782.3 else '76 (CS 단독)'}")
log("=" * 68)
