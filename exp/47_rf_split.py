# -*- coding: utf-8 -*-
"""
[47] R/F 분리 검증 — 퓨처스(2군) 세그먼트 전용 처리의 가치 (2024 폴드, CS76 기준)

배경: game_type F=퓨처스리그(공식 확인). 행의 12%, 성공률 레짐이 R과 다르게 요동
  (2022 .709 → 2023 .473). 단일 모델은 game_type을 CAT 피처로만 처리 중.

진단(학습 없음): 현 CS76 예측의 R/F 세그먼트별 성적 — F에서 얼마나 잃고 있나
실험: MoE (R 전용 모델 + F 전용 모델, 행의 game_type으로 라우팅) vs 단일 CS76

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/47_rf_split.py  (~50분)
"""

import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import log, tick, raw_score, load_train, add_features, CAT, ALL_ENG

import importlib.util
spec = importlib.util.spec_from_file_location("s10", "submissions/submit10_src/script.py")
s10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s10)
attach_cs, CS_FEATS, P_RATES, B_RATES = s10.attach_cs, s10.CS_FEATS, s10.P_RATES, s10.B_RATES

SEEDS = [42, 7]
CB_PRM = dict(iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=10.0,
              verbose=False, thread_count=10, allow_writing_files=False)


def build_const(src, id_col, n_col, rates):
    d = src.sort_values(n_col).groupby(id_col).tail(1)
    out = pd.DataFrame({"id": d[id_col].to_numpy()})
    n_last = d[n_col].fillna(0).to_numpy("float64")
    out["N_end"] = n_last + 1
    for k, col in rates.items():
        r = d[col].fillna(0).to_numpy("float64")
        if k == "succ":
            out[f"S_{k}"] = np.round(r * n_last) + d["control_success"].to_numpy("float64")
        else:
            out[f"S_{k}"] = r * (n_last + 1)
    return out


log("train 로딩...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS76 = BASE + ALL_ENG + CS_FEATS

hist = df[df.season <= 2023]
cp = build_const(hist, "pitcher_id", "asof_pitcher_n", P_RATES)
cb_tbl = build_const(hist, "batter_id", "asof_batter_n", B_RATES)
tr_raw = df[df.season <= 2023].reset_index(drop=True)
va_raw = df[df.season == 2024].reset_index(drop=True)
y_va = va_raw["control_success"].to_numpy()
prior = float(tr_raw["control_success"].mean())

# =====================================================================
# 진단: 현 CS76 예측의 세그먼트별 성적 (학습 없음 — 저장된 예측 사용)
# =====================================================================
log("\n" + "=" * 74)
log("진단: CS76(2시드)의 R/F 세그먼트별 성적 (2024 폴드)")
log("=" * 74)
p_cs = np.load("lab/41_cs_ens.npy").astype("float64")
gt = va_raw["game_type"].astype(str).to_numpy()
for seg in ["R", "F"]:
    m = gt == seg
    ys, ps = y_va[m].astype(float), p_cs[m]
    r = ys.mean()
    seg_score = 100000 * (1 - np.mean((ps - ys) ** 2) / (r * (1 - r)))
    d_ = ps.mean() - r
    pen = 100000 * d_ * d_ / (r * (1 - r))
    log(f"  {seg}: 행 {m.sum():,} ({m.mean()*100:.1f}%)  r={r:.4f}  "
        f"세그점수 {seg_score:8.1f}  (중심벌점 {pen:.1f}, d={d_:+.4f})")
log("  ※ 세그점수 = 그 세그먼트만의 r 기준 BSS. F가 크게 나쁘면 전용 처리 여지 있음")

# =====================================================================
# MoE 실험
# =====================================================================
log("\n" + "=" * 74)
log("MoE: R 전용 + F 전용 vs 단일 CS76 (782.3)")
log("=" * 74)
tick("피처 부착...")
parts = []
for S in sorted(tr_raw.season.unique()):
    h = df[df.season <= S - 1]
    rows = tr_raw[tr_raw.season == S]
    if len(h) == 0:
        cpS = pd.DataFrame({"id": [], "N_end": [], **{f"S_{k}": [] for k in P_RATES}})
        cbS = pd.DataFrame({"id": [], "N_end": [], **{f"S_{k}": [] for k in B_RATES}})
    else:
        cpS = build_const(h, "pitcher_id", "asof_pitcher_n", P_RATES)
        cbS = build_const(h, "batter_id", "asof_batter_n", B_RATES)
    parts.append(attach_cs(rows, cpS, cbS))
tr = add_features(pd.concat(parts).sort_index(), prior)
va = add_features(attach_cs(va_raw, cp, cb_tbl), prior)
del parts
y_tr = tr_raw["control_success"].to_numpy()
gt_tr = tr["game_type"].astype(str).to_numpy()

NUM = [c for c in FEATS76 if c not in CAT]


def frame(d):
    out = d[NUM].copy()
    for c in CAT:
        out[c] = d[c].astype(str)
    return out


p_moe = np.zeros((len(SEEDS), len(va)))
for seg in ["R", "F"]:
    m_tr = gt_tr == seg
    m_va = gt == seg
    tick(f"{seg} 전용 학습 ({m_tr.sum():,}행) ...")
    ptr = Pool(frame(tr[m_tr]), y_tr[m_tr], cat_features=list(CAT))
    pva = Pool(frame(va[m_va]), cat_features=list(CAT))
    for i, sd in enumerate(SEEDS):
        t0 = time.time()
        mo = CatBoostClassifier(**CB_PRM, random_seed=sd).fit(ptr)
        p_moe[i, m_va] = mo.predict_proba(pva)[:, 1]
        tick(f"  {seg} seed={sd} ({time.time()-t0:.0f}s)")
    del ptr, pva

ens = p_moe.mean(axis=0)
np.save("lab/47_moe_ens.npy", ens.astype("float32"))
s_moe = raw_score(ens, y_va)
s_cs = raw_score(p_cs, y_va)
log("\n" + "=" * 74)
log("판정")
log("=" * 74)
log(f"  단일 CS76      {s_cs:8.1f}")
log(f"  MoE (R+F 분리) {s_moe:8.1f}  ({s_moe-s_cs:+.1f})")
for seg in ["R", "F"]:
    m = gt == seg
    ys = y_va[m].astype(float)
    r = ys.mean()
    a = 100000 * (1 - np.mean((p_cs[m] - ys) ** 2) / (r * (1 - r)))
    b = 100000 * (1 - np.mean((ens[m] - ys) ** 2) / (r * (1 - r)))
    log(f"    {seg} 세그: 단일 {a:8.1f} → MoE {b:8.1f}  ({b-a:+.1f})")
log("  +15↑면 배포 후보 (F 전용 모델을 번들에 추가, game_type 라우팅은 행 독립)")
log("=" * 74)
