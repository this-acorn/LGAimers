# -*- coding: utf-8 -*-
"""
[39] 트랙맨 물리 프로필 v2 — 정확 선수매핑(97%/95%) 기반, 2024 폴드, CB 위에서

exp/38 돌파구 활용: 행 조인 투표로 투수 565명(97.2%)·타자 491명(95.5%) 정확 매핑.
이전 트랙맨 실험들과의 차이:
  · 커버리지 ~50% → ~95% (오매칭 노이즈 제거)
  · 타자 쪽 물리 노출 프로필 = 완전 신규 정보 (주최측 asof_batter_*에 없음)
  · 릴리스 분산(3회 사망)은 제외 — 수준(level)·믹스·감속만
  · 검증은 2024 폴드 (§2-5 정보 비교 표준), 모델은 현역 CatBoost

피처 (전부 시즌 as-of: 행의 season S 기준 <S 트랙맨만, (id,season) 그리드 명시 생성):
  투수 8: fb구속/회전/수직무브/수평무브, extension, 감속(rel-zone), fb구속 전시즌Δ, log표본
  타자 4: 상대한 평균구속, breaking 비율, offspeed 비율, log표본
기준: exp/36 ref (같은 시드 42/7, CB 결정론 → ens 수준 paired 비교 유효)

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/39_tm_physical_v2.py  (~45분)
"""

import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import (log, tick, raw_score, load_train, add_features, CAT, ALL_ENG)

SEEDS = [42, 7]
CB_PRM = dict(iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=10.0,
              verbose=False, thread_count=6, allow_writing_files=False)
MIN_VOTES, MIN_PURITY = 20, 0.99

# =====================================================================
# 1. 트랙맨 → (선수, 시즌) as-of 물리 프로필
# =====================================================================
log("트랙맨 로딩...")
tm = pd.read_csv("data/trackman_history.csv", encoding="utf-8-sig",
                 usecols=["season", "pitcher_trackman_id", "batter_trackman_id",
                          "pitch_type_group", "rel_speed", "spin_rate",
                          "induced_vert_break", "horz_break", "extension",
                          "zone_speed"])
tm.columns = [c.replace("﻿", "").strip() for c in tm.columns]
pmap = pd.read_csv("lab/entity_map_pitcher.csv")
bmap = pd.read_csv("lab/entity_map_batter.csv")
pmap = pmap[(pmap.purity >= MIN_PURITY) & (pmap.votes >= MIN_VOTES)]
bmap = bmap[(bmap.purity >= MIN_PURITY) & (bmap.votes >= MIN_VOTES)]
tid2pid = dict(zip(pmap.tid, pmap.id))
tid2bid = dict(zip(bmap.tid, bmap.id))
tm["pid"] = tm.pitcher_trackman_id.map(tid2pid)
tm["bid"] = tm.batter_trackman_id.map(tid2bid)
tm["is_fb"] = tm.pitch_type_group.astype(str).str.lower() == "fastball"
tm["is_brk"] = tm.pitch_type_group.astype(str).str.lower() == "breaking"
tm["is_off"] = tm.pitch_type_group.astype(str).str.lower() == "offspeed"
tm["decel"] = tm.rel_speed - tm.zone_speed
tick(f"트랙맨 {len(tm):,}행 (pid 매핑 {tm.pid.notna().mean()*100:.0f}% / "
     f"bid {tm.bid.notna().mean()*100:.0f}%)")


def asof_profile(df, id_col, spec, min_n=30):
    """(id, season S) → S 이전 시즌들의 집계. spec: {출력명: (컬럼, 마스크컬럼|None)}"""
    d = df.dropna(subset=[id_col]).copy()
    per = d.groupby([id_col, "season"]).agg(
        n=("season", "size"),
        **{f"s_{k}": (col, "sum") for k, (col, _) in spec.items()},
        **{f"c_{k}": (col, "count") for k, (col, _) in spec.items()},
    ).reset_index().sort_values([id_col, "season"])
    # 시즌 그리드 (2020~2024): 그 시즌 트랙맨이 없어도 과거가 있으면 키 생성 (exp/21-D 준수)
    rows = []
    for sid, g in per.groupby(id_col):
        g = g.set_index("season")
        for S in range(2020, 2025):
            past = g[g.index < S]
            if len(past) == 0 or past["n"].sum() < min_n:
                continue
            row = {"id": sid, "season": S, "n": past["n"].sum()}
            for k in spec:
                c = past[f"c_{k}"].sum()
                row[k] = past[f"s_{k}"].sum() / c if c > 0 else np.nan
            # 전시즌 vs 통산 델타 (첫 spec 항목에 대해)
            k0 = list(spec)[0]
            last = past.index.max()
            lc = past.loc[last, f"c_{k0}"]
            row[f"{k0}_lastdelta"] = (past.loc[last, f"s_{k0}"] / lc - row[k0]) \
                if lc > 0 and not np.isnan(row[k0]) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


tm["fb_speed"] = tm.rel_speed.where(tm.is_fb)
tm["fb_spin"] = tm.spin_rate.where(tm.is_fb)
tm["fb_ivb"] = tm.induced_vert_break.where(tm.is_fb)
tm["fb_hb"] = tm.horz_break.where(tm.is_fb)
tm["brk_f"] = tm.is_brk.astype(float)
tm["off_f"] = tm.is_off.astype(float)

P_SPEC = {"velo": ("fb_speed", None), "spin": ("fb_spin", None),
          "ivb": ("fb_ivb", None), "hb": ("fb_hb", None),
          "ext": ("extension", None), "decel": ("decel", None)}
B_SPEC = {"velo_faced": ("rel_speed", None), "brk_faced": ("brk_f", None),
          "off_faced": ("off_f", None)}
ptbl = asof_profile(tm, "pid", P_SPEC)
btbl = asof_profile(tm, "bid", B_SPEC)
tick(f"프로필: 투수 {len(ptbl):,}행 / 타자 {len(btbl):,}행 (선수-시즌)")

P_FEATS = ["p_velo", "p_spin", "p_ivb", "p_hb", "p_ext", "p_decel",
           "p_velo_lastdelta", "p_logn"]
B_FEATS = ["b_velo_faced", "b_brk_faced", "b_off_faced", "b_logn"]


def attach(d):
    d = d.copy()
    k = pd.DataFrame({"id": d.pitcher_id.to_numpy(), "season": d.season.to_numpy()})
    m = k.merge(ptbl, on=["id", "season"], how="left", validate="m:1")
    for c in P_SPEC:
        d[f"p_{c}"] = m[c].to_numpy("float32")
    d["p_velo_lastdelta"] = m["velo_lastdelta"].to_numpy("float32")
    d["p_logn"] = np.log1p(m["n"]).to_numpy("float32")
    k = pd.DataFrame({"id": d.batter_id.to_numpy(), "season": d.season.to_numpy()})
    m = k.merge(btbl, on=["id", "season"], how="left", validate="m:1")
    for c in B_SPEC:
        d[f"b_{c}"] = m[c].to_numpy("float32")
    d["b_logn"] = np.log1p(m["n"]).to_numpy("float32")
    return d


# =====================================================================
# 2. 2024 폴드 하네스 (기준 = exp/36 ref 앙상블, 같은 시드)
# =====================================================================
log("train 로딩...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS65 = BASE + ALL_ENG
NUM65 = [c for c in FEATS65 if c not in CAT]

tr_raw = df[df["season"] <= 2023].reset_index(drop=True)
va_raw = df[df["season"] == 2024].reset_index(drop=True)
y_tr = tr_raw["control_success"].to_numpy()
y_va = va_raw["control_success"].to_numpy()
prior = float(y_tr.mean())
tr = attach(add_features(tr_raw, prior))
va = attach(add_features(va_raw, prior))
del df, tr_raw, va_raw
cov_p = va["p_velo"].notna().mean() * 100
cov_b = va["b_velo_faced"].notna().mean() * 100
tick(f"2024 폴드 준비 — 행 커버리지: 투수피처 {cov_p:.1f}% / 타자피처 {cov_b:.1f}%")

p_ref = np.load("lab/36_ref_ens.npy").astype("float64")
s_ref = raw_score(p_ref, y_va)
log(f"기준 (exp/36 ref, CB 시드42/7): {s_ref:.1f}")


def run(name, extra):
    feats = NUM65 + extra
    def frame(d):
        out = d[feats].copy()
        for c in CAT:
            out[c] = d[c].astype(str)
        return out
    ptr = Pool(frame(tr), y_tr, cat_features=list(CAT))
    pva = Pool(frame(va), cat_features=list(CAT))
    preds = []
    for s in SEEDS:
        t0 = time.time()
        m = CatBoostClassifier(**CB_PRM, random_seed=s).fit(ptr)
        preds.append(m.predict_proba(pva)[:, 1])
        tick(f"{name} seed={s}  {raw_score(preds[-1], y_va):8.1f}  ({time.time()-t0:.0f}s)")
    ens = np.mean(preds, axis=0)
    np.save(f"lab/39_{name}_ens.npy", ens.astype("float32"))
    sc = raw_score(ens, y_va)
    log(f"  → {name:10s} 앙상블 {sc:8.1f}  (기준 대비 {sc-s_ref:+.1f})")
    return sc


s_p = run("tm_pitcher", P_FEATS)
s_pb = run("tm_both", P_FEATS + B_FEATS)

log("\n" + "=" * 72)
log("판정 (2024 폴드, exp/36 ref 대비)")
log("=" * 72)
log(f"  기준(65)        {s_ref:8.1f}")
log(f"  +투수물리(8)    {s_p:8.1f}  ({s_p-s_ref:+.1f})")
log(f"  +투수+타자(12)  {s_pb:8.1f}  ({s_pb-s_ref:+.1f})")
log("  +15 이상이면 독립시드 확인 → 배포 그리드(2025 = ≤2024 통산) 생성 → LB")
log("=" * 72)
