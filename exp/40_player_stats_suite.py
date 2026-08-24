# -*- coding: utf-8 -*-
"""
[40] 선수 조건부 통계 스위트 — '고원 레시피' 후보의 명시적 피처 버전 (밤샘 하네스)

목표 재설정 반영: 19등(~1162, +264) 목표 → 안전 축만으로는 불가.
  정형대회 표준인 선수 조건부 타겟통계를 그룹별로 전부 만들어 2024 폴드에서 거른다.
  전부 [F-타겟] (hand delta -22.2 전례 있음) — 로컬은 필터, 최종 판정은 LB 슬롯.

그룹 (전부 시즌 as-of: cumsum-자기시즌, exp/15 검증 패턴 / 스무딩 k):
  GA 투수×카운트군 delta   cg = 유리(s>b) / 균형 / 불리(b>s)     (3셀, k=100)
  GB 타자×카운트군 delta                                        (3셀, k=100)
  GC 투수×타자 페어 이력    n, 성공률(투수통산으로 수축 k=30)      (얇음 주의)
  GD 손 상성 페어 (hand delta 재등판 — 2025 실패 전례 명시)       (k=200)
  RM cb_rmse — CatBoostRegressor(RMSE) 목적함수 팔 (피처 65 동일)

팔: ref(65, exp/36 재사용) / +GA / +GB / +GC / +GD / +전부 / RM
비용: 6팔 × 2시드 CB ≈ 밤새. 각 팔 예측 저장(lab/40_*.npy).

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/40_player_stats_suite.py
"""

import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

sys.path.insert(0, "exp")
from common import (log, tick, raw_score, load_train, add_features, CAT, ALL_ENG)

SEEDS = [42, 7]
CB_PRM = dict(iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=10.0,
              verbose=False, thread_count=6, allow_writing_files=False)

log("train 로딩...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS65 = BASE + ALL_ENG

# =====================================================================
# 조건부 as-of delta 테이블 (hand_history_table 일반화)
# =====================================================================
def cond_asof_table(src, prior, id_col, cond_vals, k_over, k_cond):
    """(eid, season, cond) → 그 시즌 이전 기준 [조건부율 - 본인통산율]"""
    t = pd.DataFrame({"eid": src[id_col].to_numpy(),
                      "season": src["season"].to_numpy(),
                      "cond": cond_vals,
                      "y": src["control_success"].to_numpy()})
    c = (t.groupby(["eid", "season", "cond"])["y"].agg(succ="sum", n="size")
          .reset_index().sort_values(["eid", "cond", "season"]))
    g = c.groupby(["eid", "cond"])
    c["pc_s"] = g["succ"].cumsum() - c["succ"]
    c["pc_n"] = g["n"].cumsum() - c["n"]
    o = (t.groupby(["eid", "season"])["y"].agg(succ="sum", n="size")
          .reset_index().sort_values(["eid", "season"]))
    go = o.groupby("eid")
    o["po_s"] = go["succ"].cumsum() - o["succ"]
    o["po_n"] = go["n"].cumsum() - o["n"]
    tbl = c.merge(o[["eid", "season", "po_s", "po_n"]], on=["eid", "season"])
    p_over = (tbl.po_s + prior * k_over) / (tbl.po_n + k_over)
    p_cond = (tbl.pc_s + p_over * k_cond) / (tbl.pc_n + k_cond)
    tbl["delta"] = np.where(tbl.po_n > 0, p_cond - p_over, np.nan).astype("float32")
    return tbl[["eid", "season", "cond", "delta"]]


def pair_asof_table(src, prior, k_pair=30, k_over=200):
    """(pid, bid, season) → 페어 이력 n·수축 성공률 (투수통산 기준으로 수축)"""
    t = pd.DataFrame({"pid": src["pitcher_id"].to_numpy(),
                      "bid": src["batter_id"].to_numpy(),
                      "season": src["season"].to_numpy(),
                      "y": src["control_success"].to_numpy()})
    c = (t.groupby(["pid", "bid", "season"])["y"].agg(succ="sum", n="size")
          .reset_index().sort_values(["pid", "bid", "season"]))
    g = c.groupby(["pid", "bid"])
    c["pc_s"] = g["succ"].cumsum() - c["succ"]
    c["pc_n"] = g["n"].cumsum() - c["n"]
    o = (t.groupby(["pid", "season"])["y"].agg(succ="sum", n="size")
          .reset_index().sort_values(["pid", "season"]))
    go = o.groupby("pid")
    o["po_s"] = go["succ"].cumsum() - o["succ"]
    o["po_n"] = go["n"].cumsum() - o["n"]
    tbl = c.merge(o[["pid", "season", "po_s", "po_n"]], on=["pid", "season"])
    p_over = (tbl.po_s + prior * k_over) / (tbl.po_n + k_over)
    tbl["pair_rate"] = ((tbl.pc_s + p_over * k_pair) / (tbl.pc_n + k_pair)).astype("float32")
    tbl["pair_logn"] = np.log1p(tbl.pc_n).astype("float32")
    return tbl[["pid", "bid", "season", "pair_rate", "pair_logn"]]


prior_all = float(df[df.season <= 2023]["control_success"].mean())
b, s = df["balls_before"].to_numpy(), df["strikes_before"].to_numpy()
cg = np.where(s > b, 2, np.where(b > s, 0, 1)).astype("int8")   # 0불리/1균형/2유리

tick("as-of 테이블 생성 중 (GA/GB/GC/GD)...")
tGA = cond_asof_table(df, prior_all, "pitcher_id", cg, 100.0, 100.0)
tGB = cond_asof_table(df, prior_all, "batter_id", cg, 100.0, 100.0)
tGC = pair_asof_table(df, prior_all)
sh = (df["pitcher_hand"] == df["batter_hand"]).astype("int8").to_numpy()
tGD_p = cond_asof_table(df, prior_all, "pitcher_id", sh, 200.0, 200.0)
tGD_b = cond_asof_table(df, prior_all, "batter_id", sh, 200.0, 200.0)
tick(f"테이블 완료 GA{len(tGA):,} GB{len(tGB):,} GC{len(tGC):,}")


def attach_all(d):
    d = d.copy()
    d["_cg"] = np.where(d.strikes_before > d.balls_before, 2,
                        np.where(d.balls_before > d.strikes_before, 0, 1)).astype("int8")
    d["_sh"] = (d.pitcher_hand == d.batter_hand).astype("int8")
    for tbl, idc, cc, feat in [(tGA, "pitcher_id", "_cg", "f_p_cnt_delta"),
                               (tGB, "batter_id", "_cg", "f_b_cnt_delta"),
                               (tGD_p, "pitcher_id", "_sh", "f_p_hand_delta"),
                               (tGD_b, "batter_id", "_sh", "f_b_hand_delta")]:
        k = pd.DataFrame({"eid": d[idc].to_numpy(), "season": d.season.to_numpy(),
                          "cond": d[cc].to_numpy()})
        d[feat] = k.merge(tbl, on=["eid", "season", "cond"], how="left",
                          validate="m:1")["delta"].to_numpy("float32")
    k = pd.DataFrame({"pid": d.pitcher_id.to_numpy(), "bid": d.batter_id.to_numpy(),
                      "season": d.season.to_numpy()})
    m = k.merge(tGC, on=["pid", "bid", "season"], how="left", validate="m:1")
    d["f_pair_rate"] = m["pair_rate"].to_numpy("float32")
    d["f_pair_logn"] = m["pair_logn"].to_numpy("float32")
    return d.drop(columns=["_cg", "_sh"])


GROUPS = {"GA": ["f_p_cnt_delta"], "GB": ["f_b_cnt_delta"],
          "GC": ["f_pair_rate", "f_pair_logn"],
          "GD": ["f_p_hand_delta", "f_b_hand_delta"]}

tr_raw = df[df.season <= 2023].reset_index(drop=True)
va_raw = df[df.season == 2024].reset_index(drop=True)
y_tr = tr_raw["control_success"].to_numpy()
y_va = va_raw["control_success"].to_numpy()
tr = attach_all(add_features(tr_raw, prior_all))
va = attach_all(add_features(va_raw, prior_all))
del df, tr_raw, va_raw
for gname, cols in GROUPS.items():
    log(f"  {gname} 검증 커버리지: " +
        ", ".join(f"{c} {va[c].notna().mean()*100:.0f}%" for c in cols))

p_ref = np.load("lab/36_ref_ens.npy").astype("float64")
s_ref = raw_score(p_ref, y_va)
log(f"기준 (exp/36 ref): {s_ref:.1f}")
NUM65 = [c for c in FEATS65 if c not in CAT]


def run(name, extra, regressor=False):
    feats = NUM65 + extra
    def frame(d):
        out = d[feats].copy()
        for c in CAT:
            out[c] = d[c].astype(str)
        return out
    ptr = Pool(frame(tr), y_tr, cat_features=list(CAT))
    pva = Pool(frame(va), cat_features=list(CAT))
    preds = []
    for sd in SEEDS:
        t0 = time.time()
        if regressor:
            prm = {k: v for k, v in CB_PRM.items()}
            m = CatBoostRegressor(**prm, loss_function="RMSE", random_seed=sd).fit(ptr)
            p = np.clip(m.predict(pva), 0.0, 1.0)
        else:
            m = CatBoostClassifier(**CB_PRM, random_seed=sd).fit(ptr)
            p = m.predict_proba(pva)[:, 1]
        preds.append(p)
        tick(f"{name} seed={sd}  {raw_score(p, y_va):8.1f}  ({time.time()-t0:.0f}s)")
    ens = np.mean(preds, axis=0)
    np.save(f"lab/40_{name}_ens.npy", ens.astype("float32"))
    sc = raw_score(ens, y_va)
    log(f"  → {name:8s} {sc:8.1f}  (기준 대비 {sc-s_ref:+.1f})")
    return sc


results = {}
for gname, cols in GROUPS.items():
    results[gname] = run(gname, cols)
results["ALL"] = run("ALL", [c for cols in GROUPS.values() for c in cols])
results["RM"] = run("RM", [], regressor=True)

log("\n" + "=" * 70)
log(f"판정 (2024 폴드, ref {s_ref:.1f} 대비) — [F-타겟] 주의: 로컬은 필터일 뿐")
log("=" * 70)
for k, v in results.items():
    log(f"  {k:6s} {v:8.1f}  ({v-s_ref:+.1f})")
log("  +15↑ 그룹 → LB 슬롯 직접 검증 (19등 목표 공격 모드)")
log("=" * 70)
