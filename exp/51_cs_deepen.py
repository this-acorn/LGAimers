# -*- coding: utf-8 -*-
"""
[51] CS 심화 — 미분해 asof 3종(reverse/breaking/offspeed) 시즌 진행분 + prev5 delta

근거: CS 계열만 전이 1:1 증명 (submit10 +92.3). 아직 시즌 분해 안 한 누적 rate:
  reverse (의도 반대성 — 타겟 실패유형 3번!), breaking, offspeed.
  + 사용자 지적 누락분 f_form_dev5 = prev5 − overall.
팔: CS79 + 4피처 (일괄 triage — 통과 시 개별 ablation)
기준: lab/45_both_ens.npy (CS79 2시드 = 800.9), 시드 42/7

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/51_cs_deepen.py  (~35분)
"""

import importlib.util
import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import log, tick, raw_score, load_train

spec = importlib.util.spec_from_file_location("s12", "submit12_src/script.py")
s12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s12)

SEEDS = [42, 7]
CB_PRM = dict(iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=10.0,
              verbose=False, thread_count=14, allow_writing_files=False)
K_SHRINK = 50.0
MIN_CS_N = 5
EXT = {"reverse": "asof_pitcher_reverse_rate",
       "brk2": "asof_pitcher_breaking_rate",
       "off2": "asof_pitcher_offspeed_rate"}
NEW_FEATS = ["f_cs_p_reverse_d", "f_cs_p_brk2_d", "f_cs_p_off2_d", "f_form_dev5"]


def build_const_ext(src):
    d = src.sort_values("asof_pitcher_n").groupby("pitcher_id").tail(1)
    out = pd.DataFrame({"id": d["pitcher_id"].to_numpy()})
    n_last = d["asof_pitcher_n"].fillna(0).to_numpy("float64")
    out["N_end"] = n_last + 1
    for k, col in EXT.items():
        out[f"S_{k}"] = d[col].fillna(0).to_numpy("float64") * (n_last + 1)
    return out


def attach_ext(rows, ce):
    d = rows.copy()
    m = pd.DataFrame({"id": d.pitcher_id.to_numpy()}).merge(ce, on="id", how="left")
    N_end = m["N_end"].fillna(0).to_numpy("float64")
    n_now = d["asof_pitcher_n"].fillna(0).to_numpy("float64")
    cs_n = np.maximum(n_now - N_end, 0.0)
    ok = cs_n >= MIN_CS_N
    for k, col in EXT.items():
        car = d[col].to_numpy("float64")
        S_end = m[f"S_{k}"].fillna(0).to_numpy("float64")
        cs_S = np.clip(np.nan_to_num(car) * n_now - S_end, 0.0, None)
        d[f"f_cs_p_{k}_d"] = np.where(ok, cs_S / np.maximum(cs_n, 1) - car,
                                      np.nan).astype("float32")
    d["f_form_dev5"] = (d["asof_pitcher_prev5_game_success_rate"]
                        - d["asof_pitcher_success_rate"]).astype("float32")
    return d


log("train 로딩...")
df = load_train()
# 라벨 복원 (구종 피처용 — exp/48과 동일)
dd = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index()
pid = dd.pitcher_id.to_numpy()
n = dd.asof_pitcher_n.fillna(0).to_numpy("float64")
same_next = (pid[1:] == pid[:-1]) & (np.abs(n[1:] - n[:-1] - 1) < 1e-6)
for k, col in [("lab_fb", "asof_pitcher_fastball_rate"),
               ("lab_brk", "asof_pitcher_breaking_rate")]:
    S = dd[col].fillna(0).to_numpy("float64") * n
    lab = np.full(len(dd), np.nan)
    lab[:-1] = np.where(same_next, np.round(S[1:] - S[:-1]), np.nan)
    dd[k] = np.where((lab == 0) | (lab == 1), lab, np.nan)
rec = dd.set_index("index").sort_index()
df["lab_fb"], df["lab_brk"] = rec["lab_fb"], rec["lab_brk"]
b_, s_ = df.balls_before.to_numpy(), df.strikes_before.to_numpy()
df["_cg"] = np.where(s_ > b_, 2, np.where(b_ > s_, 0, 1)).astype("int8")
tick("라벨 복원 완료")

# exp/48의 prep 로직 최소 재현 (검증 스테이지만: ≤2023 → 2024)
K_MIX = 50.0


def mix_asof_train(labeled):
    t = labeled.dropna(subset=["lab_fb"])
    c = (t.groupby(["pitcher_id", "season", "_cg"])
          .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
          .reset_index().sort_values(["pitcher_id", "_cg", "season"]))
    g = c.groupby(["pitcher_id", "_cg"])
    for col in ["n", "fb", "brk"]:
        c[f"p_{col}"] = g[col].cumsum() - c[col]
    o = (t.groupby(["pitcher_id", "season"])
          .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
          .reset_index().sort_values(["pitcher_id", "season"]))
    go = o.groupby("pitcher_id")
    for col in ["n", "fb", "brk"]:
        o[f"po_{col}"] = go[col].cumsum() - o[col]
    tbl = c.merge(o[["pitcher_id", "season", "po_n", "po_fb", "po_brk"]],
                  on=["pitcher_id", "season"])
    over_fb = (tbl.po_fb + 0.5 * K_MIX) / (tbl.po_n + K_MIX)
    over_brk = (tbl.po_brk + 0.3 * K_MIX) / (tbl.po_n + K_MIX)
    tbl["mix_fb"] = np.where(tbl.po_n > 0,
                             (tbl.p_fb + K_MIX * over_fb) / (tbl.p_n + K_MIX),
                             np.nan).astype("float32")
    tbl["mix_brk"] = np.where(tbl.po_n > 0,
                              (tbl.p_brk + K_MIX * over_brk) / (tbl.p_n + K_MIX),
                              np.nan).astype("float32")
    return tbl[["pitcher_id", "season", "_cg", "mix_fb", "mix_brk"]]


def mix_career(labeled):
    t = labeled.dropna(subset=["lab_fb"])
    c = (t.groupby(["pitcher_id", "_cg"])
          .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
          .reset_index())
    o = (t.groupby("pitcher_id")
          .agg(po_n=("lab_fb", "size"), po_fb=("lab_fb", "sum"),
               po_brk=("lab_brk", "sum")).reset_index())
    tbl = c.merge(o, on="pitcher_id")
    over_fb = (tbl.po_fb + 0.5 * K_MIX) / (tbl.po_n + K_MIX)
    over_brk = (tbl.po_brk + 0.3 * K_MIX) / (tbl.po_n + K_MIX)
    tbl["mix_fb"] = ((tbl.fb + K_MIX * over_fb) / (tbl.n + K_MIX)).astype("float32")
    tbl["mix_brk"] = ((tbl.brk + K_MIX * over_brk) / (tbl.n + K_MIX)).astype("float32")
    return tbl[["pitcher_id", "_cg", "mix_fb", "mix_brk"]]


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


hist = df[df.season <= 2023]
prior = float(hist["control_success"].mean())
cp = build_const(hist, "pitcher_id", "asof_pitcher_n", s12.P_RATES)
cb_ = build_const(hist, "batter_id", "asof_batter_n", s12.B_RATES)
ce = build_const_ext(hist)
pfb_rows = hist.dropna(subset=["lab_fb"])
pfb = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=False,
                         thread_count=14, allow_writing_files=False, random_seed=42)
pfb.fit(pfb_rows[s12.PFB_IN].fillna(-999), pfb_rows["lab_fb"].astype(int))
mix_tr = mix_asof_train(hist)
mix_dep = mix_career(hist)
tick("테이블·PFB 준비 완료")

tr_rows = df[df.season <= 2023].reset_index(drop=True)
va_rows = df[df.season == 2024].reset_index(drop=True)
y_tr = tr_rows["control_success"].to_numpy()
y_va = va_rows["control_success"].to_numpy()

parts = []
for S in sorted(tr_rows.season.unique()):
    h = df[df.season <= S - 1]
    rows = tr_rows[tr_rows.season == S]
    if len(h) == 0:
        cpS = pd.DataFrame({"id": [], "N_end": [], **{f"S_{k}": [] for k in s12.P_RATES}})
        cbS = pd.DataFrame({"id": [], "N_end": [], **{f"S_{k}": [] for k in s12.B_RATES}})
        ceS = pd.DataFrame({"id": [], "N_end": [], **{f"S_{k}": [] for k in EXT}})
    else:
        cpS = build_const(h, "pitcher_id", "asof_pitcher_n", s12.P_RATES)
        cbS = build_const(h, "batter_id", "asof_batter_n", s12.B_RATES)
        ceS = build_const_ext(h)
    parts.append(attach_ext(s12.attach_cs(rows, cpS, cbS), ceS))
tr = s12.add_features(pd.concat(parts).sort_index(), prior)
key = tr[["pitcher_id", "season", "_cg"]].merge(
    mix_tr, on=["pitcher_id", "season", "_cg"], how="left")
tr["f_mixcg_fb"] = key["mix_fb"].to_numpy("float32")
tr["f_mixcg_brk"] = key["mix_brk"].to_numpy("float32")
tr["f_pfb"] = pfb.predict_proba(tr[s12.PFB_IN].fillna(-999))[:, 1].astype("float32")

va = attach_ext(s12.attach_cs(va_rows, cp, cb_), ce)
va = s12.attach_pt(s12.add_features(va, prior), mix_dep, pfb)
del parts
tick(f"부착 완료 — 신규 4피처 커버리지 " +
     ", ".join(f"{c} {va[c].notna().mean()*100:.0f}%" for c in NEW_FEATS))

p_ref = np.load("lab/45_both_ens.npy").astype("float64")
s_ref = raw_score(p_ref, y_va)
log(f"기준 (CS79 2시드): {s_ref:.1f}")

BASE = [c for c in pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1)
        .columns.str.replace("﻿", "").str.strip() if c != "row_id"]
ENG18 = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
         "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
         "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
         "f_pb_diff", "f_miss_p", "f_miss_prev1"]
FEATS = BASE + ENG18 + s12.CS_FEATS + s12.PT_FEATS + NEW_FEATS
log(f"피처 {len(FEATS)}개 (79 + 신규 4)")

Xtr = s12.build_matrix(tr, FEATS)
Xva = s12.build_matrix(va, FEATS)
ptr = Pool(Xtr, y_tr, cat_features=list(s12.CAT))
pva = Pool(Xva, cat_features=list(s12.CAT))
del tr, va, Xtr, Xva
preds = []
for sd in SEEDS:
    t0 = time.time()
    m = CatBoostClassifier(**CB_PRM, random_seed=sd).fit(ptr)
    preds.append(m.predict_proba(pva)[:, 1])
    tick(f"seed={sd}  {raw_score(preds[-1], y_va):8.1f}  ({time.time()-t0:.0f}s)")
ens = np.mean(preds, axis=0)
np.save("lab/51_ens.npy", ens.astype("float32"))
sc = raw_score(ens, y_va)
log("\n" + "=" * 70)
log(f"판정: CS79+4 = {sc:.1f}  (기준 {s_ref:.1f} 대비 {sc-s_ref:+.1f})")
log("  전이 1:1 계열이므로 +8↑면 배포 가치. 통과 시 개별 ablation")
log("=" * 70)
