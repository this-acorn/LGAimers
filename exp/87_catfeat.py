# -*- coding: utf-8 -*-
"""
[87] cat_features 확장 — 하민의 +24.51 발견을 축으로 확장 측정 (전체 데이터, 2시드 짝지음)

발견 (팀원 하민, 08-27 LB 확정):
  submit14(우리, 1033.99)와 target5_teamcat(1059.05)의 코드 실질 차이는 단 하나 —
      TEAM_CAT = ["pitcher_team_id", "batter_team_id"] 를 cat_features 로 지정
  피처 79개/CS/구종/5클래스/it500-d6-lr0.08-l2=10/8시드 전부 동일. **+25.06**.

  우리 원장의 "범주형 처리 개선 = 피해 0" 판정은 틀렸다 — 선수 ID(800명대 고카디널리티,
  cb_cat -173.9)만 테스트하고 **팀 ID(12개, 저카디널리티)** 는 본 적이 없었다.

가설: CatBoost 는 cat_features 들의 **조합(CTR combination)** 을 자동 생성한다
  (max_ctr_complexity 기본 4). 팀 ID 추가의 이득은 team x base_state, team x game_type
  같은 조합에서 나왔을 가능성이 크다. 그렇다면 저카디널리티 컬럼을 더 넣으면
  조합이 더 늘어난다 -> 더 오를 수도, 과적합으로 무너질 수도 있다. 측정한다.

★ 기준 설정을 하민 것(lr0.08)에 맞춘다 — 팀 최고 기록 위에 쌓는 게 목적이므로.
  in-run 베이스라인(cat3)을 함께 돌려 컬럼 순서 교란 없이 짝지음한다.

팔 (79피처/멀티클래스5/it500-d6-lr0.08-l2=10/thread14/시드[42,7] 전부 고정):
  cat3   현행 3개                                    <- in-run 베이스라인
  cat5   +pitcher_team_id +batter_team_id            <- 하민 (LB +24.51 / 하민 로컬 +2.0)
  cat8   cat5 +balls_before +strikes_before +outs_before
  cat12  cat8 +pitcher_hand +batter_hand +inning +game_month
  (season 은 제외 — 드리프트 축이라 레짐 베팅 위험)

판독:
  · cat5 가 로컬에서 +2 근처면 하네스가 이 축을 재현하는 것 -> 상대 순위 신뢰 가능
  · 단 이 축은 하민 실측이 로컬 +2.0 -> 실전 +24.51 (12배). 로컬 크기로 실전을 추정하지 마라.
    로컬은 **방향과 순위**만 본다. 절대 크기는 실전에서 다시 잰다.

실행: PYTHONIOENCODING=utf-8 python -u exp/87_catfeat.py   (~5.5시간)
"""

import importlib.util
import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import log, tick, raw_score, load_train

spec = importlib.util.spec_from_file_location("s12", 'submissions/submit12_src/script.py')
s12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s12)

SEEDS = [42, 7]
CB_PRM = dict(iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=10.0,
              verbose=False, thread_count=14, allow_writing_files=False,
              loss_function="MultiClass")
K_MIX = 50.0

log("train 로딩...")
df = load_train()

# ---- 라벨 복원 (succ/middle/reverse + 구종) ----
dd = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index()
pid = dd.pitcher_id.to_numpy()
n = dd.asof_pitcher_n.fillna(0).to_numpy("float64")
nxt = (pid[1:] == pid[:-1]) & (np.abs(n[1:] - n[:-1] - 1) < 1e-6)
for k, col in [("lab_mid", "asof_pitcher_middle_rate"),
               ("lab_rev", "asof_pitcher_reverse_rate"),
               ("lab_fb", "asof_pitcher_fastball_rate"),
               ("lab_brk", "asof_pitcher_breaking_rate")]:
    S = dd[col].fillna(0).to_numpy("float64") * n
    lab = np.full(len(dd), np.nan)
    diff = np.round(S[1:] - S[:-1])
    lab[:-1] = np.where(nxt, diff, np.nan)
    dd[k] = np.where((lab == 0) | (lab == 1), lab, np.nan)
rec = dd.set_index("index").sort_index()
for k in ["lab_mid", "lab_rev", "lab_fb", "lab_brk"]:
    df[k] = rec[k]
y_bin = df.control_success.to_numpy("float64")
# 5클래스: 0=성공 1=미들만 2=리버스만 3=미들∩리버스 4=빅미스
cls = np.full(len(df), -1, dtype="int8")
ok = df.lab_mid.notna().to_numpy() & df.lab_rev.notna().to_numpy()
m_ = df.lab_mid.to_numpy() == 1
r_ = df.lab_rev.to_numpy() == 1
cls[ok & (y_bin == 1)] = 0
cls[ok & (y_bin == 0) & m_ & ~r_] = 1
cls[ok & (y_bin == 0) & ~m_ & r_] = 2
cls[ok & (y_bin == 0) & m_ & r_] = 3
cls[ok & (y_bin == 0) & ~m_ & ~r_] = 4
df["_cls"] = cls
log(f"클래스 분포: " + " ".join(f"{c}:{np.mean(cls==c)*100:.1f}%" for c in range(5))
    + f"  미복원 {np.mean(cls==-1)*100:.2f}%")
b_, s_ = df.balls_before.to_numpy(), df.strikes_before.to_numpy()
df["_cg"] = np.where(s_ > b_, 2, np.where(b_ > s_, 0, 1)).astype("int8")

# ---- exp/48 검증 스테이지와 동일한 피처 준비 (79피처) ----
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


hist = df[df.season <= 2023]
prior = float(hist["control_success"].mean())
cp = build_const(hist, "pitcher_id", "asof_pitcher_n", s12.P_RATES)
cb_ = build_const(hist, "batter_id", "asof_batter_n", s12.B_RATES)
pfb_rows = hist.dropna(subset=["lab_fb"])
pfb = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=False,
                         thread_count=14, allow_writing_files=False, random_seed=42)
pfb.fit(pfb_rows[s12.PFB_IN].fillna(-999), pfb_rows["lab_fb"].astype(int))
mix_tr = mix_asof_train(hist)
mix_dep = mix_career(hist)
tick("테이블·PFB 준비")

tr_rows = df[df.season <= 2023].reset_index(drop=True)
va_rows = df[df.season == 2024].reset_index(drop=True)
y_va = va_rows["control_success"].to_numpy()
parts = []
for S in sorted(tr_rows.season.unique()):
    h = df[df.season <= S - 1]
    rows = tr_rows[tr_rows.season == S]
    if len(h) == 0:
        cpS = pd.DataFrame({"id": [], "N_end": [], **{f"S_{k}": [] for k in s12.P_RATES}})
        cbS = pd.DataFrame({"id": [], "N_end": [], **{f"S_{k}": [] for k in s12.B_RATES}})
    else:
        cpS = build_const(h, "pitcher_id", "asof_pitcher_n", s12.P_RATES)
        cbS = build_const(h, "batter_id", "asof_batter_n", s12.B_RATES)
    parts.append(s12.attach_cs(rows, cpS, cbS))
tr = s12.add_features(pd.concat(parts).sort_index(), prior)
key = tr[["pitcher_id", "season", "_cg"]].merge(
    mix_tr, on=["pitcher_id", "season", "_cg"], how="left")
tr["f_mixcg_fb"] = key["mix_fb"].to_numpy("float32")
tr["f_mixcg_brk"] = key["mix_brk"].to_numpy("float32")
tr["f_pfb"] = pfb.predict_proba(tr[s12.PFB_IN].fillna(-999))[:, 1].astype("float32")
va = s12.attach_pt(s12.add_features(s12.attach_cs(va_rows, cp, cb_), prior),
                   mix_dep, pfb)
del parts

BASE = [c for c in pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1)
        .columns.str.replace("﻿", "").str.strip() if c != "row_id"]
ENG18 = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
         "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
         "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
         "f_pb_diff", "f_miss_p", "f_miss_prev1"]
# --------------------------------------------------------------------------
FEATS = BASE + ENG18 + s12.CS_FEATS + s12.PT_FEATS
m_tr = tr["_cls"].to_numpy() >= 0
y_cls = tr["_cls"].to_numpy()[m_tr]
log(f"멀티클래스 학습 행: {m_tr.sum():,} / {len(tr):,}   피처 {len(FEATS)}")

C3 = list(s12.CAT)
ARMS = [
    ("cat3", C3),
    ("cat5", C3 + ["pitcher_team_id", "batter_team_id"]),
    ("cat8", C3 + ["pitcher_team_id", "batter_team_id",
                   "balls_before", "strikes_before", "outs_before"]),
    ("cat12", C3 + ["pitcher_team_id", "batter_team_id",
                    "balls_before", "strikes_before", "outs_before",
                    "pitcher_hand", "batter_hand", "inning", "game_month"]),
]


def build(d, cats):
    """피처 순서를 고정한 채 cats 만 문자열로. 팀ID는 float 표기 방지 위해 int64 경유."""
    out = pd.DataFrame(index=d.index)
    for c in FEATS:
        if c in cats:
            v = d[c]
            out[c] = (v.fillna(-1).astype("int64").astype(str)
                      if pd.api.types.is_numeric_dtype(v) else v.astype(str))
        else:
            out[c] = d[c]
    return out


r_ = float(y_va.mean())
DEN = r_ * (1 - r_)
res = {}
for name, cats in ARMS:
    Xtr = build(tr[m_tr], cats)
    Xva = build(va, cats)
    ptr = Pool(Xtr, y_cls, cat_features=cats)
    pva = Pool(Xva, cat_features=cats)
    del Xtr, Xva
    preds, solo = [], []
    for sd in SEEDS:
        t0 = time.time()
        m = CatBoostClassifier(**CB_PRM, random_seed=sd).fit(ptr)
        p = m.predict_proba(pva)[:, 0]
        preds.append(p)
        solo.append(raw_score(p, y_va))
        tick(f"{name} seed={sd}  {solo[-1]:8.1f}  ({time.time()-t0:.0f}s)")
        del m
    del ptr, pva
    ens = np.mean(preds, axis=0)
    np.save(f"lab/87_{name}.npy", ens.astype("float32"))
    e2 = raw_score(ens, y_va)
    ms = float(np.mean(solo))
    p8 = 1.75 * e2 - 0.75 * ms
    pen = 100000 * (ens.mean() - r_) ** 2 / DEN
    res[name] = (e2, ms, p8, pen, solo, ens)
    log(f"  -> {name:6s} 2시드 {e2:7.1f}  시드평균 {ms:7.1f}  proj8 {p8:7.1f}  벌점 {pen:5.1f}")
    log("")

log("=" * 92)
log("판정 — cat_features 확장 (in-run 베이스라인 cat3 대비)")
log("=" * 92)
b2, bm, bp = res["cat3"][0], res["cat3"][1], res["cat3"][2]
log(f"  {'팔':7s} {'2시드':>8s} {'시드평균':>9s} {'proj8':>8s} {'벌점':>6s} "
    f"{'paired 차이':>22s}")
for k, (e2, ms, p8, pen, solo, _) in res.items():
    pr = [solo[i] - res["cat3"][4][i] for i in range(len(solo))]
    ps = " / ".join(f"{v:+.1f}" for v in pr)
    log(f"  {k:7s} {e2:8.1f} {ms:9.1f} {p8:8.1f} {pen:6.1f}   {ps:>14s} "
        f"(평균 {np.mean(pr):+.1f}, 산포 {np.std(pr):.2f})")
log("")
log("  ※ 하민 실측: 이 축은 로컬 +2.0 -> 실전 +24.51 (12배). 로컬 크기로 실전 추정 금지.")
log("     로컬은 **방향과 순위**만. 최선 팔을 전체 배포 후 실전에서 다시 잰다.")
best = max(res, key=lambda k: res[k][1])
log(f"  시드평균 기준 최선: {best}  (cat3 대비 {res[best][1] - bm:+.1f})")
log("=" * 92)
