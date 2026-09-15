# -*- coding: utf-8 -*-
"""
[57] Trackman 증류 프로브 — teacher(현재 투구 물리값) → student(투구 전 정보만)

공식 승인: 손석길 08-20 Q1~3 "모두 가능", 빵투 08-14 — 학습 단계 privileged 정보 사용 허용.
직접 물리 피처(exp/39·42)가 CS 위에서 소멸한 것과 다른 메커니즘:
  물리값을 추론 입력에 넣는 게 아니라, 노이즈 많은 0/1 정답을 teacher가 부드럽게
  설명하도록 만들어 student의 학습 신호를 개선한다.

설계 (누수 차단):
  · 행 조인(exp/37 키)으로 1:1 매칭 행에만 물리값 부착 (~55%)
  · teacher OOF: 투수-경기 세그먼트 단위 2-fold — 같은 등판의 인접 투구가
    반대 폴드로 새는 것을 차단 (리뷰어 지적 반영)
  · soft label = (1-α)·y + α·p_teacher (매칭 행), 미매칭 행은 hard label 유지
  · student: CS79 피처만, CrossEntropy(확률 타깃)
기준: lab/45_both_ens.npy = CS79 이진 2시드 800.9 (2024 폴드). 프로브는 시드 42 1개.

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/57_distill_probe.py  (~40분, 6스레드)
"""

import importlib.util
import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import log, tick, raw_score, load_train

spec = importlib.util.spec_from_file_location("s12", "submissions/submit12_src/script.py")
s12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s12)

SEED = 42
NTHREAD = 6
ALPHAS = [0.25, 0.5]
CB = dict(iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=10.0,
          verbose=False, thread_count=NTHREAD, allow_writing_files=False)
K_MIX = 50.0
TEAM_MAP = {12: "DOO_BEA", 13: "LG_TWI", 14: "KIW_HER", 15: "LOT_GIA",
            16: "KIA_TIG", 17: "HAN_EAG", 18: "SAM_LIO", 19: "NC_DIN",
            20: "KT_WIZ", 21: "SSG_LAN"}
KEY = ["season", "game_month", "game_dayofweek", "inning", "top_bottom",
       "balls_before", "strikes_before", "outs_before", "ph", "bh", "pt", "bt"]
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension",
        "zone_speed"]

log("train 로딩...")
df = load_train()

# ---- 라벨 복원 (구종 피처용) ----
dd = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index()
pid = dd.pitcher_id.to_numpy()
nn = dd.asof_pitcher_n.fillna(0).to_numpy("float64")
nxt = (pid[1:] == pid[:-1]) & (np.abs(nn[1:] - nn[:-1] - 1) < 1e-6)
for k, col in [("lab_fb", "asof_pitcher_fastball_rate"),
               ("lab_brk", "asof_pitcher_breaking_rate")]:
    S = dd[col].fillna(0).to_numpy("float64") * nn
    lab = np.full(len(dd), np.nan)
    lab[:-1] = np.where(nxt, np.round(S[1:] - S[:-1]), np.nan)
    dd[k] = np.where((lab == 0) | (lab == 1), lab, np.nan)
rec = dd.set_index("index").sort_index()
df["lab_fb"], df["lab_brk"] = rec["lab_fb"], rec["lab_brk"]
b_, s_ = df.balls_before.to_numpy(), df.strikes_before.to_numpy()
df["_cg"] = np.where(s_ > b_, 2, np.where(b_ > s_, 0, 1)).astype("int8")

# ---- 투수-경기 세그먼트 (GroupKFold 그룹) ----
daykey = (dd.season.to_numpy() * 10000 + dd.game_month.to_numpy() * 100
          + dd.game_dayofweek.to_numpy()).astype("int64")
p1 = np.nan_to_num(dd.asof_pitcher_prev1_game_success_rate.to_numpy("float64"), nan=-1.0)
newg = (np.r_[True, pid[1:] != pid[:-1]] | (daykey != np.r_[0, daykey[:-1]])
        | (np.abs(p1 - np.r_[0.0, p1[:-1]]) > 1e-12))
dd["_seg"] = np.cumsum(newg)
df["_seg"] = dd.set_index("index").sort_index()["_seg"]
tick(f"세그먼트 {df._seg.nunique():,}개")

# ---- 행 단위 trackman 조인 (1:1만) ----
tm = pd.read_csv("data/trackman_history.csv", encoding="utf-8-sig",
                 usecols=["season", "game_month", "game_dayofweek", "inning",
                          "top_bottom", "balls_before", "strikes_before",
                          "outs_before", "pitcher_hand", "batter_hand",
                          "pitcher_team", "batter_team", "pitch_type_group"] + PHYS)
tm.columns = [c.replace("﻿", "").strip() for c in tm.columns]
hand = {2: "Right", 1: "Left"}
df["ph"] = df.pitcher_hand.map(hand); df["bh"] = df.batter_hand.map(hand)
df["pt"] = df.pitcher_team_id.map(TEAM_MAP); df["bt"] = df.batter_team_id.map(TEAM_MAP)
tm["pt"] = tm.pitcher_team.replace({"SK_WYV": "SSG_LAN"})
tm["bt"] = tm.batter_team.replace({"SK_WYV": "SSG_LAN"})
tm["ph"] = tm.pitcher_hand.astype(str); tm["bh"] = tm.batter_hand.astype(str)
tm["top_bottom"] = tm.top_bottom.astype(str).str[0]
df["top_bottom_k"] = df.top_bottom.astype(str).str[0]
df["top_bottom"] = df["top_bottom_k"]
for c in ["season", "game_month", "game_dayofweek", "inning",
          "balls_before", "strikes_before", "outs_before"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    tm[c] = pd.to_numeric(tm[c], errors="coerce").astype("Int64")
trk = df.dropna(subset=["pt", "bt", "ph", "bh"])
tmk = tm[tm.pt.isin(TEAM_MAP.values()) & tm.bt.isin(TEAM_MAP.values())]
c1 = trk.groupby(KEY, observed=True).size()
c2 = tmk.groupby(KEY, observed=True).size()
k11 = c1[c1 == 1].index.intersection(c2[c2 == 1].index)
tm11 = tmk.set_index(KEY).loc[k11, PHYS + ["pitch_type_group"]].reset_index()
df = df.merge(tm11, on=KEY, how="left", validate="m:1")
matched = df["rel_speed"].notna().to_numpy()
df["_decel"] = df.rel_speed - df.zone_speed
tick(f"행 조인: 매칭 {matched.sum():,} / {len(df):,} ({matched.mean()*100:.1f}%)")
del tm, tmk, trk, tm11

# ---- CS79 피처 준비 (exp/53과 동일) ----
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


def mix_tables(hist):
    t = hist.dropna(subset=["lab_fb"])
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
    tb = c.merge(o[["pitcher_id", "season", "po_n", "po_fb", "po_brk"]],
                 on=["pitcher_id", "season"])
    ofb = (tb.po_fb + 0.5 * K_MIX) / (tb.po_n + K_MIX)
    obk = (tb.po_brk + 0.3 * K_MIX) / (tb.po_n + K_MIX)
    tb["mix_fb"] = np.where(tb.po_n > 0, (tb.p_fb + K_MIX * ofb) / (tb.p_n + K_MIX),
                            np.nan).astype("float32")
    tb["mix_brk"] = np.where(tb.po_n > 0, (tb.p_brk + K_MIX * obk) / (tb.p_n + K_MIX),
                             np.nan).astype("float32")
    cc = (t.groupby(["pitcher_id", "_cg"])
           .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
           .reset_index())
    oo = (t.groupby("pitcher_id").agg(po_n=("lab_fb", "size"), po_fb=("lab_fb", "sum"),
                                      po_brk=("lab_brk", "sum")).reset_index())
    dep = cc.merge(oo, on="pitcher_id")
    ofb2 = (dep.po_fb + 0.5 * K_MIX) / (dep.po_n + K_MIX)
    obk2 = (dep.po_brk + 0.3 * K_MIX) / (dep.po_n + K_MIX)
    dep["mix_fb"] = ((dep.fb + K_MIX * ofb2) / (dep.n + K_MIX)).astype("float32")
    dep["mix_brk"] = ((dep.brk + K_MIX * obk2) / (dep.n + K_MIX)).astype("float32")
    return tb[["pitcher_id", "season", "_cg", "mix_fb", "mix_brk"]], \
        dep[["pitcher_id", "_cg", "mix_fb", "mix_brk"]]


hist = df[df.season <= 2023]
prior = float(hist["control_success"].mean())
cp = build_const(hist, "pitcher_id", "asof_pitcher_n", s12.P_RATES)
cb_ = build_const(hist, "batter_id", "asof_batter_n", s12.B_RATES)
pfb_rows = hist.dropna(subset=["lab_fb"])
pfb = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=False,
                         thread_count=NTHREAD, allow_writing_files=False, random_seed=42)
pfb.fit(pfb_rows[s12.PFB_IN].fillna(-999), pfb_rows["lab_fb"].astype(int))
mix_tr, mix_dep = mix_tables(hist)

tr_rows = df[df.season <= 2023].reset_index(drop=True)
va_rows = df[df.season == 2024].reset_index(drop=True)
y_tr = tr_rows["control_success"].to_numpy("float64")
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
k_ = tr[["pitcher_id", "season", "_cg"]].merge(mix_tr, on=["pitcher_id", "season", "_cg"],
                                               how="left")
tr["f_mixcg_fb"] = k_["mix_fb"].to_numpy("float32")
tr["f_mixcg_brk"] = k_["mix_brk"].to_numpy("float32")
tr["f_pfb"] = pfb.predict_proba(tr[s12.PFB_IN].fillna(-999))[:, 1].astype("float32")
va = s12.attach_pt(s12.add_features(s12.attach_cs(va_rows, cp, cb_), prior), mix_dep, pfb)
del parts

BASE = [c for c in pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1)
        .columns.str.replace("﻿", "").str.strip() if c != "row_id"]
ENG18 = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
         "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
         "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
         "f_pb_diff", "f_miss_p", "f_miss_prev1"]
FEATS = BASE + ENG18 + s12.CS_FEATS + s12.PT_FEATS
TFEATS = FEATS + PHYS + ["_decel"]          # teacher: + 현재 투구 물리값
CATS = list(s12.CAT) + ["pitch_type_group"]
tick(f"피처 준비 — student {len(FEATS)} / teacher {len(TFEATS)}+구종")

# =====================================================================
# 1. Teacher OOF (투수-경기 세그먼트 2-fold, 매칭 행만)
# =====================================================================
m_tr = tr["rel_speed"].notna().to_numpy()
log(f"\n학습 매칭 행: {m_tr.sum():,} / {len(tr):,} ({m_tr.mean()*100:.1f}%)")
seg = tr["_seg"].to_numpy()
fold = (pd.factorize(seg)[0] % 2)          # 세그먼트 단위 2-fold
p_teach = np.full(len(tr), np.nan)


def tframe(d):
    out = d[[c for c in TFEATS if c not in CATS]].copy()
    for c in s12.CAT:
        out[c] = d[c].astype(str)
    out["pitch_type_group"] = d["pitch_type_group"].astype(str)
    return out


for f in [0, 1]:
    trn = m_tr & (fold != f)
    hld = m_tr & (fold == f)
    t0 = time.time()
    mt = CatBoostClassifier(**CB, random_seed=SEED).fit(
        Pool(tframe(tr[trn]), y_tr[trn], cat_features=CATS))
    p_teach[hld] = mt.predict_proba(Pool(tframe(tr[hld]), cat_features=CATS))[:, 1]
    tick(f"teacher fold{f}: 학습 {trn.sum():,} → OOF {hld.sum():,} "
         f"({time.time()-t0:.0f}s)")
    del mt
tsc = raw_score(p_teach[m_tr], y_tr[m_tr])
log(f"  teacher OOF 점수(매칭행 자체 기준): {tsc:.1f}  "
    f"— 높을수록 물리값이 실제로 유용하다는 뜻")

# =====================================================================
# 2. Student: soft label 학습
# =====================================================================
p_ref = np.load("lab/45_both_ens.npy").astype("float64")
s_ref = raw_score(p_ref, y_va)
# 동일 조건 1시드 기준선 (앙상블 아님) — 공정 비교용
def sframe(d):
    out = d[[c for c in FEATS if c not in s12.CAT]].copy()
    for c in s12.CAT:
        out[c] = d[c].astype(str)
    return out


Xtr, Xva = sframe(tr), sframe(va)
t0 = time.time()
base1 = CatBoostClassifier(**CB, random_seed=SEED).fit(
    Pool(Xtr, y_tr, cat_features=list(s12.CAT)))
p_base1 = base1.predict_proba(Pool(Xva, cat_features=list(s12.CAT)))[:, 1]
s_base1 = raw_score(p_base1, y_va)
tick(f"기준 student(hard label, 1시드): {s_base1:.1f}  ({time.time()-t0:.0f}s)")
del base1

results = {}
for a in ALPHAS:
    y_soft = y_tr.copy()
    y_soft[m_tr] = (1 - a) * y_tr[m_tr] + a * p_teach[m_tr]
    t0 = time.time()
    st = CatBoostClassifier(**{**CB, "loss_function": "CrossEntropy"},
                            random_seed=SEED).fit(
        Pool(Xtr, y_soft, cat_features=list(s12.CAT)))
    p = st.predict_proba(Pool(Xva, cat_features=list(s12.CAT)))[:, 1]
    sc = raw_score(p, y_va)
    results[a] = sc
    np.save(f"lab/57_alpha{int(a*100)}.npy", p.astype("float32"))
    tick(f"student α={a}: {sc:8.1f}  (기준 대비 {sc-s_base1:+.1f})  "
         f"({time.time()-t0:.0f}s)")
    del st

log("\n" + "=" * 72)
log("판정 (2024 폴드, 1시드 동일조건)")
log("=" * 72)
log(f"  기준 student (hard label)   {s_base1:8.1f}")
for a, sc in results.items():
    log(f"  증류 student α={a:<4}         {sc:8.1f}  ({sc-s_base1:+.1f})")
log(f"  (참고: CS79 2시드 앙상블 {s_ref:.1f} / teacher OOF {tsc:.1f})")
log("  +10↑면 α 세분화 + 다시드 확증 → 멀티클래스와 결합 검토")
log("=" * 72)
