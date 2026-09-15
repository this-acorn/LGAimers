# -*- coding: utf-8 -*-
"""
[93] 저카디널 categorical 조합 + 4클래스 재구성 — 챔피언 cat5 위 짝지음 ablation (사용자 계획 08-28 밤)

팔 (exp/89 cat5 챔피언 하네스 그대로, thread14 · 시드[42,7] · it500/d6/lr0.08/l2=10 · MultiClass):
  A   + team 조합 3개  (team_matchup = p팀×b팀 / pteam_role = p팀×top_bottom×game_type / bteam_role)
  B   + count/hand 조합 2개 (count_hand = balls×strikes×p손×b손 / base_out = base_state×outs)
  C   A + B (5개)
  M4  피처 그대로(79, cat5), 타겟만 5→4클래스: 클래스 3(미들∩리버스)을 lab/92_merge.json 의
      병합 대상(2024 홀드아웃 혼동 구조로 결정)에 합침. P(성공)=클래스0 은 그대로.
  OH  (4순위) 피처·타겟 불변, one_hot_max_size=16 — cat5 전부 one-hot (CTR 없음)
  CTR1 (4순위) 피처·타겟 불변, max_ctr_complexity=1 — CatBoost 내부 cat 조합 CTR 금지
새 조합 피처는 전부 '현재 행 값'만으로 만든다 (다른 행·냉동 표·타겟 무관 → 행 독립, 규칙 준수).

기준: lab/89_cat5_probs_seed{42,7}.npy 에서 시드별 점수를 실행 시점에 재계산해 짝지음 (851.2 / 856.5).
출력: HANDOFF §1.9 표준 6줄 + d/K 혼합 이득, lab/93_{arm}_probs_seed{sd}.npy, lab/93_{arm}.npy,
      lab/93_summary.txt 한 줄 append
실행: PYTHONIOENCODING=utf-8 python -u exp/93_catcombo.py --arm A [--smoke] [--merge auto|1|2|4]
"""

import argparse
import gc
import importlib.util
import json
import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import log, tick, raw_score, load_train

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="A", choices=["A", "B", "C", "M4", "OH", "CTR1"])
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--merge", default="auto", help="M4 전용: auto(lab/92_merge.json) | 1 | 2 | 4")
args = ap.parse_args()
ARM = args.arm

spec = importlib.util.spec_from_file_location("s12", 'submissions/submit12_src/script.py')
s12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s12)

SEEDS = [42, 7]
CB_PRM = dict(iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=10.0,
              verbose=False, thread_count=14, allow_writing_files=False,
              loss_function="MultiClass")
K_MIX = 50.0
NAMES = {0: "성공", 1: "미들만", 2: "리버스만", 3: "미들∩리버스", 4: "빅미스"}

COMBO_A = ["c_team_matchup", "c_pteam_role", "c_bteam_role"]
COMBO_B = ["c_count_hand", "c_base_out"]
ARM_FEATS = {"A": COMBO_A, "B": COMBO_B, "C": COMBO_A + COMBO_B, "M4": [], "OH": [], "CTR1": []}
# 4순위 (CatBoost 범주형 설정, 피처·타겟 불변):
#   OH   one_hot_max_size=16 → 팀ID(13)·base_state(8)·CAT3 전부 one-hot, CTR 없음 (팔 A 의 CTR 드리프트 실패의 반대 방향)
#   CTR1 max_ctr_complexity=1 → CatBoost 내부 cat 조합 CTR 생성 금지 (자동 조합이 돕는지 해치는지)
if ARM == "OH":
    CB_PRM["one_hot_max_size"] = 16
elif ARM == "CTR1":
    CB_PRM["max_ctr_complexity"] = 1

MERGE = None
if ARM == "M4":
    if args.merge == "auto":
        MERGE = int(json.load(open("lab/92_merge.json"))["merge_into"])
    else:
        MERGE = int(args.merge)
    assert MERGE in (1, 2, 4)

log(f"=== exp/93 arm={ARM} smoke={args.smoke}" + (f" merge 3->{MERGE}({NAMES[MERGE]})" if MERGE else "")
    + (" CB extra " + str({k: v for k, v in CB_PRM.items() if k in ("one_hot_max_size", "max_ctr_complexity")}) if ARM in ("OH", "CTR1") else "") + " ===")
log("train 로딩...")
df = load_train()

# ---- 라벨 복원 (exp/89 와 동일) ----
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
del dd, rec, pid, n, nxt
gc.collect()

y_bin = df.control_success.to_numpy("float64")
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
del ok, m_, r_
b_, s_ = df.balls_before.to_numpy(), df.strikes_before.to_numpy()
df["_cg"] = np.where(s_ > b_, 2, np.where(b_ > s_, 0, 1)).astype("int8")


# ---- 조합 categorical (현재 행 값만 사용 — 행 독립) ----
def _s(v):
    if pd.api.types.is_numeric_dtype(v):
        return v.fillna(-1).astype("int64").astype(str)
    return v.astype(str)


def add_combos(d):
    d = d.copy()
    pt, bt = _s(d.pitcher_team_id), _s(d.batter_team_id)
    tb, gt = _s(d.top_bottom), _s(d.game_type)
    b, s = _s(d.balls_before), _s(d.strikes_before)
    ph, bh = _s(d.pitcher_hand), _s(d.batter_hand)
    bs, o = _s(d.base_state), _s(d.outs_before)
    d["c_team_matchup"] = (pt + "_" + bt).to_numpy()
    d["c_pteam_role"] = (pt + "_" + tb + "_" + gt).to_numpy()
    d["c_bteam_role"] = (bt + "_" + tb + "_" + gt).to_numpy()
    d["c_count_hand"] = (b + s + "_" + ph + bh).to_numpy()
    d["c_base_out"] = (bs + "_" + o).to_numpy()
    return d


# ---- exp/89 와 동일한 79피처 준비 ----
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
del pfb_rows
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
del parts, tr_rows, hist
gc.collect()
key = tr[["pitcher_id", "season", "_cg"]].merge(
    mix_tr, on=["pitcher_id", "season", "_cg"], how="left")
tr["f_mixcg_fb"] = key["mix_fb"].to_numpy("float32")
tr["f_mixcg_brk"] = key["mix_brk"].to_numpy("float32")
tr["f_pfb"] = pfb.predict_proba(tr[s12.PFB_IN].fillna(-999))[:, 1].astype("float32")
del key
va = s12.attach_pt(s12.add_features(s12.attach_cs(va_rows, cp, cb_), prior),
                   mix_dep, pfb)
del va_rows, df
gc.collect()
if ARM_FEATS[ARM]:
    tr = add_combos(tr)
    va = add_combos(va)
    for c in ARM_FEATS[ARM]:
        lv_tr = set(tr[c].unique())
        unseen = (~va[c].isin(lv_tr)).mean() * 100
        log(f"  조합 {c:16s} 학습 레벨 {len(lv_tr):4d}  검증 레벨 {va[c].nunique():4d}  검증 미출현행 {unseen:.3f}%")
tick("피처 부착 완료")

BASE = [c for c in pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1)
        .columns.str.replace("﻿", "").str.strip() if c != "row_id"]
ENG18 = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
         "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
         "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
         "f_pb_diff", "f_miss_p", "f_miss_prev1"]
FEATS79 = BASE + ENG18 + s12.CS_FEATS + s12.PT_FEATS
FEATS = FEATS79 + ARM_FEATS[ARM]
CAT5 = list(s12.CAT) + ["pitcher_team_id", "batter_team_id"]
CATS = CAT5 + ARM_FEATS[ARM]
r_va = float(y_va.mean())
DEN = r_va * (1 - r_va)

# ---- 짝지음 기준: lab/89 cat5 시드별 확률에서 재계산 ----
P_base = {sd: np.load(f"lab/89_cat5_probs_seed{sd}.npy").astype("float64") for sd in SEEDS}
assert all(len(v) == len(va) for v in P_base.values()), "lab/89 정렬 불일치"
BASE_SOLO = {sd: raw_score(P_base[sd][:, 0], y_va) for sd in SEEDS}
p_base = np.mean([P_base[sd][:, 0] for sd in SEEDS], axis=0)
BASE_ENS = raw_score(p_base, y_va)
BASE_PROJ8 = 1.75 * BASE_ENS - 0.75 * float(np.mean(list(BASE_SOLO.values())))
BASE_PEN = 100000 * (p_base.mean() - r_va) ** 2 / DEN
log(f"기준 cat5: 시드별 {BASE_SOLO[42]:.1f} / {BASE_SOLO[7]:.1f}  2시드 {BASE_ENS:.1f}  proj8 {BASE_PROJ8:.1f}  벌점 {BASE_PEN:.1f}")

# ---- 타겟 ----
m_tr = tr["_cls"].to_numpy() >= 0
y_cls = tr["_cls"].to_numpy()[m_tr].astype("int64")
if ARM == "M4":
    y_cls = np.where(y_cls == 3, MERGE, y_cls)
    remap = {c: i for i, c in enumerate(sorted(np.unique(y_cls)))}   # {0:0,1:1,2:2,4:3} 등 — 0 은 항상 0
    assert remap[0] == 0 and len(remap) == 4
    y_cls = np.vectorize(remap.get)(y_cls).astype("int64")
    log(f"4클래스 재구성: 3({NAMES[3]}) → {MERGE}({NAMES[MERGE]})  remap {remap}  분포 "
        + " ".join(f"{i}:{np.mean(y_cls==i)*100:.1f}%" for i in range(4)))
log(f"멀티클래스 학습 행: {m_tr.sum():,} / {len(tr):,}   피처 {len(FEATS)} (79 + {len(ARM_FEATS[ARM])})  cat {len(CATS)}개 {CATS}")


def build(d, cats):
    out = pd.DataFrame(index=d.index)
    for c in FEATS:
        if c in cats:
            v = d[c]
            out[c] = (v.fillna(-1).astype("int64").astype(str)
                      if pd.api.types.is_numeric_dtype(v) else v.astype(str))
        else:
            out[c] = d[c]
    return out


if args.smoke:
    idx = np.where(m_tr)[0][:60000]
    Xtr = build(tr.iloc[idx], CATS)
    ptr = Pool(Xtr, y_cls[:60000], cat_features=CATS)
    Xva = build(va, CATS)
    pva = Pool(Xva, cat_features=CATS)
    m = CatBoostClassifier(**{**CB_PRM, "iterations": 20}, random_seed=42).fit(ptr)
    prob = m.predict_proba(pva)
    assert list(m.classes_)[0] == 0, f"클래스0 이 첫 열이 아님: {m.classes_}"
    p = prob[:, 0]
    log(f"smoke: 6만행·20it 학습/예측 정상 — 클래스 {list(m.classes_)}  P(성공) 평균 {p.mean():.4f} 점수 {raw_score(p, y_va):.1f}")
    imp = pd.Series(m.get_feature_importance(), index=FEATS).sort_values(ascending=False)
    log("smoke 중요도 상위 12: " + ", ".join(f"{k}={v:.1f}" for k, v in imp.head(12).items()))
    if ARM_FEATS[ARM]:
        log("smoke 새 조합 중요도: " + ", ".join(f"{k}={imp[k]:.1f}" for k in ARM_FEATS[ARM]))
        log("smoke 조합 예시: " + " | ".join(f"{c}={Xva[c].iloc[0]}" for c in ARM_FEATS[ARM]))
    sys.exit(0)

Xtr = build(tr[m_tr], CATS)
del tr
gc.collect()
ptr = Pool(Xtr, y_cls, cat_features=CATS)
del Xtr
gc.collect()
Xva = build(va, CATS)
pva = Pool(Xva, cat_features=CATS)
del Xva, va
gc.collect()
tick("Pool 구성 완료")

preds, solo = [], []
for sd in SEEDS:
    t0 = time.time()
    m = CatBoostClassifier(**CB_PRM, random_seed=sd).fit(ptr)
    prob = m.predict_proba(pva)
    assert list(m.classes_)[0] == 0, f"클래스0 이 첫 열이 아님: {m.classes_}"
    np.save(f"lab/93_{ARM}_probs_seed{sd}.npy", prob.astype("float32"))
    p = prob[:, 0]
    preds.append(p)
    solo.append(raw_score(p, y_va))
    tick(f"arm {ARM} seed={sd}  {solo[-1]:8.1f}  (paired vs cat5 {solo[-1]-BASE_SOLO[sd]:+.1f}, {time.time()-t0:.0f}s)")
    imp = pd.Series(m.get_feature_importance(), index=FEATS)
    rk = imp.rank(ascending=False)
    if ARM_FEATS[ARM]:
        log("   새 조합 중요도: " + ", ".join(f"{k}={imp[k]:.2f}(#{int(rk[k])})" for k in ARM_FEATS[ARM]))
    log("   참고 팀ID 중요도: " + ", ".join(f"{k}={imp[k]:.2f}(#{int(rk[k])})" for k in ["pitcher_team_id", "batter_team_id"]))
    del m, prob
    gc.collect()

ens = np.mean(preds, axis=0)
np.save(f"lab/93_{ARM}.npy", ens.astype("float32"))
e2 = raw_score(ens, y_va)
ms = float(np.mean(solo))
p8 = 1.75 * e2 - 0.75 * ms
pen = 100000 * (ens.mean() - r_va) ** 2 / DEN
sh_new = raw_score(ens - ens.mean() + r_va, y_va)
sh_base = raw_score(p_base - p_base.mean() + r_va, y_va)
K = (100000.0 / DEN) * float(np.mean((ens - p_base) ** 2))
dK = e2 - BASE_ENS
bg = (dK + K) ** 2 / (4 * K) if abs(dK) < K else max(dK, 0.0)
w_star = min(max(0.5 + dK / (2 * K), 0.0), 1.0)
pairs = [solo[i] - BASE_SOLO[sd] for i, sd in enumerate(SEEDS)]
# game_type 별 paired 이득 (3순위 판단 재료)
gt = None
try:
    gt = pd.read_parquet("lab/90_analysis_2024.parquet", columns=["game_type"])["game_type"].astype(str).to_numpy()
    if len(gt) != len(ens):
        gt = None
except Exception:
    pass
log("")
log("=" * 80)
log(f"arm {ARM} ({len(ARM_FEATS[ARM])}조합" + (f", 4클래스 3→{MERGE}" if MERGE else "") + f")  2시드 {e2:7.1f}  시드평균 {ms:7.1f}  proj8 {p8:7.1f}  벌점 {pen:5.1f}")
log(f"1. raw paired gain (2시드)     {dK:+.1f}   시드별 {['%+.1f' % x for x in pairs]}  paired 평균 {np.mean(pairs):+.1f} 산포 {np.std(pairs):.2f}")
log(f"2. 중심 기여                    벌점 {BASE_PEN:.1f} → {pen:.1f}  ({BASE_PEN-pen:+.1f})")
log(f"3. 동일평균 후 shape gain       {sh_new - sh_base:+.1f}   (기준 {sh_base:.1f} → {sh_new:.1f})")
log(f"4. 연도별 gain                  2024 폴드 단일 ({dK:+.1f}) — 다연도 미측정"
    + (f"   game_type별 2시드 이득: " + "  ".join(f"{g} {raw_score(ens[gt==g], y_va[gt==g]) - raw_score(p_base[gt==g], y_va[gt==g]):+.1f}" for g in sorted(set(gt))) if gt is not None else ""))
log(f"5. 냉동 표 의존                 " + ("설정 변경만 (피처·타겟 불변)" if ARM in ("OH", "CTR1") else "피처 정의상 없음 — 단 cat CTR 은 ≤학습연도 타겟 통계표임(팔 A 교훈)" + (" / 타겟 구조 변경(모델축)" if MERGE else "")))
log(f"6. ★시드평균 이득 / proj8 이득  {ms - np.mean(list(BASE_SOLO.values())):+.1f} / {p8 - BASE_PROJ8:+.1f}")
log(f"   혼합 진단: d={dK:+.1f}  K={K:.1f}  w*={w_star:.2f}  혼합이득 {bg:+.1f}")
log("=" * 80)
with open("lab/93_summary.txt", "a", encoding="utf-8") as f:
    f.write(f"arm={ARM} merge={MERGE} nfeat={len(FEATS)} solo={[round(float(x),2) for x in solo]} ens={e2:.1f} seedmean={ms:.1f} "
            f"proj8={p8:.1f} pen={pen:.1f} paired={[round(float(x),2) for x in pairs]} shape={sh_new-sh_base:+.1f} d={dK:+.1f} K={K:.1f}\n")
