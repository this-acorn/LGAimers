# -*- coding: utf-8 -*-
"""
[100] 챔피언(cat5 · 5클래스 · lr0.08) 위 짝지음 팔 — 08-29 밤 워크플로 생존 제안 (P08 / P05 / P06)

팔 (exp/93 하네스 그대로: thread14 · 시드[42,7] · it500/d6/lr0.08/l2=10 · MultiClass · 79피처 · cat5):
  C6  타겟만 6클래스: 빅미스(4)를 복원 구종으로 분할 → 4 = 빅미스&브레이킹(lab_brk==1), 5 = 빅미스&비브레이킹.
      P(성공)=클래스0 은 그대로. (M4 가 "합치면 −11.8" 이었으므로 반대 방향을 잰다. 죽은 7클래스는 성공을
      판정축(볼/스트)으로 쪼갠 것 — 이 팔은 실패 하위클래스를 투구 전 피처로 예측 가능한 축(구종)으로 쪼갠다)
  HT  피처·타겟 불변, has_time=True — CTR 을 학습행의 과거 행만으로 계산(무작위 순열 대신 시간순).
      학습 Pool 은 train 원 순서(row_id = 전역 시간순, exp/88 A) 그대로 넣는다.
  IT  피처·타겟 불변, iterations=1000 + staged_predict_proba(eval_period=50) 로 it 50..1000 학습곡선.
      기본 1시드(42). it=500 단계가 lab/89 시드42(851.2) 를 재현해야 함(자기검증). 결과 lab/100_IT_curve_seed{sd}.npz
새 피처 없음 → 행 독립·규칙 문제 없음. 기준: lab/89_cat5_probs_seed{42,7}.npy 시드별 재계산(851.2 / 856.5).
출력: 표준 6줄 + 신인/베테랑/F/R 세그먼트 paired 이득(P02) + lab/100_{arm}_probs_seed{sd}.npy, lab/100_{arm}.npy, lab/100_summary.txt
실행: PYTHONIOENCODING=utf-8 python -u exp/100_champ_arms.py --arm C6 [--smoke] [--seeds 42,7]
"""

import argparse
import gc
import importlib.util
import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import log, tick, raw_score, load_train

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="C6", choices=["BASE", "C6", "HT", "IT"],
                help="BASE = 챔피언 그대로(다른 기계에서 자기 기준선 만들 때) / C6 / HT / IT")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--seeds", default=None, help="기본: BASE/C6/HT = 42,7 / IT = 42")
ap.add_argument("--threads", type=int, default=14,
                help="CatBoost thread_count. ★ 같은 기계에서 BASE 와 팔을 같은 값으로 (스레드 수는 시드와 같다, HANDOFF exp/76)")
args = ap.parse_args()
ARM = args.arm
NTHREAD = args.threads

spec = importlib.util.spec_from_file_location("s12", "submit12_src/script.py")
s12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s12)

SEEDS_BASE = [42, 7]
SEEDS = [int(s) for s in args.seeds.split(",")] if args.seeds else ([42] if ARM == "IT" else [42, 7])
CB_PRM = dict(iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=10.0,
              verbose=False, thread_count=NTHREAD, allow_writing_files=False,
              loss_function="MultiClass")
if ARM == "HT":
    CB_PRM["has_time"] = True
if ARM == "IT":
    CB_PRM["iterations"] = 1000
    EVAL_PERIOD = 50
K_MIX = 50.0
NAMES = {0: "성공", 1: "미들만", 2: "리버스만", 3: "미들∩리버스", 4: "빅미스"}
NCLS = 6 if ARM == "C6" else 5

log(f"=== exp/100 arm={ARM} smoke={args.smoke} seeds={SEEDS} threads={NTHREAD} CB extra "
    + str({k: v for k, v in CB_PRM.items() if k in ("has_time", "iterations")}) + " ===")
log("train 로딩...")
df = load_train()

# ---- 라벨 복원 (exp/89·93 와 동일) ----
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
if ARM == "C6":
    brk = df.lab_brk.to_numpy()
    big = cls == 4
    cls[big & (brk == 1)] = 4
    cls[big & (brk == 0)] = 5
    cls[big & ~((brk == 0) | (brk == 1))] = -1     # 구종 미복원 빅미스 행은 제외
    NAMES[4], NAMES[5] = "빅미스&브레이킹", "빅미스&비브레이킹"
df["_cls"] = cls
log(f"클래스 분포: " + " ".join(f"{c}:{np.mean(cls==c)*100:.1f}%" for c in range(NCLS))
    + f"  미복원 {np.mean(cls==-1)*100:.2f}%")
del ok, m_, r_
b_, s_ = df.balls_before.to_numpy(), df.strikes_before.to_numpy()
df["_cg"] = np.where(s_ > b_, 2, np.where(b_ > s_, 0, 1)).astype("int8")


# ---- exp/89·93 와 동일한 79피처 준비 ----
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
                         thread_count=NTHREAD, allow_writing_files=False, random_seed=42)
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
tr = s12.add_features(pd.concat(parts).sort_index(), prior)     # sort_index = train 원 순서(시간순)
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
tick("피처 부착 완료")

BASE = [c for c in pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1)
        .columns.str.replace("﻿", "").str.strip() if c != "row_id"]
ENG18 = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
         "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
         "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
         "f_pb_diff", "f_miss_p", "f_miss_prev1"]
FEATS = BASE + ENG18 + s12.CS_FEATS + s12.PT_FEATS
CATS = list(s12.CAT) + ["pitcher_team_id", "batter_team_id"]
assert len(FEATS) == 79 and len(CATS) == 5
r_va = float(y_va.mean())
DEN = r_va * (1 - r_va)

# ---- 짝지음 기준: 이 기계의 BASE 팔(lab/100_BASE_probs_seed*.npy)이 있으면 그것, 없으면 lab/89(두 기계, thread14) ----
import os
if all(os.path.exists(f"lab/100_BASE_probs_seed{sd}.npy") for sd in SEEDS_BASE) and ARM != "BASE":
    BASE_SRC = "lab/100_BASE (이 기계 기준선)"
    P_base = {sd: np.load(f"lab/100_BASE_probs_seed{sd}.npy").astype("float64") for sd in SEEDS_BASE}
elif all(os.path.exists(f"lab/89_cat5_probs_seed{sd}.npy") for sd in SEEDS_BASE):
    BASE_SRC = "lab/89 (두 기계 thread14 기준선)" + ("" if NTHREAD == 14 else "  ⚠ 스레드 수가 달라 비짝지음(σ≈21) — 먼저 --arm BASE 를 돌리세요")
    P_base = {sd: np.load(f"lab/89_cat5_probs_seed{sd}.npy").astype("float64") for sd in SEEDS_BASE}
else:
    raise SystemExit("기준선 없음: 먼저 `--arm BASE` 를 같은 --threads 로 돌려 lab/100_BASE_probs_seed{42,7}.npy 를 만드세요")
log(f"짝지음 기준선: {BASE_SRC}")
assert all(len(v) == len(va) for v in P_base.values()), "기준선 정렬 불일치"
BASE_SOLO = {sd: raw_score(P_base[sd][:, 0], y_va) for sd in SEEDS_BASE}
p_base = np.mean([P_base[sd][:, 0] for sd in SEEDS_BASE], axis=0)
BASE_ENS = raw_score(p_base, y_va)
BASE_PROJ8 = 1.75 * BASE_ENS - 0.75 * float(np.mean(list(BASE_SOLO.values())))
BASE_PEN = 100000 * (p_base.mean() - r_va) ** 2 / DEN
log(f"기준 cat5: 시드별 {BASE_SOLO[42]:.1f} / {BASE_SOLO[7]:.1f}  2시드 {BASE_ENS:.1f}  proj8 {BASE_PROJ8:.1f}  벌점 {BASE_PEN:.1f}")

# ---- 세그먼트 (P02): 신인/베테랑/F/R — lab/90 parquet 정렬 동일 ----
SEG = {}
try:
    a90 = pd.read_parquet("lab/90_analysis_2024.parquet", columns=["game_type", "p_first_season", "asof_pitcher_n"])
    if len(a90) == len(va):
        gt = a90["game_type"].astype(str).to_numpy()
        rk = (a90["p_first_season"].to_numpy() == 2024)
        SEG = {"신인": rk, "베테랑": ~rk, "F": gt == "F", "R": gt == "R"}
except Exception as e:
    log(f"  (세그먼트 정보 없음: {e})")


def seg_gain(p_new, p_old):
    """세그먼트별 paired 이득을 전체 점수 척도(100000/(N·DEN))로"""
    out = []
    for name, m in SEG.items():
        dsse = float(np.sum((p_old[m] - y_va[m]) ** 2) - np.sum((p_new[m] - y_va[m]) ** 2))
        out.append(f"{name} {100000 * dsse / (len(y_va) * DEN):+.1f}")
    return "  ".join(out)


# ---- 타겟 ----
m_tr = tr["_cls"].to_numpy() >= 0
y_cls = tr["_cls"].to_numpy()[m_tr].astype("int64")
log(f"멀티클래스 학습 행: {m_tr.sum():,} / {len(tr):,}   피처 {len(FEATS)}  cat {len(CATS)}개  클래스 {NCLS}개 "
    + " ".join(f"{i}:{np.mean(y_cls==i)*100:.1f}%" for i in range(NCLS)))


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
    prm = {**CB_PRM, "iterations": 40 if ARM == "IT" else 20}
    m = CatBoostClassifier(**prm, random_seed=42).fit(ptr)
    prob = m.predict_proba(pva)
    assert list(m.classes_)[0] == 0, f"클래스0 이 첫 열이 아님: {m.classes_}"
    p = prob[:, 0]
    log(f"smoke: 6만행·{prm['iterations']}it 학습/예측 정상 — 클래스 {list(m.classes_)}  P(성공) 평균 {p.mean():.4f} 점수 {raw_score(p, y_va):.1f}")
    if ARM == "IT":
        st = [raw_score(pr[:, 0], y_va) for pr in m.staged_predict_proba(pva, eval_period=10)]
        log(f"smoke staged(10it 단위): {['%.1f' % x for x in st]}  (마지막 = 전체 예측 {raw_score(p, y_va):.1f} 과 일치해야 함)")
    if SEG:
        log("smoke 세그먼트 paired(vs cat5 2시드): " + seg_gain(p, p_base))
    imp = pd.Series(m.get_feature_importance(), index=FEATS).sort_values(ascending=False)
    log("smoke 중요도 상위 8: " + ", ".join(f"{k}={v:.1f}" for k, v in imp.head(8).items()))
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
    assert list(m.classes_)[0] == 0, f"클래스0 이 첫 열이 아님: {m.classes_}"
    if ARM == "IT":
        its, scores, pens = [], [], []
        best = (None, -1e9)
        for j, pr in enumerate(m.staged_predict_proba(pva, eval_period=EVAL_PERIOD), start=1):
            it = j * EVAL_PERIOD
            pj = pr[:, 0]
            sc = raw_score(pj, y_va)
            its.append(it); scores.append(sc); pens.append(100000 * (pj.mean() - r_va) ** 2 / DEN)
            if it == 500:
                p500 = pj.copy()
                log(f"   it=500 자기검증: {sc:.1f} vs lab/89 시드{sd} {BASE_SOLO.get(sd, float('nan')):.1f}  최대차 {np.max(np.abs(pj - P_base[sd][:, 0])) if sd in P_base else float('nan'):.2e}")
            if sc > best[1]:
                best = (it, sc)
        np.save(f"lab/100_IT_curve_seed{sd}.npy", np.array([its, scores, pens]))
        log("   학습곡선 it:점수(벌점) " + "  ".join(f"{i}:{s:.1f}({q:.0f})" for i, s, q in zip(its, scores, pens)))
        log(f"   ★ 정점 it={best[0]} {best[1]:.1f}  (it500 대비 {best[1] - scores[its.index(500)]:+.1f})")
        prob = m.predict_proba(pva)
        p = p500          # 6줄 표는 it500 기준(짝지음 검증), 곡선은 위 로그
    else:
        prob = m.predict_proba(pva)
        p = prob[:, 0]
    np.save(f"lab/100_{ARM}_probs_seed{sd}.npy", prob.astype("float32"))
    preds.append(p)
    solo.append(raw_score(p, y_va))
    base_sd = BASE_SOLO.get(sd, float("nan"))
    tick(f"arm {ARM} seed={sd}  {solo[-1]:8.1f}  (paired vs cat5 {solo[-1]-base_sd:+.1f}, {time.time()-t0:.0f}s)")
    if SEG and sd in P_base:
        log("   세그먼트 paired: " + seg_gain(p, P_base[sd][:, 0]))
    imp = pd.Series(m.get_feature_importance(), index=FEATS)
    rk_ = imp.rank(ascending=False)
    log("   참고 팀ID 중요도: " + ", ".join(f"{k}={imp[k]:.2f}(#{int(rk_[k])})" for k in ["pitcher_team_id", "batter_team_id"]))
    del m, prob
    gc.collect()

ens = np.mean(preds, axis=0)
np.save(f"lab/100_{ARM}.npy", ens.astype("float32"))
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
pairs = [solo[i] - BASE_SOLO[sd] for i, sd in enumerate(SEEDS) if sd in BASE_SOLO]
nseed = len(SEEDS)
if nseed == 1:
    log("  ⚠ 1시드 실행 — 2시드/proj8 줄은 시드42 단독 기준이라 짝지음 기준(2시드)과 직접 비교 불가")
log("")
log("=" * 80)
log(f"arm {ARM}  {nseed}시드 {e2:7.1f}  시드평균 {ms:7.1f}  proj8 {p8:7.1f}  벌점 {pen:5.1f}")
log(f"1. raw paired gain ({nseed}시드)     {dK:+.1f}   시드별 {['%+.1f' % x for x in pairs]}  paired 평균 {np.mean(pairs):+.1f} 산포 {np.std(pairs):.2f}")
log(f"2. 중심 기여                    벌점 {BASE_PEN:.1f} → {pen:.1f}  ({BASE_PEN-pen:+.1f})")
log(f"3. 동일평균 후 shape gain       {sh_new - sh_base:+.1f}   (기준 {sh_base:.1f} → {sh_new:.1f})")
log(f"4. 연도별 gain                  2024 폴드 단일 ({dK:+.1f}) — 다연도 미측정"
    + ("   세그먼트 앙상블 paired: " + seg_gain(ens, p_base) if SEG else ""))
log(f"5. 냉동 표 의존                 " + {"BASE": "(기준선 재현 — 다른 기계/스레드 격차 측정용)", "C6": "없음 — 타겟 구조 변경(모델축), 피처 불변", "HT": "없음 — CTR 계산 순서만 변경(피처·타겟 불변)", "IT": "없음 — 학습 반복수만"}[ARM])
log(f"6. ★시드평균 이득 / proj8 이득  {ms - np.mean([BASE_SOLO[sd] for sd in SEEDS if sd in BASE_SOLO]):+.1f} / {p8 - BASE_PROJ8:+.1f}")
log(f"   혼합 진단: d={dK:+.1f}  K={K:.1f}  w*={w_star:.2f}  혼합이득 {bg:+.1f}")
log("=" * 80)
with open("lab/100_summary.txt", "a", encoding="utf-8") as f:
    f.write(f"arm={ARM} seeds={SEEDS} ncls={NCLS} solo={[round(float(x),2) for x in solo]} ens={e2:.1f} seedmean={ms:.1f} "
            f"proj8={p8:.1f} pen={pen:.1f} paired={[round(float(x),2) for x in pairs]} shape={sh_new-sh_base:+.1f} d={dK:+.1f} K={K:.1f}\n")
