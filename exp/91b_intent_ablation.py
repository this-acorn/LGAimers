# -*- coding: utf-8 -*-
"""
[91b] 의도·도착 영역 프로필 ablation — 사용자 제안(08-28)의 '라벨 기반 상한' 측정

배경 (exp/91a 확정):
  · ball/strike 는 위치가 아니라 판정 결과 라벨 (볼→balls+1 93%, 인플레이→새 타석 100%)
  · 3영역 = {0 볼(심판 볼, 비미들) / 1 가운데(mid=1) / 2 비가운데 존(스트라이크·파울·인플레이, 비미들)}
    인플레이·비미들(14.5%)은 성공률 .734 ≈ 엣지 .717 이라 영역 2 에 병합
  · 성공 투구 중 P(볼) 은 카운트로 0.27~0.50, 투수 간 진짜 sd 는 0.035 — 얇은 신호

왜 Trackman 없이 먼저 재는가:
  Trackman 도착모델 p_A 를 투수×상황 셀로 평균하면 그 셀의 경험적 영역 빈도로 수렴한다
  (보정된 모델의 그룹 평균 = 그룹 빈도). 셀 빈도는 Δ라벨로 100% 커버리지에서 정확히
  계산된다. 따라서 이 스크립트의 D2 는 사용자 설계 D 의 셀 수준 상한이다.
  D2 ≈ B 이면 Trackman 단계는 착수 전에 종결. D2 > B 여야 Trackman 정밀화 가치가 생긴다.

팔 (exp/89 cat5 챔피언 하네스 위에 피처만 추가, thread14 · 시드[42,7] · it500/d6/lr0.08):
  B  : + 투수×카운트×타자손 smoothed 성공률 (+ n, fallback 단계)             4피처
  C  : B + 의도 P(볼|성공) 표                                                6피처
  D  : C + 도착 P(볼), P(미들), 일치도, P(성공|볼), P(성공|존), 기대실행률   12피처
표는 시즌 as-of 냉동 (S 행 ← 시즌<S 라벨 행) — 2025 배포 시 ≤2024 표를 얼려 조인하는
상황을 그대로 모사한다. 수축 λ=50, 계층: (p,b,s,bh) → (p,b,s) → 투수전체+카운트오프셋 → 리그.

기준: lab/89_cat5 (시드42 851.2 / 시드7 856.5 / 2시드 873.9 / proj8 889.0 / 벌점 37.0)
출력: HANDOFF §1.9 표준 6줄 + d/K 혼합 이득
실행: python -u exp/91b_intent_ablation.py --arm D  [--smoke]
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
ap.add_argument("--arm", default="D", choices=["B", "C", "D", "T0", "T"])
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
ARM = args.arm
# T0 = D′ + 의도모델 4피처 / T = T0 + Trackman 프로필 9피처 (exp/91c 산출물, 사용자 설계 전체)
IM_FEATS = ["f_im_ball", "f_im_exec_ball", "f_im_exec_edge", "f_im_exec_exp"]
TM_FEATS = ["f_tm_agree", "f_tm_ctrl", "f_tm_arr_ball", "f_tm_arr_mid", "f_tm_logn",
            "f_tm_agree_sd", "f_tm_agree_fb", "f_tm_agree_nfb", "f_tm_agree_pt"]

spec = importlib.util.spec_from_file_location("s12", 'submissions/submit12_src/script.py')
s12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s12)

SEEDS = [42, 7]
CB_PRM = dict(iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=10.0,
              verbose=False, thread_count=14, allow_writing_files=False,
              loss_function="MultiClass")
K_MIX = 50.0
LAM = 50.0
MIN_LVL_N = 30
BASE_SOLO = {42: 851.2, 7: 856.5}
BASE_ENS = 873.9
BASE_PROJ8 = 889.0
BASE_PEN = 37.0

log(f"=== exp/91b arm={ARM} smoke={args.smoke} ===")
log("train 로딩...")
df = load_train()

# ---- 라벨 복원 (exp/89 + ball/strike) ----
dd = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index()
pid = dd.pitcher_id.to_numpy()
n = dd.asof_pitcher_n.fillna(0).to_numpy("float64")
nxt = (pid[1:] == pid[:-1]) & (np.abs(n[1:] - n[:-1] - 1) < 1e-6)
for k, col in [("lab_mid", "asof_pitcher_middle_rate"),
               ("lab_rev", "asof_pitcher_reverse_rate"),
               ("lab_fb", "asof_pitcher_fastball_rate"),
               ("lab_brk", "asof_pitcher_breaking_rate"),
               ("lab_ball", "asof_pitcher_ball_rate"),
               ("lab_strike", "asof_pitcher_strike_rate")]:
    S = dd[col].fillna(0).to_numpy("float64") * n
    lab = np.full(len(dd), np.nan)
    diff = np.round(S[1:] - S[:-1])
    lab[:-1] = np.where(nxt, diff, np.nan)
    dd[k] = np.where((lab == 0) | (lab == 1), lab, np.nan)
rec = dd.set_index("index").sort_index()
for k in ["lab_mid", "lab_rev", "lab_fb", "lab_brk", "lab_ball", "lab_strike"]:
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
b_, s_ = df.balls_before.to_numpy(), df.strikes_before.to_numpy()
df["_cg"] = np.where(s_ > b_, 2, np.where(b_ > s_, 0, 1)).astype("int8")

# ---- 3영역 + 표 지표 컬럼 ----
labeled = ok & df.lab_ball.notna().to_numpy()
ball1 = df.lab_ball.to_numpy() == 1
reg = np.full(len(df), -1, dtype="int8")
reg[labeled & m_] = 1
reg[labeled & ~m_ & ball1] = 0
reg[labeled & ~m_ & ~ball1] = 2
df["_reg"] = reg
succ = labeled & (y_bin == 1)
IND = {"i_n": labeled, "i_succ": succ, "i_ball": reg == 0, "i_mid": reg == 1,
       "i_edge": reg == 2, "i_succ_ball": succ & (reg == 0), "i_succ_edge": succ & (reg == 2)}
for k, v in IND.items():
    df[k] = v.astype("float32")
log(f"영역 분포 (라벨행 {labeled.mean()*100:.2f}%): "
    + " ".join(f"{r}:{np.mean(reg[labeled]==r)*100:.1f}%" for r in (0, 1, 2))
    + f"   성공 중 볼 {np.mean(reg[succ]==0)*100:.1f}% / 존 {np.mean(reg[succ]==2)*100:.1f}%")
del ok, m_, r_, labeled, ball1, succ, reg
gc.collect()

# 표 수량: (분자, 분모)
QTY = {"succ": ("i_succ", "i_n"), "int": ("i_succ_ball", "i_succ"),
       "arr_ball": ("i_ball", "i_n"), "arr_mid": ("i_mid", "i_n"),
       "sgb": ("i_succ_ball", "i_ball"), "sge": ("i_succ_edge", "i_edge")}
ICOLS = list(IND.keys())
K1 = ["pitcher_id", "balls_before", "strikes_before", "batter_hand"]
K2 = ["pitcher_id", "balls_before", "strikes_before"]
K3 = ["pitcher_id"]
K4 = ["pitcher_hand", "balls_before", "strikes_before"]
KPH = ["pitcher_hand"]
KBS = ["balls_before", "strikes_before"]
LEVELS = [("T1", K1), ("T2", K2), ("T3", K3), ("T4", K4), ("TPH", KPH), ("TBS", KBS)]


def level_tables(hist):
    """hist: 시즌<S 행. 각 계층의 지표 합계 표."""
    h = hist[hist.i_n > 0]
    out = {}
    for name, key in LEVELS:
        out[name] = h.groupby(key, observed=True)[ICOLS].sum().reset_index()
    out["G"] = h[ICOLS].sum()
    return out


def _shr(num, den, prior):
    return (num + LAM * prior) / (den + LAM)


def attach_profile(rows, T):
    """rows 에 프로필 피처 부착. T=None 이면 전부 NaN."""
    d = rows.copy()
    feats = {}
    if T is None:
        for q in QTY:
            feats[f"R1_{q}"] = np.full(len(d), np.nan)
            feats[f"P2_{q}"] = np.full(len(d), np.nan)
        n1 = n2 = n3 = np.zeros(len(d))
    else:
        keyf = d[["pitcher_id", "balls_before", "strikes_before", "batter_hand", "pitcher_hand"]].copy()
        for c in keyf.columns:
            keyf[c] = keyf[c].fillna(0).astype("int64")
        sums = {}
        for name, key in LEVELS:
            t = T[name].copy()
            for c in key:
                t[c] = t[c].fillna(0).astype("int64")
            mg = keyf[key].merge(t, on=key, how="left")
            sums[name] = {c: mg[c].fillna(0).to_numpy("float64") for c in ICOLS}
        G = T["G"]
        n1, n2, n3 = sums["T1"]["i_n"], sums["T2"]["i_n"], sums["T3"]["i_n"]
        for q, (nu, de) in QTY.items():
            g = float(G[nu]) / max(float(G[de]), 1.0)
            Rbs = _shr(sums["TBS"][nu], sums["TBS"][de], g)
            Rph = _shr(sums["TPH"][nu], sums["TPH"][de], g)
            R4 = _shr(sums["T4"][nu], sums["T4"][de], Rbs)
            R3 = _shr(sums["T3"][nu], sums["T3"][de], g)
            P2 = np.clip(R3 + (R4 - Rph), 0.0, 1.0)          # 투수전체 + 카운트 오프셋
            R2 = _shr(sums["T2"][nu], sums["T2"][de], P2)
            R1 = _shr(sums["T1"][nu], sums["T1"][de], R2)
            feats[f"R1_{q}"] = np.where(n3 > 0, R1, np.nan)
            feats[f"P2_{q}"] = np.where(n3 > 0, P2, np.nan)
    lvl = np.where(n1 >= MIN_LVL_N, 1, np.where(n2 >= MIN_LVL_N, 2, np.where(n3 >= MIN_LVL_N, 3, 4)))
    # B
    d["f_pc_succ"] = np.asarray(feats["R1_succ"], "float32")
    d["f_pc_succ_d"] = np.asarray(feats["R1_succ"] - feats["P2_succ"], "float32")
    d["f_pc_logn"] = np.log1p(n1).astype("float32")
    d["f_pc_lvl"] = lvl.astype("int8")
    # C
    d["f_pc_int"] = np.asarray(feats["R1_int"], "float32")
    d["f_pc_int_d"] = np.asarray(feats["R1_int"] - feats["P2_int"], "float32")
    # D
    ib = feats["R1_int"]; ab = feats["R1_arr_ball"]; am = feats["R1_arr_mid"]
    ae = np.clip(1.0 - ab - am, 0.0, 1.0)
    d["f_pc_arr_ball"] = np.asarray(ab, "float32")
    d["f_pc_arr_mid"] = np.asarray(am, "float32")
    d["f_pc_sgb"] = np.asarray(feats["R1_sgb"], "float32")
    d["f_pc_sge"] = np.asarray(feats["R1_sge"], "float32")
    d["f_pc_agree"] = np.asarray(ib * ab + (1 - ib) * ae, "float32")
    d["f_pc_exec"] = np.asarray(ib * feats["R1_sgb"] + (1 - ib) * feats["R1_sge"], "float32")
    return d


ARM_FEATS = {"B": ["f_pc_succ", "f_pc_succ_d", "f_pc_logn", "f_pc_lvl"]}
ARM_FEATS["C"] = ARM_FEATS["B"] + ["f_pc_int", "f_pc_int_d"]
ARM_FEATS["D"] = ARM_FEATS["C"] + ["f_pc_arr_ball", "f_pc_arr_mid", "f_pc_sgb", "f_pc_sge",
                                   "f_pc_agree", "f_pc_exec"]
ARM_FEATS["T0"] = ARM_FEATS["D"] + IM_FEATS
ARM_FEATS["T"] = ARM_FEATS["T0"] + TM_FEATS
ALL_PC = ARM_FEATS[ARM] if ARM in ("T0", "T") else ARM_FEATS["D"]

if ARM in ("T0", "T"):
    # exp/91c 산출물 부착 (전체 train 행 순서, 키로 정렬 검증)
    z = np.load("lab/91c_feats_smoke.npz" if args.smoke else "lab/91c_feats.npz")
    key_now = (df.pitcher_id.to_numpy("int64") * 10_000_000
               + df.asof_pitcher_n.fillna(0).to_numpy("int64"))
    assert len(z["key"]) == len(df) and np.array_equal(z["key"], key_now), "91c 키 정렬 불일치"
    for c in ARM_FEATS[ARM]:
        if c in IM_FEATS + TM_FEATS:
            df[c] = z[c].astype("float32")
    log(f"91c 피처 부착: {len([c for c in ARM_FEATS[ARM] if c in IM_FEATS + TM_FEATS])}개  "
        f"(2024 커버리지 f_im_exec_exp {df.loc[df.season==2024,'f_im_exec_exp'].notna().mean()*100:.1f}%"
        + (f" / f_tm_agree {df.loc[df.season==2024,'f_tm_agree'].notna().mean()*100:.1f}%)" if ARM == "T" else ")"))
    del z, key_now


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
T_dep = level_tables(hist)               # 2024 검증용 (≤2023 냉동)
tick(f"테이블·PFB 준비 — 프로필 셀 T1 {len(T_dep['T1']):,} / T2 {len(T_dep['T2']):,} / 투수 {len(T_dep['T3']):,}")

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
        TS = None
    else:
        cpS = build_const(h, "pitcher_id", "asof_pitcher_n", s12.P_RATES)
        cbS = build_const(h, "batter_id", "asof_batter_n", s12.B_RATES)
        TS = level_tables(h)
    parts.append(attach_profile(s12.attach_cs(rows, cpS, cbS), TS))
    tick(f"  시즌 {S}: 프로필 커버리지 {parts[-1].f_pc_succ.notna().mean()*100:.1f}%")
tr = s12.add_features(pd.concat(parts).sort_index(), prior)
del parts, tr_rows, hist
gc.collect()
key = tr[["pitcher_id", "season", "_cg"]].merge(
    mix_tr, on=["pitcher_id", "season", "_cg"], how="left")
tr["f_mixcg_fb"] = key["mix_fb"].to_numpy("float32")
tr["f_mixcg_brk"] = key["mix_brk"].to_numpy("float32")
tr["f_pfb"] = pfb.predict_proba(tr[s12.PFB_IN].fillna(-999))[:, 1].astype("float32")
del key
va = s12.attach_pt(s12.add_features(attach_profile(s12.attach_cs(va_rows, cp, cb_), T_dep), prior),
                   mix_dep, pfb)
del va_rows, df
gc.collect()
tick(f"피처 부착 완료 — 검증 프로필 커버리지 {va.f_pc_succ.notna().mean()*100:.1f}%  "
     f"lvl 분포 {va.f_pc_lvl.value_counts(normalize=True).sort_index().round(3).to_dict()}")

BASE = [c for c in pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1)
        .columns.str.replace("﻿", "").str.strip() if c != "row_id"]
ENG18 = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
         "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
         "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
         "f_pb_diff", "f_miss_p", "f_miss_prev1"]
FEATS79 = BASE + ENG18 + s12.CS_FEATS + s12.PT_FEATS
FEATS = FEATS79 + ARM_FEATS[ARM]
r_va = float(y_va.mean())
DEN = r_va * (1 - r_va)

# =====================================================================
# 값싼 사전 게이트 — 챔피언 잔차에 새 피처가 선형으로 얼마나 붙는가 (학습 없음)
# =====================================================================
p_base = np.load("lab/89_cat5.npy").astype("float64")
assert len(p_base) == len(va), "lab/89_cat5.npy 정렬 불일치"
resid = y_va - p_base
log("\n--- 사전 게이트: corr(챔피언 잔차, 피처) / 결측 제외 ---")
for c in ["f_smooth_p", "f_cs_p_rate", "f_mixcg_fb"] + ALL_PC:
    v = va[c].to_numpy("float64")
    ok_ = ~np.isnan(v)
    cc = np.corrcoef(resid[ok_], v[ok_])[0, 1] if ok_.sum() > 1000 and np.nanstd(v[ok_]) > 0 else np.nan
    log(f"  {c:16s} corr {cc:+.4f}  cover {ok_.mean()*100:5.1f}%  mean {np.nanmean(v):.4f}")


def lin_gate(cols, name):
    X = []
    for c in cols:
        v = va[c].to_numpy("float64")
        mu = np.nanmean(v)
        X.append(np.where(np.isnan(v), mu, v))
        X.append(np.isnan(v).astype("float64"))
    X = np.column_stack(X + [np.ones(len(va))])
    beta, *_ = np.linalg.lstsq(X, resid, rcond=None)
    fit = X @ beta
    r2 = 1 - np.var(resid - fit) / np.var(resid)
    gain = 100000.0 * (np.var(resid) - np.var(resid - fit)) / DEN
    log(f"  선형 게이트 [{name:3s}] R²(잔차) {r2:.5f} → 점수 상당 {gain:+.1f}  (in-sample OLS, {X.shape[1]-1}개 열)")
    return gain


for a in (["B", "C", "D"] + ([ARM] if ARM in ("T0", "T") else [])):
    lin_gate(ARM_FEATS[a], a)
log("  (참고: 기존 피처의 잔차 R² 는 모델이 이미 흡수해 0 근처여야 정상)")
lin_gate(["f_smooth_p", "f_cs_p_rate"], "ref")

m_tr = tr["_cls"].to_numpy() >= 0
y_cls = tr["_cls"].to_numpy()[m_tr]
log(f"\n멀티클래스 학습 행: {m_tr.sum():,} / {len(tr):,}   피처 {len(FEATS)} (79 + {len(ARM_FEATS[ARM])})")
CAT5 = list(s12.CAT) + ["pitcher_team_id", "batter_team_id"]


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
    Xtr = build(tr.iloc[idx], CAT5)
    ptr = Pool(Xtr, tr["_cls"].to_numpy()[idx], cat_features=CAT5)
    Xva = build(va, CAT5)
    pva = Pool(Xva, cat_features=CAT5)
    m = CatBoostClassifier(**{**CB_PRM, "iterations": 20}, random_seed=42).fit(ptr)
    p = m.predict_proba(pva)[:, 0]
    log(f"smoke: 6만행·20it 학습/예측 정상 — P(성공) 평균 {p.mean():.4f} 점수 {raw_score(p, y_va):.1f}")
    imp = pd.Series(m.get_feature_importance(), index=FEATS).sort_values(ascending=False)
    log("smoke 중요도 상위 15: " + ", ".join(f"{k}={v:.1f}" for k, v in imp.head(15).items()))
    log("smoke 중 새 피처 중요도: " + ", ".join(f"{k}={imp[k]:.1f}" for k in ARM_FEATS[ARM]))
    sys.exit(0)

Xtr = build(tr[m_tr], CAT5)
del tr
gc.collect()
ptr = Pool(Xtr, y_cls, cat_features=CAT5)
del Xtr
gc.collect()
Xva = build(va, CAT5)
pva = Pool(Xva, cat_features=CAT5)
del Xva, va
gc.collect()
tick("Pool 구성 완료")

preds, solo = [], []
for sd in SEEDS:
    t0 = time.time()
    m = CatBoostClassifier(**CB_PRM, random_seed=sd).fit(ptr)
    prob = m.predict_proba(pva)
    np.save(f"lab/91b_{ARM}_probs_seed{sd}.npy", prob.astype("float32"))
    p = prob[:, 0]
    preds.append(p)
    solo.append(raw_score(p, y_va))
    tick(f"arm {ARM} seed={sd}  {solo[-1]:8.1f}  (paired vs cat5 {solo[-1]-BASE_SOLO[sd]:+.1f}, {time.time()-t0:.0f}s)")
    imp = pd.Series(m.get_feature_importance(), index=FEATS)
    log("   새 피처 중요도: " + ", ".join(f"{k}={imp[k]:.2f}" for k in ARM_FEATS[ARM])
        + f"  (순위 상위 {int((imp.rank(ascending=False)[ARM_FEATS[ARM]]).min())}위)")
    del m, prob
    gc.collect()

ens = np.mean(preds, axis=0)
np.save(f"lab/91b_{ARM}.npy", ens.astype("float32"))
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
log("")
log("=" * 80)
log(f"arm {ARM} ({len(ARM_FEATS[ARM])}피처)  2시드 {e2:7.1f}  시드평균 {ms:7.1f}  proj8 {p8:7.1f}  벌점 {pen:5.1f}")
log(f"1. raw paired gain (2시드)     {dK:+.1f}   시드별 {['%+.1f' % x for x in pairs]}  paired 평균 {np.mean(pairs):+.1f} 산포 {np.std(pairs):.2f}")
log(f"2. 중심 기여                    벌점 {BASE_PEN:.1f} → {pen:.1f}  ({BASE_PEN-pen:+.1f})")
log(f"3. 동일평균 후 shape gain       {sh_new - sh_base:+.1f}   (기준 {sh_base:.1f} → {sh_new:.1f})")
log(f"4. 연도별 gain                  2024 폴드 단일 ({dK:+.1f}) — 다연도 미측정")
log(f"5. 냉동 표 의존                 있음 — 시즌 as-of 냉동 [F-타겟] 계열 (2025 전이 위험 상)")
log(f"6. ★시드평균 이득 / proj8 이득  {ms - np.mean(list(BASE_SOLO.values())):+.1f} / {p8 - BASE_PROJ8:+.1f}")
log(f"   혼합 진단: d={dK:+.1f}  K={K:.1f}  w*={w_star:.2f}  혼합이득 {bg:+.1f}")
log("=" * 80)
with open("lab/91b_summary.txt", "a", encoding="utf-8") as f:
    f.write(f"arm={ARM} feats={len(ARM_FEATS[ARM])} solo={solo} ens={e2:.1f} seedmean={ms:.1f} "
            f"proj8={p8:.1f} pen={pen:.1f} paired={pairs} shape={sh_new-sh_base:+.1f} d={dK:+.1f} K={K:.1f}\n")
