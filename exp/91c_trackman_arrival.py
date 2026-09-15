# -*- coding: utf-8 -*-
"""
[91c] Trackman 도착 영역 모델 + 의도 모델 + 투수×상황 실행 프로필 (사용자 설계 2~8단계 구현)

설계 (사용자 08-28 제안, exp/91a·91b 위에서):
  2단계 의도(IM): P(볼 | 성공, 투구 전 상황) — 성공 투구(영역 ∈{볼, 존}, 가운데 0%)로 학습.
        투수-경기 세그먼트 2-fold OOF → 모든 ≤2023 행에 p_I(볼). 2024 행은 두 폴드 모델 평균.
  3단계 도착(AM): P(영역 | 실제 구종·Trackman 물리) 3클래스 — 행 조인(exp/57 키, 1:1) 매칭
        ≤2023 행(≈54%)에서 세그먼트 2-fold OOF. 입력은 물리값+구종+양손만 (상황·투수ID 제외).
        + 직접 성공모델(CM): P(성공 | 물리) 2-fold OOF (사용자 표의 tm_control_prob).
  4~5단계 일치도: agree = p_I·pA_볼 + (1−p_I)·pA_존  (가운데는 의도 0 → 기여 0)
  6~7단계 프로필: 투수×볼×스트×타자손 → 투수×카운트 → 투수전체+카운트오프셋 → 리그, λ=50,
        시즌 as-of 냉동 (S 행 ← 시즌<S). IM 실행률은 라벨 행 100%, TM 프로필은 매칭 행만.
  8단계 행 피처: f_im_ball(행 자신의 p_I) · 목표별 실행률 · 기대실행률 = p_I·exec_볼 + (1−p_I)·exec_존
        TM: 평균 일치도·직접성공확률·도착성향·표본·일치도 sd·구종군별 일치도(asof fb_rate 가중)
  최근 시즌 가중 평균은 넣지 않는다 (HANDOFF §1.13: 레짐 베팅 2/2 실패).

출력: lab/91c_feats.npz  — 전체 train 행 순서, key=pitcher_id*1e7+asof_n (행 유일) + 13피처
      exp/91b --arm T0 (D′+IM 4) / --arm T (D′+IM 4+TM 9) 가 이 파일을 읽는다.
실행: python -u exp/91c_trackman_arrival.py [--smoke]   (~25분 / smoke ~6분)
"""

import argparse
import gc
import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import log, tick, raw_score, load_train

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
NTHREAD = 4 if args.smoke else 14
IT = 10 if args.smoke else 300
LAM = 50.0
MIN_SD_N = 30
OUT = "lab/91c_feats_smoke.npz" if args.smoke else "lab/91c_feats.npz"
TEAM_MAP = {12: "DOO_BEA", 13: "LG_TWI", 14: "KIW_HER", 15: "LOT_GIA",
            16: "KIA_TIG", 17: "HAN_EAG", 18: "SAM_LIO", 19: "NC_DIN",
            20: "KT_WIZ", 21: "SSG_LAN"}
KEY = ["season", "game_month", "game_dayofweek", "inning", "top_bottom",
       "balls_before", "strikes_before", "outs_before", "ph", "bh", "pt", "bt"]
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension",
        "rel_height", "rel_side", "zone_speed"]
TM_CAT = ["tagged_pitch_type", "auto_pitch_type", "pitch_type_group"]
IM_IN = ["balls_before", "strikes_before", "outs_before", "inning", "num_runners_on",
         "runner_on_1b", "runner_on_2b", "runner_on_3b", "pitcher_hand", "batter_hand",
         "score_diff_pitcher_team", "li", "asof_pitcher_n", "asof_pitcher_success_rate",
         "asof_pitcher_ball_rate", "asof_pitcher_strike_rate", "asof_pitcher_middle_rate",
         "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
         "asof_pitcher_prev1_game_success_rate", "asof_batter_n",
         "asof_batter_success_rate", "asof_batter_middle_rate"]
CB_MC = dict(iterations=IT, depth=6, learning_rate=0.1, l2_leaf_reg=10.0, verbose=False,
             thread_count=NTHREAD, allow_writing_files=False, random_seed=42)

log(f"=== exp/91c smoke={args.smoke} thread={NTHREAD} it={IT} ===")
log("train 로딩...")
df = load_train()
N = len(df)
key_row = (df.pitcher_id.to_numpy("int64") * 10_000_000
           + df.asof_pitcher_n.fillna(0).to_numpy("int64"))
assert len(np.unique(key_row)) == N, "행 키 비유일"

# ---- 라벨 복원 + 3영역 (exp/91b 와 동일) ----
dd = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index()
pid = dd.pitcher_id.to_numpy()
n = dd.asof_pitcher_n.fillna(0).to_numpy("float64")
nxt = (pid[1:] == pid[:-1]) & (np.abs(n[1:] - n[:-1] - 1) < 1e-6)
for k, col in [("lab_mid", "asof_pitcher_middle_rate"), ("lab_rev", "asof_pitcher_reverse_rate"),
               ("lab_ball", "asof_pitcher_ball_rate")]:
    S = dd[col].fillna(0).to_numpy("float64") * n
    lab = np.full(len(dd), np.nan)
    lab[:-1] = np.where(nxt, np.round(S[1:] - S[:-1]), np.nan)
    dd[k] = np.where((lab == 0) | (lab == 1), lab, np.nan)
# 투수-경기 세그먼트 (exp/57 패턴)
daykey = (dd.season.to_numpy() * 10000 + dd.game_month.to_numpy() * 100
          + dd.game_dayofweek.to_numpy()).astype("int64")
p1 = np.nan_to_num(dd.asof_pitcher_prev1_game_success_rate.to_numpy("float64"), nan=-1.0)
newg = (np.r_[True, pid[1:] != pid[:-1]] | (daykey != np.r_[0, daykey[:-1]])
        | (np.abs(p1 - np.r_[0.0, p1[:-1]]) > 1e-12))
dd["_seg"] = np.cumsum(newg)
rec = dd.set_index("index").sort_index()
for k in ["lab_mid", "lab_rev", "lab_ball", "_seg"]:
    df[k] = rec[k]
del dd, rec, pid, n, nxt, daykey, p1, newg
gc.collect()
y = df.control_success.to_numpy("float64")
labeled = df.lab_mid.notna().to_numpy() & df.lab_rev.notna().to_numpy() & df.lab_ball.notna().to_numpy()
m1 = df.lab_mid.to_numpy() == 1
b1 = df.lab_ball.to_numpy() == 1
reg = np.full(N, -1, dtype="int8")
reg[labeled & m1] = 1
reg[labeled & ~m1 & b1] = 0
reg[labeled & ~m1 & ~b1] = 2
df["_reg"] = reg
fold = (pd.factorize(df._seg.to_numpy())[0] % 2).astype("int8")
season = df.season.to_numpy()
tr_mask = (season <= 2023) & labeled
log(f"라벨 행 {labeled.mean()*100:.2f}%  영역 0/1/2 = "
    + "/".join(f"{np.mean(reg[labeled]==r)*100:.1f}%" for r in (0, 1, 2))
    + f"  세그먼트 {df._seg.nunique():,}  fold0 비율 {np.mean(fold==0):.3f}")

# ---- 행 단위 trackman 조인 (exp/57 키, 1:1 만) ----
tm = pd.read_csv("data/trackman_history.csv", encoding="utf-8-sig",
                 usecols=["season", "game_month", "game_dayofweek", "inning", "top_bottom",
                          "balls_before", "strikes_before", "outs_before", "pitcher_hand",
                          "batter_hand", "pitcher_team", "batter_team"] + TM_CAT + PHYS)
tm.columns = [c.replace("﻿", "").strip() for c in tm.columns]
hand = {2: "Right", 1: "Left"}
jk = df[["season", "game_month", "game_dayofweek", "inning", "top_bottom", "balls_before",
         "strikes_before", "outs_before", "pitcher_hand", "batter_hand",
         "pitcher_team_id", "batter_team_id"]].copy()
jk["ph"] = jk.pitcher_hand.map(hand); jk["bh"] = jk.batter_hand.map(hand)
jk["pt"] = jk.pitcher_team_id.map(TEAM_MAP); jk["bt"] = jk.batter_team_id.map(TEAM_MAP)
jk["top_bottom"] = jk.top_bottom.astype(str).str[0]
tm["pt"] = tm.pitcher_team.replace({"SK_WYV": "SSG_LAN"})
tm["bt"] = tm.batter_team.replace({"SK_WYV": "SSG_LAN"})
tm["ph"] = tm.pitcher_hand.astype(str); tm["bh"] = tm.batter_hand.astype(str)
tm["top_bottom"] = tm.top_bottom.astype(str).str[0]
for c in ["season", "game_month", "game_dayofweek", "inning", "balls_before", "strikes_before",
          "outs_before"]:
    jk[c] = pd.to_numeric(jk[c], errors="coerce").astype("Int64")
    tm[c] = pd.to_numeric(tm[c], errors="coerce").astype("Int64")
jk["_row"] = np.arange(N)
trk = jk.dropna(subset=["pt", "bt", "ph", "bh"])
tmk = tm[tm.pt.isin(TEAM_MAP.values()) & tm.bt.isin(TEAM_MAP.values())]
c1 = trk.groupby(KEY, observed=True).size()
c2 = tmk.groupby(KEY, observed=True).size()
k11 = c1[c1 == 1].index.intersection(c2[c2 == 1].index)
tm11 = tmk.set_index(KEY).loc[k11, PHYS + TM_CAT].reset_index()
mg = trk[KEY + ["_row"]].merge(tm11, on=KEY, how="inner", validate="m:1")
assert mg._row.is_unique, "행 조인 결과 중복"
phys = pd.DataFrame(index=np.arange(N), columns=PHYS + TM_CAT, dtype=object)
for c in PHYS:
    a = np.full(N, np.nan); a[mg._row.to_numpy()] = mg[c].to_numpy("float64"); phys[c] = a
for c in TM_CAT:
    a = np.full(N, "NA", dtype=object)
    a[mg._row.to_numpy()] = mg[c].fillna("NA").astype(str).to_numpy().astype(object)
    phys[c] = a
matched = ~np.isnan(phys["rel_speed"].to_numpy("float64"))
phys["decel"] = phys["rel_speed"].astype("float64") - phys["zone_speed"].astype("float64")
is_fb = np.array([v == "fastball" for v in phys["pitch_type_group"].to_numpy()], dtype=bool)
del tm, tmk, trk, tm11, mg, jk, c1, c2, k11
gc.collect()
tick(f"행 조인: 매칭 {matched.sum():,} / {N:,} ({matched.mean()*100:.1f}%)  "
     f"≤2023 라벨∩매칭 {(matched & tr_mask).sum():,}")


def fr_im(idx):
    return df.iloc[idx][IM_IN].fillna(-999)


def fr_am(idx):
    o = phys.iloc[idx][PHYS + ["decel"]].astype("float64").copy()
    o["pitcher_hand"] = df.pitcher_hand.to_numpy()[idx]
    o["batter_hand"] = df.batter_hand.to_numpy()[idx]
    for c in TM_CAT:
        o[c] = np.array([str(v) if v is not None and v == v else "NA"
                         for v in phys[c].to_numpy()[idx]], dtype=object)
    return o


# =====================================================================
# 2단계 IM: P(볼 | 성공, 상황) — 세그먼트 2-fold OOF
# =====================================================================
p_im = np.full(N, np.nan)
im_models = []
for f in (0, 1):
    trn = np.where(tr_mask & (fold != f) & (reg != 1) & (y == 1))[0]
    hld = np.where((season <= 2023) & (fold == f))[0]
    t0 = time.time()
    m = CatBoostClassifier(**CB_MC).fit(fr_im(trn), (reg[trn] == 0).astype(int))
    p_im[hld] = m.predict_proba(fr_im(hld))[:, 1]
    im_models.append(m)
    tick(f"IM fold{f}: 학습 {len(trn):,} 성공행 → OOF {len(hld):,} ({time.time()-t0:.0f}s)")
va_idx = np.where(season == 2024)[0]
p_im[va_idx] = np.mean([m.predict_proba(fr_im(va_idx))[:, 1] for m in im_models], axis=0)
q = tr_mask & (reg != 1) & (y == 1)
bs_im = np.mean((p_im[q] - (reg[q] == 0)) ** 2)
pr = np.mean(reg[q] == 0)
log(f"  IM OOF(성공행): Brier {bs_im:.4f} vs 사전 {pr*(1-pr):.4f}  → 설명비율 {1-bs_im/(pr*(1-pr)):.3f}"
    f"   p_I 평균 {np.nanmean(p_im[season<=2023]):.3f} / 2024 {p_im[va_idx].mean():.3f}")
del im_models
gc.collect()

# =====================================================================
# 3단계 AM: P(영역 | 물리·구종) 3클래스 + CM: P(성공 | 물리·구종) — 매칭행 세그먼트 2-fold OOF
# =====================================================================
pA = np.full((N, 3), np.nan)
p_cm = np.full(N, np.nan)
mm = tr_mask & matched
for f in (0, 1):
    trn = np.where(mm & (fold != f))[0]
    hld = np.where(mm & (fold == f))[0]
    t0 = time.time()
    Xtr, Xh = fr_am(trn), fr_am(hld)
    am = CatBoostClassifier(**{**CB_MC, "loss_function": "MultiClass"}).fit(
        Pool(Xtr, reg[trn].astype(int), cat_features=TM_CAT))
    pA[hld] = am.predict_proba(Pool(Xh, cat_features=TM_CAT))
    cm = CatBoostClassifier(**CB_MC).fit(Pool(Xtr, y[trn].astype(int), cat_features=TM_CAT))
    p_cm[hld] = cm.predict_proba(Pool(Xh, cat_features=TM_CAT))[:, 1]
    tick(f"AM/CM fold{f}: 학습 {len(trn):,} → OOF {len(hld):,} ({time.time()-t0:.0f}s)")
    del am, cm, Xtr, Xh
    gc.collect()
# 진단
onehot = np.eye(3)[reg[mm]]
ll = -np.mean(np.log(np.clip(pA[mm][np.arange(mm.sum()), reg[mm]], 1e-9, 1)))
prior3 = onehot.mean(0)
ll0 = -np.sum(prior3 * np.log(prior3))
log(f"  AM OOF: multiclass logloss {ll:.4f} vs 사전 {ll0:.4f}  설명비율 {1-ll/ll0:.3f}"
    f"   pA 평균 볼/미들/존 = {pA[mm].mean(0).round(3).tolist()}")
log(f"  CM OOF(매칭행, 물리만): 점수 {raw_score(p_cm[mm], y[mm]):.1f}   (exp/57 teacher=물리+상황 1632)")
agree = p_im * pA[:, 0] + (1 - p_im) * pA[:, 2]
log(f"  일치도: 평균 {np.nanmean(agree[mm]):.3f}  corr(agree,y) {np.corrcoef(agree[mm], y[mm])[0,1]:+.4f}"
    f"  corr(p_cm,y) {np.corrcoef(p_cm[mm], y[mm])[0,1]:+.4f}"
    f"  성공/실패 평균 일치도 {np.nanmean(agree[mm & (y==1)]):.3f}/{np.nanmean(agree[mm & (y==0)]):.3f}")

# =====================================================================
# 6~8단계 프로필 (시즌 as-of) → 행 피처
# =====================================================================
base = pd.DataFrame({
    "pitcher_id": df.pitcher_id.to_numpy("int64"),
    "balls_before": df.balls_before.to_numpy("int64"),
    "strikes_before": df.strikes_before.to_numpy("int64"),
    "batter_hand": df.batter_hand.fillna(0).to_numpy("int64"),
    "pitcher_hand": df.pitcher_hand.fillna(0).to_numpy("int64"),
    "season": season})
V = pd.DataFrame({
    "n_lab": tr_mask.astype("float64"),
    "im_yb": np.where(tr_mask, y * p_im, 0.0), "im_b": np.where(tr_mask, p_im, 0.0),
    "im_ye": np.where(tr_mask, y * (1 - p_im), 0.0), "im_e": np.where(tr_mask, 1 - p_im, 0.0),
    "n_tm": mm.astype("float64"),
    "tm_ag": np.where(mm, agree, 0.0), "tm_ag2": np.where(mm, agree ** 2, 0.0),
    "tm_cm": np.where(mm, p_cm, 0.0),
    "tm_ab": np.where(mm, pA[:, 0], 0.0), "tm_am": np.where(mm, pA[:, 1], 0.0),
    "n_fb": (mm & is_fb).astype("float64"), "ag_fb": np.where(mm & is_fb, agree, 0.0),
    "n_nfb": (mm & ~is_fb).astype("float64"), "ag_nfb": np.where(mm & ~is_fb, agree, 0.0)})
V = V.fillna(0.0)
VC = list(V.columns)
K1 = ["pitcher_id", "balls_before", "strikes_before", "batter_hand"]
K2 = ["pitcher_id", "balls_before", "strikes_before"]
K3 = ["pitcher_id"]
K4 = ["pitcher_hand", "balls_before", "strikes_before"]
KPH = ["pitcher_hand"]
KBS = ["balls_before", "strikes_before"]
LEVELS = [("T1", K1), ("T2", K2), ("T3", K3), ("T4", K4), ("TPH", KPH), ("TBS", KBS)]
QTY = {"exec_b": ("im_yb", "im_b"), "exec_e": ("im_ye", "im_e"),
       "tm_agree": ("tm_ag", "n_tm"), "tm_ctrl": ("tm_cm", "n_tm"),
       "tm_arr_ball": ("tm_ab", "n_tm"), "tm_arr_mid": ("tm_am", "n_tm")}
FEAT_NAMES = ["f_im_ball", "f_im_exec_ball", "f_im_exec_edge", "f_im_exec_exp",
              "f_tm_agree", "f_tm_ctrl", "f_tm_arr_ball", "f_tm_arr_mid", "f_tm_logn",
              "f_tm_agree_sd", "f_tm_agree_fb", "f_tm_agree_nfb", "f_tm_agree_pt"]
F = {k: np.full(N, np.nan, dtype="float32") for k in FEAT_NAMES}
F["f_im_ball"][:] = p_im.astype("float32")


def _shr(num, den, prior):
    return (num + LAM * prior) / (den + LAM)


for S in range(2020, 2025):
    hmask = season < S
    rows = np.where(season == S)[0]
    if len(rows) == 0:
        continue
    hb = base[hmask]; hv = V[hmask]
    T = {name: pd.concat([hb[key], hv], axis=1).groupby(key, observed=True)[VC].sum().reset_index()
         for name, key in LEVELS}
    G = hv[VC].sum()
    keyf = base.iloc[rows]
    sums = {}
    for name, key in LEVELS:
        m_ = keyf[key].merge(T[name], on=key, how="left")
        sums[name] = {c: m_[c].fillna(0).to_numpy("float64") for c in VC}
    n1, n3 = sums["T1"]["n_tm"], sums["T3"]["n_tm"]
    nl3 = sums["T3"]["n_lab"]
    R = {}
    for q, (nu, de) in QTY.items():
        g = float(G[nu]) / max(float(G[de]), 1.0)
        Rbs = _shr(sums["TBS"][nu], sums["TBS"][de], g)
        Rph = _shr(sums["TPH"][nu], sums["TPH"][de], g)
        R4 = _shr(sums["T4"][nu], sums["T4"][de], Rbs)
        R3 = _shr(sums["T3"][nu], sums["T3"][de], g)
        P2 = np.clip(R3 + (R4 - Rph), 0.0, 1.0)
        R2 = _shr(sums["T2"][nu], sums["T2"][de], P2)
        R[q] = _shr(sums["T1"][nu], sums["T1"][de], R2)
    has_lab = nl3 > 0
    has_tm = n3 > 0
    pi = p_im[rows]
    F["f_im_exec_ball"][rows] = np.where(has_lab, R["exec_b"], np.nan)
    F["f_im_exec_edge"][rows] = np.where(has_lab, R["exec_e"], np.nan)
    F["f_im_exec_exp"][rows] = np.where(has_lab, pi * R["exec_b"] + (1 - pi) * R["exec_e"], np.nan)
    for q in ["tm_agree", "tm_ctrl", "tm_arr_ball", "tm_arr_mid"]:
        F[f"f_{q}"][rows] = np.where(has_tm, R[q], np.nan)
    F["f_tm_logn"][rows] = np.log1p(n1)
    s3 = sums["T3"]
    mean3 = s3["tm_ag"] / np.maximum(n3, 1)
    var3 = s3["tm_ag2"] / np.maximum(n3, 1) - mean3 ** 2
    F["f_tm_agree_sd"][rows] = np.where(n3 >= MIN_SD_N, np.sqrt(np.clip(var3, 0, None)), np.nan)
    g_fb = float(G["ag_fb"]) / max(float(G["n_fb"]), 1.0)
    g_nfb = float(G["ag_nfb"]) / max(float(G["n_nfb"]), 1.0)
    afb = _shr(s3["ag_fb"], s3["n_fb"], g_fb)
    anfb = _shr(s3["ag_nfb"], s3["n_nfb"], g_nfb)
    fbr = np.nan_to_num(df.asof_pitcher_fastball_rate.to_numpy("float64")[rows], nan=g_fb)
    F["f_tm_agree_fb"][rows] = np.where(has_tm, afb, np.nan)
    F["f_tm_agree_nfb"][rows] = np.where(has_tm, anfb, np.nan)
    F["f_tm_agree_pt"][rows] = np.where(has_tm, fbr * afb + (1 - fbr) * anfb, np.nan)
    tick(f"  시즌 {S}: 행 {len(rows):,}  IM 실행률 커버 {has_lab.mean()*100:.1f}%  "
         f"TM 프로필 커버 {has_tm.mean()*100:.1f}%  평균 일치도 {np.nanmean(F['f_tm_agree'][rows]):.3f}")

np.savez_compressed(OUT, key=key_row, names=np.array(FEAT_NAMES), **F)
log("\n2024 행 피처 요약:")
for k in FEAT_NAMES:
    v = F[k][va_idx]
    log(f"  {k:16s} cover {np.mean(~np.isnan(v))*100:5.1f}%  mean {np.nanmean(v):.4f}  sd {np.nanstd(v):.4f}")
log(f"저장: {OUT}  ({N:,}행 × {len(FEAT_NAMES)}피처)")
