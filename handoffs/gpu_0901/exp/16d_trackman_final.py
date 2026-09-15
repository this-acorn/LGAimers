"""
[16d] Trackman 지문 매칭 4차(최종) — 월×요일 등판 패턴 + 팀 하드블록 + 필링 배정

실행:  PYTHONIOENCODING=utf-8 py -3.12 -u exp/16d_trackman_final.py

3차(16c)까지의 확정:
  - 팀 대응표 완성 (순도 100%): 12=DOO 13=LG 14=KIW 15=LOT 16=KIA 17=HAN
    18=SAM 19=NC 20=KT 21=SSG(SK 통합)
  - 시즌 6칸 지문으로는 저볼륨 투수 분리 불가 → 커버리지 27~50%에서 정체

4차의 새 지문: 양쪽 테이블에 공통으로 존재하는 game_month × game_dayofweek.
  투수의 (시즌, 월, 요일)별 투구 점유율 벡터(~300차원)는 등판 스케줄 서명이라
  커버리지 80% 노이즈에도 개인 식별력이 훨씬 강하다.

배정: 라운드별로 헝가리안 → 확신 쌍(d·격차 임계)만 잠그고 양쪽에서 제거 →
      임계를 늦추며 반복 (앵커 필링). 남은 투수는 conf=False로 최선 추정만 기록.

검증(독립): 믹스 잔차, 커버리지 비율 분포, 시즌 활동 일치.
산출: lab/trackman_match_v3.csv
"""

import sys
import time
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, "exp")
from common import log

from scipy.optimize import linear_sum_assignment

T0 = time.time()
FIRST = ["DOO_BEA", "HAN_EAG", "KIA_TIG", "KIW_HER", "KT_WIZ", "LG_TWI",
         "LOT_GIA", "NC_DIN", "SAM_LIO", "SK_WYV", "SSG_LAN"]
TEAM_MAP = {12: "DOO_BEA", 13: "LG_TWI", 14: "KIW_HER", 15: "LOT_GIA",
            16: "KIA_TIG", 17: "HAN_EAG", 18: "SAM_LIO", 19: "NC_DIN",
            20: "KT_WIZ", 21: "SSG_LAN"}
HAND_MAP = {2: "Right", 1: "Left"}
SEASONS = list(range(2019, 2025))


def tick(msg):
    log(f"  [{time.time()-T0:6.0f}s] {msg}")


# =====================================================================
# 로드
# =====================================================================
tm = pd.read_csv("data/trackman_history.csv",
                 usecols=["season", "game_month", "game_dayofweek",
                          "pitcher_trackman_id", "pitcher_hand",
                          "pitcher_team", "pitch_type_group"],
                 encoding="utf-8-sig")
tm.columns = [c.replace("﻿", "").strip() for c in tm.columns]
tm1 = tm[tm["pitcher_team"].isin(FIRST)].copy()
tm1["franchise"] = tm1["pitcher_team"].replace({"SK_WYV": "SSG_LAN"})

trn = pd.read_csv("data/train.csv",
                  usecols=["pitcher_id", "season", "game_month",
                           "game_dayofweek", "pitcher_hand", "pitcher_team_id",
                           "asof_pitcher_pitchmix_n",
                           "asof_pitcher_fastball_rate",
                           "asof_pitcher_breaking_rate",
                           "asof_pitcher_offspeed_rate"],
                  encoding="utf-8-sig")
trn.columns = [c.replace("﻿", "").strip() for c in trn.columns]
tick(f"로드 — trackman 1군 {len(tm1):,} / train {len(trn):,}")

# =====================================================================
# 지문: (season, month, dow) 점유율 벡터
# =====================================================================
trn["key"] = (trn["season"] * 1000 + trn["game_month"] * 10
              + trn["game_dayofweek"]).astype("int32")
tm1["key"] = (tm1["season"] * 1000 + tm1["game_month"] * 10
              + tm1["game_dayofweek"]).astype("int32")

P_cnt = trn.groupby(["pitcher_id", "key"]).size().unstack(fill_value=0)
T_cnt = tm1.groupby(["pitcher_trackman_id", "key"]).size().unstack(fill_value=0)
ALL_KEYS = sorted(set(P_cnt.columns) | set(T_cnt.columns))
P_cnt = P_cnt.reindex(columns=ALL_KEYS, fill_value=0)
T_cnt = T_cnt.reindex(columns=ALL_KEYS, fill_value=0)
tick(f"지문 벡터 {len(ALL_KEYS)}차원 — train {len(P_cnt)}명 / trackman {len(T_cnt)}명")

P_sh = P_cnt.div(P_cnt.sum(axis=1).clip(lower=1), axis=0).to_numpy("float64")
T_sh = T_cnt.div(T_cnt.sum(axis=1).clip(lower=1), axis=0).to_numpy("float64")

# 믹스 (시즌말 누적 3비율) 프로파일
trn_s = trn.sort_values(["pitcher_id", "season", "asof_pitcher_pitchmix_n"],
                        na_position="first")
last = (trn_s.groupby(["pitcher_id", "season"]).tail(1)
        [["pitcher_id", "season", "asof_pitcher_fastball_rate",
          "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]])
grp = tm1["pitch_type_group"].astype(str).str.lower()
tmix = (pd.DataFrame({"tid": tm1["pitcher_trackman_id"].to_numpy(),
                      "season": tm1["season"].to_numpy(),
                      "fb": (grp == "fastball").astype("int32").to_numpy(),
                      "br": (grp == "breaking").astype("int32").to_numpy(),
                      "os": (grp == "offspeed").astype("int32").to_numpy()})
        .groupby(["tid", "season"]).agg(n=("season", "size"), fb=("fb", "sum"),
                                        br=("br", "sum"), os=("os", "sum"))
        .reset_index().sort_values(["tid", "season"]))
gt = tmix.groupby("tid")
for c in ["n", "fb", "br", "os"]:
    tmix["cum_" + c] = gt[c].cumsum()
for c in ["fb", "br", "os"]:
    tmix["r_" + c] = tmix["cum_" + c] / tmix["cum_n"].clip(lower=1)


def profile(df_, idc, valc, index):
    return (df_.pivot_table(index=idc, columns="season", values=valc,
                            aggfunc="first")
               .reindex(columns=SEASONS).reindex(index))


MIX_P, MIX_T = [], []
for tr_c, tm_c in [("asof_pitcher_fastball_rate", "r_fb"),
                   ("asof_pitcher_breaking_rate", "r_br"),
                   ("asof_pitcher_offspeed_rate", "r_os")]:
    MIX_P.append(profile(last, "pitcher_id", tr_c, P_cnt.index).to_numpy("float64"))
    MIX_T.append(profile(tmix, "tid", tm_c, T_cnt.index).to_numpy("float64"))

# 팀 궤적 (하드블록용)
tr_team = (trn.groupby(["pitcher_id", "season"])["pitcher_team_id"]
              .agg(lambda s: s.mode().iloc[0]).reset_index()
              .rename(columns={"pitcher_team_id": "team"}))
tr_team["fr"] = tr_team["team"].map(TEAM_MAP)
tm_team = (tm1.groupby(["pitcher_trackman_id", "season"])["franchise"]
              .agg(lambda s: s.mode().iloc[0]).reset_index()
              .rename(columns={"pitcher_trackman_id": "tid",
                               "franchise": "fr"}))
P_team = profile(tr_team, "pitcher_id", "fr", P_cnt.index).to_numpy("object")
T_team = profile(tm_team, "tid", "fr", T_cnt.index).to_numpy("object")

tr_hand = trn.groupby("pitcher_id")["pitcher_hand"].agg(lambda s: s.mode().iloc[0])
tm_hand = (tm1.groupby("pitcher_trackman_id")["pitcher_hand"]
              .agg(lambda s: s.mode().iloc[0]))
vol = trn.groupby("pitcher_id").size()
tick("프로파일 완료")


# =====================================================================
# 거리 행렬 (좌우 블록별, 청크)
# =====================================================================
def build_block(pmask, tmask):
    """pmask/tmask: P_cnt.index / T_cnt.index 상의 불리언"""
    pi = np.where(pmask)[0]
    ti = np.where(tmask)[0]
    A, B = P_sh[pi], T_sh[ti]
    D = np.zeros((len(pi), len(ti)))
    for s0 in range(0, len(pi), 64):
        s1 = min(s0 + 64, len(pi))
        D[s0:s1] = 0.5 * np.abs(A[s0:s1, None, :] - B[None, :, :]).sum(axis=2)
    D *= 2.0                                   # share 가중치
    # 믹스
    mixd = np.zeros_like(D)
    for Pm, Tm in zip(MIX_P, MIX_T):
        a, b = Pm[pi][:, None, :], Tm[ti][None, :, :]
        both = ~np.isnan(a) & ~np.isnan(b)
        mixd += np.nan_to_num(
            np.nanmean(np.where(both, np.abs(a - b), np.nan), axis=1 + 1),
            nan=0.4)
    D += 1.0 * mixd
    # 팀 하드블록: 공통으로 알려진 시즌 중 2개 이상 어긋나면 배제 (트레이드 1회 허용)
    Pt, Tt = P_team[pi], T_team[ti]
    knP = np.array([[x is not None and not (isinstance(x, float) and np.isnan(x))
                     for x in row] for row in Pt])
    knT = np.array([[x is not None and not (isinstance(x, float) and np.isnan(x))
                     for x in row] for row in Tt])
    valid = knP[:, None, :] & knT[None, :, :]
    mism = valid & (Pt[:, None, :] != Tt[None, :, :])
    D += np.where(mism.sum(2) >= 2, 1e6, 0.0) + 0.3 * mism.sum(2)
    return D, pi, ti


# =====================================================================
# 필링 배정
# =====================================================================
log("\n" + "=" * 88)
log("필링 배정")
log("=" * 88)
ROUNDS = [(0.25, 0.10), (0.40, 0.08), (0.55, 0.05)]
locked = []          # (p_global_idx, t_global_idx, d, margin, round)
p_used = np.zeros(len(P_cnt), dtype=bool)
t_used = np.zeros(len(T_cnt), dtype=bool)
p_hand_arr = tr_hand.reindex(P_cnt.index).to_numpy()
t_hand_arr = tm_hand.reindex(T_cnt.index).to_numpy()

for rnd, (dth, mth) in enumerate(ROUNDS, 1):
    n_new = 0
    for h_tr, h_tm in HAND_MAP.items():
        pmask = (p_hand_arr == h_tr) & ~p_used
        tmask = (t_hand_arr == h_tm) & ~t_used
        if pmask.sum() == 0 or tmask.sum() == 0:
            continue
        D, pi, ti = build_block(pmask, tmask)
        Df = np.where(np.isfinite(D), D, 1e6)
        ri, ci = linear_sum_assignment(Df)
        # 격차: 배정 열을 제외한 그 행의 최솟값 − 배정값
        for r_, c_ in zip(ri, ci):
            d_ = D[r_, c_]
            if d_ >= 1e5:
                continue
            row = D[r_].copy()
            row[c_] = np.inf
            mg = np.min(row) - d_
            if d_ < dth and mg > mth:
                locked.append((pi[r_], ti[c_], d_, mg, rnd))
                p_used[pi[r_]] = True
                t_used[ti[c_]] = True
                n_new += 1
    cum_cov = vol.reindex(P_cnt.index[[l[0] for l in locked]]).sum() / vol.sum()
    log(f"  라운드 {rnd} (d<{dth} 격차>{mth}): 신규 {n_new}명, "
        f"누적 {len(locked)}명, 누적 커버리지 {cum_cov*100:.1f}%")

# =====================================================================
# 검증
# =====================================================================
log("\n" + "=" * 88)
log("검증 (잠긴 매칭 기준)")
log("=" * 88)
lk = pd.DataFrame(locked, columns=["p_i", "t_i", "d", "margin", "round"])
lk["pid"] = P_cnt.index.to_numpy()[lk["p_i"]]
lk["tid"] = T_cnt.index.to_numpy()[lk["t_i"]]

# 믹스 잔차 (독립)
snapm = last.merge(lk[["pid", "tid"]], left_on="pitcher_id", right_on="pid")
snapm = snapm.merge(tmix[["tid", "season", "r_fb", "r_br", "r_os"]],
                    on=["tid", "season"], how="inner")
mixerr = (np.abs(snapm["r_fb"] - snapm["asof_pitcher_fastball_rate"])
          + np.abs(snapm["r_br"] - snapm["asof_pitcher_breaking_rate"])
          + np.abs(snapm["r_os"] - snapm["asof_pitcher_offspeed_rate"]))
log(f"  믹스 잔차: 중앙값 {mixerr.median():.4f}  p90 {mixerr.quantile(0.9):.4f}")

# 커버리지 비율 분포
tm_tot = tm1.groupby("pitcher_trackman_id").size()
ratio = (tm_tot.reindex(lk["tid"]).to_numpy(dtype="float64")
         / vol.reindex(lk["pid"]).to_numpy(dtype="float64"))
log(f"  투수별 trackman/train 비율: p10/50/90 = "
    f"{np.nanpercentile(ratio, 10):.2f}/{np.nanpercentile(ratio, 50):.2f}/"
    f"{np.nanpercentile(ratio, 90):.2f}")

# 시즌 활동 일치 (자카드)
P_act = (P_cnt.to_numpy() > 0)
act_seasons_p = {i: set(np.array(ALL_KEYS)[P_cnt.to_numpy()[i] > 0] // 1000)
                 for i in lk["p_i"]}
act_seasons_t = {i: set(np.array(ALL_KEYS)[T_cnt.to_numpy()[i] > 0] // 1000)
                 for i in lk["t_i"]}
jac = [len(act_seasons_p[p] & act_seasons_t[t])
       / max(len(act_seasons_p[p] | act_seasons_t[t]), 1)
       for p, t in zip(lk["p_i"], lk["t_i"])]
log(f"  시즌 활동 자카드: 중앙값 {np.median(jac):.3f}  최소 {np.min(jac):.3f}")

final_cov = vol.reindex(lk["pid"]).sum() / vol.sum()
log(f"\n  ★ 최종: {len(lk)}명 잠금 / train 투수 {len(P_cnt)}명  "
    f"투구량 커버리지 {final_cov*100:.1f}%")

out = lk[["pid", "tid", "d", "margin", "round"]].copy()
path = os.path.join("lab", "trackman_match_v3.csv")
out.to_csv(path, index=False)
log(f"  저장: {path}")

log("\n" + "=" * 88)
log("판정")
log("=" * 88)
if final_cov >= 0.6 and mixerr.median() < 0.03:
    log("  [GO] exp/17로 진행 — 매칭 투수의 릴리스 일관성/구속/무브먼트 as-of 피처 실험")
else:
    log("  [부분 GO/보류] 커버리지 미달 — 잠긴 부분집합만으로 피처를 만들되")
    log("  미매칭 투수 NaN 처리로 exp/17을 강등 실행할지 판단할 것")
log("=" * 88)
