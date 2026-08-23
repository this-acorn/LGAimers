"""
[16c] Trackman 지문 매칭 3차 — 점유율 형태 + 믹스 + 팀 궤적 부트스트랩 + 일대일 배정

실행:  PYTHONIOENCODING=utf-8 py -3.12 -u exp/16c_trackman_match.py

2차(16b)의 확정 사실:
  - pitchmix_n == asof_n (100%) → 카운트 '정수 일치' 가설 기각
  - 1군 필터 후 trackman은 train의 ~80%만 커버 (시즌별 0.79~0.83, 팀별 편차 큼:
    KIA 6.4만 vs 두산 12.8만 → 구장별 장비 차이) → 카운트 레벨 매칭은 구조적 한계
  - 그러나 확신 57명의 팀 대응표가 거의 완벽 일대일 (팀13=LG_TWI 등)
    + 믹스 오차 중앙값 0.017 → 매칭 신호 자체는 진짜

3차 거리 설계 (커버리지 차이에 강건하게):
  share : 투수의 시즌별 투구 '점유율' 벡터 L1/2 — 투수별 커버리지 비율이
          시즌 간 일정하면(같은 홈구장) 분자·분모에서 상쇄됨 ← 주력 신호
  mix   : 시즌말 누적 구종 믹스 3비율 L1 (2차에서 정밀함 검증됨)
  level : 시즌별 전역 커버리지 비율로 보정한 카운트 로그비 (보조)
  team  : 유도된 팀 대응표(부트스트랩)와 어긋나는 공통 시즌 비율
  hand  : 하드 블록 (train 2=Right, 1=Left)
  배정  : scipy 헝가리안으로 좌우 블록별 전역 일대일 최적화

검증: 팀 순도(부트스트랩 순환성 있음 — 참고용), 믹스 잔차(독립),
      배정거리 임계별 커버리지, 투수별 커버리지 비율 분포
산출: lab/trackman_match_v2.csv (pid, tid, 거리, 확신도)
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

try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

T0 = time.time()
SEASONS = list(range(2019, 2025))


def tick(msg):
    log(f"  [{time.time()-T0:6.0f}s] {msg}")


# =====================================================================
# 로드 & 정리
# =====================================================================
tm = pd.read_csv("data/trackman_history.csv",
                 usecols=["season", "pitcher_trackman_id", "pitcher_hand",
                          "pitcher_team", "pitch_type_group"],
                 encoding="utf-8-sig")
tm.columns = [c.replace("﻿", "").strip() for c in tm.columns]
trn = pd.read_csv("data/train.csv",
                  usecols=["pitcher_id", "season", "pitcher_hand",
                           "pitcher_team_id", "asof_pitcher_pitchmix_n",
                           "asof_pitcher_fastball_rate",
                           "asof_pitcher_breaking_rate",
                           "asof_pitcher_offspeed_rate"],
                  encoding="utf-8-sig")
trn.columns = [c.replace("﻿", "").strip() for c in trn.columns]

FIRST = ["DOO_BEA", "HAN_EAG", "KIA_TIG", "KIW_HER", "KT_WIZ", "LG_TWI",
         "LOT_GIA", "NC_DIN", "SAM_LIO", "SK_WYV", "SSG_LAN"]
tm1 = tm[tm["pitcher_team"].isin(FIRST)].copy()
# SK 와이번스 → SSG 랜더스 리브랜딩 통합 (같은 구단)
tm1["franchise"] = tm1["pitcher_team"].replace({"SK_WYV": "SSG_LAN"})
tick(f"로드 — trackman 1군 {len(tm1):,} / train {len(trn):,}")

# 시즌별 전역 커버리지 비율
tm_yr = tm1.groupby("season").size()
tr_yr = trn.groupby("season").size()
R_S = {s: tm_yr.get(s, 0) / max(tr_yr.get(s, 1), 1) for s in SEASONS}
log(f"  시즌별 커버리지 비율: " + "  ".join(f"{s}:{R_S[s]:.3f}" for s in SEASONS))

# =====================================================================
# 스냅샷
# =====================================================================
snap = (trn.groupby(["pitcher_id", "season"])
           .agg(rows=("pitcher_id", "size")).reset_index())
trn_s = trn.sort_values(["pitcher_id", "season", "asof_pitcher_pitchmix_n"],
                        na_position="first")
last = (trn_s.groupby(["pitcher_id", "season"]).tail(1)
        [["pitcher_id", "season", "asof_pitcher_fastball_rate",
          "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]])
snap = snap.merge(last, on=["pitcher_id", "season"], how="left")
tr_hand = trn.groupby("pitcher_id")["pitcher_hand"].agg(lambda s: s.mode().iloc[0])
tr_team = (trn.groupby(["pitcher_id", "season"])["pitcher_team_id"]
              .agg(lambda s: s.mode().iloc[0]).reset_index()
              .rename(columns={"pitcher_team_id": "team"}))

grp = tm1["pitch_type_group"].astype(str).str.lower()
tm2 = pd.DataFrame({"tid": tm1["pitcher_trackman_id"].to_numpy(),
                    "season": tm1["season"].to_numpy(),
                    "fb": (grp == "fastball").astype("int32").to_numpy(),
                    "br": (grp == "breaking").astype("int32").to_numpy(),
                    "os": (grp == "offspeed").astype("int32").to_numpy()})
tsnap = (tm2.groupby(["tid", "season"])
            .agg(n=("season", "size"), fb=("fb", "sum"),
                 br=("br", "sum"), os=("os", "sum"))
            .reset_index().sort_values(["tid", "season"]))
gt = tsnap.groupby("tid")
for c in ["n", "fb", "br", "os"]:
    tsnap["cum_" + c] = gt[c].cumsum()
for c in ["fb", "br", "os"]:
    tsnap["r_" + c] = tsnap["cum_" + c] / tsnap["cum_n"].clip(lower=1)
tm_hand = (tm1.groupby("pitcher_trackman_id")["pitcher_hand"]
              .agg(lambda s: s.mode().iloc[0]))
tm_team = (tm1.groupby(["pitcher_trackman_id", "season"])["franchise"]
              .agg(lambda s: s.mode().iloc[0]).reset_index()
              .rename(columns={"pitcher_trackman_id": "tid", "franchise": "team"}))
HAND_MAP = {2: "Right", 1: "Left"}
tick("스냅샷 완료")


def profile(df_, idc, valc):
    return (df_.pivot_table(index=idc, columns="season", values=valc,
                            aggfunc="first").reindex(columns=SEASONS))


def team_matrix(df_, idc):
    return (df_.pivot_table(index=idc, columns="season", values="team",
                            aggfunc="first").reindex(columns=SEASONS))


def build_D(pids_blk, tids_blk, team_map):
    sn = snap[snap["pitcher_id"].isin(pids_blk)]
    ts = tsnap[tsnap["tid"].isin(tids_blk)]

    Pc = profile(sn, "pitcher_id", "rows")
    Tc = profile(ts, "tid", "n")
    pid_idx, tid_idx = Pc.index.to_numpy(), Tc.index.to_numpy()
    A = Pc.to_numpy(dtype="float64")
    B = Tc.to_numpy(dtype="float64")

    # share — 시즌별 점유율 (결측=0), L1/2
    A_sh = np.nan_to_num(A / np.nansum(A, axis=1, keepdims=True), nan=0.0)
    B_sh = np.nan_to_num(B / np.nansum(B, axis=1, keepdims=True), nan=0.0)
    share = 0.5 * np.abs(A_sh[:, None, :] - B_sh[None, :, :]).sum(axis=2)

    # level — 전역 커버리지 보정 카운트 로그비 (공통 시즌 평균)
    r_vec = np.array([R_S[s] for s in SEASONS])
    B_adj = B / r_vec[None, :]
    A3, B3 = A[:, None, :], B_adj[None, :, :]
    both = ~np.isnan(A3) & ~np.isnan(B3)
    rel = np.where(both, np.abs(np.log((A3 + 1.0) / (B3 + 1.0))), np.nan)
    level = np.nanmean(rel, axis=2)
    level = np.where(np.isnan(level), 5.0, level)   # 공통 시즌 없음 → 사실상 배제

    # mix — 시즌말 누적 믹스 L1 (공통 시즌 평균)
    mixd = np.zeros_like(share)
    for tr_c, tm_c in [("asof_pitcher_fastball_rate", "r_fb"),
                       ("asof_pitcher_breaking_rate", "r_br"),
                       ("asof_pitcher_offspeed_rate", "r_os")]:
        Am = profile(sn, "pitcher_id", tr_c).to_numpy(dtype="float64")[:, None, :]
        Bm = profile(ts, "tid", tm_c).to_numpy(dtype="float64")[None, :, :]
        mixd += np.nan_to_num(np.nanmean(np.where(both, np.abs(Am - Bm), np.nan),
                                         axis=2), nan=0.4)

    # team — 유도 대응표와 어긋나는 공통 시즌 비율
    teamd = np.zeros_like(share)
    if team_map:
        Ptm = team_matrix(tr_team[tr_team["pitcher_id"].isin(pids_blk)],
                          "pitcher_id").reindex(pid_idx)
        Ttm = team_matrix(tm_team[tm_team["tid"].isin(tids_blk)],
                          "tid").reindex(tid_idx)
        Pm = Ptm.to_numpy(dtype="object")
        Tm = Ttm.to_numpy(dtype="object")
        Pmap = np.empty_like(Pm)
        for i in range(Pm.shape[0]):
            for j in range(Pm.shape[1]):
                v = Pm[i, j]
                Pmap[i, j] = team_map.get(v, None) if pd.notna(v) else None
        knownP = np.array([[x is not None for x in row] for row in Pmap])
        knownT = Ttm.notna().to_numpy()
        valid = knownP[:, None, :] & knownT[None, :, :]        # (n_tr, n_tm, 6)
        mism = valid & (Pmap[:, None, :] != Tm[None, :, :])
        teamd = mism.sum(2) / np.maximum(valid.sum(2), 1)

    D = 2.0 * share + 1.5 * mixd + 0.5 * level + 1.0 * teamd
    return D, pid_idx, tid_idx


def round_match(team_map, label):
    rows = []
    for h_tr, h_tm in HAND_MAP.items():
        pids_blk = tr_hand[tr_hand == h_tr].index
        tids_blk = tm_hand[tm_hand == h_tm].index
        D, pid_idx, tid_idx = build_D(pids_blk, tids_blk, team_map)
        order = np.argsort(D, axis=1)
        r = np.arange(D.shape[0])
        b1, b2 = order[:, 0], order[:, 1]
        d1, d2 = D[r, b1], D[r, b2]
        if HAVE_SCIPY:
            ri, ci = linear_sum_assignment(np.where(np.isfinite(D), D, 1e6))
            assigned = dict(zip(ri, ci))
            a_col = np.array([assigned.get(i, b1[i]) for i in range(D.shape[0])])
        else:
            a_col = b1
        rows.append(pd.DataFrame({
            "pid": pid_idx, "tid": tid_idx[a_col],
            "d": D[r, a_col], "d1": d1, "margin": d2 - d1,
            "moved": a_col != b1}))
    pairs = pd.concat(rows, ignore_index=True)
    log(f"  [{label}] 배정거리 중앙값 {pairs['d'].median():.4f}  "
        f"격차 중앙값 {pairs['margin'].median():.4f}  "
        f"헝가리안 이동 {int(pairs['moved'].sum())}명")
    return pairs


def induce_team_map(pairs, dth, mth):
    c = pairs[(pairs["d"] < dth) & (pairs["margin"] > mth)]
    tt = (c[["pid", "tid"]]
          .merge(tr_team, left_on="pid", right_on="pitcher_id")
          .merge(tm_team, on=["tid", "season"], how="inner",
                 suffixes=("_tr", "_tm")))
    if not len(tt):
        return {}
    xt = pd.crosstab(tt["team_tr"], tt["team_tm"])
    return {int(k): xt.loc[k].idxmax() for k in xt.index}


# =====================================================================
# 부트스트랩 3라운드
# =====================================================================
log("\n" + "=" * 88)
log("부트스트랩 매칭")
log("=" * 88)
pairs = round_match({}, "1라운드 (팀 제약 없음)")
tmap = induce_team_map(pairs, dth=0.15, mth=0.15)
log(f"  → 유도된 팀 대응 {len(tmap)}개: {tmap}")
pairs = round_match(tmap, "2라운드 (팀 제약)")
tmap = induce_team_map(pairs, dth=0.20, mth=0.15)
log(f"  → 팀 대응 갱신 {len(tmap)}개")
pairs = round_match(tmap, "3라운드 (최종)")

# =====================================================================
# 검증
# =====================================================================
log("\n" + "=" * 88)
log("검증")
log("=" * 88)
vol = snap.groupby("pitcher_id")["rows"].sum()

for dth, mth in [(0.10, 0.15), (0.15, 0.10), (0.20, 0.10), (0.30, 0.05)]:
    c = (pairs["d"] < dth) & (pairs["margin"] > mth)
    cov = vol.reindex(pairs.loc[c, "pid"]).sum() / vol.sum()
    sub = pairs[c]
    # 믹스 잔차 (독립 검증)
    mg = (snap.merge(sub[["pid", "tid"]], left_on="pitcher_id", right_on="pid")
              .merge(tsnap[["tid", "season", "r_fb", "r_br", "r_os"]],
                     on=["tid", "season"], how="inner"))
    mixerr = (np.abs(mg["r_fb"] - mg["asof_pitcher_fastball_rate"])
              + np.abs(mg["r_br"] - mg["asof_pitcher_breaking_rate"])
              + np.abs(mg["r_os"] - mg["asof_pitcher_offspeed_rate"])).median()
    # 팀 순도 (부트스트랩 순환성 있음 — 참고)
    tt = (sub[["pid", "tid"]]
          .merge(tr_team, left_on="pid", right_on="pitcher_id")
          .merge(tm_team, on=["tid", "season"], how="inner",
                 suffixes=("_tr", "_tm")))
    xt = pd.crosstab(tt["team_tr"], tt["team_tm"])
    pur = xt.max(axis=1).sum() / xt.to_numpy().sum() if xt.size else float("nan")
    log(f"  d<{dth:.2f} 격차>{mth:.2f}: {int(c.sum()):3d}명  "
        f"커버리지 {cov*100:5.1f}%  믹스잔차(중앙) {mixerr:.4f}  팀순도 {pur*100:5.1f}%")

# 매칭된 투수별 커버리지 비율 분포 (tm 총투구 / train 총투구)
CONF = (pairs["d"] < 0.20) & (pairs["margin"] > 0.10)
sub = pairs[CONF]
tm_tot = tsnap.groupby("tid")["n"].sum()
ratio = tm_tot.reindex(sub["tid"]).to_numpy() / vol.reindex(sub["pid"]).to_numpy()
log(f"\n  [확신 {int(CONF.sum())}명] 투수별 커버리지 비율: "
    f"p10/50/90 = {np.nanpercentile(ratio, 10):.2f}/"
    f"{np.nanpercentile(ratio, 50):.2f}/{np.nanpercentile(ratio, 90):.2f}")

out = pairs.copy()
out["conf"] = CONF
path = os.path.join("lab", "trackman_match_v2.csv")
out.to_csv(path, index=False)
log(f"  전체 배정 저장: {path} (conf 컬럼 = d<0.20 & 격차>0.10)")

log("\n" + "=" * 88)
log("판정 가이드")
log("=" * 88)
log("  GO  : 확신 커버리지 60%+ 그리고 믹스잔차 ~0.02 유지")
log("        → exp/17: 매칭 투수의 릴리스 일관성/구속 등 as-of 피처 실험")
log("  NO-GO: 커버리지가 그래도 낮으면 → 확신 부분집합만으로 피처를 만들되")
log("         미매칭 투수는 NaN (GBDT 네이티브) — 부분 커버리지 실험으로 강등")
log("=" * 88)
