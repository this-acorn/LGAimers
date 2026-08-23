"""
[16b] Trackman 지문 매칭 2차 정찰 — 1군 필터 + 좌우 제약 + 시즌별 카운트

실행:  PYTHONIOENCODING=utf-8 py -3.12 -u exp/16b_trackman_filter.py

1차 정찰(exp/16)의 발견:
  - trackman에는 2군(MIN_*)·상무/경찰(KBO_ARM/KBO_POL)·기타(ACE_*) 경기가 섞여 있다
    → train(1군 기록)과 투수별 카운트가 구조적으로 어긋남 (중앙값 10%)
  - 그럼에도 카운트만으로 잡힌 강한매칭 19명은 카운트 차 중앙값 6 → 신호는 실재
  - 좌우 코딩: train 2=Right(74.1%↔74.9%), 1=Left
  - pitchmix_n과 asof_n은 매칭 결과가 동일 → 같은 값인지 직접 확인 필요

이번에 하는 것:
  1. pitchmix_n == asof_n 직접 검사
  2. 팀 코드 26개 전수 출력 → 1군 코드만 필터
  3. 시즌별(비누적) 카운트 + 좌우 하드 제약 + 누적 믹스 비율 거리로 재매칭
  4. 검증: 팀 순도 / tid 충돌 / 정수 일치율 / 투구량 가중 커버리지
  5. 확신 매칭 쌍을 CSV로 저장 (다음 단계 피처 실험용)
"""

import sys
import time
import os
import numpy as np
import pandas as pd

sys.path.insert(0, "exp")
from common import log

T0 = time.time()
OUT = os.environ.get("SCOUT_OUT", "lab")
os.makedirs(OUT, exist_ok=True)


def tick(msg):
    log(f"  [{time.time()-T0:6.0f}s] {msg}")


# =====================================================================
# 로드
# =====================================================================
tm = pd.read_csv("data/trackman_history.csv",
                 usecols=["season", "pitcher_trackman_id", "pitcher_hand",
                          "pitcher_team", "pitch_type_group"],
                 encoding="utf-8-sig")
tm.columns = [c.replace("﻿", "").strip() for c in tm.columns]
trn = pd.read_csv("data/train.csv",
                  usecols=["pitcher_id", "season", "pitcher_hand",
                           "pitcher_team_id", "asof_pitcher_n",
                           "asof_pitcher_pitchmix_n",
                           "asof_pitcher_fastball_rate",
                           "asof_pitcher_breaking_rate",
                           "asof_pitcher_offspeed_rate"],
                  encoding="utf-8-sig")
trn.columns = [c.replace("﻿", "").strip() for c in trn.columns]
tick(f"로드 완료 — trackman {len(tm):,} / train {len(trn):,}")

# =====================================================================
# 1. pitchmix_n == asof_n ?
# =====================================================================
log("\n" + "=" * 88)
log("1. asof_pitcher_pitchmix_n vs asof_pitcher_n")
log("=" * 88)
a, b = trn["asof_pitcher_n"], trn["asof_pitcher_pitchmix_n"]
both = a.notna() & b.notna()
eq = (a[both] == b[both]).mean()
log(f"  둘 다 존재 {both.mean()*100:.1f}% · 그중 완전 일치 {eq*100:.2f}%")
if eq < 0.999:
    d = (b[both] - a[both])
    log(f"  차이 분포: 평균 {d.mean():+.1f}  중앙값 {d.median():+.1f}  "
        f"p5/p95 {d.quantile(0.05):+.0f}/{d.quantile(0.95):+.0f}")

# =====================================================================
# 2. 팀 코드 전수 + 1군 필터
# =====================================================================
log("\n" + "=" * 88)
log("2. 팀 코드 26개 전수")
log("=" * 88)
tc = tm.groupby("pitcher_team").size().sort_values(ascending=False)
for code, n in tc.items():
    log(f"  {str(code):12s} {n:9,d}")
FIRST = [c for c in tc.index
         if not (str(c).startswith("MIN_") or str(c).startswith("KBO_")
                 or str(c).startswith("ACE_"))]
log(f"\n  1군 판정 코드 {len(FIRST)}개: {sorted(map(str, FIRST))}")
tm1 = tm[tm["pitcher_team"].isin(FIRST)].copy()
log(f"  필터 후 행수 {len(tm1):,} (전체의 {len(tm1)/len(tm)*100:.1f}%)")
log("  시즌별 행수 비교 (1군 필터 trackman vs train):")
tm_yr = tm1.groupby("season").size()
tr_yr = trn.groupby("season").size()
for s in range(2019, 2025):
    log(f"    {s}: trackman {tm_yr.get(s, 0):9,d}  vs  train {tr_yr.get(s, 0):9,d}"
        f"   비율 {tm_yr.get(s, 0)/max(tr_yr.get(s, 1), 1):.3f}")

# =====================================================================
# 3. 스냅샷 + 매칭 (좌우 블록, 시즌별 카운트 + 누적 믹스)
# =====================================================================
log("\n" + "=" * 88)
log("3. 재매칭")
log("=" * 88)

# train 쪽: 시즌별 행수(=1군 투구수), 시즌말 누적 믹스, 좌우
snap = (trn.groupby(["pitcher_id", "season"])
           .agg(rows=("pitcher_id", "size")).reset_index())
trn_s = trn.sort_values(["pitcher_id", "season", "asof_pitcher_pitchmix_n"],
                        na_position="first")
last = (trn_s.groupby(["pitcher_id", "season"]).tail(1)
        [["pitcher_id", "season", "asof_pitcher_fastball_rate",
          "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]])
snap = snap.merge(last, on=["pitcher_id", "season"], how="left")
tr_hand = trn.groupby("pitcher_id")["pitcher_hand"].agg(lambda s: s.mode().iloc[0])

# trackman 쪽(1군 필터): 시즌별 카운트, 누적 믹스, 좌우
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
HAND_MAP = {2: "Right", 1: "Left"}   # 1차 정찰의 빈도 대응

SEASONS = list(range(2019, 2025))


def profile(df_, idc, valc):
    return (df_.pivot_table(index=idc, columns="season", values=valc,
                            aggfunc="first").reindex(columns=SEASONS))


def block_match(pids_blk, tids_blk):
    Pc = profile(snap[snap["pitcher_id"].isin(pids_blk)], "pitcher_id", "rows")
    Tc = profile(tsnap[tsnap["tid"].isin(tids_blk)], "tid", "cum_n")
    # 시즌별(비누적) 카운트: trackman은 n, train은 rows
    Tn = profile(tsnap[tsnap["tid"].isin(tids_blk)], "tid", "n")
    A = Pc.to_numpy(dtype="float64")[:, None, :]
    B = Tn.reindex(Pc.columns, axis=1).to_numpy(dtype="float64")[None, :, :]
    both = ~np.isnan(A) & ~np.isnan(B)
    either = ~np.isnan(A) | ~np.isnan(B)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(both, np.abs(np.log((A + 1.0) / (B + 1.0))), np.nan)
        cnt = np.nanmean(rel, axis=2)
    jac = both.sum(2) / np.maximum(either.sum(2), 1)
    # 누적 믹스 거리
    mixd = np.zeros_like(cnt)
    for tr_c, tm_c in [("asof_pitcher_fastball_rate", "r_fb"),
                       ("asof_pitcher_breaking_rate", "r_br"),
                       ("asof_pitcher_offspeed_rate", "r_os")]:
        Am = profile(snap[snap["pitcher_id"].isin(pids_blk)],
                     "pitcher_id", tr_c).to_numpy(dtype="float64")[:, None, :]
        Bm = profile(tsnap[tsnap["tid"].isin(tids_blk)],
                     "tid", tm_c).to_numpy(dtype="float64")[None, :, :]
        with np.errstate(invalid="ignore"):
            mixd += np.nan_to_num(np.nanmean(np.where(both, np.abs(Am - Bm),
                                                      np.nan), axis=2), nan=1.0)
    D = np.where(np.isnan(cnt), np.inf, cnt) + 0.5 * (1 - jac) + 1.0 * mixd
    order = np.argsort(D, axis=1)
    r = np.arange(D.shape[0])
    b1, b2 = order[:, 0], order[:, 1]
    return (Pc.index.to_numpy(), Tc.index.to_numpy()[b1],
            D[r, b1], D[r, b2] - D[r, b1])


all_pairs = []
for h_tr, h_tm in HAND_MAP.items():
    pids_blk = tr_hand[tr_hand == h_tr].index
    tids_blk = tm_hand[tm_hand == h_tm].index
    pid_a, tid_a, d1_a, mg_a = block_match(pids_blk, tids_blk)
    log(f"  [{h_tm}] {len(pid_a)}명 매칭 — 1등거리 중앙값 {np.median(d1_a):.4f}  "
        f"격차 중앙값 {np.median(mg_a):.4f}")
    all_pairs.append(pd.DataFrame({"pid": pid_a, "tid": tid_a,
                                   "d1": d1_a, "margin": mg_a}))
pairs = pd.concat(all_pairs, ignore_index=True)

# =====================================================================
# 4. 검증
# =====================================================================
log("\n" + "=" * 88)
log("4. 검증")
log("=" * 88)
vol = snap.groupby("pitcher_id")["rows"].sum()
for dth, mth in [(0.05, 0.05), (0.10, 0.10), (0.20, 0.10)]:
    c = (pairs["d1"] < dth) & (pairs["margin"] > mth)
    cov = vol.reindex(pairs.loc[c, "pid"]).sum() / vol.sum()
    dup = pairs.loc[c, "tid"].duplicated().sum()
    log(f"  d1<{dth:.2f} & 격차>{mth:.2f}: {c.sum():3d}명  "
        f"커버리지 {cov*100:5.1f}%  tid중복 {dup}건")

CONF = (pairs["d1"] < 0.10) & (pairs["margin"] > 0.10)
conf_pairs = pairs[CONF].copy()

mg = snap.merge(conf_pairs[["pid", "tid"]], left_on="pitcher_id", right_on="pid")
mg = mg.merge(tsnap[["tid", "season", "n", "r_fb", "r_br", "r_os"]],
              on=["tid", "season"], how="inner")
diff = np.abs(mg["rows"].to_numpy(dtype="float64") - mg["n"].to_numpy(dtype="float64"))
log(f"\n  [확신 매칭 {CONF.sum()}명 기준]")
log(f"  시즌별 카운트 차: 중앙값 {np.median(diff):.0f}  ≤3 {np.mean(diff<=3)*100:.1f}%  "
    f"≤20 {np.mean(diff<=20)*100:.1f}%   ({len(diff):,}투수-시즌)")
mixerr = (np.abs(mg["r_fb"] - mg["asof_pitcher_fastball_rate"])
          + np.abs(mg["r_br"] - mg["asof_pitcher_breaking_rate"])
          + np.abs(mg["r_os"] - mg["asof_pitcher_offspeed_rate"]))
log(f"  믹스 절대오차 합: 중앙값 {mixerr.median():.4f}  평균 {mixerr.mean():.4f}")

tm_team = (tm1.groupby(["pitcher_trackman_id", "season"])["pitcher_team"]
              .agg(lambda s: s.mode().iloc[0])
              .reset_index().rename(columns={"pitcher_trackman_id": "tid"}))
tr_team = (trn.groupby(["pitcher_id", "season"])["pitcher_team_id"]
              .agg(lambda s: s.mode().iloc[0]).reset_index())
tt = (conf_pairs[["pid", "tid"]]
      .merge(tr_team, left_on="pid", right_on="pitcher_id")
      .merge(tm_team, on=["tid", "season"], how="inner"))
xt = pd.crosstab(tt["pitcher_team_id"], tt["pitcher_team"])
purity = xt.max(axis=1).sum() / xt.to_numpy().sum() if xt.size else float("nan")
log(f"  팀 대응 순도: {purity*100:.1f}%  (투수-시즌 {len(tt):,}쌍)")
log("\n  팀 대응표 (train ID → trackman 코드 최빈):")
for tid_, row in xt.iterrows():
    log(f"    팀 {tid_:>3} → {row.idxmax():12s} ({row.max()}/{row.sum()})")

path = os.path.join(OUT, "trackman_match.csv")
conf_pairs.to_csv(path, index=False)
log(f"\n  확신 매칭 쌍 저장: {path}")

log("\n" + "=" * 88)
log("판정 가이드")
log("=" * 88)
log("  GO  : 커버리지 70%+ · 팀 순도 97%+ · 카운트 차 중앙값 한 자릿수")
log("  NO-GO: 1군 필터 후에도 커버리지/순도가 낮음 → trackman 축 폐기")
log("=" * 88)
