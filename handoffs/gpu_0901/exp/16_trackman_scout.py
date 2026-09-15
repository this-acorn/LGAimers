"""
[16] Trackman 지문 매칭 정찰 — pitcher_id ↔ pitcher_trackman_id 연결 가능성 측정

실행:  PYTHONIOENCODING=utf-8 py -3.12 -u exp/16_trackman_scout.py

배경:
  두 ID 체계의 교집합은 0 (HANDOFF §5-4 확정). 그러나 양쪽에 공통 신호가 많다:
  시즌·팀·좌우·시즌별 투구량·구종 믹스. 특히 train의 asof_pitcher_pitchmix_n이
  Trackman류 소스에서 계산됐다면, 시즌말 누적값이 trackman 누적 투구수와
  거의 정수 단위로 일치할 것이다 → 지문 매칭으로 익명화를 우회할 수 있다.

이 스크립트는 **정찰만** 한다 (모델 학습 없음, 피처 생성 없음):
  A. 두 파일의 스키마·코딩 확인 (hand/team/구종군 값 체계)
  B. 시즌말 스냅샷 구축
     train:    (pid, season)별 행수 / max asof_pitcher_n / max asof_pitcher_pitchmix_n
     trackman: (tid, season)별 누적 투구수 / 누적 구종 믹스
  C. 세 가지 카운트 가설(행수/asof_n/pitchmix_n) 각각으로 지문 매칭
     → 어느 카운트가 trackman과 정렬되는지 판별
  D. 최선 가설로 매칭 품질 측정: 확신도(1등 vs 2등 거리), 정수 일치율,
     믹스 비율 일치, 팀 대응 순도, 좌우 일치율, 투구량 가중 커버리지

합법성: 둘 다 공식 대회 데이터. 설명서가 trackman으로 "투수 단위 요약값 등
추가 피처" 생성을 명시 허용. 외부 데이터 아님. (피처를 만든다면 시즌 as-of
규율은 exp/14와 동일하게 적용 — 그건 다음 단계 스크립트의 몫)
"""

import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "exp")
from common import log

T0 = time.time()


def tick(msg):
    log(f"  [{time.time()-T0:6.0f}s] {msg}")


# =====================================================================
# A. 로드 & 스키마 정찰
# =====================================================================
log("=" * 88)
log("A. 스키마 정찰")
log("=" * 88)

TM_COLS = ["season", "pitcher_trackman_id", "pitcher_hand", "pitcher_team",
           "pitch_type_group", "rel_speed", "spin_rate", "rel_height", "rel_side",
           "extension"]
tm = pd.read_csv("data/trackman_history.csv", usecols=TM_COLS, encoding="utf-8-sig")
tm.columns = [c.replace("﻿", "").strip() for c in tm.columns]
tick(f"trackman {len(tm):,}행 로드")
log(f"  투수 수: {tm['pitcher_trackman_id'].nunique()}")
log(f"  pitcher_hand 값: {dict(tm['pitcher_hand'].value_counts(dropna=False).head(6))}")
log(f"  pitch_type_group 값: {dict(tm['pitch_type_group'].value_counts(dropna=False).head(8))}")
log(f"  pitcher_team 종류 {tm['pitcher_team'].nunique()}개: "
    f"{sorted(map(str, tm['pitcher_team'].dropna().unique()))[:15]}")
log(f"  시즌별 행수: {dict(tm['season'].value_counts().sort_index())}")
log("  측정 결측률: " + "  ".join(
    f"{c}={tm[c].isna().mean()*100:.1f}%"
    for c in ["rel_speed", "spin_rate", "rel_height", "rel_side", "extension"]))

TR_COLS = ["pitcher_id", "season", "pitcher_hand", "pitcher_team_id",
           "asof_pitcher_n", "asof_pitcher_pitchmix_n",
           "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
           "asof_pitcher_offspeed_rate"]
trn = pd.read_csv("data/train.csv", usecols=TR_COLS, encoding="utf-8-sig")
trn.columns = [c.replace("﻿", "").strip() for c in trn.columns]
tick(f"train {len(trn):,}행 로드")
log(f"  투수 수: {trn['pitcher_id'].nunique()}")
log(f"  pitcher_hand 값: {dict(trn['pitcher_hand'].value_counts(dropna=False).head(6))}")
log(f"  pitcher_team_id 종류 {trn['pitcher_team_id'].nunique()}개")

# =====================================================================
# B. 시즌말 스냅샷
# =====================================================================
log("\n" + "=" * 88)
log("B. 시즌말 스냅샷")
log("=" * 88)

snap = (trn.groupby(["pitcher_id", "season"])
           .agg(rows=("pitcher_id", "size"),
                asof_n=("asof_pitcher_n", "max"),
                mix_n=("asof_pitcher_pitchmix_n", "max"))
           .reset_index())
# 시즌 마지막(=pitchmix_n 최대) 시점의 누적 믹스 비율 스냅샷
trn_s = trn.sort_values(["pitcher_id", "season", "asof_pitcher_pitchmix_n"],
                        na_position="first")
last = (trn_s.groupby(["pitcher_id", "season"]).tail(1)
        [["pitcher_id", "season", "pitcher_hand", "pitcher_team_id",
          "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
          "asof_pitcher_offspeed_rate"]])
snap = snap.merge(last, on=["pitcher_id", "season"], how="left")
tick(f"train 스냅샷 {len(snap):,}개 (투수-시즌)")
log(f"  mix_n 보유율: {snap['mix_n'].notna().mean()*100:.1f}%")

grp = tm["pitch_type_group"].astype(str).str.lower()
tm2 = pd.DataFrame({
    "tid": tm["pitcher_trackman_id"].to_numpy(),
    "season": tm["season"].to_numpy(),
    "fb": (grp == "fastball").astype("int32"),
    "br": (grp == "breaking").astype("int32"),
    "os": (grp == "offspeed").astype("int32"),
})
tsnap = (tm2.groupby(["tid", "season"])
            .agg(n=("season", "size"), fb=("fb", "sum"),
                 br=("br", "sum"), os=("os", "sum"))
            .reset_index().sort_values(["tid", "season"]))
gt = tsnap.groupby("tid")
for c in ["n", "fb", "br", "os"]:
    tsnap["cum_" + c] = gt[c].cumsum()
tick(f"trackman 스냅샷 {len(tsnap):,}개 (투수-시즌)")

SEASONS = list(range(2019, 2025))


def profile(df_, idc, valc):
    return (df_.pivot_table(index=idc, columns="season", values=valc, aggfunc="first")
               .reindex(columns=SEASONS))


def match_quality(P_tr, P_tm, label):
    """행=train 투수, 열=trackman 투수 거리행렬 → 1등/2등 거리와 인덱스"""
    A = P_tr.to_numpy(dtype="float64")[:, None, :]
    B = P_tm.to_numpy(dtype="float64")[None, :, :]
    both = ~np.isnan(A) & ~np.isnan(B)
    either = ~np.isnan(A) | ~np.isnan(B)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(np.log((A + 1.0) / (B + 1.0)))
    rel = np.where(both, rel, np.nan)
    with np.errstate(invalid="ignore"):
        d = np.nanmean(rel, axis=2)                     # 공통 시즌 평균 로그비율
    jac = both.sum(2) / np.maximum(either.sum(2), 1)    # 활동 시즌 자카드
    D = np.where(np.isnan(d), np.inf, d) + 1.0 * (1.0 - jac)
    order = np.argsort(D, axis=1)
    b1, b2 = order[:, 0], order[:, 1]
    r = np.arange(D.shape[0])
    d1, d2 = D[r, b1], D[r, b2]
    log(f"  [{label}] 1등거리 중앙값 {np.median(d1):.4f}  "
        f"확신(2등-1등) 중앙값 {np.median(d2-d1):.4f}  "
        f"강한매칭(d1<0.02 & 격차>0.1) {int(((d1<0.02)&(d2-d1>0.1)).sum())}명"
        f" / {D.shape[0]}명")
    return D, b1, d1, d2


log("\n" + "=" * 88)
log("C. 카운트 가설 3종 매칭 — 어느 카운트가 trackman과 정렬되나")
log("=" * 88)
P_tm_n = profile(tsnap, "tid", "cum_n")
results = {}
for valc, label in [("rows", "train 행수"), ("asof_n", "asof_pitcher_n"),
                    ("mix_n", "pitchmix_n")]:
    P_tr = profile(snap, "pitcher_id", valc)
    results[valc] = match_quality(P_tr, P_tm_n, label)

best_key = min(results, key=lambda k: np.median(results[k][2]))
log(f"\n  ▶ 최적 카운트 가설: {best_key}")

# =====================================================================
# D. 최적 가설로 매칭 검증
# =====================================================================
log("\n" + "=" * 88)
log("D. 매칭 검증 (최적 가설 기준)")
log("=" * 88)
D, b1, d1, d2 = results[best_key]
P_tr = profile(snap, "pitcher_id", best_key)
pids = P_tr.index.to_numpy()
tids = P_tm_n.index.to_numpy()
margin = d2 - d1
conf = (d1 < 0.02) & (margin > 0.1)
pairs = pd.DataFrame({"pid": pids, "tid": tids[b1],
                      "d1": d1, "margin": margin, "conf": conf})
log(f"  강한 매칭 {conf.sum()}명 / train 투수 {len(pids)}명")

# 투구량 가중 커버리지
vol = snap.groupby("pitcher_id")["rows"].sum()
cov = vol.reindex(pairs.loc[pairs["conf"], "pid"]).sum() / vol.sum()
log(f"  투구량 가중 커버리지: {cov*100:.1f}% (train 투구 중 강한매칭 투수의 비중)")

# 정수 일치율 — 카운트가 거의 그대로 일치하는가 (스모킹건)
mg = snap.merge(pairs[pairs["conf"]][["pid", "tid"]],
                left_on="pitcher_id", right_on="pid")
mg = mg.merge(tsnap[["tid", "season", "cum_n", "cum_fb", "cum_br", "cum_os"]],
              on=["tid", "season"], how="inner")
cnt = mg[best_key].to_numpy(dtype="float64")
cum = mg["cum_n"].to_numpy(dtype="float64")
ok = ~np.isnan(cnt)
diff = np.abs(cnt[ok] - cum[ok])
log(f"  |카운트 차| 중앙값 {np.median(diff):.0f}  "
    f"≤5 비율 {np.mean(diff<=5)*100:.1f}%  ≤50 비율 {np.mean(diff<=50)*100:.1f}%"
    f"   (투수-시즌 {ok.sum():,}쌍)")

# 믹스 비율 일치 — 분모 가설 2종
mgm = mg.dropna(subset=["asof_pitcher_fastball_rate", "cum_n"])
den_all = np.maximum(mgm["cum_n"].to_numpy(dtype="float64"), 1.0)
den_cls = np.maximum((mgm["cum_fb"] + mgm["cum_br"] + mgm["cum_os"])
                     .to_numpy(dtype="float64"), 1.0)
for nm, den in [("분모=전체", den_all), ("분모=other제외", den_cls)]:
    err = (np.abs(mgm["cum_fb"] / den - mgm["asof_pitcher_fastball_rate"]).mean()
           + np.abs(mgm["cum_br"] / den - mgm["asof_pitcher_breaking_rate"]).mean()
           + np.abs(mgm["cum_os"] / den - mgm["asof_pitcher_offspeed_rate"]).mean())
    log(f"  믹스 절대오차 합({nm}): {err:.4f}   (0.03 이하면 사실상 같은 소스)")

# 팀 대응 순도 — 매칭이 맞다면 팀ID 대응이 깨끗한 일대일이어야 함
tm_team = (tm.groupby(["pitcher_trackman_id", "season"])["pitcher_team"]
             .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
             .reset_index().rename(columns={"pitcher_trackman_id": "tid"}))
tt = mg.merge(tm_team, on=["tid", "season"], how="left")
xt = pd.crosstab(tt["pitcher_team_id"], tt["pitcher_team"])
purity = xt.max(axis=1).sum() / xt.to_numpy().sum() if xt.size else float("nan")
log(f"  팀 대응 순도: {purity*100:.1f}%   (100% 근처면 매칭 정합 강력 증거)")

# 좌우 일치율 (빈도 순위 대응으로)
tm_hand = tm.groupby("pitcher_trackman_id")["pitcher_hand"] \
            .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
tr_hand = snap.groupby("pitcher_id")["pitcher_hand"] \
              .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
ph = pairs[pairs["conf"]].copy()
ph["h_tr"] = tr_hand.reindex(ph["pid"]).to_numpy()
ph["h_tm"] = tm_hand.reindex(ph["tid"]).to_numpy()
hx = pd.crosstab(ph["h_tr"], ph["h_tm"])
hpur = hx.max(axis=1).sum() / hx.to_numpy().sum() if hx.size else float("nan")
log(f"  좌우 대응 순도: {hpur*100:.1f}%")

# 중복 배정 — 서로 다른 train 투수가 같은 trackman 투수를 가리키는 충돌
dup = pairs[pairs["conf"]]["tid"].duplicated().sum()
log(f"  강한매칭 중 tid 중복 배정: {dup}건 (0이어야 정상)")

log("\n" + "=" * 88)
log("판정 가이드")
log("=" * 88)
log("  GO  : 강한매칭 커버리지 70%+ · 팀 순도 95%+ · 카운트 차 중앙값 수십 이하")
log("        → 다음 단계: 매칭 테이블 저장 + trackman 요약 피처(릴리스 일관성 등) 실험")
log("  NO-GO: 확신 격차가 안 벌어지거나 팀 순도가 낮음 → trackman 축 최종 폐기")
log("=" * 88)
