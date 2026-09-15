# -*- coding: utf-8 -*-
"""
[37] train ↔ trackman 행 단위 조인 가능성 테스트 (순수 진단, 학습 없음)

가설: 운영진이 베이스라인에서 "trackman을 직접 활용해 보라"고 명시했다.
  공통 컨텍스트 키 (season, game_month, game_dayofweek, inning, top_bottom,
  balls, strikes, outs, pitcher_hand, batter_hand, pitcher_team, batter_team)로
  train 투구 ↔ trackman 투구가 행 단위로 (준)유일하게 붙는지 잰다.

주의: 좌우 코딩 train 2=R/1=L vs trackman R/L 문자, 초말 T/B 동일 여부,
  팀 대응(exp/16에서 확정: 12=DOO 13=LG 14=KIW 15=LOT 16=KIA 17=HAN 18=SAM
  19=NC 20=KT 21=SSG/SK) 반영 필요.

산출: 키별 매칭 카디널리티 분포 (1:1 비율), 커버리지, 시즌별 상황.
"""

import time
import numpy as np
import pandas as pd

T0 = time.time()


def log(m):
    print(f"  [{time.time()-T0:6.0f}s] {m}", flush=True)


TEAM_MAP = {12: "DOO_BEA", 13: "LG_TWI", 14: "KIW_HER", 15: "LOT_GIA",
            16: "KIA_TIG", 17: "HAN_EAG", 18: "SAM_LIO", 19: "NC_DIN",
            20: "KT_WIZ", 21: "SSG_LAN"}
# SK는 21로 통합 (exp/16). SK_WYV도 21로 매핑되는 trackman 팀명.
TM_TEAM_ALIAS = {"SK_WYV": "SSG_LAN"}

log("train 로딩...")
tr = pd.read_csv("data/train.csv", encoding="utf-8-sig",
                 usecols=["season", "game_month", "game_dayofweek", "inning",
                          "top_bottom", "balls_before", "strikes_before",
                          "outs_before", "pitcher_hand", "batter_hand",
                          "pitcher_team_id", "batter_team_id", "pitcher_id",
                          "control_success"])
tr.columns = [c.replace("﻿", "").strip() for c in tr.columns]
log(f"train {tr.shape}")

log("trackman 로딩...")
tm = pd.read_csv("data/trackman_history.csv", encoding="utf-8-sig",
                 usecols=["season", "game_month", "game_dayofweek", "inning",
                          "top_bottom", "balls_before", "strikes_before",
                          "outs_before", "pitcher_hand", "batter_hand",
                          "pitcher_team", "batter_team", "pitcher_trackman_id",
                          "rel_speed", "pitch_type_group"])
tm.columns = [c.replace("﻿", "").strip() for c in tm.columns]
log(f"trackman {tm.shape}")

# ---- 코딩 정규화 ----
log(f"train pitcher_hand 값: {sorted(tr.pitcher_hand.dropna().unique())[:5]}")
log(f"tm    pitcher_hand 값: {sorted(tm.pitcher_hand.dropna().unique().astype(str))[:5]}")
log(f"train top_bottom: {sorted(tr.top_bottom.unique())} / "
    f"tm: {sorted(tm.top_bottom.astype(str).unique())[:4]}")

# train 손: 2=Right, 1=Left (exp/16 확정) → 문자로
hand_map = {2: "Right", 1: "Left", 2.0: "Right", 1.0: "Left"}
tm_hand_vals = set(tm.pitcher_hand.astype(str).unique())
log(f"tm 손 표기 전체: {tm_hand_vals}")
# trackman 손 표기가 'Right'/'Left'면 그대로, 'R'/'L'이면 축약 매핑
if tm_hand_vals & {"Right", "Left"}:
    pass
elif tm_hand_vals & {"R", "L"}:
    hand_map = {2: "R", 1: "L", 2.0: "R", 1.0: "L"}

tr["ph"] = tr.pitcher_hand.map(hand_map)
tr["bh"] = tr.batter_hand.map(hand_map)
tr["pt"] = tr.pitcher_team_id.map(TEAM_MAP)
tr["bt"] = tr.batter_team_id.map(TEAM_MAP)
tm["pt"] = tm.pitcher_team.replace(TM_TEAM_ALIAS)
tm["bt"] = tm.batter_team.replace(TM_TEAM_ALIAS)
tm["ph"] = tm.pitcher_hand.astype(str)
tm["bh"] = tm.batter_hand.astype(str)
# ★ 초말 표기 정규화: train 'T'/'B' vs trackman 'Top'/'Bottom'
tm["top_bottom"] = tm.top_bottom.astype(str).str[0]
tr["top_bottom"] = tr.top_bottom.astype(str).str[0]
# ★ 수치 키 dtype 통일 (int vs float 이면 groupby 키가 어긋난다)
for c in ["season", "game_month", "game_dayofweek", "inning",
          "balls_before", "strikes_before", "outs_before"]:
    tr[c] = pd.to_numeric(tr[c], errors="coerce").astype("Int64")
    tm[c] = pd.to_numeric(tm[c], errors="coerce").astype("Int64")
log(f"정규화 후 tm top_bottom: {sorted(tm.top_bottom.unique())}")

cover_t = tr["pt"].notna().mean()
log(f"train 팀 매핑 커버리지 {cover_t*100:.1f}% (미매핑 팀 22/23/25 등은 제외됨)")

KEY = ["season", "game_month", "game_dayofweek", "inning", "top_bottom",
       "balls_before", "strikes_before", "outs_before", "ph", "bh", "pt", "bt"]
trk = tr.dropna(subset=["pt", "bt", "ph", "bh"]).copy()
tmk = tm[tm.pt.isin(TEAM_MAP.values()) & tm.bt.isin(TEAM_MAP.values())].copy()
log(f"키 대상: train {len(trk):,} / trackman {len(tmk):,}")

# ---- 키 카디널리티 ----
g_tr = trk.groupby(KEY, observed=True).size().rename("n_tr")
g_tm = tmk.groupby(KEY, observed=True).size().rename("n_tm")
log(f"고유 키: train {len(g_tr):,} / trackman {len(g_tm):,}")

j = pd.concat([g_tr, g_tm], axis=1, join="inner")
log(f"교집합 키 {len(j):,}")
both11 = ((j.n_tr == 1) & (j.n_tm == 1)).sum()
log(f"양쪽 모두 1:1인 키 {both11:,} ({both11/max(len(j),1)*100:.1f}%)")

# train 행 기준 커버리지: 자기 키가 trackman에 존재하고 1:1인 행 비율
trk2 = trk.merge(j.reset_index()[KEY + ["n_tr", "n_tm"]], on=KEY, how="left")
has = trk2.n_tm.notna()
uniq = has & (trk2.n_tr == 1) & (trk2.n_tm == 1)
log(f"train 행 중 키가 trackman에 존재: {has.mean()*100:.1f}%")
log(f"train 행 중 1:1 유일 매칭: {uniq.mean()*100:.1f}%")
for s, grp in trk2.groupby("season"):
    u = ((grp.n_tr == 1) & (grp.n_tm == 1) & grp.n_tm.notna()).mean()
    e = grp.n_tm.notna().mean()
    print(f"    {s}: 존재 {e*100:5.1f}%  1:1 {u*100:5.1f}%", flush=True)

# ---- 1:1 표본으로 정합성 검증: 같은 투수 지문인가 ----
# 1:1 매칭 행에서 (train pitcher_id ↔ tm pitcher_trackman_id) 대응을 세면,
# exp/16 매칭 테이블과 일치해야 진짜 조인이다
sample_keys = j[(j.n_tr == 1) & (j.n_tm == 1)].head(50000).index
tr_s = trk.set_index(KEY).loc[sample_keys, ["pitcher_id"]].reset_index()
tm_s = tmk.set_index(KEY).loc[sample_keys, ["pitcher_trackman_id"]].reset_index()
pair = tr_s.merge(tm_s, on=KEY)
match = pd.read_csv("lab/trackman_match_v3.csv")
known = dict(zip(match.pid, match.tid))
pair["known_tid"] = pair.pitcher_id.map(known)
chk = pair.dropna(subset=["known_tid"])
agree = (chk.pitcher_trackman_id == chk.known_tid).mean() if len(chk) else float("nan")
log(f"1:1 표본 {len(pair):,}건 중 기존 매칭과 대조 가능 {len(chk):,}건 — "
    f"투수 ID 일치율 {agree*100:.1f}%")
log("일치율이 높으면(>90%) 행 단위 조인은 실재한다 → 물리 측정값을 train에 부착 가능")
log(f"총 {time.time()-T0:.0f}초")
