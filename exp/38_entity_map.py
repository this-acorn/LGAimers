# -*- coding: utf-8 -*-
"""
[38] 행 조인 기반 완전 선수 매핑 — 투수 확정 + 타자 신규 (exp/37 돌파구의 즉시 활용)

exp/37 확정: 컨텍스트 키(시즌·월·요일·이닝·초말·카운트·아웃·양손·양팀)로
  train 행의 54.9%가 trackman과 1:1 유일 매칭, 투수 ID 일치율 100%.

이 스크립트: 1:1 행 전체(~80만)에서 (train_id ↔ trackman_id) 투표를 모아
  투수·타자 매핑 테이블을 만든다. 판정: 해당 train_id의 투표 중 최다 tid 비율(순도).
  순도 ≥ 99% & 표 ≥ 20 → 확정. 산출:
    lab/entity_map_pitcher.csv  (pid, tid, votes, purity)
    lab/entity_map_batter.csv   (bid, tid, votes, purity)

규칙 메모: train+trackman(≤2024, 공식 데이터)만 사용. test 무관. 행 독립성 무관.
  현재 투구 측정값을 입력으로 쓰는 것이 아니라 '선수 신원 대응'만 만든다.
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
TM_TEAM_ALIAS = {"SK_WYV": "SSG_LAN"}
KEY = ["season", "game_month", "game_dayofweek", "inning", "top_bottom",
       "balls_before", "strikes_before", "outs_before", "ph", "bh", "pt", "bt"]

log("train 로딩...")
tr = pd.read_csv("data/train.csv", encoding="utf-8-sig",
                 usecols=["season", "game_month", "game_dayofweek", "inning",
                          "top_bottom", "balls_before", "strikes_before",
                          "outs_before", "pitcher_hand", "batter_hand",
                          "pitcher_team_id", "batter_team_id",
                          "pitcher_id", "batter_id"])
tr.columns = [c.replace("﻿", "").strip() for c in tr.columns]
log("trackman 로딩...")
tm = pd.read_csv("data/trackman_history.csv", encoding="utf-8-sig",
                 usecols=["season", "game_month", "game_dayofweek", "inning",
                          "top_bottom", "balls_before", "strikes_before",
                          "outs_before", "pitcher_hand", "batter_hand",
                          "pitcher_team", "batter_team",
                          "pitcher_trackman_id", "batter_trackman_id"])
tm.columns = [c.replace("﻿", "").strip() for c in tm.columns]

hand_map = {2: "Right", 1: "Left"}
tr["ph"] = tr.pitcher_hand.map(hand_map)
tr["bh"] = tr.batter_hand.map(hand_map)
tr["pt"] = tr.pitcher_team_id.map(TEAM_MAP)
tr["bt"] = tr.batter_team_id.map(TEAM_MAP)
tm["pt"] = tm.pitcher_team.replace(TM_TEAM_ALIAS)
tm["bt"] = tm.batter_team.replace(TM_TEAM_ALIAS)
tm["ph"] = tm.pitcher_hand.astype(str)
tm["bh"] = tm.batter_hand.astype(str)
tm["top_bottom"] = tm.top_bottom.astype(str).str[0]
tr["top_bottom"] = tr.top_bottom.astype(str).str[0]
for c in ["season", "game_month", "game_dayofweek", "inning",
          "balls_before", "strikes_before", "outs_before"]:
    tr[c] = pd.to_numeric(tr[c], errors="coerce").astype("Int64")
    tm[c] = pd.to_numeric(tm[c], errors="coerce").astype("Int64")

trk = tr.dropna(subset=["pt", "bt", "ph", "bh"])
tmk = tm[tm.pt.isin(TEAM_MAP.values()) & tm.bt.isin(TEAM_MAP.values())]
log(f"키 대상 train {len(trk):,} / tm {len(tmk):,}")

# 1:1 키만 남기기
c_tr = trk.groupby(KEY, observed=True).size()
c_tm = tmk.groupby(KEY, observed=True).size()
k1_tr = c_tr[c_tr == 1].index
k1_tm = c_tm[c_tm == 1].index
keys11 = k1_tr.intersection(k1_tm)
log(f"1:1 키 {len(keys11):,}")

a = trk.set_index(KEY).loc[keys11, ["pitcher_id", "batter_id"]]
b = tmk.set_index(KEY).loc[keys11, ["pitcher_trackman_id", "batter_trackman_id"]]
pair = pd.concat([a.reset_index(drop=True), b.reset_index(drop=True)], axis=1)
log(f"투표 쌍 {len(pair):,}")


def build_map(df, id_col, tid_col, name):
    votes = df.groupby([id_col, tid_col]).size().rename("v").reset_index()
    tot = votes.groupby(id_col)["v"].sum().rename("total")
    top = votes.sort_values("v", ascending=False).drop_duplicates(id_col)
    top = top.merge(tot, on=id_col)
    top["purity"] = top.v / top.total
    ok = top[(top.purity >= 0.99) & (top.v >= 20)]
    soft = top[(top.purity >= 0.95) & (top.v >= 5)]
    log(f"[{name}] 후보 {len(top):,}명 / 확정(순도99%+표20) {len(ok):,}명 / "
        f"느슨(95%+5표) {len(soft):,}명")
    out = top.rename(columns={id_col: "id", tid_col: "tid", "v": "votes"})
    out[["id", "tid", "votes", "total", "purity"]].to_csv(
        f"lab/entity_map_{name}.csv", index=False)
    return top, ok


p_top, p_ok = build_map(pair, "pitcher_id", "pitcher_trackman_id", "pitcher")
b_top, b_ok = build_map(pair, "batter_id", "batter_trackman_id", "batter")

# 기존 지문 매칭과 대조 (투수)
old = pd.read_csv("lab/trackman_match_v3.csv")
mg = p_ok.merge(old, left_on="pitcher_id", right_on="pid", how="inner")
agree = (mg.pitcher_trackman_id == mg.tid).mean() if len(mg) else float("nan")
log(f"기존 지문매칭 {len(old)}명과 겹침 {len(mg)}명 — 일치율 {agree*100:.1f}%")

# 커버리지: train 투구량 기준
pw = tr.groupby("pitcher_id").size().rename("rows").reset_index()
pw["ok"] = pw.pitcher_id.isin(set(p_ok.pitcher_id))
bw = tr.groupby("batter_id").size().rename("rows").reset_index()
bw["ok"] = bw.batter_id.isin(set(b_ok.batter_id))
log(f"투수: 확정 {pw.ok.sum()}/{len(pw)}명, 투구량 커버리지 "
    f"{pw.loc[pw.ok, 'rows'].sum()/pw.rows.sum()*100:.1f}%")
log(f"타자: 확정 {bw.ok.sum()}/{len(bw)}명, 투구량 커버리지 "
    f"{bw.loc[bw.ok, 'rows'].sum()/bw.rows.sum()*100:.1f}%")
log(f"총 {time.time()-T0:.0f}초 — 산출: lab/entity_map_pitcher.csv / entity_map_batter.csv")
