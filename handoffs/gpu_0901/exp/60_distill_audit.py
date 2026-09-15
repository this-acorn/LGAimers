# -*- coding: utf-8 -*-
"""
[60] 증류 사후 감사 — exp/57 결과의 중심/변별력 분해 + 매칭·비매칭 분리 평가

리뷰 지적 반영:
  · 조인율 54.4%라 매칭 행이 특정 연도·R/F·선수에 편중되면 teacher 이득이
    전체로 전이되지 않는다 → 2024 검증행을 매칭/비매칭으로 갈라 각각 평가
  · 모든 후보에 중심(평균) vs 변별력(정보) 분해를 적용 (멀티클래스와 동일 절차)

실행: exp/57 완료 후  PYTHONIOENCODING=utf-8 py -3.12 -u exp/60_distill_audit.py
"""

import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, "exp")
from common import log, load_train

TEAM_MAP = {12: "DOO_BEA", 13: "LG_TWI", 14: "KIW_HER", 15: "LOT_GIA",
            16: "KIA_TIG", 17: "HAN_EAG", 18: "SAM_LIO", 19: "NC_DIN",
            20: "KT_WIZ", 21: "SSG_LAN"}
KEY = ["season", "game_month", "game_dayofweek", "inning", "top_bottom",
       "balls_before", "strikes_before", "outs_before", "ph", "bh", "pt", "bt"]

df = load_train()
va = df[df.season == 2024].reset_index(drop=True)
y = va["control_success"].to_numpy("float64")
r = y.mean()
DEN = r * (1 - r)


def sc(p, yy=None, dd=None):
    yy = y if yy is None else yy
    dd = DEN if dd is None else dd
    return 100000 * (1 - np.mean((p - yy) ** 2) / dd)


def decomp(p):
    d = p.mean() - r
    pen = 100000 * d * d / DEN
    return sc(p), sc(p) + pen, pen


# ---- 2024 검증행의 trackman 매칭 마스크 재구성 ----
tm = pd.read_csv("data/trackman_history.csv", encoding="utf-8-sig",
                 usecols=["season", "game_month", "game_dayofweek", "inning",
                          "top_bottom", "balls_before", "strikes_before",
                          "outs_before", "pitcher_hand", "batter_hand",
                          "pitcher_team", "batter_team", "rel_speed"])
tm.columns = [c.replace("﻿", "").strip() for c in tm.columns]
hand = {2: "Right", 1: "Left"}
va["ph"] = va.pitcher_hand.map(hand); va["bh"] = va.batter_hand.map(hand)
va["pt"] = va.pitcher_team_id.map(TEAM_MAP); va["bt"] = va.batter_team_id.map(TEAM_MAP)
tm["pt"] = tm.pitcher_team.replace({"SK_WYV": "SSG_LAN"})
tm["bt"] = tm.batter_team.replace({"SK_WYV": "SSG_LAN"})
tm["ph"] = tm.pitcher_hand.astype(str); tm["bh"] = tm.batter_hand.astype(str)
tm["top_bottom"] = tm.top_bottom.astype(str).str[0]
va["top_bottom"] = va.top_bottom.astype(str).str[0]
for c in ["season", "game_month", "game_dayofweek", "inning",
          "balls_before", "strikes_before", "outs_before"]:
    va[c] = pd.to_numeric(va[c], errors="coerce").astype("Int64")
    tm[c] = pd.to_numeric(tm[c], errors="coerce").astype("Int64")
vk = va.dropna(subset=["pt", "bt", "ph", "bh"])
tk = tm[tm.pt.isin(TEAM_MAP.values()) & tm.bt.isin(TEAM_MAP.values())]
c1 = vk.groupby(KEY, observed=True).size()
c2 = tk.groupby(KEY, observed=True).size()
k11 = c1[c1 == 1].index.intersection(c2[c2 == 1].index)
tm11 = tk.set_index(KEY).loc[k11, ["rel_speed"]].reset_index()
mm = va.merge(tm11, on=KEY, how="left", validate="m:1")["rel_speed"].notna().to_numpy()
log(f"2024 검증행 trackman 매칭: {mm.sum():,} / {len(va):,} ({mm.mean()*100:.1f}%)")
log(f"  매칭행 r={y[mm].mean():.4f} / 비매칭 r={y[~mm].mean():.4f}  "
    f"(편중 확인 — 크게 다르면 teacher 이득 전이 주의)")
gt = va.game_type.astype(str).to_numpy()
log(f"  매칭행 중 R 비율 {np.mean(gt[mm]=='R')*100:.1f}% / "
    f"비매칭 중 R {np.mean(gt[~mm]=='R')*100:.1f}%")

# ---- 후보 예측 로드 ----
P = {"CS79(2시드)": np.load("lab/45_both_ens.npy").astype("float64"),
     "MC(2시드)": np.load("lab/53_mc_ens.npy").astype("float64")}
for f in sorted(glob.glob("lab/57_alpha*.npy")):
    P[f"증류 {f.split('alpha')[1].replace('.npy','')}%"] = np.load(f).astype("float64")
for f in sorted(glob.glob("lab/58_curve*_final.npy")):
    P[f.split("/")[-1].replace("58_", "").replace("_final.npy", "")] = \
        np.load(f).astype("float64")

log("\n" + "=" * 92)
log("전체 후보: 총점 = 변별력 − 중심벌점 / 매칭·비매칭 분리")
log("=" * 92)
log(f"  {'모델':16s} {'총점':>9s} {'변별력':>9s} {'중심벌점':>9s} "
    f"{'매칭행':>9s} {'비매칭행':>9s}")
for k, p in P.items():
    t, d_, pen = decomp(p)
    ym, yu = y[mm], y[~mm]
    sm = 100000 * (1 - np.mean((p[mm] - ym) ** 2) / (ym.mean() * (1 - ym.mean())))
    su = 100000 * (1 - np.mean((p[~mm] - yu) ** 2) / (yu.mean() * (1 - yu.mean())))
    log(f"  {k:16s} {t:>9.1f} {d_:>9.1f} {pen:>9.1f} {sm:>9.1f} {su:>9.1f}")

log("\n  판독:")
log("   · 증류가 매칭행에서만 좋아지면 → teacher 지식이 전이 안 된 것 (기각)")
log("   · 매칭·비매칭 모두 좋아지면 → 진짜 학습신호 개선 (승격)")
log("   · 변별력이 아니라 중심벌점만 줄었으면 → 2024 평균 우연 적중 (전이 위험)")
log("=" * 92)
