# -*- coding: utf-8 -*-
"""
[56] 챔피언 오차 지도 — 2024 폴드에서 어디서 틀리는가 (순수 진단, 학습 없음)

저장된 예측 3종 비교:
  lab/41_cs_ens.npy   CS76 이진 (782.3)
  lab/45_both_ens.npy CS79 이진 (800.9)  ← 현 챔피언 계열 (LB 993.63)
  lab/53_mc_ens.npy   멀티클래스 P(성공) (837.0)  ← 학습 중인 후보

산출:
  A. 그룹별 편향(mean y−p)·세그먼트 점수·행수 — cs_n 분위, 콜드스타트, R/F, 카운트, 손조합
  B. 멀티클래스가 이진을 이긴 구간 (왜 +36.1인가) → 다음 수의 표적
  C. 잔여 개선 상한: 각 그룹을 완벽 보정했을 때의 이득 (스무딩으로 회수 가능한 몫)

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/56_error_map.py  (~3분)
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "exp")
from common import log, tick, load_train

df = load_train()
va = df[df.season == 2024].reset_index(drop=True)
y = va["control_success"].to_numpy("float64")
r_all = y.mean()
DEN = r_all * (1 - r_all)

P = {"CS76": np.load("lab/41_cs_ens.npy").astype("float64"),
     "CS79": np.load("lab/45_both_ens.npy").astype("float64"),
     "MC": np.load("lab/53_mc_ens.npy").astype("float64")}
for k, v in P.items():
    log(f"{k}: {100000*(1-np.mean((v-y)**2)/DEN):8.1f}   예측평균 {v.mean():.4f} "
        f"(실제 r={r_all:.4f})")

# ---- CS 재구성 (cs_n 필요) ----
hist = df[df.season <= 2023]
last = hist.sort_values("asof_pitcher_n").groupby("pitcher_id").tail(1)
Nend = dict(zip(last.pitcher_id, last.asof_pitcher_n.fillna(0) + 1))
cs_n = np.maximum(va.asof_pitcher_n.fillna(0).to_numpy("float64")
                  - va.pitcher_id.map(Nend).fillna(0).to_numpy("float64"), 0.0)
lastb = hist.sort_values("asof_batter_n").groupby("batter_id").tail(1)
Nendb = dict(zip(lastb.batter_id, lastb.asof_batter_n.fillna(0) + 1))
cs_nb = np.maximum(va.asof_batter_n.fillna(0).to_numpy("float64")
                   - va.batter_id.map(Nendb).fillna(0).to_numpy("float64"), 0.0)

GROUPS = {
    "투수 cs_n": pd.cut(cs_n, [-1, 0, 20, 100, 400, 1000, 1e9],
                        labels=["0", "1-20", "21-100", "101-400", "401-1000", "1000+"]),
    "타자 cs_n": pd.cut(cs_nb, [-1, 0, 20, 100, 400, 1e9],
                        labels=["0", "1-20", "21-100", "101-400", "400+"]),
    "game_type": va.game_type.astype(str),
    "카운트": va.balls_before.astype(str) + "-" + va.strikes_before.astype(str),
    "손조합": (va.pitcher_hand.astype(str) + "/" + va.batter_hand.astype(str)),
    "이닝": pd.cut(va.inning, [0, 3, 6, 9, 99], labels=["1-3", "4-6", "7-9", "10+"]),
    "월": va.game_month.astype(str),
}


def seg_stats(mask, p):
    ys, ps = y[mask], p[mask]
    if len(ys) < 500:
        return None
    bias = float((ys - ps).mean())
    # 전체 r 기준 기여도: 이 그룹이 총점에 미치는 영향 (행수 가중)
    contrib = -100000 * np.sum((ps - ys) ** 2 - DEN) / (DEN * len(y))
    # 완벽 중심보정 시 이득 (이 그룹만 평균 편향 제거)
    gain = 100000 * len(ys) * bias ** 2 / (DEN * len(y))
    return bias, contrib, gain, len(ys)


log("\n" + "=" * 92)
log("A. 그룹별 편향 (mean y−p) 및 중심보정 상한 — CS79 기준")
log("=" * 92)
log(f"  {'그룹':>10s} {'구간':>10s} {'행수':>9s} {'편향':>9s} {'CS79 기여':>10s} "
    f"{'보정상한':>9s} {'MC 편향':>9s}")
total_gain = 0.0
for gname, gv in GROUPS.items():
    for lvl in pd.Series(gv).dropna().unique():
        m = (pd.Series(gv) == lvl).to_numpy()
        s79 = seg_stats(m, P["CS79"])
        smc = seg_stats(m, P["MC"])
        if s79 is None:
            continue
        b, c, g, n_ = s79
        total_gain += g
        log(f"  {gname:>10s} {str(lvl):>10s} {n_:>9,} {b:>+9.4f} {c:>10.1f} "
            f"{g:>9.1f} {smc[0]:>+9.4f}")
log(f"\n  ※ 보정상한 합 {total_gain:.1f} (그룹 중복 있으므로 상한의 상한)")

log("\n" + "=" * 92)
log("B. 멀티클래스가 이진을 이긴 구간 (+36.1의 출처)")
log("=" * 92)
d_mc = (P["CS79"] - y) ** 2 - (P["MC"] - y) ** 2      # 양수 = MC 승
log(f"  {'그룹':>10s} {'구간':>10s} {'행수':>9s} {'MC 이득':>9s} {'행당':>10s}")
for gname, gv in GROUPS.items():
    rows = []
    for lvl in pd.Series(gv).dropna().unique():
        m = (pd.Series(gv) == lvl).to_numpy()
        if m.sum() < 500:
            continue
        gain = 100000 * d_mc[m].sum() / (DEN * len(y))
        rows.append((str(lvl), int(m.sum()), gain, gain / m.sum() * 1e4))
    rows.sort(key=lambda x: -x[2])
    for lvl, n_, gain, per in rows[:4]:
        log(f"  {gname:>10s} {lvl:>10s} {n_:>9,} {gain:>+9.1f} {per:>+10.3f}")

log("\n" + "=" * 92)
log("C. CS79가 CS76보다 나빠진 구간 (구종 피처의 부작용 탐지)")
log("=" * 92)
d_79 = (P["CS76"] - y) ** 2 - (P["CS79"] - y) ** 2
worst = []
for gname, gv in GROUPS.items():
    for lvl in pd.Series(gv).dropna().unique():
        m = (pd.Series(gv) == lvl).to_numpy()
        if m.sum() < 500:
            continue
        gain = 100000 * d_79[m].sum() / (DEN * len(y))
        worst.append((gname, str(lvl), int(m.sum()), gain))
worst.sort(key=lambda x: x[3])
log(f"  {'그룹':>10s} {'구간':>10s} {'행수':>9s} {'CS79 이득':>10s}")
for gname, lvl, n_, gain in worst[:8]:
    log(f"  {gname:>10s} {lvl:>10s} {n_:>9,} {gain:>+10.1f}")
log("  (음수 = 구종 피처가 그 구간에서 해로웠음)")
log("\n" + "=" * 92)
