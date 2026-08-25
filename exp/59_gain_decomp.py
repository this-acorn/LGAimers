# -*- coding: utf-8 -*-
"""
[59] 이득 분해 — 멀티클래스 +36.1이 진짜 정보인가, 2024 평균을 우연히 맞힌 것인가
     + cs_n 구간별 CS79 vs CB65 (전문가 혼합 생존 여부)  [순수 계산, 학습 없음]

분해:
  총점 = 변별력 − 중심벌점,   중심벌점 = 100000·(p̄−ȳ)²/DEN
  · 평균을 서로 맞춘 뒤에도 남는 차이 = 진짜 shape(정보) 이득
  · 평균 통일 후 사라지면 = 2024 전역 오프셋을 우연히 맞힌 것 → LB 전이 위험

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/59_gain_decomp.py  (~2분)
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "exp")
from common import log, load_train

df = load_train()
va = df[df.season == 2024].reset_index(drop=True)
y = va["control_success"].to_numpy("float64")
r = y.mean()
DEN = r * (1 - r)

P = {"CB65": np.load("lab/36_ref_ens.npy").astype("float64"),
     "CS76": np.load("lab/41_cs_ens.npy").astype("float64"),
     "CS79": np.load("lab/45_both_ens.npy").astype("float64"),
     "MC": np.load("lab/53_mc_ens.npy").astype("float64")}


def sc(p):
    return 100000 * (1 - np.mean((p - y) ** 2) / DEN)


def decomp(p):
    d = p.mean() - r
    pen = 100000 * d * d / DEN
    return sc(p), sc(p) + pen, pen, p.mean(), p.std()


log("=" * 88)
log("A. 총점 = 변별력 − 중심벌점  (2024 폴드, 실제 r=%.4f)" % r)
log("=" * 88)
log(f"  {'모델':8s} {'총점':>9s} {'변별력':>9s} {'중심벌점':>9s} {'예측평균':>9s} {'예측std':>8s}")
D = {}
for k, p in P.items():
    D[k] = decomp(p)
    log(f"  {k:8s} {D[k][0]:>9.1f} {D[k][1]:>9.1f} {D[k][2]:>9.1f} "
        f"{D[k][3]:>9.4f} {D[k][4]:>8.4f}")

log("\n" + "=" * 88)
log("B. ★ 멀티클래스 +36.1의 분해 (CS79 기준)")
log("=" * 88)
raw = D["MC"][0] - D["CS79"][0]
shape = D["MC"][1] - D["CS79"][1]
center = D["CS79"][2] - D["MC"][2]
log(f"  원시 이득            {raw:+7.1f}")
log(f"  ├ 중심(평균) 기여    {center:+7.1f}   ← 2024 평균을 더 잘 맞힌 몫 (전이 불확실)")
log(f"  └ 변별력(정보) 기여  {shape:+7.1f}   ← 진짜 shape 이득 (전이 기대)")

# 검증: MC 평균을 CS79 평균과 동일하게 강제 이동 후 재비교
shift = D["CS79"][3] - D["MC"][3]
mc_shifted = np.clip(P["MC"] + shift, 0, 1)
log(f"\n  [교차검증] MC 평균을 CS79와 동일하게 이동(+{shift:.4f}) 후:")
log(f"    MC(평균통일) {sc(mc_shifted):8.1f}  vs CS79 {D['CS79'][0]:8.1f}  "
    f"→ 차이 {sc(mc_shifted)-D['CS79'][0]:+.1f}")
log(f"\n  [완전보정 비교] 둘 다 실제 r로 중심 이동 시:")
for k in ["CS79", "MC"]:
    pc = np.clip(P[k] - (P[k].mean() - r), 0, 1)
    log(f"    {k:6s} {sc(pc):8.1f}")

log("\n  판독: 변별력 기여가 +20↑ → 진짜 정보, LB 전이 기대")
log("        +5~15 → 정보와 우연 보정 혼재 / ~0 → 전이 위험 큼")

# =====================================================================
log("\n" + "=" * 88)
log("C. cs_n 구간별 CS79 vs CB65 — 전문가 혼합 생존 여부")
log("=" * 88)
hist = df[df.season <= 2023]
last = hist.sort_values("asof_pitcher_n").groupby("pitcher_id").tail(1)
Nend = dict(zip(last.pitcher_id, last.asof_pitcher_n.fillna(0) + 1))
cs_n = np.maximum(va.asof_pitcher_n.fillna(0).to_numpy("float64")
                  - va.pitcher_id.map(Nend).fillna(0).to_numpy("float64"), 0.0)
bins = pd.cut(cs_n, [-1, 0, 10, 30, 100, 300, 1000, 1e9],
              labels=["0", "1-10", "11-30", "31-100", "101-300", "301-1000", "1000+"])
log(f"  {'cs_n':>10s} {'행수':>9s} {'CB65':>9s} {'CS79':>9s} {'MC':>9s} "
    f"{'CS79-CB65':>10s} {'MC-CS79':>9s}")
for lvl in bins.categories:
    m = np.asarray(bins == lvl)
    if m.sum() < 300:
        continue
    ys = y[m]
    rr = ys.mean()
    dn = rr * (1 - rr)
    s = {k: 100000 * (1 - np.mean((P[k][m] - ys) ** 2) / dn) for k in P}
    log(f"  {str(lvl):>10s} {m.sum():>9,} {s['CB65']:>9.1f} {s['CS79']:>9.1f} "
        f"{s['MC']:>9.1f} {s['CS79']-s['CB65']:>+10.1f} {s['MC']-s['CS79']:>+9.1f}")
log("\n  판독: 낮은 cs_n에서 CB65가 CS79를 이기면 → 전문가 혼합 생존")
log("        전 구간 CS79 우세면 → 완전 기각")
log("=" * 88)
