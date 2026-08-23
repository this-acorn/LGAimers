# -*- coding: utf-8 -*-
"""
[24] no_pid 효과 분해 — '어디서 이기는가'를 찾고, 그걸 배포 가능한 형태로 바꾼다

exp/22 결과: no_pid 연도별 이득 +44.0 / +1.8 / +54.7 / -4.5 (평균 +24.0)
  체크리스트는 통과했지만 df=3에 학습셋이 중첩(2021⊂2022⊂2023⊂2024)이라
  t값은 참고치일 뿐이고, '전부 빼기 vs 전부 두기' 이분법 자체가 조악하다.

이 스크립트가 답하는 것:
  A. 투수 ID와 타자 ID 중 어느 쪽이 문제인가  (지금까지 둘을 같이 빼서 효과가 섞였다)
  B. 어떤 행에서 ID 제거가 이득인가 — 표본수(asof_n) · 신인 여부 · 데뷔세대별 분해
  C. 전량 제거 대신 혼합(base ⊗ no_pid)이 더 나은가, 최적 비율은 연도 간 안정적인가

★ 왜 이게 '새 시드 재측정'보다 먼저인가:
  · 시드 재측정은 노이즈만 줄인다. 왜 2022/2024에서 졌는지는 알려주지 않는다.
  · 반면 구간을 찾아내면 asof_pitcher_n 조건부 가중이라는 **배포 가능한 피처**가 나온다.
    asof_pitcher_n은 주최측 공식 컬럼이고 그 행 자신의 값이므로 행 독립성 위반이 아니다.

★ exp/22가 예측 배열을 버렸으므로(점수만 저장) 여기서 재학습하되,
  혼합 스윕에 필요한 예측을 **같은 실행에서** 전부 저장해 재학습을 두 번 하지 않는다.

팔 4개 (전부 [S] 구조형 — 동결 테이블 없음):
  base65        기준 (현재 제출, LB 830.322760105)
  no_pitcher    pitcher_id만 제거
  no_batter     batter_id만 제거
  no_both       둘 다 제거 (= exp/22의 no_pid, 재현 확인용)

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/24_nopid_decomp.py
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "exp")
from common import (log, tick, raw_score, load_train, add_features,
                    fit_encoder, to_matrix, ALL_ENG, HGB_FAST)

YEARS = [2021, 2022, 2023, 2024]
SEEDS = [42, 7, 123, 2024]      # exp/22와 동일 — 재현 확인이 목적이므로 일부러 같게
# 혼합비율은 2021~2023에서 고르고 2024에서 확인한다 (선택연도와 확인연도 분리)
PICK_YEARS, CONFIRM_YEAR = [2021, 2022, 2023], 2024
W_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

log("train.csv 로딩 중...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS65 = BASE + ALL_ENG

ARMS = [
    ("base65", FEATS65),
    ("no_pitcher", [c for c in FEATS65 if c != "pitcher_id"]),
    ("no_batter", [c for c in FEATS65 if c != "batter_id"]),
    ("no_both", [c for c in FEATS65 if c not in ("pitcher_id", "batter_id")]),
]
tick(f"로드 {df.shape} — 팔 {len(ARMS)} × 연도 {len(YEARS)} × 시드 {len(SEEDS)} "
     f"= {len(ARMS)*len(YEARS)*len(SEEDS)}회 학습")

# 각 연도의 앙상블 예측과 검증 메타데이터를 보관 (혼합 스윕·구간분해용)
store = {}     # year -> {"y":..., "meta": DataFrame, "preds": {arm: ndarray}}

for Y in YEARS:
    tr_raw = df[df["season"] <= Y - 1].reset_index(drop=True)
    va_raw = df[df["season"] == Y].reset_index(drop=True)
    y_tr = tr_raw["control_success"].to_numpy()
    y_va = va_raw["control_success"].to_numpy()
    prior = float(y_tr.mean())
    tr, va = add_features(tr_raw, prior), add_features(va_raw, prior)
    enc = fit_encoder(tr, FEATS65)

    # 신인 판정: 그 해 검증행의 선수가 학습기간(≤Y-1)에 등장했는가
    seen_p = set(tr_raw["pitcher_id"].unique())
    seen_b = set(tr_raw["batter_id"].unique())
    meta = pd.DataFrame({
        "pitcher_id": va_raw["pitcher_id"].to_numpy(),
        "batter_id": va_raw["batter_id"].to_numpy(),
        "p_n": va["asof_pitcher_n"].fillna(0).to_numpy(),
        "b_n": va["asof_batter_n"].fillna(0).to_numpy(),
        "p_new": (~va_raw["pitcher_id"].isin(seen_p)).to_numpy(),
        "b_new": (~va_raw["batter_id"].isin(seen_b)).to_numpy(),
    })

    log(f"\n{'='*88}")
    log(f"검증연도 {Y}  학습 {len(tr):,} / 검증 {len(va):,}  "
        f"(신인투수행 {meta.p_new.mean()*100:.1f}% / 신인타자행 {meta.b_new.mean()*100:.1f}%)")
    log("=" * 88)

    preds = {}
    for arm_name, feats in ARMS:
        Xtr, Xva = to_matrix(tr, feats, enc), to_matrix(va, feats, enc)
        acc = np.zeros(len(va))
        for s in SEEDS:
            tick(f"{Y} {arm_name} seed={s}")
            m = HistGradientBoostingClassifier(
                **{**HGB_FAST, "random_state": s}).fit(Xtr, y_tr)
            acc += m.predict_proba(Xva)[:, 1]
        preds[arm_name] = acc / len(SEEDS)
        log(f"    {arm_name:11s} 앙상블 {raw_score(preds[arm_name], y_va):8.1f}")
        del Xtr, Xva

    store[Y] = {"y": y_va, "meta": meta, "preds": preds}
    del tr, va, tr_raw, va_raw

# =====================================================================
log("\n" + "=" * 88)
log("A. 투수 ID / 타자 ID 분리 — 어느 쪽이 문제인가")
log("=" * 88)
log(f"  {'연도':>6s} {'no_pitcher':>12s} {'no_batter':>12s} {'no_both':>12s}")
sep = {a: [] for a, _ in ARMS if a != "base65"}
for Y in YEARS:
    s0 = raw_score(store[Y]["preds"]["base65"], store[Y]["y"])
    row = []
    for a in ["no_pitcher", "no_batter", "no_both"]:
        g = raw_score(store[Y]["preds"][a], store[Y]["y"]) - s0
        sep[a].append(g)
        row.append(f"{g:+12.1f}")
    log(f"  {Y:>6d} {''.join(row)}")
log(f"  {'─'*46}")
for a in ["no_pitcher", "no_batter", "no_both"]:
    v = np.array(sep[a])
    log(f"  {a:11s} 평균 {v.mean():+7.1f}  최악 {v.min():+7.1f}  양수 {(v>0).sum()}/4")
log("\n  해석: no_pitcher와 no_batter의 합이 no_both와 크게 다르면 두 ID가 상호작용한다.")

# =====================================================================
log("\n" + "=" * 88)
log("B. 구간 분해 — 행별 이득 = base 제곱오차 - no_both 제곱오차  (양수면 제거가 이득)")
log("=" * 88)


def bucket_report(Y, arm="no_both"):
    d = store[Y]
    y, m = d["y"], d["meta"]
    gain = (d["preds"]["base65"] - y) ** 2 - (d["preds"][arm] - y) ** 2
    r = y.mean()
    scale = 100000.0 / (r * (1 - r))       # 행별 이득 → 점수 단위 환산
    out = []
    pn_bin = pd.cut(m.p_n, [-1, 0, 50, 200, 1000, 1e9],
                    labels=["0", "1-50", "51-200", "201-1000", "1000+"])
    for name, grp in [("투수 asof_n", pn_bin),
                      ("신인투수", m.p_new.map({True: "신인", False: "기존"})),
                      ("신인타자", m.b_new.map({True: "신인", False: "기존"}))]:
        agg = pd.DataFrame({"g": gain, "k": grp}).groupby("k", observed=True)["g"]
        for k, v in agg.agg(["mean", "size"]).iterrows():
            out.append((name, str(k), v["mean"] * scale, int(v["size"])))
    return out


for Y in YEARS:
    log(f"\n  ── {Y} (총 이득 {sep['no_both'][YEARS.index(Y)]:+.1f}) ──")
    log(f"    {'구분':<12s} {'구간':<10s} {'기여(점수단위)':>14s} {'행수':>10s}")
    for name, k, contrib, n in bucket_report(Y):
        log(f"    {name:<12s} {k:<10s} {contrib:>14.1f} {n:>10,}")

# =====================================================================
log("\n" + "=" * 88)
log("C. 혼합 스윕 — 전량 제거 대신 base ⊗ no_both")
log("=" * 88)
log(f"  비율 선택: {PICK_YEARS} / 확인: {CONFIRM_YEAR}  (선택연도와 확인연도 분리)")
log(f"\n  {'w':>5s} " + " ".join(f"{Y:>9d}" for Y in YEARS) + f" {'선택평균':>10s}")
best_w, best_v = 0.0, -1e9
for w in W_GRID:
    gains = []
    for Y in YEARS:
        d = store[Y]
        p = (1 - w) * d["preds"]["base65"] + w * d["preds"]["no_both"]
        gains.append(raw_score(p, d["y"]) - raw_score(d["preds"]["base65"], d["y"]))
    pick = float(np.mean([gains[YEARS.index(Y)] for Y in PICK_YEARS]))
    log(f"  {w:>5.1f} " + " ".join(f"{g:>+9.1f}" for g in gains) + f" {pick:>+10.1f}")
    if pick > best_v:
        best_w, best_v = w, pick

conf = [g for g in [None]]
d = store[CONFIRM_YEAR]
p = (1 - best_w) * d["preds"]["base65"] + best_w * d["preds"]["no_both"]
conf_gain = raw_score(p, d["y"]) - raw_score(d["preds"]["base65"], d["y"])
log(f"\n  선택된 w = {best_w:.1f}  (2021~2023 평균 이득 {best_v:+.1f})")
log(f"  → {CONFIRM_YEAR} 확인 이득 {conf_gain:+.1f}   "
    f"(no_both 전량 제거는 {sep['no_both'][YEARS.index(CONFIRM_YEAR)]:+.1f} 였다)")
log(f"\n  판정: 확인연도 이득이 0 이상이면서 전량제거보다 나으면 혼합이 답이다.")
log(f"        w가 0 또는 1 끝으로 몰리면 혼합은 의미 없다.")
log("=" * 88)
