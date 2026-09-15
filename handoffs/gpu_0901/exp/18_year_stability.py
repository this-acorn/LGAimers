"""
[18] hand delta 연도 안정성 진단 — 리더보드 역행(-22.2)의 원인 규명

실행:  PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 py -3.12 -u exp/18_year_stability.py

사건:
  hand delta는 2024 검증에서 독립 3회 +13.9/+17.5/+22.9로 채택됐다.
  그런데 실제 2025 리더보드에서 830.32 → 808.09 (-22.2)로 역행했다.
  제출 경로 버그 가능성은 낮음(가짜 서버 245,789행 조인 정상, 조인 실패였다면
  NaN→기존 모델과 비슷한 점수가 나왔어야 함).

가설: 손 매치업 delta는 연도 안정성이 없는 신호다.
  2024 한 해로만 검증했는데, 이 대회 타겟은 해마다 판정 기준이 바뀐다
  (r 계단식 하락, 시즌 내 방향 제각각). ±0.02짜리 신호가 연 단위 변화에 취약할 수 있다.

검증: Y = 2021 / 2022 / 2023 각각에 대해
  train ≤Y-1 → validate Y, 기준(65피처) vs +hand delta(67피처), 4시드.
  2024 결과(+13.9/+17.5/+22.9)와 합쳐 연도별 이득 분포를 본다.

판정:
  전 연도 일관 양수 → 2025만 특이 (불운/2025 기준 변화) — 피처 자체는 정상
  연도별 널뛰기(음수 섞임) → 연도 불안정 신호 — 피처 철회, 제출 65피처로 복귀
  그리고 어느 쪽이든: 앞으로 모든 새 피처는 다연도 안정성 검사를 통과해야 채택
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "exp")
from common import (log, tick, decompose, load_train, add_features,
                    fit_encoder, to_matrix, ALL_ENG, HAND_ENG, HGB_FAST,
                    hand_history_table, attach_hand_deltas_asof)

SEEDS = [42, 7, 123, 2024]
YEARS = [2021, 2022, 2023]
KNOWN_2024 = [13.9, 17.5, 22.9]   # 이미 측정된 2024 검증 이득 (독립 3회)

log("train.csv 로딩 중...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS65 = BASE + ALL_ENG
FEATS67 = FEATS65 + HAND_ENG


def run_year(Y):
    tr_raw = df[df["season"] <= Y - 1].reset_index(drop=True)
    va_raw = df[df["season"] == Y].reset_index(drop=True)
    y_tr = tr_raw["control_success"].to_numpy()
    y_va = va_raw["control_success"].to_numpy()
    prior = float(y_tr.mean())
    tbl_p = hand_history_table(df, prior, "pitcher_id")
    tbl_b = hand_history_table(df, prior, "batter_id")
    tr = attach_hand_deltas_asof(add_features(tr_raw, prior), tbl_p, tbl_b)
    va = attach_hand_deltas_asof(add_features(va_raw, prior), tbl_p, tbl_b)
    log(f"\n  === 검증연도 {Y}: 학습 {len(tr):,} / 검증 {len(va):,} "
        f"(va delta 커버리지 {va['f_hand_delta'].notna().mean()*100:.0f}%) ===")
    out = {}
    for nm, feats in [("기준65", FEATS65), ("+hand67", FEATS67)]:
        enc = fit_encoder(tr, feats)
        Xtr, Xva = to_matrix(tr, feats, enc), to_matrix(va, feats, enc)
        acc = np.zeros(len(va))
        for i, s in enumerate(SEEDS, 1):
            tick(f"{Y} {nm} seed={s} ({i}/{len(SEEDS)})...")
            m = HistGradientBoostingClassifier(
                **{**HGB_FAST, "random_state": s}).fit(Xtr, y_tr)
            acc += m.predict_proba(Xva)[:, 1]
        z = decompose(acc / len(SEEDS), y_va)
        out[nm] = z
        log(f"    {nm:10s} 총점 {z['총점']:7.1f}  변별력 {z['변별력']:7.1f}")
    gain = out["+hand67"]["총점"] - out["기준65"]["총점"]
    gain_d = out["+hand67"]["변별력"] - out["기준65"]["변별력"]
    log(f"    ▶ {Y} 이득: 총점 {gain:+.1f}  변별력 {gain_d:+.1f}")
    return gain, gain_d


results = {}
for Y in YEARS:
    results[Y] = run_year(Y)

log("\n" + "=" * 88)
log("연도별 이득 분포")
log("=" * 88)
log(f"  {'검증연도':>6s} {'총점이득':>9s} {'변별력이득':>10s}")
for Y in YEARS:
    g, gd = results[Y]
    log(f"  {Y:6d} {g:+9.1f} {gd:+10.1f}")
log(f"  {2024:6d}   (기측정: {', '.join(f'{v:+.1f}' for v in KNOWN_2024)} — 독립 3회)")
log(f"  {2025:6d}     -22.2   (실제 리더보드, 전이율 0.958 역산 시 로컬 약 -23)")

gains = [results[Y][0] for Y in YEARS] + [float(np.mean(KNOWN_2024))]
log(f"\n  2021~2024 이득: 평균 {np.mean(gains):+.1f}  표준편차 {np.std(gains):.1f}  "
    f"최소 {min(gains):+.1f}  최대 {max(gains):+.1f}")
neg = sum(1 for g in gains for _ in [0] if g < 0) + 1   # 2025 포함 음수 개수
log("\n" + "=" * 88)
log("판정 가이드")
log("=" * 88)
log("  전 연도(21~24) 일관 양수 + 2025만 음수 → 2025 특이. 피처 자체는 건전하나")
log("      2025에 배신당한 것 — 제출은 65피처로 복귀하되 원인(기준 변화) 기록")
log("  연도별 부호 널뛰기 → 연도 불안정 신호 확정 — 피처 철회 + 검증 프로토콜에")
log("      '다연도 안정성 검사' 추가 (모든 새 피처는 3개 연도 이상 일관성 요구)")
log("=" * 88)
