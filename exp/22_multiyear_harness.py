# -*- coding: utf-8 -*-
"""
[22] 다연도 paired 검증 하네스 — 앞으로 모든 피처 판정의 표준 절차

배경 (왜 이게 0순위인가):
  · 로컬↔LB 전이율 0.958이 반증됐다. 데이터포인트 3개:
      415.6→549.51 (+133.9) / 708.7→830.32 (+121.6) / 726.2→808.09 (+81.9 ← 이탈)
    이제 로컬 점수로 LB를 예측할 방법이 없다. 다연도 일관성이 유일한 증거원이다.
  · 2024는 이미 홀드아웃이 아니다. G2 매치업 선택, 노이즈바닥, 하이퍼파라미터,
    ablation 전부 2024 정답을 보고 결정했다. '독립 3회'는 독립 시드였을 뿐이다.

★ 채택 기준을 왜 리뷰 제안보다 엄하게 잡는가:
  hand delta의 연도별 이득 = [2021 +8.7, 2022 +4.9, 2023 +35.4, 2024 +18.1]
    평균 +16.8 / 중앙값 +13.4 / 최악 +4.9 / t=2.47 (n=4)
  → "4년 중 3년 양수 + 평균 양수 + 붕괴 없음" 기준을 **4개 전부 통과**한다.
  → 그런데 실제 2025 리더보드는 -22.2 였다.
  그러므로 그 기준은 이미 한 번 실패한 기준이다. 아래 위험등급별로 분리한다.

★ 위험등급 (이 프로젝트에서 실측으로 갈린 두 계열):
  [S] 구조형  — 행 안의 값 비교/산술, 또는 주최측 asof_* 로부터 계산.
               asof_*는 2025 시점 기준으로 주최측이 다시 계산해 주므로 레짐이 최신이다.
               예: f_same_hand(+107, 전이 성공), pb_mean/pb_logit_add
  [F] 동결테이블형 — train(≤2024)에서 우리가 집계해 model.pkl에 얼려 넣고
               2025 행에 조인하는 값. 타겟 판정기준이 해마다 바뀌는데
               (r: 0.5647→0.4861 단조 하락) 테이블만 옛 레짐에 고정된다.
               예: hand delta(-22.2 실패), CatBoost의 선수 범주 타겟통계, trackman 집계
  → [F] 계열은 다연도 통과해도 신뢰도가 낮다. 임계를 크게 올린다.

이번 배치 (전부 [S] 계열):
  base65     현재 제출 구성 (LB 830.322760105 확정)
  no_pid     pitcher_id / batter_id 제거 — 수치 ID 분할이 도움인가 해악인가
             ※ corr(pitcher_id, 첫등장시즌)=+0.704 → ID 분할은 사실상 '데뷔시기' 분할
  pb_add     f_pb_mean, f_pb_logit_add 추가 — 확률 결합의 정석 형태(로그오즈 덧셈)

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/22_multiyear_harness.py
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "exp")
from common import (log, tick, raw_score, load_train, add_features,
                    fit_encoder, to_matrix, ALL_ENG, HGB_FAST)

YEARS = [2021, 2022, 2023, 2024]
SEEDS = [42, 7, 123, 2024]
SMOOTH_A = 200.0

# hand delta 실측 기준점 — 이 값들을 통과시킨 기준은 이미 실패했다
HAND_REF = {"mean": 16.8, "min": 4.9, "lb": -22.2}


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def add_pb_features(d, prior):
    """[S] 구조형. f_smooth_p/f_smooth_b(주최측 asof에서 계산)로부터 만든다.

    기존 f_pb_diff = p - b 는 '차이'인데, 두 성공률은 타겟에 같은 방향으로 작용한다.
    자연스러운 결합은 차이가 아니라 합이며, 확률의 합은 로그오즈에서 덧셈이다.
    트리는 두 연속값의 합을 축평행 계단으로만 근사하므로 직접 만들기 어렵다.
    """
    d = d.copy()
    pp, pb = d["f_smooth_p"], d["f_smooth_b"]
    d["f_pb_mean"] = ((pp + pb) / 2.0).astype("float32")
    d["f_pb_logit_add"] = (logit(pp) + logit(pb) - logit(prior)).astype("float32")
    return d


PB_ENG = ["f_pb_mean", "f_pb_logit_add"]

# =====================================================================
log("train.csv 로딩 중...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS65 = BASE + ALL_ENG
FEATS_NOPID = [c for c in FEATS65 if c not in ("pitcher_id", "batter_id")]
FEATS_PB = FEATS65 + PB_ENG

ARMS = [
    ("base65", FEATS65, "-"),
    ("no_pid", FEATS_NOPID, "S"),
    ("pb_add", FEATS_PB, "S"),
]
tick(f"로드 완료 {df.shape} — 팔 {len(ARMS)}개 × 연도 {len(YEARS)}개 × 시드 {len(SEEDS)}개 "
     f"= {len(ARMS)*len(YEARS)*len(SEEDS)}회 학습")
log(f"  base65    {len(FEATS65)}피처 (현재 제출 = LB 830.322760105)")
log(f"  no_pid    {len(FEATS_NOPID)}피처 (pitcher_id/batter_id 제거)")
log(f"  pb_add    {len(FEATS_PB)}피처 (+pb_mean, +pb_logit_add)")

results = {}          # (year, arm) -> {"seed_scores": [...], "ens": float}

for Y in YEARS:
    tr_raw = df[df["season"] <= Y - 1].reset_index(drop=True)
    va_raw = df[df["season"] == Y].reset_index(drop=True)
    y_tr = tr_raw["control_success"].to_numpy()
    y_va = va_raw["control_success"].to_numpy()
    prior = float(y_tr.mean())

    tr = add_pb_features(add_features(tr_raw, prior), prior)
    va = add_pb_features(add_features(va_raw, prior), prior)
    enc = fit_encoder(tr, FEATS65)          # CAT 3개는 모든 팔에서 동일
    log(f"\n{'='*86}")
    log(f"검증연도 {Y}  학습 {len(tr):,} / 검증 {len(va):,}  (prior={prior:.4f}, "
        f"실제 r={y_va.mean():.4f})")
    log("=" * 86)

    for arm_name, feats, _cls in ARMS:
        Xtr = to_matrix(tr, feats, enc)
        Xva = to_matrix(va, feats, enc)
        seed_scores, preds = [], []
        for s in SEEDS:
            tick(f"{Y} {arm_name} seed={s}")
            m = HistGradientBoostingClassifier(**{**HGB_FAST, "random_state": s}).fit(Xtr, y_tr)
            p = m.predict_proba(Xva)[:, 1]
            preds.append(p)
            seed_scores.append(raw_score(p, y_va))
        ens = raw_score(np.mean(preds, axis=0), y_va)
        results[(Y, arm_name)] = {"seed_scores": seed_scores, "ens": ens}
        log(f"    {arm_name:8s} 시드별 {[f'{v:.0f}' for v in seed_scores]}  앙상블 {ens:7.1f}")
    del tr, va, tr_raw, va_raw

# =====================================================================
log("\n" + "=" * 86)
log("연도별 paired 결과  (같은 시드끼리 짝지어 차이를 낸다 → 모델 노이즈 상쇄)")
log("=" * 86)

summary = {}
for arm_name, _f, cls in ARMS:
    if arm_name == "base65":
        continue
    log(f"\n■ {arm_name}   [위험등급 {cls}]")
    log(f"  {'검증연도':>8s} {'paired 평균이득':>15s} {'시드별 이득범위':>20s} {'앙상블 이득':>12s}")
    per_year = []
    for Y in YEARS:
        b = results[(Y, "base65")]
        c = results[(Y, arm_name)]
        pg = [ci - bi for ci, bi in zip(c["seed_scores"], b["seed_scores"])]
        eg = c["ens"] - b["ens"]
        per_year.append(np.mean(pg))
        log(f"  {Y:>8d} {np.mean(pg):>+15.1f} "
            f"{f'[{min(pg):+.1f}, {max(pg):+.1f}]':>20s} {eg:>+12.1f}")
    a = np.array(per_year)
    summary[arm_name] = a
    se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")
    log(f"  {'─'*66}")
    log(f"  평균 {a.mean():+.1f} / 중앙값 {np.median(a):+.1f} / 최악 {a.min():+.1f} / "
        f"최고 {a.max():+.1f} / 양수 {(a>0).sum()}/{len(a)}")
    log(f"  연도간 표준편차 {a.std(ddof=1):.1f}  표준오차 {se:.1f}  t={a.mean()/se:+.2f}")

# =====================================================================
log("\n" + "=" * 86)
log("채택 판정")
log("=" * 86)
log(f"  ※ 보정 기준점: hand delta는 평균 {HAND_REF['mean']:+.1f} / 최악 "
    f"{HAND_REF['min']:+.1f} / 4년 전부 양수 / t=2.47 로 통과했지만")
log(f"     실제 2025 리더보드에서 {HAND_REF['lb']:+.1f} 였다. 그 수준은 증거가 아니다.\n")

for arm_name, _f, cls in ARMS:
    if arm_name == "base65":
        continue
    a = summary[arm_name]
    npos, mean_g, min_g = int((a > 0).sum()), a.mean(), a.min()
    if cls == "S":
        ok = (npos >= 3) and (mean_g >= 10.0) and (min_g >= -10.0)
        rule = "[S] 구조형: 4년중 3년+ 양수 & 평균≥+10 & 최악≥-10"
    else:
        ok = (npos == 4) and (mean_g >= 40.0) and (min_g >= 15.0)
        rule = "[F] 동결형: 4년 전부 양수 & 평균≥+40 & 최악≥+15 (hand delta 2.4배)"
    log(f"  {arm_name:8s} {rule}")
    log(f"           양수 {npos}/4, 평균 {mean_g:+.1f}, 최악 {min_g:+.1f}  →  "
        f"{'채택 후보 (독립시드 재확인 권장)' if ok else '기각 / 보류'}")

log("\n" + "=" * 86)
log("다음: CatBoost는 [F] 동결형이므로 이 하네스로 반드시 검증할 것.")
log("      ID를 수치+범주 두 벌로 넣고, has_time=True(시즌 정렬) 팔도 함께 볼 것 —")
log("      CatBoost 기본값은 '무작위 순열' 타겟통계라 시즌 경계를 존중하지 않는다.")
log("=" * 86)
