# -*- coding: utf-8 -*-
"""
[76] thread_count 결정성 직접 측정 — 인용이 아니라 이 기계에서 재는다

배경: HANDOFF §4-2 와 내 이전 판단은 증류 위양성(+37.3 → +0.5)의 원인을
  thread_count 6 vs 14 로 귀속했다. 리뷰어는 CatBoost 공식 문서가 CPU 의
  thread_count 에 대해 "속도만 최적화하고 결과에 영향 없음"이라 명시한다고 반박했다.
  문서와 실제가 다를 수 있으므로 **직접 잰다.** 오프라인이라 문서 확인은 불가능하고,
  어차피 우리에게 필요한 건 "이 버전, 이 CPU, 이 데이터에서 그런가"이다.

동시에 진짜 용의자도 같이 잰다:
  exp/57:96   df[c] = ...astype("Int64")   <- 원본 df 의 dtype 을 변형
  exp/64:95   "df 의 dtype 을 바꾸지 않는다 — 사본에서만 캐스팅"
  dtype 변형은 CatBoost 입력값 자체를 바꾸므로 훨씬 유력한 원인이다.

측정 (전부 같은 시드·같은 행·같은 파라미터, 한 프로세스 안에서):
  A. thread 14 를 두 번    -> 같은 스레드면 완전 재현되는가 (결정성 바닥)
  B. thread 14 vs thread 8 -> 스레드가 결과를 바꾸는가            ★핵심
  C. thread 14 vs thread 6 -> exp/57 이 쓴 값
  D. float64 vs Int64 캐스팅 -> dtype 변형이 결과를 바꾸는가      ★진짜 용의자

exp/75 가 14스레드로 돌고 있으므로 가볍게 잡는다 (20만행, 100라운드).
실행: PYTHONIOENCODING=utf-8 python -u exp/76_thread_determinism.py   (~5분)
"""

import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import log

N = 200_000
SEED = 42
CAT = ["top_bottom", "game_type", "base_state"]
NUM = ["balls_before", "strikes_before", "outs_before", "inning", "num_runners_on",
       "runner_on_1b", "runner_on_2b", "runner_on_3b", "li", "run_total_before",
       "score_diff_pitcher_team", "home_win_expectancy", "asof_pitcher_n",
       "asof_pitcher_success_rate", "asof_pitcher_middle_rate",
       "asof_pitcher_ball_rate", "asof_pitcher_strike_rate", "asof_batter_n",
       "asof_batter_success_rate", "pitcher_id", "batter_id",
       "pitcher_team_id", "batter_team_id"]
PRM = dict(iterations=100, depth=6, learning_rate=0.08, l2_leaf_reg=10.0,
           verbose=False, allow_writing_files=False, random_seed=SEED)

log(f"train.csv 상위 {N:,}행 로딩 (필요 컬럼만)...")
df = pd.read_csv("data/train.csv", encoding="utf-8-sig",
                 usecols=NUM + CAT + ["control_success"], nrows=N)
df.columns = [c.replace("﻿", "").strip() for c in df.columns]
y = df["control_success"].to_numpy()
log(f"  {len(df):,}행 · 수치 {len(NUM)} + 범주 {len(CAT)}  성공률 {y.mean():.4f}")


def frame(int64_keys=False):
    X = df[NUM].copy()
    if int64_keys:
        # exp/57 이 원본 df 에 했던 것과 동일한 변형
        for c in ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("Int64")
    for c in CAT:
        X[c] = df[c].astype(str)
    return X


def fit_predict(threads, int64_keys=False):
    X = frame(int64_keys)
    m = CatBoostClassifier(**PRM, thread_count=threads).fit(
        Pool(X, y, cat_features=CAT))
    return m.predict_proba(Pool(X, cat_features=CAT))[:, 1]


log("")
log("=" * 84)
log("측정")
log("=" * 84)
runs = {}
for tag, th, i64 in [("t14_a", 14, False), ("t14_b", 14, False),
                     ("t8", 8, False), ("t6", 6, False),
                     ("t14_int64", 14, True)]:
    import time
    t0 = time.time()
    runs[tag] = fit_predict(th, i64)
    log(f"  {tag:10s} thread={th:2d} int64={str(i64):5s}  "
        f"({time.time()-t0:5.1f}s)  예측평균 {runs[tag].mean():.10f}")


def cmp(a, b, label):
    d = np.abs(runs[a] - runs[b])
    ident = bool(np.array_equal(runs[a], runs[b]))
    # 점수 영향 (이 부분집합 기준)
    r = y.mean()
    DEN = r * (1 - r)
    sa = 100000 * (1 - np.mean((runs[a] - y) ** 2) / DEN)
    sb = 100000 * (1 - np.mean((runs[b] - y) ** 2) / DEN)
    log(f"  {label:34s} 완전동일={str(ident):5s}  최대차 {d.max():.3e}  "
        f"평균차 {d.mean():.3e}  점수차 {sb-sa:+8.2f}")
    return ident, d.max(), sb - sa


log("")
log("=" * 84)
log("비교")
log("=" * 84)
same_thread = cmp("t14_a", "t14_b", "A. thread14 vs thread14 (결정성 바닥)")
th8 = cmp("t14_a", "t8", "B. thread14 vs thread8  ★핵심")
th6 = cmp("t14_a", "t6", "C. thread14 vs thread6  (exp/57 값)")
dt = cmp("t14_a", "t14_int64", "D. float vs Int64 캐스팅 ★진짜 용의자")

log("")
log("=" * 84)
log("판정")
log("=" * 84)
if not same_thread[0]:
    log("  ⚠ 같은 스레드에서도 재현이 안 된다 — 이 비교 자체가 무의미. 다른 원인 조사 필요")
else:
    log("  · 같은 스레드에서는 완전 재현됨 (결정성 확인)")
    if th8[0] and th6[0]:
        log("  ★ 스레드 수는 결과를 바꾸지 않는다 — 리뷰어 지적이 옳다.")
        log("    HANDOFF §4-2 의 'thread_count 가 바뀌면 결과가 바뀐다' 항목은 오류.")
        log("    exp/62 의 +10.9 를 '스레드 인공물'이라 부른 것도 철회해야 한다.")
    else:
        log("  ★ 스레드 수가 결과를 바꾼다 — 문서와 달리 이 환경에서는 비결정적.")
        log(f"    최대차 thread8 {th8[1]:.3e} / thread6 {th6[1]:.3e}")
    if not dt[0]:
        log(f"  ★ dtype 캐스팅은 결과를 바꾼다 (최대차 {dt[1]:.3e}, "
            f"점수차 {dt[2]:+.1f}) — 증류 위양성의 유력한 진짜 원인.")
    else:
        log("  · dtype 캐스팅도 결과를 바꾸지 않는다 — 위양성 원인은 제3의 것.")
log("")
log("  ※ 100라운드/20만행이라 절대 점수차는 작게 나온다. 판정은 '완전동일 여부'로 하라.")
log("    500라운드/122만행에서는 같은 방향의 차이가 증폭된다.")
log("=" * 84)
