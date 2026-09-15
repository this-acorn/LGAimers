"""
[05] 피처 2차 — 가지치기 + 새 조합 검증

실행:  PYTHONIOENCODING=utf-8 py -3.12 -u exp/05_features_v2.py

04 ablation이 알려준 원칙:
  "트리가 구조적으로 못 하는 계산"만 피처로 가치가 있다.
    ✅ A == B  (비교)   → 트리는 두 컬럼을 동시에 못 봄
    ✅ A − B   (차이)   → 트리는 뺄셈을 못 함
    ❌ log(A)           → 단조변환 불변. 같은 행을 나눔
    ❌ 결측 플래그       → GBDT가 NaN을 이미 처리
    ❌ 원본 재조합       → 트리가 알아서 함

검증할 4가지:
  v1  현재 65피처                     (기준, 741.8 재현되어야 함)
  v2  가지치기 54피처  (G1·G3·G6 제거)  → 쓰레기 11개를 버리면 오르나?
  v3  v2 + 새 '차이' 피처 5개          → 차이 피처를 더 넣으면 오르나?
  v4  v3 + '매치업×상황' 조합 3개       → 명시적 상호작용이 도움이 되나?

v4가 v3보다 안 오르면 → 트리가 상호작용은 스스로 한다는 뜻 (원칙 정교화)
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "exp")
from common import (log, tick, decompose, load_train, add_features,
                    fit_encoder, to_matrix, FEATURE_GROUPS, ALL_ENG,
                    HGB_FAST, LB_BASELINE, LB_TOP)

log("train.csv 로딩 중...")
df = load_train()
tick(f"로딩 완료 {df.shape}")

BASE = [c for c in df.columns if c not in ("row_id", "control_success")]

tr = df[df["season"] <= 2023].reset_index(drop=True)
va = df[df["season"] == 2024].reset_index(drop=True)
y_tr = tr["control_success"].to_numpy()
y_va = va["control_success"].to_numpy()
PRIOR = float(y_tr.mean())

tr = add_features(tr, PRIOR)      # 기존 18개 (04에서 쓴 것과 동일)
va = add_features(va, PRIOR)


# =====================================================================
# 새 피처 — 전부 '트리가 못 하는 계산' 형태로만
# =====================================================================
NEW_DIFF = ["f_hand_combo", "f_pb_middle", "f_strike_ball",
            "f_prev1_prev3", "f_prev3_prev5"]
NEW_INTER = ["f_hand_x_count", "f_hand_x_runner", "f_hand_x_2str"]


def add_v2(d):
    d = d.copy()

    # --- 비교형 (G2가 63점을 만든 그 형태) ---
    # same_hand는 '같다/다르다' 2종뿐. LL/LR/RL/RR 4종으로 세분화한다.
    # 야구에서 좌투좌타와 우투우타는 전혀 다른 상황이라 구분할 가치가 있다.
    d["f_hand_combo"] = (d["pitcher_hand"] * 10 + d["batter_hand"]).astype("int16")

    # --- 차이형 (트리가 뺄셈을 못 하므로) ---
    # middle_rate는 타겟 정의('스트라이크존 가운데'가 실패)와 직결된 지표
    d["f_pb_middle"] = (d["asof_pitcher_middle_rate"]
                        - d["asof_batter_middle_rate"]).astype("float32")
    # 이 투수가 스트라이크 성향인가 볼 성향인가
    d["f_strike_ball"] = (d["asof_pitcher_strike_rate"]
                          - d["asof_pitcher_ball_rate"]).astype("float32")
    # 단계별 추세 (기존 form_trend는 prev1-prev5 한 칸만 봤음)
    d["f_prev1_prev3"] = (d["asof_pitcher_prev1_game_success_rate"]
                          - d["asof_pitcher_prev3_game_success_rate"]).astype("float32")
    d["f_prev3_prev5"] = (d["asof_pitcher_prev3_game_success_rate"]
                          - d["asof_pitcher_prev5_game_success_rate"]).astype("float32")

    # --- 상호작용형 (G2가 '촉매'였으니 직접 엮어본다) ---
    hc = d["f_hand_combo"].astype("int32")
    d["f_hand_x_count"] = (hc * 100 + d["balls_before"] * 10
                           + d["strikes_before"]).astype("int32")
    d["f_hand_x_runner"] = (hc * 10 + d["num_runners_on"]).astype("int32")
    d["f_hand_x_2str"] = (hc * 10 + (d["strikes_before"] == 2)).astype("int32")
    return d


tr = add_v2(tr)
va = add_v2(va)
tick("피처 생성 완료")

# --- 피처 세트 정의 ---
KEEP = FEATURE_GROUPS["G2 매치업/압박"] + FEATURE_GROUPS["G4 최근폼"] + FEATURE_GROUPS["G5 투타차이"]
DROP = FEATURE_GROUPS["G1 볼카운트"] + FEATURE_GROUPS["G3 asof신뢰도"] + FEATURE_GROUPS["G6 결측플래그"]

SETS = {
    "v1 현재 65피처": BASE + ALL_ENG,
    "v2 가지치기": BASE + KEEP,
    "v3 +차이 5개": BASE + KEEP + NEW_DIFF,
    "v4 +상호작용 3개": BASE + KEEP + NEW_DIFF + NEW_INTER,
}

log(f"\n학습 2019~2023 {len(tr):,}행 (r={PRIOR:.6f})  /  검증 2024 {len(va):,}행 "
    f"(r={y_va.mean():.6f})")
log(f"버리는 것 {len(DROP)}개: {', '.join(DROP)}\n")


def run(feats):
    enc = fit_encoder(tr, feats)
    m = HistGradientBoostingClassifier(**HGB_FAST).fit(to_matrix(tr, feats, enc), y_tr)
    p = m.predict_proba(to_matrix(va, feats, enc))[:, 1]
    return decompose(p, y_va)


log("=" * 88)
log("피처 세트 비교")
log("=" * 88)

res = {}
for i, (name, feats) in enumerate(SETS.items(), 1):
    tick(f"{name} 학습 중 ({i}/{len(SETS)})... [{len(feats)}피처]")
    z = run(feats)
    res[name] = (len(feats), z)
    log(f"  {name:20s} {len(feats):3d}피처  총점 {z['총점']:8.1f} = "
        f"변별력 {z['변별력']:7.1f} - 벌점 {z['중심벌점']:6.1f}")

# =====================================================================
log("\n" + "=" * 88)
log("판정")
log("=" * 88)

v1 = res["v1 현재 65피처"][1]["변별력"]
log(f"  {'세트':20s} {'피처수':>6s} {'변별력':>9s} {'v1대비':>9s}")
log("  " + "-" * 50)
for name, (n, z) in res.items():
    log(f"  {name:20s} {n:6d} {z['변별력']:9.1f} {z['변별력']-v1:+9.1f}")

best = max(res, key=lambda k: res[k][1]["변별력"])
log(f"\n  ▶ 최고: {best}  (변별력 {res[best][1]['변별력']:.1f}, "
    f"총점 {res[best][1]['총점']:.1f})")

d_v2 = res["v2 가지치기"][1]["변별력"] - v1
d_v3 = res["v3 +차이 5개"][1]["변별력"] - res["v2 가지치기"][1]["변별력"]
d_v4 = res["v4 +상호작용 3개"][1]["변별력"] - res["v3 +차이 5개"][1]["변별력"]
log(f"\n  가지치기 효과   (v1→v2) : {d_v2:+7.1f}   쓰레기 11개 제거")
log(f"  차이피처 효과   (v2→v3) : {d_v3:+7.1f}   새 차이 5개 추가")
log(f"  상호작용 효과   (v3→v4) : {d_v4:+7.1f}   매치업×상황 3개 추가")
log(f"\n  ※ 상호작용이 0에 가까우면 → 트리가 상호작용은 스스로 한다는 뜻")
log(f"  참고: 리더보드 베이스라인 {LB_BASELINE:.1f}, 1위 {LB_TOP:.1f}")
log("=" * 88)
