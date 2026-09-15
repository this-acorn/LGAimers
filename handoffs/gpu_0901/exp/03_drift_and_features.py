"""
[03] 드리프트 대응 전략 + 피처 엔지니어링 + 목적함수 비교   (경량판)

실행:  py -3.12 -u exp/03_drift_and_features.py
       ↑ -u 필수! 붙이지 않으면 진행 상황이 안 보인다.

구조 (전부 "2019~2023 학습 → 2024 검증" = 실전(2019~2024→2025)의 축소판):
  Part 1  RandomForest 재현        → 비어있던 ② 숫자 확보
  Part 2  드리프트 전략 5종 비교    → 어떻게 학습해야 중심이 맞나
  Part 3  기본피처 vs 엔지니어링    → 피처가 몇 점짜리인가
  Part 4  분류 vs 회귀             → Brier에 뭐가 맞나

경량화 내역 (이전 버전이 40분 넘게 걸려서):
  - RF in-sample 예측 삭제 (이미 약 2260점인 걸 알고 있음)
  - GBDT max_iter 300 → 120, learning_rate 0.05 → 0.08
  - 모든 print에 flush=True (진행률 실시간 표시)
  - GBDT 학습은 총 5회 (④⑤는 ①의 예측을 이동만 하므로 재학습 없음)

핵심 도구: Brier 분해
  Brier(p) = Brier(중심 맞춘 p) + d²      (d = 예측평균 - 실제평균)
  → 총점 = 변별력 - 중심오차벌점
    이걸 봐야 "피처가 좋아진 건지, 중심이 우연히 맞은 건지" 구분된다.
"""

import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

LB_BASELINE = 549.51
LB_TOP = 1421.99852
CAT = ["top_bottom", "game_type", "base_state"]
T_START = time.time()


def log(*a):
    """진행 상황을 즉시 출력 (버퍼링 방지)."""
    print(*a, flush=True)


def tick(msg):
    log(f"  [{time.time()-T_START:6.0f}s] {msg}")


# =====================================================================
# 점수 도구
# =====================================================================
def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def raw_score(p, y):
    """대회 점수. 단 max(0,·)를 적용하지 않는다 (음수도 봐야 하므로)."""
    y = np.asarray(y, float)
    r = y.mean()
    return 100000.0 * (1.0 - brier(p, y) / (r * (1.0 - r)))


def decompose(p, y):
    """총점 = 변별력 - 중심오차벌점 으로 분해."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    r = y.mean()
    d = p.mean() - r
    total = raw_score(p, y)
    penalty = 100000.0 * d * d / (r * (1 - r))
    return {"총점": total, "변별력": total + penalty, "중심벌점": penalty, "d": d}


def report(name, p, y):
    z = decompose(p, y)
    log(f"  {name:32s} 총점 {z['총점']:8.1f} = 변별력 {z['변별력']:7.1f} "
        f"- 벌점 {z['중심벌점']:7.1f}   (d={z['d']:+.4f})")
    return z


# =====================================================================
# 데이터
# =====================================================================
log("train.csv 로딩 중... (전체 49컬럼)")
df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
df.columns = [c.replace("﻿", "").strip() for c in df.columns]
for c in df.select_dtypes("float64").columns:
    df[c] = df[c].astype("float32")
tick(f"로딩 완료 {df.shape}  {df.memory_usage(deep=True).sum()/1024**2:.0f}MB")

BASE_FEATS = [c for c in df.columns if c not in ("row_id", "control_success")]

tr = df[df["season"] <= 2023].reset_index(drop=True)
va = df[df["season"] == 2024].reset_index(drop=True)
y_tr = tr["control_success"].to_numpy()
y_va = va["control_success"].to_numpy()
R_TR = y_tr.mean()      # 학습기간 평균 (2024 정보 아님 → 사용 가능)
R_2023 = df.loc[df["season"] == 2023, "control_success"].mean()

log(f"\n학습 2019~2023 : {len(tr):,}행  r={R_TR:.6f}")
log(f"검증 2024      : {len(va):,}행  r={y_va.mean():.6f}")
log(f"(참고) 2023 r  : {R_2023:.6f}   ← 학습기간 중 가장 최근\n")


# =====================================================================
# 피처 엔지니어링
# =====================================================================
def add_features(d):
    """투구 '직전' 정보만 사용. 다른 행을 참조하지 않으므로 test에도 그대로 적용 가능."""
    d = d.copy()
    b, s = d["balls_before"], d["strikes_before"]

    # ① 볼카운트 — 01번 D파트에서 r 폭 0.0345 실측됨
    d["f_count_state"] = b * 3 + s
    d["f_count_diff"] = b - s
    d["f_is_3ball"] = (b == 3).astype("int8")
    d["f_is_2strike"] = (s == 2).astype("int8")
    d["f_is_full"] = ((b == 3) & (s == 2)).astype("int8")

    # ② 매치업 / 상황 압박
    d["f_same_hand"] = (d["pitcher_hand"] == d["batter_hand"]).astype("int8")
    d["f_scoring_pos"] = ((d["runner_on_2b"] == 1) | (d["runner_on_3b"] == 1)).astype("int8")
    d["f_any_runner"] = (d["num_runners_on"] > 0).astype("int8")

    # ③ asof 신뢰도 — 표본이 적을수록 rate를 믿으면 안 된다
    pn = d["asof_pitcher_n"].fillna(0)
    bn = d["asof_batter_n"].fillna(0)
    d["f_log_pn"] = np.log1p(pn).astype("float32")
    d["f_log_bn"] = np.log1p(bn).astype("float32")
    a = 200.0
    d["f_smooth_p"] = ((d["asof_pitcher_success_rate"].fillna(R_TR) * pn + R_TR * a)
                       / (pn + a)).astype("float32")
    d["f_smooth_b"] = ((d["asof_batter_success_rate"].fillna(R_TR) * bn + R_TR * a)
                       / (bn + a)).astype("float32")

    # ④ 최근 폼 — 통산 rate는 드리프트에 오염됨. 최근 창은 덜하다.
    d["f_form_dev1"] = (d["asof_pitcher_prev1_game_success_rate"]
                        - d["asof_pitcher_success_rate"]).astype("float32")
    d["f_form_dev3"] = (d["asof_pitcher_prev3_game_success_rate"]
                        - d["asof_pitcher_success_rate"]).astype("float32")
    d["f_form_trend"] = (d["asof_pitcher_prev1_game_success_rate"]
                         - d["asof_pitcher_prev5_game_success_rate"]).astype("float32")

    # ⑤ 투수 vs 타자 매치업 차이
    d["f_pb_diff"] = (d["asof_pitcher_success_rate"]
                      - d["asof_batter_success_rate"]).astype("float32")

    # ⑥ 결측 자체가 신호 (신인/첫등판 표시)
    d["f_miss_p"] = d["asof_pitcher_success_rate"].isna().astype("int8")
    d["f_miss_prev1"] = d["asof_pitcher_prev1_game_success_rate"].isna().astype("int8")
    return d


ENG_FEATS = BASE_FEATS + [
    "f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
    "f_same_hand", "f_scoring_pos", "f_any_runner",
    "f_log_pn", "f_log_bn", "f_smooth_p", "f_smooth_b",
    "f_form_dev1", "f_form_dev3", "f_form_trend", "f_pb_diff",
    "f_miss_p", "f_miss_prev1",
]

tr_e, va_e = add_features(tr), add_features(va)
tick(f"피처 생성 완료: 기본 {len(BASE_FEATS)}개 → {len(ENG_FEATS)}개 (+{len(ENG_FEATS)-len(BASE_FEATS)})")


def encode(train_df, valid_df, feats):
    """범주형 3개는 OrdinalEncoder, 나머지는 그대로. 결측(NaN)은 GBDT가 알아서 처리."""
    cats = [c for c in CAT if c in feats]
    nums = [c for c in feats if c not in cats]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xtr = np.column_stack([enc.fit_transform(train_df[cats]).astype("float32"),
                           train_df[nums].to_numpy("float32")])
    Xva = np.column_stack([enc.transform(valid_df[cats]).astype("float32"),
                           valid_df[nums].to_numpy("float32")])
    return Xtr, Xva


# =====================================================================
# Part 1 — RandomForest 재현 (비어있던 ② 숫자)
# =====================================================================
log("\n" + "=" * 92)
log("Part 1. RandomForest 재현   (train 2260 → ??? → 실제 549.51 의 가운데를 채운다)")
log("=" * 92)

pre = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT),
    ("num", SimpleImputer(strategy="median"), [c for c in BASE_FEATS if c not in CAT]),
])
rf = Pipeline([("pre", pre),
               ("clf", RandomForestClassifier(n_estimators=100, max_depth=10,
                                              min_samples_leaf=200, n_jobs=-1,
                                              random_state=42))])
tick("RF 학습 시작 (트리 100개)...")
rf.fit(tr[BASE_FEATS], y_tr)
tick("RF 학습 완료 → 2024 예측 중")
p_rf = rf.predict_proba(va[BASE_FEATS])[:, 1]
tick("RF 예측 완료")
z_rf = report("RF 홀드아웃(2024)  ★", p_rf, y_va)

log(f"\n  ▶ 리더보드 실제(2025) = {LB_BASELINE:.2f}")
log(f"  ▶ 여기 2024 홀드아웃  = {z_rf['총점']:.2f}")
log(f"     비슷하면 → 로컬검증이 LB를 잘 예측 (제출 없이 개선 판단 가능)")
log(f"     홀드아웃이 훨씬 높으면 → 2025가 특별히 어려운 해")


# =====================================================================
# Part 2 — 드리프트 전략
# =====================================================================
log("\n" + "=" * 92)
log("Part 2. 드리프트 대응 전략 비교 (모델 = HistGradientBoosting 고정)")
log("=" * 92)

Xtr_b, Xva_b = encode(tr, va, BASE_FEATS)
HGB = dict(max_iter=120, learning_rate=0.08, max_leaf_nodes=31,
           min_samples_leaf=200, l2_regularization=10.0,
           early_stopping=False, random_state=42)
results = {}

tick("① 전체 균등 학습 중 (1/3)...")
m1 = HistGradientBoostingClassifier(**HGB).fit(Xtr_b, y_tr)
p1 = m1.predict_proba(Xva_b)[:, 1]
results["① 전체 균등"] = report("① 전체 균등 (2019~23)", p1, y_va)

tick("② 최근2시즌만 학습 중 (2/3)...")
mask = (tr["season"] >= 2022).to_numpy()
m2 = HistGradientBoostingClassifier(**HGB).fit(Xtr_b[mask], y_tr[mask])
p2 = m2.predict_proba(Xva_b)[:, 1]
results["② 최근2시즌만"] = report("② 최근2시즌만 (2022~23)", p2, y_va)

tick("③ 최근가중 학습 중 (3/3)...")
w = np.power(2.0, tr["season"].to_numpy() - 2023).astype("float64")
m3 = HistGradientBoostingClassifier(**HGB).fit(Xtr_b, y_tr, sample_weight=w)
p3 = m3.predict_proba(Xva_b)[:, 1]
results["③ 최근가중"] = report("③ 최근가중 (2배씩)", p3, y_va)

# ④⑤ 는 ①의 예측을 이동만 함 (재학습 없음)
p4 = np.clip(p1 - p1.mean() + R_2023, 0.001, 0.999)
results["④ 중심을 2023으로"] = report("④ 중심을 2023으로 이동", p4, y_va)

p5 = np.clip(p1 - p1.mean() + y_va.mean(), 0.001, 0.999)
report("⑤ 중심=2024정답 (누수/상한)", p5, y_va)

best = max(results, key=lambda k: results[k]["총점"])
log(f"\n  ▶ 최고 전략: {best}  ({results[best]['총점']:.1f}점)")
log(f"  ▶ ⑤는 정답을 쓴 반칙 — '중심만 완벽하면 몇 점인가'의 천장")


# =====================================================================
# Part 3 — 피처 엔지니어링 효과
# =====================================================================
log("\n" + "=" * 92)
log("Part 3. 기본피처 vs 엔지니어링피처  (전략은 ① 전체균등 고정)")
log("=" * 92)

Xtr_e, Xva_e = encode(tr_e, va_e, ENG_FEATS)
tick(f"엔지니어링 {len(ENG_FEATS)}피처 학습 중...")
m_e = HistGradientBoostingClassifier(**HGB).fit(Xtr_e, y_tr)
p_e = m_e.predict_proba(Xva_e)[:, 1]
z_b = decompose(p1, y_va)
log(f"  {'기본 47개 (Part2 ①)':32s} 총점 {z_b['총점']:8.1f} = 변별력 {z_b['변별력']:7.1f} "
    f"- 벌점 {z_b['중심벌점']:7.1f}   (d={z_b['d']:+.4f})")
z_e = report(f"엔지니어링 {len(ENG_FEATS)}개", p_e, y_va)
log(f"\n  ▶ 피처 추가 효과: 총점 {z_e['총점']-z_b['총점']:+.1f}점 / "
    f"변별력 {z_e['변별력']-z_b['변별력']:+.1f}점")
log(f"     ※ '변별력' 증가분이 진짜 피처 효과. 총점엔 중심 운이 섞여있다.")


# =====================================================================
# Part 4 — 분류 vs 회귀
# =====================================================================
log("\n" + "=" * 92)
log("Part 4. 목적함수: 분류(log loss) vs 회귀(MSE = Brier 직접최적화)")
log("=" * 92)

tick("회귀 모델 학습 중...")
reg = HistGradientBoostingRegressor(loss="squared_error", **HGB).fit(Xtr_e, y_tr)
p_r = np.clip(reg.predict(Xva_e), 0.001, 0.999)
log(f"  {'분류 logloss (엔지니어링)':32s} 총점 {z_e['총점']:8.1f} = 변별력 {z_e['변별력']:7.1f} "
    f"- 벌점 {z_e['중심벌점']:7.1f}   (d={z_e['d']:+.4f})")
report("회귀 MSE   (엔지니어링)", p_r, y_va)


# =====================================================================
log("\n" + "=" * 92)
log("요약")
log("=" * 92)
log(f"  리더보드 베이스라인(2025) = {LB_BASELINE:8.2f}")
log(f"  리더보드 현재 1위(2025)   = {LB_TOP:8.2f}")
log(f"  로컬 RF 홀드아웃(2024)    = {z_rf['총점']:8.2f}   ← 이 둘의 관계가 앞으로의 나침반")
log(f"\n  총 소요 {time.time()-T_START:.0f}초")
log("=" * 92)
