"""
[19] hand delta 철회 — 65피처로 복귀 (830.32 재현)

배경: exp/18로 hand delta의 다연도 안정성을 쟀다.
  2021 +8.7 / 2022 +4.9 / 2023 +35.4 / 2024 +13.9~22.9(독립3회, 평균+18.1)
  → 4개 연도 전부 양수. 그런데 실제 2025 리더보드는 -22.2 (830.32 → 808.09).
  판정 가이드(사전 등록, exp/18 docstring): "전 연도 일관 양수 + 2025만 음수
  → 2025 특이. 제출은 65피처로 복귀하되 원인 기록." 그대로 적용한다.

이 스크립트는 hand delta를 완전히 빼고 원래 65피처 구성(원본47+엔지니어링18)만으로
8시드 앙상블을 다시 학습해 submit/model/model.pkl을 덮어쓴다.
exp/07_train_submit.py와 동일한 절차, HAND_ENG만 제외.

실행 (★ 반드시 서버 버전 venv311로):
  <venv311>/Scripts/python.exe -u exp/19_revert_65feat.py
"""

import importlib.util
import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

T0 = time.time()
SEEDS = [42, 7, 123, 2024, 99, 555, 31337, 1]
HGB = dict(max_iter=120, learning_rate=0.08, max_leaf_nodes=31,
           min_samples_leaf=200, l2_regularization=10.0, early_stopping=False)
CAT = ["top_bottom", "game_type", "base_state"]
LOCAL_TARGET = 708.7        # exp/06 노이즈바닥 실험의 8시드 앙상블 총점 (65피처, hand delta 없음)


def log(*a):
    print(*a, flush=True)


def tick(m):
    log(f"  [{time.time()-T0:6.0f}s] {m}")


# ---- submit/script.py 에서 피처 함수 가져오기 (add_features만. attach_hand_deltas는 안 씀) ----
spec = importlib.util.spec_from_file_location("subm", "submit/script.py")
subm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subm)
add_features, to_matrix = subm.add_features, subm.to_matrix
log("피처 함수 출처: submit/script.py add_features (hand delta 미사용)")
log(f"실행 환경: python {sys.version.split()[0]} / numpy {np.__version__} / "
    f"pandas {pd.__version__}")


def decompose(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    r = y.mean()
    d = p.mean() - r
    total = 100000.0 * (1.0 - float(np.mean((p - y) ** 2)) / (r * (1 - r)))
    pen = 100000.0 * d * d / (r * (1 - r))
    return {"총점": total, "변별력": total + pen, "중심벌점": pen, "d": d}


# =====================================================================
log("\ntrain.csv 로딩 중...")
df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
df.columns = [c.replace("﻿", "").strip() for c in df.columns]
for c in df.select_dtypes("float64").columns:
    df[c] = df[c].astype("float32")
tick(f"로딩 완료 {df.shape}")

test_cols = list(pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1).columns)
test_cols = [c.replace("﻿", "").strip() for c in test_cols]
BASE = [c for c in test_cols if c != "row_id"]
missing = [c for c in BASE if c not in df.columns]
if missing:
    raise SystemExit(f"train.csv에 없는 test 컬럼: {missing}")

ENG = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
       "f_same_hand", "f_scoring_pos", "f_any_runner",
       "f_log_pn", "f_log_bn", "f_smooth_p", "f_smooth_b",
       "f_form_dev1", "f_form_dev3", "f_form_trend", "f_pb_diff",
       "f_miss_p", "f_miss_prev1"]      # ← hand delta 2개 제외 (65피처 복귀)
FEATS = BASE + ENG
log(f"  피처 {len(FEATS)}개 = 원본 {len(BASE)} + 엔지니어링 {len(ENG)}  (hand delta 제외)")


def train_ensemble(train_df, prior, tag):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train_df[CAT])
    ft = add_features(train_df, prior)
    X = to_matrix(ft, FEATS, enc)
    y = train_df["control_success"].to_numpy()
    models = []
    for i, s in enumerate(SEEDS, 1):
        tick(f"[{tag}] seed={s} 학습 중 ({i}/{len(SEEDS)})...")
        prm = dict(HGB, random_state=s)
        models.append(HistGradientBoostingClassifier(**prm).fit(X, y))
    return models, enc


def predict_ensemble(models, enc, part_df, prior):
    ft = add_features(part_df, prior)
    X = to_matrix(ft, FEATS, enc)
    acc = np.zeros(len(part_df))
    for m in models:
        acc += m.predict_proba(X)[:, 1]
    return acc / len(models)


# =====================================================================
# 1단계 — 검증
# =====================================================================
log("\n" + "=" * 84)
log("1단계 검증: 2019~2023 학습 → 2024 채점  (708.7 재현 확인)")
log("=" * 84)

tr = df[df["season"] <= 2023].reset_index(drop=True)
va = df[df["season"] == 2024].reset_index(drop=True)
prior_v = float(tr["control_success"].mean())
models_v, enc_v = train_ensemble(tr, prior_v, "검증")
p_va = predict_ensemble(models_v, enc_v, va, prior_v)
z = decompose(p_va, va["control_success"].to_numpy())
log(f"\n  8시드 앙상블 → 총점 {z['총점']:.1f} = 변별력 {z['변별력']:.1f} "
    f"- 벌점 {z['중심벌점']:.1f}  (d={z['d']:+.4f})")
log(f"  기대값(06 실험) {LOCAL_TARGET:.1f}   차이 {z['총점']-LOCAL_TARGET:+.1f}")
if abs(z["총점"] - LOCAL_TARGET) > 30:
    log("  경고: 30점 넘게 벗어남 — 파이프라인 점검 필요. 저장 단계로 진행하지 않는다.")
    raise SystemExit(1)
log("  재현 확인. 제출 코드 경로가 정상 동작한다.")

del models_v, enc_v, tr, va

# =====================================================================
# 2단계 — 최종 학습 + 저장 (기존 submit/model/model.pkl 덮어씀)
# =====================================================================
log("\n" + "=" * 84)
log("2단계 최종: 2019~2024 전체 학습 → 저장 (65피처, hand delta 제외)")
log("=" * 84)

prior = float(df["control_success"].mean())
log(f"  prior (전체 평균 성공률) = {prior:.6f}")
models, enc = train_ensemble(df, prior, "최종")

os.makedirs("submit/model", exist_ok=True)
bundle = {"models": models, "encoder": enc, "prior": prior, "feats": FEATS,
          "seeds": SEEDS, "hgb": HGB, "n_train": int(len(df)),
          "hand_tbl_p": None, "hand_tbl_b": None}   # hand delta 비활성 표시
joblib.dump(bundle, "submit/model/model.pkl", compress=3)
size = os.path.getsize("submit/model/model.pkl") / 1024 ** 2
tick(f"저장 완료 submit/model/model.pkl  ({size:.1f} MB, 65피처)")

log("\n  저장본 재로드 검증 (5행 test.csv, hand delta 없이)...")
b2 = joblib.load("submit/model/model.pkl")
t5 = pd.read_csv("data/test.csv", encoding="utf-8-sig")
t5.columns = [c.replace("﻿", "").strip() for c in t5.columns]
ft5 = subm.add_features(t5, b2["prior"])
X5 = to_matrix(ft5, b2["feats"], b2["encoder"])
acc5 = np.zeros(len(t5))
for m in b2["models"]:
    acc5 += m.predict_proba(X5)[:, 1]
p5 = acc5 / len(b2["models"])
log(f"  진짜 test.csv 5행 예측 = {np.round(p5, 6).tolist()}")
log(f"  범위 확인: min={p5.min():.6f} max={p5.max():.6f} "
    f"(0~1 이내: {bool((p5 >= 0).all() and (p5 <= 1).all())})")

log(f"\n총 소요 {time.time()-T0:.0f}초")
log("=" * 84)
