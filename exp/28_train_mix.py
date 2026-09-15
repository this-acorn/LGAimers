# -*- coding: utf-8 -*-
"""
[28] HGB+CatBoost 혼합 제출용 학습 — CatBoost 8시드 전체 데이터 학습 + 번들 확장

근거 사슬:
  exp/23 분해 프로브 → exp/25 다연도(선택시드) 통과 → exp/27 독립시드 확증 통과
  → 8+8시드 풀링: w=0.3 에서 4/4 연도 양수 (평균 +20.7, 최악 +2.2)
  w는 번들 스칼라로 저장 — 바꿀 때 재학습 불필요 (추론 시 적용)

이 스크립트가 하는 일:
  1. 기존 submissions/submit/model/model.pkl 로드 (8 HGB, 전체 2019~2024 학습, LB 830.32 확정)
     → HGB 재학습 없음. 검증: models=8, feats=65, hand_tbl 비활성
  2. CatBoost 8시드를 전체 데이터로 학습 (exp/25/27과 동일 설정: iter500 lr0.08 d6 L2=10)
  3. 번들 확장 저장: + cb_models(8), w_cb=0.3, cb_params
     원본 백업: 스크래치패드/model_65feat_backup.pkl (+ artifacts/submissions/submit4.zip 안에도 원본 보존)
  4. 5행 test.csv 서빙 경로 검증

★ 반드시 venv311(서버 버전 numpy 1.26.4)로 실행:
  <venv311>/Scripts/python.exe -u exp/28_train_mix.py
소요 예상: CatBoost 8 × ~650s ≈ 90분
"""

import os
import shutil
import sys
import time
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier, Pool

T0 = time.time()
SEEDS = [42, 7, 123, 2024, 99, 555, 31337, 1]
W_CB = 0.3
CB_PARAMS = dict(iterations=500, learning_rate=0.08, depth=6,
                 l2_leaf_reg=10.0, verbose=False, thread_count=6,
                 allow_writing_files=False)
CAT = ["top_bottom", "game_type", "base_state"]
BUNDLE_PATH = "submissions/submit/model/model.pkl"
BACKUP = ("C:/Users/gwonn/AppData/Local/Temp/claude/c--Users-gwonn-Desktop-open/"
          "ac9f75b2-20b2-4bd0-a3a5-9f91191c2d24/scratchpad/model_65feat_backup.pkl")


def log(*a):
    print(*a, flush=True)


def tick(m):
    log(f"  [{time.time()-T0:6.0f}s] {m}")


log(f"실행 환경: python {sys.version.split()[0]} / numpy {np.__version__} / "
    f"pandas {pd.__version__}")
import catboost
log(f"catboost {catboost.__version__}")

# ---- 1. 기존 번들 로드 + 검증 + 백업 ----
b = joblib.load(BUNDLE_PATH)
assert len(b["models"]) == 8, f"HGB 모델 수 {len(b['models'])} != 8"
assert len(b["feats"]) == 65, f"피처 수 {len(b['feats'])} != 65"
assert b.get("hand_tbl_p") is None, "hand delta가 남아있다 — exp/19 번들이 아님"
log(f"기존 번들 확인: HGB {len(b['models'])}개 / 피처 {len(b['feats'])} / "
    f"prior {b['prior']:.6f}  (LB 830.32 확정본)")
shutil.copy(BUNDLE_PATH, BACKUP)
log(f"백업 완료 → {BACKUP}")

# ---- 2. CatBoost 8시드 전체 학습 ----
log("\ntrain.csv 로딩 중...")
df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
for c in df.select_dtypes("float64").columns:
    df[c] = df[c].astype("float32")
y = df["control_success"].to_numpy()
tick(f"로딩 완료 {df.shape}")

# 피처 생성은 script.py의 add_features (학습·추론 단일 원본 유지)
import importlib.util
spec = importlib.util.spec_from_file_location("subm", "submissions/submit/script.py")
subm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subm)
ft = subm.add_features(df, b["prior"])
NUM_FEATS = [c for c in b["feats"] if c not in CAT]
frame = ft[NUM_FEATS].copy()
for c in CAT:
    frame[c] = ft[c].astype(str)
del df, ft
tick(f"피처 준비 완료 — CatBoost 입력 {frame.shape}")

pool = Pool(frame, y, cat_features=list(CAT))
del frame
cb_models = []
for i, s in enumerate(SEEDS, 1):
    t0 = time.time()
    m = CatBoostClassifier(**CB_PARAMS, random_seed=s).fit(pool)
    cb_models.append(m)
    tick(f"catboost seed={s} 완료 ({i}/8, {time.time()-t0:.0f}s)")
del pool

# ---- 3. 번들 확장 저장 ----
b["cb_models"] = cb_models
b["w_cb"] = W_CB
b["cb_params"] = CB_PARAMS
b["cb_feats_num"] = NUM_FEATS
b["cb_feats_cat"] = list(CAT)
joblib.dump(b, BUNDLE_PATH, compress=3)
size = os.path.getsize(BUNDLE_PATH) / 1024 ** 2
tick(f"저장 완료 {BUNDLE_PATH}  ({size:.1f} MB)  w_cb={W_CB}")

# ---- 4. 5행 서빙 경로 검증 ----
log("\n저장본 재로드 + 5행 test.csv 혼합 예측 검증...")
b2 = joblib.load(BUNDLE_PATH)
t5 = pd.read_csv("data/test.csv", encoding="utf-8-sig")
t5.columns = [c.replace("\ufeff", "").strip() for c in t5.columns]
ft5 = subm.add_features(t5, b2["prior"])
X5 = subm.to_matrix(ft5, b2["feats"], b2["encoder"])
p_h = np.mean([m.predict_proba(X5)[:, 1] for m in b2["models"]], axis=0)
f5 = ft5[b2["cb_feats_num"]].copy()
for c in b2["cb_feats_cat"]:
    f5[c] = ft5[c].astype(str)
p_c = np.mean([m.predict_proba(f5)[:, 1] for m in b2["cb_models"]], axis=0)
p = (1 - b2["w_cb"]) * p_h + b2["w_cb"] * p_c
log(f"  HGB   5행: {np.round(p_h, 6).tolist()}")
log(f"  CB    5행: {np.round(p_c, 6).tolist()}")
log(f"  혼합  5행: {np.round(p, 6).tolist()}")
log(f"  범위: [{p.min():.6f}, {p.max():.6f}]  (0~1 이내: "
    f"{bool((p >= 0).all() and (p <= 1).all())})")
log(f"\n총 소요 {time.time()-T0:.0f}초")
log("다음: script.py 혼합 추론 개편 → exp/29 artifacts/submissions/submit5.zip 빌드+검증")
