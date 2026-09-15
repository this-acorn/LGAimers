# -*- coding: utf-8 -*-
"""
[33] CatBoost 시드 8→16 증량 — 2025가 CB 친화 해로 확정된 지금, CB 앙상블 강화

근거:
  · 실측 2025 곡선의 최적이 w≈1.0 → 예측의 CB 비중이 클수록 CB 앙상블 분산이 점수를 깎는다
  · CB 시드 분산은 HGB의 수 배 (같은 조건 단일시드 130점 차 실측)
    → 시드 증량 효과가 HGB의 +2.4보다 훨씬 클 것 (1/n 감쇠, 분산이 크면 이득도 큼)

이 스크립트: 새 시드 8개를 전체 데이터로 학습해 번들에 추가 (기존 8개 유지 → 16개).
  w_cb는 건드리지 않는다 (제출 시점에 결정).

★ venv311 필수:  <venv311>/Scripts/python.exe -u exp/33_cb_more_seeds.py
소요: 8 × ~470s ≈ 65분
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
NEW_SEEDS = [11, 222, 3333, 44444, 55555, 666, 77, 8888]
CB_PARAMS = dict(iterations=500, learning_rate=0.08, depth=6,
                 l2_leaf_reg=10.0, verbose=False, thread_count=6,
                 allow_writing_files=False)
CAT = ["top_bottom", "game_type", "base_state"]
BUNDLE = "submit/model/model.pkl"
BACKUP = ("C:/Users/gwonn/AppData/Local/Temp/claude/c--Users-gwonn-Desktop-open/"
          "ac9f75b2-20b2-4bd0-a3a5-9f91191c2d24/scratchpad/model_mix8_backup.pkl")


def log(*a):
    print(*a, flush=True)


def tick(m):
    log(f"  [{time.time()-T0:6.0f}s] {m}")


log(f"환경: python {sys.version.split()[0]} / numpy {np.__version__}")
b = joblib.load(BUNDLE)
assert len(b["cb_models"]) == 8, f"cb 모델 {len(b['cb_models'])}개 — 이미 증량됐나?"
shutil.copy(BUNDLE, BACKUP)
log(f"백업: {BACKUP}")

log("train.csv 로딩...")
df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
for c in df.select_dtypes("float64").columns:
    df[c] = df[c].astype("float32")
y = df["control_success"].to_numpy()

import importlib.util
spec = importlib.util.spec_from_file_location("subm", "submit/script.py")
subm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subm)
ft = subm.add_features(df, b["prior"])
frame = ft[b["cb_feats_num"]].copy()
for c in CAT:
    frame[c] = ft[c].astype(str)
del df, ft
tick(f"준비 완료 {frame.shape}")

pool = Pool(frame, y, cat_features=list(CAT))
del frame
for i, s in enumerate(NEW_SEEDS, 1):
    t0 = time.time()
    m = CatBoostClassifier(**CB_PARAMS, random_seed=s).fit(pool)
    b["cb_models"].append(m)
    tick(f"seed={s} 완료 ({i}/8, {time.time()-t0:.0f}s)")
del pool

b["cb_seeds_extra"] = NEW_SEEDS
joblib.dump(b, BUNDLE, compress=3)
tick(f"저장: {BUNDLE} ({os.path.getsize(BUNDLE)/1024**2:.1f} MB, CB {len(b['cb_models'])}개)")

# 5행 검증
b2 = joblib.load(BUNDLE)
t5 = pd.read_csv("data/test.csv", encoding="utf-8-sig")
t5.columns = [c.replace("\ufeff", "").strip() for c in t5.columns]
ft5 = subm.add_features(t5, b2["prior"])
f5 = ft5[b2["cb_feats_num"]].copy()
for c in b2["cb_feats_cat"]:
    f5[c] = ft5[c].astype(str)
p16 = np.mean([m.predict_proba(f5)[:, 1] for m in b2["cb_models"]], axis=0)
p8 = np.mean([m.predict_proba(f5)[:, 1] for m in b2["cb_models"][:8]], axis=0)
log(f"\n  CB8  5행: {np.round(p8, 6).tolist()}")
log(f"  CB16 5행: {np.round(p16, 6).tolist()}")
log(f"  차이 최대 {np.abs(p16-p8).max():.6f} (시드 증량 효과 존재 확인)")
log(f"\n총 {time.time()-T0:.0f}초. 다음: exp/32 재실행으로 원하는 w의 zip 재빌드")
