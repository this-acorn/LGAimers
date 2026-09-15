# -*- coding: utf-8 -*-
"""
[81] 번들 재포장 1단계 — 서버 호환 형식으로 내보내기 (시스템 python 3.12 에서 실행)

사고 경위:
  exp/80 배포 학습을 venv311 이 아니라 시스템 python 으로 돌렸다.
  시스템 환경이 **pandas 3.0.1 / numpy 2.1.3** 이고, pandas 3.0 은 pickle 내부에서
  PyArrow 를 참조한다. 그 결과 submit16_src/model/model.pkl 이

      ModuleNotFoundError: No module named 'pyarrow'

  로 venv311(=평가 서버와 동일: python 3.11 / numpy 1.26.4 / pandas 2.0.3)에서
  로드에 실패한다. **평가 서버에도 pyarrow 가 없으므로 그대로 제출하면 실행 오류다.**

  다행히 번들 안 DataFrame 의 dtype 은 전부 numpy 계열(int64/float64/float32/int8)이라
  내용 자체는 멀쩡하다. 5시간 재학습 대신 **재포장**으로 해결한다.

이 스크립트가 하는 일:
  · CatBoost 모델 9개(본체 8 + P(구종) 1)를 CatBoost 자체 포맷 .cbm 으로 저장
    (버전 이식 가능한 바이너리. joblib pickle 과 달리 pandas/numpy 에 의존하지 않는다)
  · 표 3개를 컬럼별 numpy 배열로 .npz 에 저장 (.npy 포맷은 numpy 1.x/2.x 호환)
  · 스칼라/리스트는 JSON 으로
  · ★ 진짜 test.csv 5행에 대한 **기준 예측값**을 저장 — 2단계에서 재조립한 번들이
    비트 단위로 같은 값을 내는지 검증하는 데 쓴다

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/81_export_bundle.py   (~1분)
다음: venv311 로 exp/82_rebuild_submit16.py
"""

import importlib.util
import json
import os
import shutil
import numpy as np
import pandas as pd
import joblib

SRC = 'submissions/submit16_src/model/model.pkl'
STAGE = 'submissions/submit16_src/_stage'

print(f"저장 환경: numpy {np.__version__} / pandas {pd.__version__}")
b = joblib.load(SRC)
print(f"로드 완료: 모델 {len(b['cb_models'])}개, 피처 {len(b['feats'])}개, "
      f"version={b.get('version')}")

if os.path.exists(STAGE):
    shutil.rmtree(STAGE)
os.makedirs(STAGE)

# ---- 1. CatBoost 모델 → .cbm ----
for i, m in enumerate(b["cb_models"]):
    m.save_model(f"{STAGE}/cb_{i}.cbm")
b["pfb_model"].save_model(f"{STAGE}/pfb.cbm")
print(f"모델 저장: cb_0~{len(b['cb_models'])-1}.cbm + pfb.cbm")

# ---- 2. 표 → .npz (컬럼별 numpy 배열) ----
tables = {}
for key in ["cs_const_p", "cs_const_b", "mix_tbl"]:
    df = b[key]
    cols = {}
    for c in df.columns:
        cols[c] = df[c].to_numpy()
    np.savez(f"{STAGE}/{key}.npz", **cols)
    tables[key] = {"columns": list(df.columns),
                   "dtypes": {c: str(df[c].dtype) for c in df.columns},
                   "shape": list(df.shape)}
    print(f"표 저장: {key}  {df.shape}  {list(df.columns)}")

# ---- 3. 스칼라/리스트 → JSON ----
meta = {
    "feats": list(b["feats"]),
    "prior": float(b["prior"]),
    "seeds": [int(s) for s in b["seeds"]],
    "version": b.get("version"),
    "season_decay": b.get("season_decay"),
    "classes": b.get("classes"),
    "n_models": len(b["cb_models"]),
    "tables": tables,
}
with open(f"{STAGE}/meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
print(f"메타 저장: 피처 {len(meta['feats'])}개, prior={meta['prior']:.10f}, "
      f"decay={meta['season_decay']}")

# ---- 4. 기준 예측값 (진짜 test.csv 5행) ----
spec = importlib.util.spec_from_file_location("s16", 'submissions/submit16_src/script.py')
s16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s16)

t5 = pd.read_csv("data/test.csv", encoding="utf-8-sig")
t5.columns = [c.replace("﻿", "").strip() for c in t5.columns]
ft = s16.attach_pt(
    s16.attach_cs(s16.add_features(t5, b["prior"]), b["cs_const_p"], b["cs_const_b"]),
    b["mix_tbl"], b["pfb_model"])
X = s16.build_matrix(ft, b["feats"])
per_seed = np.array([m.predict_proba(X)[:, 0] for m in b["cb_models"]], dtype="float64")
ref = per_seed.mean(axis=0)
np.savez(f"{STAGE}/ref_pred.npz", per_seed=per_seed, ens=ref,
         X_shape=np.array(X.shape))
print("\n기준 예측 (5행, 8시드 평균 P(성공)):")
print("  " + "  ".join(f"{v:.10f}" for v in ref))
print("  시드별 산포:", " ".join(f"{v:.6f}" for v in per_seed.max(0) - per_seed.min(0)))

print(f"\n스테이징 완료: {STAGE}/  "
      f"({sum(os.path.getsize(f'{STAGE}/{f}') for f in os.listdir(STAGE))/1024**2:.1f} MB)")
print("다음: venv311/python.exe -u exp/82_rebuild_submit16.py")
