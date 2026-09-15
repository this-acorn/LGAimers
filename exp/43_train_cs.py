# -*- coding: utf-8 -*-
"""
[43] CS 76피처 배포 학습 — CatBoost 8시드 전체 데이터 + 상수표 번들

근거: exp/41 시즌 진행분 +107.9 (역대 최대), exp/42 적층 확인(물리 흡수 → CS만).
피처 로직 단일 원본 = submissions/submit10_src/script.py (여기서 import).

절차:
  1. 검증 재현: 2019~2023 학습 → 2024 채점, 8시드 — exp/41(2시드 782.3) 재현 확인
  2. 최종: 2019~2024 전체 학습 8시드 + 배포 상수표(≤2024) → submissions/submit10_src/model/model.pkl
  3. 5행 test.csv 서빙 경로 검증 (CS 커버리지 포함)

★ venv311 필수:  <venv311>/Scripts/python.exe -u exp/43_train_cs.py   (~2.5시간)
"""

import importlib.util
import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier, Pool

T0 = time.time()
SEEDS = [42, 7, 123, 2024, 99, 555, 31337, 1]
CB_PRM = dict(iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=10.0,
              verbose=False, thread_count=14, allow_writing_files=False)
LOCAL_REF = 782.3          # exp/41 2시드 앙상블 (8시드는 이보다 소폭 높아야 정상)

spec = importlib.util.spec_from_file_location("subm", "submissions/submit10_src/script.py")
subm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subm)
add_features, attach_cs, build_matrix = subm.add_features, subm.attach_cs, subm.build_matrix
CAT, P_RATES, B_RATES, CS_FEATS = subm.CAT, subm.P_RATES, subm.B_RATES, subm.CS_FEATS


def log(*a):
    print(*a, flush=True)


def tick(m):
    log(f"  [{time.time()-T0:6.0f}s] {m}")


def raw_score(p, y):
    y = np.asarray(y, float)
    r = y.mean()
    return 100000.0 * (1.0 - float(np.mean((np.asarray(p, float) - y) ** 2)) / (r * (1 - r)))


def build_const(src, id_col, n_col, rates):
    d = src.sort_values(n_col).groupby(id_col).tail(1)
    out = pd.DataFrame({"id": d[id_col].to_numpy()})
    n_last = d[n_col].fillna(0).to_numpy("float64")
    out["N_end"] = n_last + 1
    for k, col in rates.items():
        r = d[col].fillna(0).to_numpy("float64")
        if k == "succ":
            out[f"S_{k}"] = np.round(r * n_last) + d["control_success"].to_numpy("float64")
        else:
            out[f"S_{k}"] = r * (n_last + 1)
    return out


log(f"환경: python {sys.version.split()[0]} / numpy {np.__version__} / pandas {pd.__version__}")
import catboost
log(f"catboost {catboost.__version__}")
log("train 로딩...")
df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
df.columns = [c.replace("﻿", "").strip() for c in df.columns]
for c in df.select_dtypes("float64").columns:
    df[c] = df[c].astype("float32")
tick(f"로딩 {df.shape}")

test_cols = [c.replace("﻿", "").strip() for c in
             pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1).columns]
BASE = [c for c in test_cols if c != "row_id"]
ENG18 = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
         "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
         "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
         "f_pb_diff", "f_miss_p", "f_miss_prev1"]
FEATS = BASE + ENG18 + CS_FEATS
log(f"피처 {len(FEATS)} = 원본 {len(BASE)} + 기본 {len(ENG18)} + CS {len(CS_FEATS)}")


def attach_train_rows(rows_df):
    """학습 행: 시즌 S ← ≤S-1 상수표 (배포와 동일 방법)"""
    parts = []
    for S in sorted(rows_df.season.unique()):
        h = df[df.season <= S - 1]
        rows = rows_df[rows_df.season == S]
        if len(h) == 0:
            cpS = pd.DataFrame({"id": [], "N_end": [],
                                **{f"S_{k}": [] for k in P_RATES}})
            cbS = pd.DataFrame({"id": [], "N_end": [],
                                **{f"S_{k}": [] for k in B_RATES}})
        else:
            cpS = build_const(h, "pitcher_id", "asof_pitcher_n", P_RATES)
            cbS = build_const(h, "batter_id", "asof_batter_n", B_RATES)
        parts.append(attach_cs(rows, cpS, cbS))
    return pd.concat(parts).sort_index()


def train_ens(tr_rows, prior, tag):
    ft = add_features(attach_train_rows(tr_rows), prior)
    X = build_matrix(ft, FEATS)
    y = tr_rows["control_success"].to_numpy()
    pool = Pool(X, y, cat_features=list(CAT))
    del ft, X
    models = []
    for i, s in enumerate(SEEDS, 1):
        t0 = time.time()
        models.append(CatBoostClassifier(**CB_PRM, random_seed=s).fit(pool))
        tick(f"[{tag}] seed={s} ({i}/8, {time.time()-t0:.0f}s)")
    del pool
    return models


# =====================================================================
log("\n" + "=" * 80)
log("1단계 검증: 2019~2023 학습 → 2024 채점 (8시드, exp/41 재현 확인)")
log("=" * 80)
tr = df[df.season <= 2023].reset_index(drop=True)
va_raw = df[df.season == 2024].reset_index(drop=True)
prior_v = float(tr["control_success"].mean())
models_v = train_ens(tr, prior_v, "검증")
cp_v = build_const(df[df.season <= 2023], "pitcher_id", "asof_pitcher_n", P_RATES)
cb_v = build_const(df[df.season <= 2023], "batter_id", "asof_batter_n", B_RATES)
va = add_features(attach_cs(va_raw, cp_v, cb_v), prior_v)
Xva = build_matrix(va, FEATS)
acc = np.zeros(len(va))
for m in models_v:
    acc += m.predict_proba(Xva)[:, 1]
p8 = acc / len(models_v)
s8 = raw_score(p8, va_raw["control_success"].to_numpy())
log(f"\n  8시드 앙상블 = {s8:.1f}  (exp/41 2시드 {LOCAL_REF} — 이보다 높거나 비슷해야 정상)")
if s8 < LOCAL_REF - 30:
    log("  ⚠️ 재현 실패 의심 — 저장 중단")
    raise SystemExit(1)
log("  재현 확인.")
del models_v, tr, va, Xva, va_raw

# =====================================================================
log("\n" + "=" * 80)
log("2단계 최종: 2019~2024 전체 학습 + 배포 상수표 저장")
log("=" * 80)
prior = float(df["control_success"].mean())
models = train_ens(df, prior, "최종")
cp_d = build_const(df, "pitcher_id", "asof_pitcher_n", P_RATES)
cb_d = build_const(df, "batter_id", "asof_batter_n", B_RATES)
log(f"  배포 상수표: 투수 {len(cp_d)}명 / 타자 {len(cb_d)}명 (≤2024 전체)")

bundle = {"cb_models": models, "feats": FEATS, "prior": prior,
          "cs_const_p": cp_d, "cs_const_b": cb_d,
          "seeds": SEEDS, "cb_params": CB_PRM, "version": "cs76"}
os.makedirs("submissions/submit10_src/model", exist_ok=True)
joblib.dump(bundle, "submissions/submit10_src/model/model.pkl", compress=3)
tick(f"저장: submissions/submit10_src/model/model.pkl "
     f"({os.path.getsize('submissions/submit10_src/model/model.pkl')/1024**2:.1f} MB)")

# ---- 5행 서빙 경로 검증 ----
b2 = joblib.load("submissions/submit10_src/model/model.pkl")
t5 = pd.read_csv("data/test.csv", encoding="utf-8-sig")
t5.columns = [c.replace("﻿", "").strip() for c in t5.columns]
ft5 = subm.attach_cs(subm.add_features(t5, b2["prior"]), b2["cs_const_p"], b2["cs_const_b"])
X5 = subm.build_matrix(ft5, b2["feats"])
acc5 = np.zeros(len(t5))
for m in b2["cb_models"]:
    acc5 += m.predict_proba(X5)[:, 1]
p5 = acc5 / len(b2["cb_models"])
log(f"  진짜 test.csv 5행(2025) 예측 = {np.round(p5, 6).tolist()}")
log(f"  5행 CS 커버리지: f_cs_p_rate {ft5['f_cs_p_rate'].notna().sum()}/5, "
    f"cs_p_n = {np.expm1(ft5['f_cs_p_logn'].to_numpy()).round(0).tolist()}")
log(f"\n총 {time.time()-T0:.0f}초. 다음: exp/44 artifacts/submissions/submit10.zip 빌드+검증")
