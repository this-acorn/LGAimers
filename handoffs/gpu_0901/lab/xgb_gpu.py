"""
[연구실 GPU용] XGBoost vs HistGradientBoosting 비교

이 파일 하나만 있으면 돌아간다 (외부 의존 없음).
필요한 것: train.csv, test.csv 를 ./data/ 아래 두기

────────────────────────────────────────────────────────────────
실행 방법 (연구실 리눅스에서)

  # 1) 환경 (평가 서버가 python 3.11 이므로 맞춰둔다)
  conda create -n aimers python=3.11 -y
  conda activate aimers
  pip install numpy==1.26.4 pandas==2.0.3 scikit-learn==1.8.0 joblib==1.5.3
  pip install xgboost

  # 2) 폴더 구조
  #   작업폴더/
  #     ├── xgb_gpu.py      ← 이 파일
  #     └── data/
  #         ├── train.csv
  #         └── test.csv

  # 3) 실행
  python -u xgb_gpu.py 2>&1 | tee xgb_result.txt

  # 4) xgb_result.txt 내용을 그대로 복사해서 보내주세요
────────────────────────────────────────────────────────────────

무엇을 확인하는가:
  노트북에서는 sklearn HistGradientBoosting(CPU 전용)을 쓰고 있다.
  XGBoost는 같은 계열(히스토그램 GBDT)이지만 구현이 다르고 GPU를 쓸 수 있다.
  → 같은 피처 · 같은 검증(2019~2023 학습 → 2024 채점)으로 공정 비교한다.

  ★ 노트북 실험과 딱 하나만 다르게 한다: 라이브러리.
    피처, 데이터 분할, 점수 산식은 완전히 동일하게 맞췄다.

기준값 (노트북에서 측정한 것):
  HistGradientBoosting 8시드 앙상블 → 2024 홀드아웃 총점 708.7 / 변별력 753.2
  이 모델의 실제 리더보드 점수 830.32
  측정 노이즈 σ=15.3 → 30점 미만 차이는 무시할 것
"""

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

T0 = time.time()
SEEDS = [42, 7, 123, 2024]
CAT = ["top_bottom", "game_type", "base_state"]
HGB_REF_TOTAL = 708.7      # 노트북 8시드 앙상블 총점
HGB_REF_DISC = 753.2       # 그 변별력
NOISE_SD = 15.3


def log(*a):
    print(*a, flush=True)


def tick(m):
    log(f"  [{time.time()-T0:6.0f}s] {m}")


# =====================================================================
# 점수 (대회 공식 산식. max(0,·)는 빼서 음수도 보이게)
# =====================================================================
def decompose(p, y):
    """총점 = 변별력 - 중심오차벌점"""
    p, y = np.asarray(p, float), np.asarray(y, float)
    r = y.mean()
    d = p.mean() - r
    total = 100000.0 * (1.0 - float(np.mean((p - y) ** 2)) / (r * (1 - r)))
    pen = 100000.0 * d * d / (r * (1 - r))
    return {"총점": total, "변별력": total + pen, "중심벌점": pen, "d": d}


# =====================================================================
# 피처 엔지니어링  ★ 노트북 submit/script.py 와 완전히 동일
# =====================================================================
def add_features(d, prior):
    d = d.copy()
    b, s = d["balls_before"], d["strikes_before"]

    d["f_count_state"] = (b * 3 + s).astype("int8")
    d["f_count_diff"] = (b - s).astype("int8")
    d["f_is_3ball"] = (b == 3).astype("int8")
    d["f_is_2strike"] = (s == 2).astype("int8")
    d["f_is_full"] = ((b == 3) & (s == 2)).astype("int8")

    d["f_same_hand"] = (d["pitcher_hand"] == d["batter_hand"]).astype("int8")
    d["f_scoring_pos"] = ((d["runner_on_2b"] == 1) | (d["runner_on_3b"] == 1)).astype("int8")
    d["f_any_runner"] = (d["num_runners_on"] > 0).astype("int8")

    pn = d["asof_pitcher_n"].fillna(0)
    bn = d["asof_batter_n"].fillna(0)
    d["f_log_pn"] = np.log1p(pn).astype("float32")
    d["f_log_bn"] = np.log1p(bn).astype("float32")
    a = 200.0
    d["f_smooth_p"] = ((d["asof_pitcher_success_rate"].fillna(prior) * pn + prior * a)
                       / (pn + a)).astype("float32")
    d["f_smooth_b"] = ((d["asof_batter_success_rate"].fillna(prior) * bn + prior * a)
                       / (bn + a)).astype("float32")

    d["f_form_dev1"] = (d["asof_pitcher_prev1_game_success_rate"]
                        - d["asof_pitcher_success_rate"]).astype("float32")
    d["f_form_dev3"] = (d["asof_pitcher_prev3_game_success_rate"]
                        - d["asof_pitcher_success_rate"]).astype("float32")
    d["f_form_trend"] = (d["asof_pitcher_prev1_game_success_rate"]
                         - d["asof_pitcher_prev5_game_success_rate"]).astype("float32")

    d["f_pb_diff"] = (d["asof_pitcher_success_rate"]
                      - d["asof_batter_success_rate"]).astype("float32")

    d["f_miss_p"] = d["asof_pitcher_success_rate"].isna().astype("int8")
    d["f_miss_prev1"] = d["asof_pitcher_prev1_game_success_rate"].isna().astype("int8")
    return d


ENG = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
       "f_same_hand", "f_scoring_pos", "f_any_runner",
       "f_log_pn", "f_log_bn", "f_smooth_p", "f_smooth_b",
       "f_form_dev1", "f_form_dev3", "f_form_trend", "f_pb_diff",
       "f_miss_p", "f_miss_prev1"]


# =====================================================================
# 환경 점검
# =====================================================================
log("=" * 92)
log("환경 점검")
log("=" * 92)
import sys
log(f"  python  {sys.version.split()[0]}")
log(f"  numpy   {np.__version__}")
log(f"  pandas  {pd.__version__}")
try:
    import xgboost as xgb
    log(f"  xgboost {xgb.__version__}")
except ImportError:
    raise SystemExit("xgboost 미설치 → pip install xgboost")

# GPU 사용 가능한지 실제로 시험해본다 (되면 GPU, 안 되면 CPU로 자동 폴백)
DEVICE = "cpu"
try:
    _t = xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=2, verbosity=0)
    _t.fit(np.random.rand(64, 3), np.random.randint(0, 2, 64))
    _t.predict_proba(np.random.rand(8, 3))
    DEVICE = "cuda"
    log("  GPU     사용 가능 ✅ (device=cuda)")
except Exception as e:
    log(f"  GPU     사용 불가 → CPU로 진행 ({type(e).__name__}: {str(e)[:80]})")


# =====================================================================
# 데이터
# =====================================================================
log("\ntrain.csv 로딩 중...")
df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
df.columns = [c.replace("﻿", "").strip() for c in df.columns]
for c in df.select_dtypes("float64").columns:
    df[c] = df[c].astype("float32")
tick(f"로딩 완료 {df.shape}")

BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS = BASE + ENG

tr = df[df["season"] <= 2023].reset_index(drop=True)
va = df[df["season"] == 2024].reset_index(drop=True)
y_tr, y_va = tr["control_success"].to_numpy(), va["control_success"].to_numpy()
PRIOR = float(y_tr.mean())
tr, va = add_features(tr, PRIOR), add_features(va, PRIOR)

cats = [c for c in CAT if c in FEATS]
nums = [c for c in FEATS if c not in cats]
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(tr[cats])


def to_matrix(d):
    return np.column_stack([enc.transform(d[cats]).astype("float32"),
                            d[nums].to_numpy("float32")])


Xtr, Xva = to_matrix(tr), to_matrix(va)
tick(f"준비 완료 — {len(FEATS)}피처, 학습 {len(tr):,} / 검증 {len(va):,}")
log(f"\n비교 기준: 노트북 HistGradientBoosting 8시드 → 총점 {HGB_REF_TOTAL} / 변별력 {HGB_REF_DISC}")
log(f"          노이즈 σ={NOISE_SD} → 30점 미만 차이는 무시\n")


# =====================================================================
# XGBoost 설정들
# =====================================================================
def run(name, **params):
    """4시드 학습 → 앙상블 예측 → 점수"""
    t0 = time.time()
    acc = np.zeros(len(va))
    singles = []
    for s in SEEDS:
        m = xgb.XGBClassifier(
            device=DEVICE, tree_method="hist", random_state=s,
            n_jobs=-1, verbosity=0, eval_metric="logloss", **params)
        m.fit(Xtr, y_tr)
        p = m.predict_proba(Xva)[:, 1]
        acc += p
        singles.append(decompose(p, y_va)["변별력"])
    z = decompose(acc / len(SEEDS), y_va)
    log(f"  {name:30s} 총점 {z['총점']:7.1f}  변별력 {z['변별력']:7.1f}  "
        f"(단일 {min(singles):.0f}~{max(singles):.0f})  [{time.time()-t0:.0f}s]")
    return z


log("=" * 92)
log("XGBoost 설정 비교 (전부 4시드 앙상블)")
log("=" * 92)

CONFIGS = [
    # 노트북 HistGB 설정을 XGBoost 용어로 옮긴 것 (max_leaf_nodes 31 ≈ max_leaves 31)
    ("① HistGB 설정 이식", dict(n_estimators=120, learning_rate=0.08, max_leaves=31,
                                max_depth=0, grow_policy="lossguide",
                                min_child_weight=200, reg_lambda=10.0)),
    ("② 용량 확장", dict(n_estimators=400, learning_rate=0.03, max_leaves=31,
                         max_depth=0, grow_policy="lossguide",
                         min_child_weight=200, reg_lambda=10.0)),
    ("③ 깊이제한 방식", dict(n_estimators=400, learning_rate=0.03, max_depth=6,
                             min_child_weight=200, reg_lambda=10.0)),
    ("④ 서브샘플 추가", dict(n_estimators=400, learning_rate=0.03, max_depth=6,
                             min_child_weight=200, reg_lambda=10.0,
                             subsample=0.8, colsample_bytree=0.7)),
    ("⑤ 대용량", dict(n_estimators=1000, learning_rate=0.015, max_depth=6,
                      min_child_weight=200, reg_lambda=10.0,
                      subsample=0.8, colsample_bytree=0.7)),
]

res = {}
for i, (name, prm) in enumerate(CONFIGS, 1):
    tick(f"{i}/{len(CONFIGS)} 학습 중 — {name}")
    try:
        res[name] = run(name, **prm)
    except Exception as e:
        log(f"  {name:30s} 실패: {type(e).__name__}: {str(e)[:120]}")


# =====================================================================
log("\n" + "=" * 92)
log("종합 — XGBoost vs HistGradientBoosting")
log("=" * 92)
log(f"  {'설정':30s} {'총점':>8s} {'변별력':>8s} {'HistGB대비':>11s}")
log("  " + "-" * 64)
log(f"  {'[노트북] HistGB 8시드':30s} {HGB_REF_TOTAL:8.1f} {HGB_REF_DISC:8.1f} "
    f"{0.0:+11.1f}")
for name, z in sorted(res.items(), key=lambda x: -x[1]["변별력"]):
    log(f"  {name:30s} {z['총점']:8.1f} {z['변별력']:8.1f} "
        f"{z['변별력']-HGB_REF_DISC:+11.1f}")

if res:
    best = max(res, key=lambda k: res[k]["변별력"])
    gap = res[best]["변별력"] - HGB_REF_DISC
    log(f"\n  ▶ XGBoost 최고: {best}  (HistGB 대비 {gap:+.1f}점)")
    if gap >= 30:
        log("     → XGBoost가 확실히 낫다. 갈아탈 가치 있음")
    elif gap <= -30:
        log("     → HistGB가 확실히 낫다. XGBoost 접기")
    else:
        log("     → 노이즈 범위(±30). 성능은 비등 —")
        log("        다만 계열이 다르므로 '앙상블 재료'로는 여전히 가치 있음")
        log("        (서로 다르게 틀리는 모델을 섞으면 분산이 줄어든다)")

log(f"\n  총 소요 {time.time()-T0:.0f}초  (device={DEVICE})")
log("=" * 92)
