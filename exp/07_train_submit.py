"""
[07] 제출용 모델 학습 + 저장

실행 (★ 반드시 서버와 같은 버전의 venv로):
  PYTHONIOENCODING=utf-8 <venv311>/Scripts/python.exe -u exp/07_train_submit.py

왜 venv인가:
  로컬 기본 파이썬은 numpy 2.1.3 / pandas 3.0.1인데 평가 서버는 numpy 1.26.4 / pandas 2.0.3.
  numpy 2.x로 만든 pickle은 numpy 1.x에서 ModuleNotFoundError('numpy._core')로 못 연다.
  → 서버와 같은 버전에서 저장해야 한다.

단계:
  1단계 검증  2019~2023 학습 → 2024 채점.  로컬 실험값(총점 708.7)이 재현되는지 확인.
              여기서 재현이 안 되면 파이프라인 어딘가가 틀린 것이므로 저장하지 않는다.
  2단계 최종  2019~2024 전체 학습 → submissions/submit/model/model.pkl 저장

★ 피처 생성 함수는 submissions/submit/script.py 에서 import한다 (복사 금지).
  학습과 추론이 물리적으로 같은 코드를 쓰게 만드는 장치.
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
LOCAL_TARGET = 726.2        # exp/15 확증 실험의 8시드 앙상블 총점 (hand delta 포함)
A1 = 200.0                  # 통산을 prior 쪽으로 당기는 스무딩 (exp/14~15와 동일)
A2 = 200.0                  # 손 조건부를 본인 통산 쪽으로 당기는 스무딩


def log(*a):
    print(*a, flush=True)


def tick(m):
    log(f"  [{time.time()-T0:6.0f}s] {m}")


# ---- submissions/submit/script.py 에서 피처 함수 가져오기 (단일 원본) ----
spec = importlib.util.spec_from_file_location("subm", "submissions/submit/script.py")
subm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subm)
add_features, to_matrix = subm.add_features, subm.to_matrix
log(f"피처 함수 출처: submissions/submit/script.py  (학습·추론 동일 코드 보장)")
log(f"실행 환경: python {sys.version.split()[0]} / numpy {np.__version__} / "
    f"pandas {pd.__version__}")


# ---- 점수 ----
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

# 피처 목록: test.csv 컬럼 순서를 기준으로 잡는다 (서버 입력과 정확히 맞추기 위해)
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
       "f_miss_p", "f_miss_prev1",
       "f_hand_delta", "f_bhand_delta"]      # ← exp/14·15에서 채택 (+17.5)
FEATS = BASE + ENG
log(f"  피처 {len(FEATS)}개 = 원본 {len(BASE)} + 엔지니어링 {len(ENG)}")


# =====================================================================
# hand delta 테이블 (exp/15에서 토이 실행으로 as-of 누수 없음 검증된 로직)
# =====================================================================
def hand_history_table(src, prior, id_col):
    """(eid, season, sh) → 그 시즌 '이전' 시즌들 기준 delta. 학습/검증 행 부착용."""
    t = pd.DataFrame({
        "eid": src[id_col].to_numpy(),
        "season": src["season"].to_numpy(),
        "sh": (src["pitcher_hand"] == src["batter_hand"]).astype("int8").to_numpy(),
        "y": src["control_success"].to_numpy()})
    cond = (t.groupby(["eid", "season", "sh"])["y"].agg(succ="sum", n="size")
             .reset_index().sort_values(["eid", "sh", "season"]))
    g = cond.groupby(["eid", "sh"])
    cond["pc_succ"] = g["succ"].cumsum() - cond["succ"]
    cond["pc_n"] = g["n"].cumsum() - cond["n"]
    over = (t.groupby(["eid", "season"])["y"].agg(succ="sum", n="size")
             .reset_index().sort_values(["eid", "season"]))
    go = over.groupby("eid")
    over["po_succ"] = go["succ"].cumsum() - over["succ"]
    over["po_n"] = go["n"].cumsum() - over["n"]
    tbl = cond.merge(over[["eid", "season", "po_succ", "po_n"]],
                     on=["eid", "season"], how="left")
    p_over = (tbl["po_succ"] + prior * A1) / (tbl["po_n"] + A1)
    p_cond = (tbl["pc_succ"] + p_over * A2) / (tbl["pc_n"] + A2)
    tbl["delta"] = np.where(tbl["po_n"] > 0, p_cond - p_over, np.nan).astype("float32")
    return tbl[["eid", "season", "sh", "delta"]]


def deployment_table(src, prior, id_col):
    """(eid, sh) → 전체 train(≤2024) 합계 기준 delta.
    2025 test 행 입장에서는 '자기 시즌 이전 전부'이므로 as-of 의미가 정확히 같다.
    ★ train에 있는 선수는 sh 0/1 전조합을 만든다 — 해당 손 표본이 0이어도
      as-of 수식과 동일하게 delta=0 (p_cond가 p_over로 완전 수축) [감사 중간-1 수정]"""
    t = pd.DataFrame({
        "eid": src[id_col].to_numpy(),
        "sh": (src["pitcher_hand"] == src["batter_hand"]).astype("int8").to_numpy(),
        "y": src["control_success"].to_numpy()})
    cond = t.groupby(["eid", "sh"])["y"].agg(succ="sum", n="size").reset_index()
    over = (t.groupby("eid")["y"].agg(po_succ="sum", po_n="size").reset_index())
    grid = pd.MultiIndex.from_product(
        [over["eid"].to_numpy(), np.array([0, 1], dtype="int8")],
        names=["eid", "sh"]).to_frame(index=False)
    tbl = grid.merge(cond, on=["eid", "sh"], how="left")
    tbl[["succ", "n"]] = tbl[["succ", "n"]].fillna(0.0)
    tbl = tbl.merge(over, on="eid", how="left")
    p_over = (tbl["po_succ"] + prior * A1) / (tbl["po_n"] + A1)
    p_cond = (tbl["succ"] + p_over * A2) / (tbl["n"] + A2)
    tbl["delta"] = (p_cond - p_over).astype("float32")
    return tbl[["eid", "sh", "delta"]]


def attach_asof(d, tbl_p, tbl_b):
    """학습·검증 행에 시즌 as-of delta 부착 (3-키 조인). add_features 이후 호출."""
    d = d.copy()
    for tbl, idc, feat in [(tbl_p, "pitcher_id", "f_hand_delta"),
                           (tbl_b, "batter_id", "f_bhand_delta")]:
        key = pd.DataFrame({"eid": d[idc].to_numpy(),
                            "season": d["season"].to_numpy(),
                            "sh": d["f_same_hand"].to_numpy("int8")})
        d[feat] = key.merge(tbl, on=["eid", "season", "sh"], how="left",
                            validate="m:1")["delta"].to_numpy("float32")
    return d


def train_ensemble(train_df, prior, tag, tbl_p, tbl_b):
    """8시드 학습 → (모델리스트, 인코더)"""
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train_df[CAT])
    ft = attach_asof(add_features(train_df, prior), tbl_p, tbl_b)
    log(f"  [{tag}] f_hand_delta 커버리지 {ft['f_hand_delta'].notna().mean()*100:.1f}% / "
        f"f_bhand_delta {ft['f_bhand_delta'].notna().mean()*100:.1f}%")
    X = to_matrix(ft, FEATS, enc)
    y = train_df["control_success"].to_numpy()
    models = []
    for i, s in enumerate(SEEDS, 1):
        tick(f"[{tag}] seed={s} 학습 중 ({i}/{len(SEEDS)})...")
        prm = dict(HGB, random_state=s)
        models.append(HistGradientBoostingClassifier(**prm).fit(X, y))
    return models, enc


def predict_ensemble(models, enc, part_df, prior, tbl_p, tbl_b):
    ft = attach_asof(add_features(part_df, prior), tbl_p, tbl_b)
    X = to_matrix(ft, FEATS, enc)
    acc = np.zeros(len(part_df))
    for m in models:
        acc += m.predict_proba(X)[:, 1]
    return acc / len(models)


# =====================================================================
# 1단계 — 검증 (파이프라인이 맞는지 확인)
# =====================================================================
log("\n" + "=" * 84)
log("1단계 검증: 2019~2023 학습 → 2024 채점  (실험값 재현 확인)")
log("=" * 84)

tr = df[df["season"] <= 2023].reset_index(drop=True)
va = df[df["season"] == 2024].reset_index(drop=True)
prior_v = float(tr["control_success"].mean())
# as-of 테이블은 전체 df로 만들어도 안전 (season=S 키에는 S 이전만 들어감, exp/15 검증)
tbl_pv = hand_history_table(df, prior_v, "pitcher_id")
tbl_bv = hand_history_table(df, prior_v, "batter_id")
models_v, enc_v = train_ensemble(tr, prior_v, "검증", tbl_pv, tbl_bv)
p_va = predict_ensemble(models_v, enc_v, va, prior_v, tbl_pv, tbl_bv)
z = decompose(p_va, va["control_success"].to_numpy())
log(f"\n  8시드 앙상블 → 총점 {z['총점']:.1f} = 변별력 {z['변별력']:.1f} "
    f"- 벌점 {z['중심벌점']:.1f}  (d={z['d']:+.4f})")
log(f"  기대값(15 실험) {LOCAL_TARGET:.1f}   차이 {z['총점']-LOCAL_TARGET:+.1f}")
if abs(z["총점"] - LOCAL_TARGET) > 30:
    log("  ⚠️ 30점 넘게 벗어남 (앙상블 재현 2σ≈15의 2배) — 파이프라인 점검 필요")
    log("  ★ 저장 단계로 진행하지 않는다. 기존 model.pkl을 보존한다.")
    raise SystemExit(1)
log("  ✅ 재현 확인. 제출 코드 경로가 정상 동작한다.")

del models_v, enc_v, tr, va, tbl_pv, tbl_bv


# =====================================================================
# 2단계 — 최종 학습 (2019~2024 전체) + 저장
# =====================================================================
log("\n" + "=" * 84)
log("2단계 최종: 2019~2024 전체 학습 → 저장")
log("=" * 84)

prior = float(df["control_success"].mean())
log(f"  prior (전체 평균 성공률) = {prior:.6f}")
tbl_p = hand_history_table(df, prior, "pitcher_id")
tbl_b = hand_history_table(df, prior, "batter_id")
models, enc = train_ensemble(df, prior, "최종", tbl_p, tbl_b)

# 배포 테이블: (eid, sh)별 전체 train 합계 — 2025 행의 '자기 이전 전부'
dep_p = deployment_table(df, prior, "pitcher_id")
dep_b = deployment_table(df, prior, "batter_id")
log(f"  배포 테이블: 투수 {dep_p['eid'].nunique()}명 {len(dep_p)}행 / "
    f"타자 {dep_b['eid'].nunique()}명 {len(dep_b)}행")

os.makedirs("submissions/submit/model", exist_ok=True)
bundle = {"models": models, "encoder": enc, "prior": prior, "feats": FEATS,
          "seeds": SEEDS, "hgb": HGB, "n_train": int(len(df)),
          "hand_tbl_p": dep_p, "hand_tbl_b": dep_b}
joblib.dump(bundle, "submissions/submit/model/model.pkl", compress=3)
size = os.path.getsize("submissions/submit/model/model.pkl") / 1024 ** 2
tick(f"저장 완료 submissions/submit/model/model.pkl  ({size:.1f} MB)")

# 저장한 걸 다시 열어서 5행 test.csv로 ★서빙 경로 그대로★ 예측되는지 확인
log("\n  저장본 재로드 검증 (script.py 서빙 경로)...")
b2 = joblib.load("submissions/submit/model/model.pkl")
t5 = pd.read_csv("data/test.csv", encoding="utf-8-sig")
t5.columns = [c.replace("﻿", "").strip() for c in t5.columns]
ft5 = subm.add_features(t5, b2["prior"])
ft5 = subm.attach_hand_deltas(ft5, b2["hand_tbl_p"], b2["hand_tbl_b"])
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
