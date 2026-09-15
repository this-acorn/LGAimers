"""
평가 서버가 실행하는 추론 코드.

  ./data/test.csv  +  ./model/model.pkl   →   ./output/submission.csv

★ 이 파일이 피처 생성 로직의 '단일 원본'이다.
  학습 스크립트(exp/07_train_submit.py)가 여기서 add_features()를 import해서 쓴다.
  → 학습과 추론이 물리적으로 같은 코드를 쓰므로 불일치가 원천 차단된다.

★ 행 간 독립성:
  각 행의 피처는 그 행의 값만으로 계산된다. test.csv의 다른 행을 참조하는
  집계·정렬·groupby·누적이 전혀 없다. (대회 규칙 필수 요건)
"""

import os
import numpy as np
import pandas as pd
import joblib

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT = ["top_bottom", "game_type", "base_state"]


# =====================================================================
# 피처 엔지니어링  ★ 학습·추론 공용 (exp/common.py 와 동일 내용)
# =====================================================================
def add_features(d, prior):
    """
    투구 '직전' 정보만 사용. 다른 행을 참조하지 않으므로 test에도 그대로 적용 가능.

    prior : 베이지안 스무딩의 사전확률. 학습 데이터의 control_success 평균.
            ★ 모델과 함께 저장해두고 추론 시 같은 값을 쓴다.
    """
    d = d.copy()
    b, s = d["balls_before"], d["strikes_before"]

    # G1 볼카운트
    d["f_count_state"] = (b * 3 + s).astype("int8")
    d["f_count_diff"] = (b - s).astype("int8")
    d["f_is_3ball"] = (b == 3).astype("int8")
    d["f_is_2strike"] = (s == 2).astype("int8")
    d["f_is_full"] = ((b == 3) & (s == 2)).astype("int8")

    # G2 매치업 / 상황 압박  ← ablation에서 유일하게 노이즈를 넘은 그룹 (+107)
    d["f_same_hand"] = (d["pitcher_hand"] == d["batter_hand"]).astype("int8")
    d["f_scoring_pos"] = ((d["runner_on_2b"] == 1) | (d["runner_on_3b"] == 1)).astype("int8")
    d["f_any_runner"] = (d["num_runners_on"] > 0).astype("int8")

    # G3 asof 신뢰도
    pn = d["asof_pitcher_n"].fillna(0)
    bn = d["asof_batter_n"].fillna(0)
    d["f_log_pn"] = np.log1p(pn).astype("float32")
    d["f_log_bn"] = np.log1p(bn).astype("float32")
    a = 200.0
    d["f_smooth_p"] = ((d["asof_pitcher_success_rate"].fillna(prior) * pn + prior * a)
                       / (pn + a)).astype("float32")
    d["f_smooth_b"] = ((d["asof_batter_success_rate"].fillna(prior) * bn + prior * a)
                       / (bn + a)).astype("float32")

    # G4 최근 폼
    d["f_form_dev1"] = (d["asof_pitcher_prev1_game_success_rate"]
                        - d["asof_pitcher_success_rate"]).astype("float32")
    d["f_form_dev3"] = (d["asof_pitcher_prev3_game_success_rate"]
                        - d["asof_pitcher_success_rate"]).astype("float32")
    d["f_form_trend"] = (d["asof_pitcher_prev1_game_success_rate"]
                         - d["asof_pitcher_prev5_game_success_rate"]).astype("float32")

    # G5 투수 vs 타자 차이
    d["f_pb_diff"] = (d["asof_pitcher_success_rate"]
                      - d["asof_batter_success_rate"]).astype("float32")

    # G6 결측 플래그
    d["f_miss_p"] = d["asof_pitcher_success_rate"].isna().astype("int8")
    d["f_miss_prev1"] = d["asof_pitcher_prev1_game_success_rate"].isna().astype("int8")
    return d


def attach_hand_deltas(d, p_tbl, b_tbl):
    """
    투수/타자별 손 매치업 이력 delta를 (선수ID, 손일치여부) 키로 조인.

    p_tbl / b_tbl : DataFrame[eid, sh, delta]
      학습 데이터(train.csv)만으로 사전 계산되어 model.pkl에 저장된 고정 테이블.
      각 행은 자기 자신의 (ID, 손일치) 키 하나만 조회한다 — test의 다른 행을
      전혀 참조하지 않는 행 독립 연산 (대회 규칙 요건).
      train에 없는 선수는 NaN → GBDT 네이티브 결측 처리 (학습 때와 같은 의미).
    """
    d = d.copy()
    for tbl, idc, feat in [(p_tbl, "pitcher_id", "f_hand_delta"),
                           (b_tbl, "batter_id", "f_bhand_delta")]:
        key = pd.DataFrame({"eid": d[idc].to_numpy(),
                            "sh": d["f_same_hand"].to_numpy("int8")})
        d[feat] = key.merge(tbl, on=["eid", "sh"], how="left",
                            validate="m:1")["delta"].to_numpy("float32")
    return d


def to_matrix(df, feats, enc):
    """DataFrame → 모델 입력 행렬. 결측(NaN)은 GBDT가 알아서 처리한다."""
    cats = [c for c in CAT if c in feats]
    nums = [c for c in feats if c not in cats]
    return np.column_stack([enc.transform(df[cats]).astype("float32"),
                            df[nums].to_numpy("float32")])


# =====================================================================
# main
# =====================================================================
def main():
    DATA_DIR = "./data"
    MODEL_DIR = "./model"
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(DATA_DIR, "test.csv")
    SAMPLE_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
    MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 ----
    print("Load model...")
    bundle = joblib.load(MODEL_PATH)
    models = bundle["models"]
    enc = bundle["encoder"]
    prior = bundle["prior"]
    feats = bundle["feats"]
    print(f"  models={len(models)}  features={len(feats)}  prior={prior:.6f}")

    # ---- 테스트 데이터 ----
    print("Load test data...")
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    test.columns = [c.replace("﻿", "").strip() for c in test.columns]
    print(f"  test rows={len(test)}")

    # ---- 피처 (행 단위 계산. 다른 행 참조 없음) ----
    print("Build features...")
    ft = add_features(test, prior)
    if bundle.get("hand_tbl_p") is not None:
        ft = attach_hand_deltas(ft, bundle["hand_tbl_p"], bundle["hand_tbl_b"])
        for c in ("f_hand_delta", "f_bhand_delta"):
            if len(ft):
                print(f"  {c}: NaN {ft[c].isna().mean()*100:.1f}%")
    else:
        print("  hand delta 비활성 (65피처 모델)")
    X = to_matrix(ft, feats, enc)
    print(f"  X={X.shape}")

    # ---- 예측: 시드 앙상블 평균 ----
    print(f"Inference ({len(models)} models)...")
    if len(test) == 0:
        preds = np.zeros(0, dtype=float)
    else:
        acc = np.zeros(len(test), dtype=np.float64)
        for i, m in enumerate(models, 1):
            acc += m.predict_proba(X)[:, 1]
            print(f"  model {i}/{len(models)} done")
        preds = acc / len(models)

    # ---- CatBoost 혼합 (있을 때만) ----
    # p = (1-w)*HGB평균 + w*CatBoost평균 — 행별 산술이므로 행 독립성 유지.
    # 두 모델 계열 모두 train.csv만으로 사전 학습되어 model.pkl에 저장된 것.
    cb_models = bundle.get("cb_models")
    if cb_models and len(test) > 0:
        w = float(bundle["w_cb"])
        print(f"CatBoost inference ({len(cb_models)} models, w={w})...")
        fcb = ft[bundle["cb_feats_num"]].copy()
        for c in bundle["cb_feats_cat"]:
            fcb[c] = ft[c].astype(str)
        acc_cb = np.zeros(len(test), dtype=np.float64)
        for i, m in enumerate(cb_models, 1):
            acc_cb += m.predict_proba(fcb)[:, 1]
            print(f"  cb model {i}/{len(cb_models)} done")
        preds = (1.0 - w) * preds + w * (acc_cb / len(cb_models))
        print(f"  mixed: w_cb={w}")
    preds = np.clip(preds, 0.0, 1.0)
    if len(preds):
        print(f"  pred mean={preds.mean():.6f}  min={preds.min():.6f}  max={preds.max():.6f}")

    # ---- 제출 파일 ----
    # sample_submission이 있고 행 수가 맞으면 그 row_id 순서를 따른다.
    # 없거나 행 수가 안 맞으면 test.csv 순서로 직접 만든다. (방어 코드)
    print("Build submission...")
    sub = None
    if os.path.exists(SAMPLE_PATH):
        s = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
        s.columns = [c.replace("﻿", "").strip() for c in s.columns]
        if ID_COL in s.columns and len(s) == len(test):
            pred_map = dict(zip(test[ID_COL], preds))
            vals = [pred_map.get(r, float(prior)) for r in s[ID_COL]]
            sub = pd.DataFrame({ID_COL: s[ID_COL], TARGET_COL: vals})
            print(f"  sample_submission 순서 사용 (rows={len(sub)})")
        else:
            print(f"  sample_submission 불일치(rows={len(s)} vs test={len(test)}) → test 순서 사용")
    else:
        print("  sample_submission 없음 → test 순서 사용")

    if sub is None:
        sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: preds})

    os.makedirs(OUT_DIR, exist_ok=True)
    sub.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved: {OUT_PATH}  rows={len(sub)}")


if __name__ == "__main__":
    main()
