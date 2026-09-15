"""
평가 서버 추론 코드 — v8: 타겟 분해 멀티클래스(v6) + team_cat cat_features(v5) 콤보
                        CatBoost 8시드, 79피처, cat_features 5개

  v6(제출자: dou, submit14, LB 1033.99)의 5클래스 타겟 분해{성공, 미들, 리버스,
  미들∩리버스, 빅미스}는 그대로 두고, 거기에 team_cat(우리 팀 검증, LB 1018.1353에서
  CS79 대비 +24.51 확인)의 pitcher_team_id/batter_team_id cat_features 추가만 얹었다.
  피처 79개·CS/PT 로직은 v5/v6과 완전히 동일 — cat_features만 3개→5개.

  ./data/test.csv + ./model/model.pkl → ./output/submission.csv

★ 행 간 독립성 (대회 규칙):
  모든 피처는 각 행 자신의 값 + model.pkl 안의 '학습 데이터 유래 고정 상수표'만으로
  계산된다. test의 다른 행을 참조하는 연산(집계/정렬/누적/분포)은 전혀 없다.
  CS 피처 = 행의 asof_*(공식 입력) − 학습종료 시점 누적 상수 — 공식 Q&A가 명시
  허용한 방식이다 ("학습 데이터에서 추출한 정보를 바탕으로 독립 적용 → 가능").
  5클래스 타겟의 미들/리버스 라벨도 구종 라벨과 동일한 방식(train 내 asof 차분 복원)으로
  train에서만 만들어지고, 추론 입력에는 전혀 들어가지 않는다.
★ 이 파일이 피처 로직의 단일 원본이다 — 학습 스크립트(exp/73_train_target5_teamcat.py)가
  여기서 import한다.

리더보드 결과: 1059.0501189623 (팀 최고 기록, 2026-08-27 확정)
"""
import os
import numpy as np
import pandas as pd
import joblib

NTREE_END = 500   # 예측에 사용할 트리 수 (학습된 8시드 모델 각 500트리 이하; 500 = 전체 사용)
SHIFT_DELTA = 0.012   # 학습 데이터(train.csv) 연도별 성공률 하락 추세(2019 .565 → 2024 .486, 연평균 −0.016)에 근거한
                       # 사전 중심 보정 상수. 학습 데이터 유래 고정값이며 각 test 행에 독립 적용 (다른 test 행·집계 미참조)

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT = ["top_bottom", "game_type", "base_state"]
TEAM_CAT = ["pitcher_team_id", "batter_team_id"]  # team_cat 콤보: cat_features로 추가
K_SHRINK = 50.0
MIN_CS_N = 5

P_RATES = {"succ": "asof_pitcher_success_rate", "ball": "asof_pitcher_ball_rate",
           "strike": "asof_pitcher_strike_rate", "middle": "asof_pitcher_middle_rate",
           "fb": "asof_pitcher_fastball_rate"}
B_RATES = {"succ": "asof_batter_success_rate", "middle": "asof_batter_middle_rate"}
CS_FEATS = ["f_cs_p_logn", "f_cs_p_rate", "f_cs_p_delta", "f_cs_p_ball_d",
            "f_cs_p_strike_d", "f_cs_p_middle_d", "f_cs_p_fb_d",
            "f_cs_b_logn", "f_cs_b_rate", "f_cs_b_delta", "f_cs_b_middle_d"]

# 구종 3피처 (v5/v6과 동일): train에서 복원한 투구별 구종 라벨(공식 Q&A로 train 내 사용 허용)로
# 사전 학습된 성향 테이블·확률 모델을 각 행에 행 독립 적용. 현재 투구의 실제 구종은
# 추론 입력에 결코 들어가지 않는다 (P(구종|투구 전 정보) 예측값만 사용).
PT_FEATS = ["f_mixcg_fb", "f_mixcg_brk", "f_pfb"]
PFB_IN = ["balls_before", "strikes_before", "outs_before", "inning", "num_runners_on",
          "runner_on_1b", "runner_on_2b", "runner_on_3b", "pitcher_hand", "batter_hand",
          "score_diff_pitcher_team", "li", "asof_pitcher_fastball_rate",
          "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]


def attach_pt(d, mix_tbl, pfb_model):
    """구종 성향(투수×카운트군, 학습 유래 고정표) + P(fastball|상황) 모델 출력"""
    d = d.copy()
    b, s = d["balls_before"].to_numpy(), d["strikes_before"].to_numpy()
    cg = np.where(s > b, 2, np.where(b > s, 0, 1)).astype("int8")
    key = pd.DataFrame({"pitcher_id": d["pitcher_id"].to_numpy(), "_cg": cg})
    m = key.merge(mix_tbl, on=["pitcher_id", "_cg"], how="left")
    d["f_mixcg_fb"] = m["mix_fb"].to_numpy("float32")
    d["f_mixcg_brk"] = m["mix_brk"].to_numpy("float32")
    d["f_pfb"] = pfb_model.predict_proba(d[PFB_IN].fillna(-999))[:, 1].astype("float32")
    return d


# =====================================================================
# 기본 엔지니어링 18개 (기존 65피처 구성과 동일)
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


# =====================================================================
# 시즌 진행분(CS) 11개 — 행의 asof − 상수표 (행 독립 산술)
# =====================================================================
def attach_cs(rows, cp, cb_):
    """cp/cb_: DataFrame[id, N_end, S_succ, ...] — 학습 데이터 유래 고정 상수표"""
    d = rows.copy()
    m = pd.DataFrame({"id": d.pitcher_id.to_numpy()}).merge(cp, on="id", how="left")
    N_end = m["N_end"].fillna(0).to_numpy("float64")
    n_now = d["asof_pitcher_n"].fillna(0).to_numpy("float64")
    cs_n = np.maximum(n_now - N_end, 0.0)
    ok = cs_n >= MIN_CS_N
    d["f_cs_p_logn"] = np.log1p(cs_n).astype("float32")
    for k, col in P_RATES.items():
        car = d[col].to_numpy("float64")
        S_end = m[f"S_{k}"].fillna(0).to_numpy("float64")
        cs_S = np.clip(np.nan_to_num(car) * n_now - S_end, 0.0, None)
        rate = np.where(ok, (cs_S + K_SHRINK * np.nan_to_num(car)) / (cs_n + K_SHRINK),
                        np.nan)
        delta = np.where(ok, cs_S / np.maximum(cs_n, 1) - car, np.nan)
        if k == "succ":
            d["f_cs_p_rate"] = rate.astype("float32")
            d["f_cs_p_delta"] = delta.astype("float32")
        else:
            d[f"f_cs_p_{k}_d"] = delta.astype("float32")
    m = pd.DataFrame({"id": d.batter_id.to_numpy()}).merge(cb_, on="id", how="left")
    N_end = m["N_end"].fillna(0).to_numpy("float64")
    n_now = d["asof_batter_n"].fillna(0).to_numpy("float64")
    cs_n = np.maximum(n_now - N_end, 0.0)
    ok = cs_n >= MIN_CS_N
    d["f_cs_b_logn"] = np.log1p(cs_n).astype("float32")
    for k, col in B_RATES.items():
        car = d[col].to_numpy("float64")
        S_end = m[f"S_{k}"].fillna(0).to_numpy("float64")
        cs_S = np.clip(np.nan_to_num(car) * n_now - S_end, 0.0, None)
        rate = np.where(ok, (cs_S + K_SHRINK * np.nan_to_num(car)) / (cs_n + K_SHRINK),
                        np.nan)
        delta = np.where(ok, cs_S / np.maximum(cs_n, 1) - car, np.nan)
        if k == "succ":
            d["f_cs_b_rate"] = rate.astype("float32")
            d["f_cs_b_delta"] = delta.astype("float32")
        else:
            d[f"f_cs_b_{k}_d"] = delta.astype("float32")
    return d


def build_matrix(d, feats):
    """CatBoost 입력 프레임. CAT 3개는 기존과 동일하게 str 캐스팅. TEAM_CAT 2개(숫자형)는
    float 표기("13.0")로 쪼개지는 걸 막기 위해 int64로 정규화한 뒤 문자열로 캐스팅
    (script_v5_team_cat.py의 build_matrix와 동일한 처리)."""
    out = d[[c for c in feats if c not in CAT and c not in TEAM_CAT]].copy()
    for c in CAT:
        out[c] = d[c].astype(str)
    for c in TEAM_CAT:
        out[c] = d[c].fillna(-1).astype("int64").astype(str)
    return out


# =====================================================================
# main
# =====================================================================
def main():
    DATA_DIR, MODEL_DIR, OUT_DIR = "./data", "./model", "./output"
    print("Load model...")
    b = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    models = b["cb_models"]
    feats = b["feats"]
    prior = b["prior"]
    print(f"  cb_models={len(models)}  features={len(feats)}  prior={prior:.6f}")

    print("Load test data...")
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig")
    test.columns = [c.replace("﻿", "").strip() for c in test.columns]
    print(f"  test rows={len(test)}")

    print("Build features...")
    ft = attach_cs(add_features(test, prior), b["cs_const_p"], b["cs_const_b"])
    ft = attach_pt(ft, b["mix_tbl"], b["pfb_model"])
    if len(ft):
        print(f"  f_cs_p_rate coverage={ft['f_cs_p_rate'].notna().mean()*100:.1f}%")
        print(f"  f_mixcg_fb coverage={ft['f_mixcg_fb'].notna().mean()*100:.1f}%  "
              f"f_pfb mean={ft['f_pfb'].mean():.3f}")
    X = build_matrix(ft, feats)
    print(f"  X={X.shape}")

    print(f"Inference ({len(models)} models)...")
    if len(test) == 0:
        preds = np.zeros(0, dtype=float)
    else:
        acc = np.zeros(len(test), dtype=np.float64)
        for i, m in enumerate(models, 1):
            acc += m.predict_proba(X, ntree_end=NTREE_END)[:, 0]      # 멀티클래스: 클래스0 = 성공
            print(f"  cb model {i}/{len(models)} done")
        preds = acc / len(models)
        print(f"  multiclass P(success) averaged, version={b.get('version')}  ntree_end={NTREE_END} shift={SHIFT_DELTA} tree_count={[m.tree_count_ for m in models]}")
    preds = np.clip(preds - SHIFT_DELTA, 0.0, 1.0)   # 학습 유래 고정 상수 차감 — 행 독립 (행 하나만 있어도 결과 동일)
    if len(preds):
        print(f"  pred mean={preds.mean():.6f}  min={preds.min():.6f}  max={preds.max():.6f}")

    print("Build submission...")
    sub = None
    sp = os.path.join(DATA_DIR, "sample_submission.csv")
    if os.path.exists(sp):
        s = pd.read_csv(sp, encoding="utf-8-sig")
        s.columns = [c.replace("﻿", "").strip() for c in s.columns]
        if ID_COL in s.columns and len(s) == len(test):
            pred_map = dict(zip(test[ID_COL], preds))
            vals = [pred_map.get(r, float(prior)) for r in s[ID_COL]]
            sub = pd.DataFrame({ID_COL: s[ID_COL], TARGET_COL: vals})
            print(f"  sample_submission 순서 사용 (rows={len(sub)})")
    if sub is None:
        sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: preds})
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "submission.csv")
    sub.to_csv(out, index=False, encoding="utf-8")
    print(f"Saved: {out}  rows={len(sub)}")


if __name__ == "__main__":
    main()
