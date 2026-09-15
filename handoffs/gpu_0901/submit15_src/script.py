"""
평가 서버 추론 코드 — v7: 3-way 앙상블 (CB65 / CS76 / CS79)

  p = w65·CB65평균 + w76·CS76평균 + w79·CS79평균

각 계열은 이미 리더보드에서 단독 측정된 확정 모델이다:
  CB65 898.62 / CS76 990.96 / CS79 993.63
가중치는 그 LB 점수 + 실측 다양성(CS79⊗CB65 50:50 = 975.50)으로 산출.
앙상블 가중치를 리더보드 점수 참고로 선택하는 것은 공식 Q&A(08-12) 명시 허용.

★ 행 독립성: 세 경로 모두 각 행 자신의 값 + model.pkl 안의 학습 유래 고정 상수표만
  사용한다. 혼합은 행별 산술. test의 다른 행을 참조하는 연산 없음.
"""

import os
import numpy as np
import pandas as pd
import joblib

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT = ["top_bottom", "game_type", "base_state"]
K_SHRINK = 50.0
MIN_CS_N = 5

P_RATES = {"succ": "asof_pitcher_success_rate", "ball": "asof_pitcher_ball_rate",
           "strike": "asof_pitcher_strike_rate", "middle": "asof_pitcher_middle_rate",
           "fb": "asof_pitcher_fastball_rate"}
B_RATES = {"succ": "asof_batter_success_rate", "middle": "asof_batter_middle_rate"}
PFB_IN = ["balls_before", "strikes_before", "outs_before", "inning", "num_runners_on",
          "runner_on_1b", "runner_on_2b", "runner_on_3b", "pitcher_hand", "batter_hand",
          "score_diff_pitcher_team", "li", "asof_pitcher_fastball_rate",
          "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]


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


def attach_cs(rows, cp, cb_):
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


def attach_pt(d, mix_tbl, pfb_model):
    d = d.copy()
    b, s = d["balls_before"].to_numpy(), d["strikes_before"].to_numpy()
    cg = np.where(s > b, 2, np.where(b > s, 0, 1)).astype("int8")
    key = pd.DataFrame({"pitcher_id": d["pitcher_id"].to_numpy(), "_cg": cg})
    m = key.merge(mix_tbl, on=["pitcher_id", "_cg"], how="left")
    d["f_mixcg_fb"] = m["mix_fb"].to_numpy("float32")
    d["f_mixcg_brk"] = m["mix_brk"].to_numpy("float32")
    d["f_pfb"] = pfb_model.predict_proba(d[PFB_IN].fillna(-999))[:, 1].astype("float32")
    return d


def frame_of(d, feats):
    out = d[[c for c in feats if c not in CAT]].copy()
    for c in CAT:
        out[c] = d[c].astype(str)
    return out


def avg_proba(models, X, tag):
    acc = np.zeros(len(X), dtype=np.float64)
    for i, m in enumerate(models, 1):
        acc += m.predict_proba(X)[:, 1]
        print(f"  {tag} {i}/{len(models)} done")
    return acc / len(models)


def main():
    DATA_DIR, MODEL_DIR, OUT_DIR = "./data", "./model", "./output"
    print("Load model...")
    b = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    W = b["weights"]
    prior = b["prior"]
    print(f"  weights: CB65={W['CB65']} CS76={W['CS76']} CS79={W['CS79']}  "
          f"prior={prior:.6f}")

    print("Load test data...")
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig")
    test.columns = [c.replace("﻿", "").strip() for c in test.columns]
    print(f"  test rows={len(test)}")

    print("Build features...")
    ft = add_features(test, prior)
    ft_cs = attach_cs(ft, b["cs_const_p"], b["cs_const_b"])
    ft_79 = attach_pt(ft_cs, b["mix_tbl"], b["pfb_model"])
    X65 = frame_of(ft, b["feats65"])
    X76 = frame_of(ft_cs, b["feats76"])
    X79 = frame_of(ft_79, b["feats79"])
    print(f"  X65={X65.shape} X76={X76.shape} X79={X79.shape}")
    if len(ft_cs):
        print(f"  f_cs_p_rate coverage={ft_cs['f_cs_p_rate'].notna().mean()*100:.1f}%")

    print("Inference...")
    if len(test) == 0:
        preds = np.zeros(0, dtype=float)
    else:
        p65 = avg_proba(b["cb65_models"], X65, "cb65")
        p76 = avg_proba(b["cs76_models"], X76, "cs76")
        p79 = avg_proba(b["cs79_models"], X79, "cs79")
        preds = W["CB65"] * p65 + W["CS76"] * p76 + W["CS79"] * p79
        print(f"  blended 3-way")
    preds = np.clip(preds, 0.0, 1.0)
    if len(preds):
        print(f"  pred mean={preds.mean():.6f}  min={preds.min():.6f}  "
              f"max={preds.max():.6f}")

    print("Build submission...")
    sub = None
    sp = os.path.join(DATA_DIR, "sample_submission.csv")
    if os.path.exists(sp):
        s = pd.read_csv(sp, encoding="utf-8-sig")
        s.columns = [c.replace("﻿", "").strip() for c in s.columns]
        if ID_COL in s.columns and len(s) == len(test):
            pm = dict(zip(test[ID_COL], preds))
            sub = pd.DataFrame({ID_COL: s[ID_COL],
                                TARGET_COL: [pm.get(r, float(prior)) for r in s[ID_COL]]})
            print(f"  sample_submission 순서 사용 (rows={len(sub)})")
    if sub is None:
        sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: preds})
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "submission.csv")
    sub.to_csv(out, index=False, encoding="utf-8")
    print(f"Saved: {out}  rows={len(sub)}")


if __name__ == "__main__":
    main()
