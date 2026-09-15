# script.py
"""[실험적 제출] 조아님(김조아) F-Regime075 모델(anchor, 원본 그대로) 위에,
R행에만 우리 모델(XGBoost+LightGBM+CatBoost 15-seed 앙상블) 예측값을
입력으로 하는 소형 CatBoost 잔차보정을 scale=0.15로 얹은 버전.

judgement:
- [배경] JM(이정민)님의 R Residual(Candidate4 메타피처, +1.02점 실측)과
  같은 구조 - "약한 모델 전체를 블렌딩"(이미 실패, F/R조건부 -3.63)이
  아니라 "anchor 위에 독립 신호로 만든 작은 잔차보정만 추가". 다만 JM은
  Candidate4(ExtraTrees+Beta) 피처를 썼고, 여기서는 **우리 모델의 예측값
  자체**를 입력으로 썼다 - team_asof(우리 최대 성공 피처)는 이미 이
  구조(check_team_asof_residual.py)로도 기각(-7.33)됐기 때문에 다른 신호를
  시도.
- [로컬 검증] `joa_1126_work/experiments/check_our_model_residual.py`
  (season<vs→vs out-of-sample, anchor는 v3_decay55-style 재학습, 우리
  모델은 61피처 LightGBM 프록시, R행만, pitcher-cluster 부트스트랩 CI):
  scale=0.10~0.15에서 **2022·2024 둘 다 cluster CI 전부 양수**(통계적
  유의미), 2023은 중립(손해 아님, 이 시즌 자체 예측 난이도 문제로 추정).
  team_asof·main_weights·is_same_hand 등 오늘 시도한 다른 모든 후보와
  달리 두 시즌에서 유의미한 개선을 보인 유일한 후보.
- [scale 선택] 점추정 최댓값(0.25)이 아니라, cluster CI 하한이 확실히
  양수로 유지되는 범위(0.10~0.15) 중 **0.15**를 선택 - JM님이 점추정
  최댓값(0.100) 대신 분산까지 고려해 0.075를 고른 것과 같은 논리.
- [한계] 이건 anchor+우리 모델 예측값 조합을 **로컬 프록시**(재학습된
  v3_decay55 anchor + LightGBM 61피처 proxy)로 검증한 것이지, 이 패키지가
  실제로 쓰는 조합(그녀의 진짜 pretrained anchor + 우리의 진짜 15-seed
  앙상블)을 직접 out-of-sample로 검증한 건 아니다 - 오늘 main_weights
  사례처럼 "일부만 뗀 프록시가 실제와 다를 수 있다"는 위험은 여전히
  존재. 다만 이번엔 프록시 자체가 진짜 out-of-sample(season<vs→vs
  재학습)이라 in-sample이었던 main_weights 사례보다는 신뢰도가 높다.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

MODEL_DIR = "./model"
sys.path.insert(0, MODEL_DIR)

from src.preprocessing_v2 import build_v2_features, build_v3_features  # noqa: E402
from src.adaptive_gate import build_gate_features  # noqa: E402
from src.psych_latent import build_production_features, apply_linear_residual  # noqa: E402
from src.psych_film_expert import apply_saved_regime  # noqa: E402
from src.split_priors import apply_split_priors  # noqa: E402
from src.league_transition import transition_features  # noqa: E402

ID_COL = "row_id"
TARGET_COL = "control_success"
RESIDUAL_SCALE = 0.15


def predict_teammate(test):
    meta = json.loads((open(os.path.join(MODEL_DIR, "manifest.json"), encoding="utf-8")).read())
    ps = pd.read_pickle(os.path.join(MODEL_DIR, "pitcher_snapshots.pkl"))
    bs = pd.read_pickle(os.path.join(MODEL_DIR, "batter_snapshots.pkl"))
    ms = pd.read_pickle(os.path.join(MODEL_DIR, "pitchmix_snapshots.pkl"))
    tm = os.path.join(MODEL_DIR, "trackman_prior_features.csv")

    x2, base2 = build_v2_features(test, meta["prior"], ps, tm)
    x3, base3 = build_v3_features(test, meta["prior"], ps, bs, ms, tm)

    seeds = meta.get("seeds")

    def load_reg(stem):
        names = [f"{stem}_seed{s}.cbm" for s in seeds] if seeds else [f"{stem}.cbm"]
        out = []
        for filename in names:
            model = CatBoostRegressor()
            model.load_model(os.path.join(MODEL_DIR, filename))
            out.append(model)
        return out

    predictions = []
    for stem, x, base in [
        ("v2_decay55", x2, base2),
        ("v3_decay55", x3, base3),
        ("v3_decay30", x3, base3),
    ]:
        member = [np.clip(base + m.predict(x), 1e-6, 1 - 1e-6) for m in load_reg(stem)]
        predictions.append(np.mean(member, axis=0))

    regime = json.loads(open(os.path.join(MODEL_DIR, "f_regime_meta.json")).read())
    futures = test["game_type"].eq("F").to_numpy()

    def f_reg_mean(stem, count, x, base):
        member = []
        for j in range(count):
            m = CatBoostRegressor()
            m.load_model(os.path.join(MODEL_DIR, f"{stem}_{j}.cbm"))
            member.append(np.clip(base + m.predict(x), 1e-6, 1 - 1e-6))
        return np.mean(member, axis=0)

    if futures.any():
        f2 = f_reg_mean("f_v2_all", 4, x2, base2)
        predictions[0] = np.where(futures, predictions[0] + regime["v2_scale"] * (f2 - predictions[0]), predictions[0])
        f55 = f_reg_mean("f_v355_recent", 6, x3, base3)
        predictions[1] = np.where(futures, predictions[1] + regime["v355_scale"] * (f55 - predictions[1]), predictions[1])
        f30a = f_reg_mean("f_v330_all", 4, x3, base3)
        f30r = f_reg_mean("f_v330_recent", 2, x3, base3)
        recent_inner = predictions[2] + regime["v330_recent_inner_scale"] * (f30r - predictions[2])
        f30 = regime["v330_all_weight"] * f30a + (1 - regime["v330_all_weight"]) * recent_inner
        predictions[2] = np.where(futures, predictions[2] + regime["v330_scale"] * (f30 - predictions[2]), predictions[2])

    risks = []
    for name in ("middle", "wild", "reverse"):
        stems = [f"subtype_{name}_seed{s}.cbm" for s in seeds] if seeds else [f"subtype_{name}.cbm"]
        member = []
        for filename in stems:
            model = CatBoostClassifier()
            model.load_model(os.path.join(MODEL_DIR, filename))
            member.append(model.predict_proba(x3)[:, 1])
        risk = np.mean(member, axis=0)
        if futures.any():
            fm = CatBoostClassifier()
            fm.load_model(os.path.join(MODEL_DIR, f"f_subtype_{name}.cbm"))
            fr = fm.predict_proba(x3)[:, 1]
            risk = np.where(futures, risk + regime["subtype_scale"] * (fr - risk), risk)
        risks.append(risk)

    original_main = np.average(
        np.vstack(predictions), axis=0, weights=[0.27358084, 0.26512224, 0.46129691]
    )
    original_z = np.column_stack([original_main] + risks)
    original_p = 0.0300329767 + original_z @ np.asarray(
        [0.93505266, -0.00520129, 0.01091677, -0.02528331]
    )

    if "direct_coefficients" in meta:
        z = np.column_stack(predictions + risks)
        p = meta["stack_intercept"] + z @ np.asarray(meta["direct_coefficients"])
    else:
        main_p = np.average(np.vstack(predictions), axis=0, weights=meta["main_weights"])
        z = np.column_stack([main_p] + risks)
        p = meta["stack_intercept"] + z @ np.asarray(meta["stack_coefficients"])

    if meta.get("adaptive_gate", False):
        gate_x = build_gate_features(test, predictions, risks, np.clip(original_p, 1e-6, 1 - 1e-6))
        gate = CatBoostRegressor()
        gate.load_model(os.path.join(MODEL_DIR, "adaptive_gate.cbm"))
        p = original_p + float(meta.get("gate_scale", 1.0)) * gate.predict(gate_x)

    if "channel_weights" in meta:
        w = meta["channel_weights"]
        gate_p = np.asarray(p, dtype=float).copy()
        p = gate_p.copy()

        if w.get("psych"):
            residual_x = build_production_features(
                test, os.path.join(MODEL_DIR, "psych_profile.pkl"),
                os.path.join(MODEL_DIR, "latent_pitch_context.csv"))
            p = p + w["psych"] * apply_linear_residual(
                residual_x, os.path.join(MODEL_DIR, "psych_latent_meta.npz"))

        if w.get("split"):
            tables = pd.read_pickle(os.path.join(MODEL_DIR, "split_prior_tables.pkl"))
            p = p + w["split"] * apply_split_priors(test, tables)

        film_weight = float(w.get("film", 0.0)) + 1.25 * float(w.get("minimax", 0.0))
        if film_weight:
            profile = pd.read_pickle(os.path.join(MODEL_DIR, "psych_regime_profile.pkl"))
            p = p + film_weight * apply_saved_regime(
                test, gate_p, profile, os.path.join(MODEL_DIR, "psych_regime_assets.npz"),
                os.path.join(MODEL_DIR, "psych_regime.pt"))

        if w.get("minimax"):
            from src.stable_experts import context_ridge_correction, platoon_correction
            stable = pd.read_pickle(os.path.join(MODEL_DIR, "stable_context_profile.pkl"))
            expert = CatBoostRegressor()
            expert.load_model(os.path.join(MODEL_DIR, "stable_platoon.cbm"))
            p = p + w["minimax"] * (
                context_ridge_correction(test, stable, os.path.join(MODEL_DIR, "stable_context_ridge.npz"))
                + platoon_correction(test, expert, 0.30))

        if w.get("hier"):
            hier_z = np.column_stack([
                np.average(np.vstack(predictions), axis=0, weights=meta["main_weights"])] + risks)
            hier_p = meta["stack_intercept"] + hier_z @ np.asarray(meta["stack_coefficients"])
            p = p + w["hier"] * (hier_p - gate_p)
    else:
        blend = float(meta.get("correction_scale", 1.0))
        if meta.get("psych_latent_residual", False):
            residual_x = build_production_features(
                test, os.path.join(MODEL_DIR, "psych_profile.pkl"),
                os.path.join(MODEL_DIR, "latent_pitch_context.csv"))
            p = p + blend * apply_linear_residual(residual_x, os.path.join(MODEL_DIR, "psych_latent_meta.npz"))
        if meta.get("psych_regime_film", False):
            profile = pd.read_pickle(os.path.join(MODEL_DIR, "psych_regime_profile.pkl"))
            p = p + apply_saved_regime(test, p, profile, os.path.join(MODEL_DIR, "psych_regime_assets.npz"),
                                        os.path.join(MODEL_DIR, "psych_regime.pt"))
        if meta.get("split_priors", False):
            tables = pd.read_pickle(os.path.join(MODEL_DIR, "split_prior_tables.pkl"))
            p = p + blend * apply_split_priors(test, tables)

    p = p + float(meta.get("global_shift", 0.0))

    transition = CatBoostRegressor()
    transition.load_model(os.path.join(MODEL_DIR, "transition_gate.cbm"))
    tx = transition_features(test, p, os.path.join(MODEL_DIR, "prior_type.pkl"))
    p = p + regime["transition_scale"] * transition.predict(tx)

    return np.clip(p, 1e-5, 1 - 1e-5)


# ==== 우리 모델 (원본 그대로) ====

OUR_SEEDS = [42, 1, 2, 3, 4]
OUR_MODEL_SPECS = [
    (kind, f"{kind}_seed{seed}.pkl")
    for kind in ["xgboost", "lightgbm", "catboost"]
    for seed in OUR_SEEDS
]
OUR_HIERARCHICAL_PRIOR = 0.523766
OUR_RECENT_COLS = [f"asof_pitcher_prev{k}_game_success_rate" for k in (1, 3, 5)]
OUR_PITCHER_TEAM_RATE = {
    12: 0.531235, 13: 0.538699, 14: 0.521553, 15: 0.519699, 16: 0.515797,
    17: 0.521577, 18: 0.540460, 19: 0.505792, 20: 0.520104, 21: 0.511935,
    22: 0.692308, 23: 0.609872, 25: 0.424658,
}
OUR_BATTER_TEAM_RATE = {
    12: 0.518156, 13: 0.536698, 14: 0.520958, 15: 0.521550, 16: 0.522809,
    17: 0.523435, 18: 0.518872, 19: 0.520518, 20: 0.521957, 21: 0.522071,
    22: 0.684347, 23: 0.607369, 25: 0.482877,
}
OUR_GLOBAL_TEAM_RATE = 0.523766
OUR_FEATURE_COLS = [
    "top_bottom", "game_type", "base_state",
    "season", "game_month", "game_dayofweek", "inning",
    "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "home_win_expectancy", "away_win_expectancy", "li",
    "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    "is_same_hand",
    "asof_pitcher_team_success_rate", "asof_batter_team_success_rate",
    "team_matchup_gap",
    "recent_success_std", "dynamic_smoothing_strength", "career_dynamic_base",
    "season_form_estimate", "season_pitcher_n", "season_success_reliability",
    "season_recent_gap", "stable_momentum", "pitcher_success_logit",
    "season_success_logit",
]
OUR_CAT_LOW = ["top_bottom", "game_type", "base_state"]


def our_add_derived_features(df):
    df = df.copy()
    df["is_same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    df["asof_pitcher_team_success_rate"] = (
        df["pitcher_team_id"].map(OUR_PITCHER_TEAM_RATE).fillna(OUR_GLOBAL_TEAM_RATE))
    df["asof_batter_team_success_rate"] = (
        df["batter_team_id"].map(OUR_BATTER_TEAM_RATE).fillna(OUR_GLOBAL_TEAM_RATE))
    df["team_matchup_gap"] = (
        df["asof_pitcher_team_success_rate"] - df["asof_batter_team_success_rate"])
    return df


def our_build_hierarchical_base(df, snapshots):
    prior = OUR_HIERARCHICAL_PRIOR
    n = pd.to_numeric(df["asof_pitcher_n"], errors="coerce").fillna(0).clip(lower=0)
    career_rate = pd.to_numeric(df["asof_pitcher_success_rate"], errors="coerce").fillna(prior)

    rv = df[OUR_RECENT_COLS].apply(pd.to_numeric, errors="coerce")
    recent_std = rv.std(axis=1).fillna(0.15).clip(0, 0.5)
    recent_mean = rv.mean(axis=1).fillna(prior).to_numpy(float)

    dynamic_strength = (55 + 220 * recent_std + 40 / (1 + np.log1p(n))).clip(50, 180)
    career_base = (career_rate * n + prior * dynamic_strength) / (n + dynamic_strength)

    snap = snapshots.copy()
    snap["pitcher_id"] = snap["pitcher_id"].astype(str)
    snap = snap.set_index(["pitcher_id", "season"])
    keys = pd.MultiIndex.from_arrays([df["pitcher_id"].astype(str), df["season"].astype(int)])
    prev_n = snap["snapshot_n"].reindex(keys).fillna(0).to_numpy(float)
    prev_success = snap["snapshot_success_count"].reindex(keys).fillna(0).to_numpy(float)

    n_arr = n.to_numpy(float)
    season_n = np.maximum(n_arr - prev_n, 0)
    total_success = n_arr * career_rate.to_numpy(float)
    season_success_count = np.maximum(total_success - prev_success, 0)
    season_raw = np.divide(season_success_count, season_n,
                            out=np.full(len(df), prior), where=season_n > 0)

    season_strength = 30.0
    season_est = (season_raw * season_n + prior * season_strength) / (season_n + season_strength)
    season_reliability = season_n / (season_n + 80)
    season_weight = 0.15 + 0.30 * season_reliability
    career_base_arr = career_base.to_numpy(float)
    hierarchical_base = career_base_arr + season_weight * (season_est - career_base_arr)

    def safe_logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    extra = pd.DataFrame(index=df.index)
    extra["recent_success_std"] = recent_std.to_numpy(float).astype("float32")
    extra["dynamic_smoothing_strength"] = dynamic_strength.to_numpy(float).astype("float32")
    extra["career_dynamic_base"] = career_base_arr.astype("float32")
    extra["season_form_estimate"] = season_est.astype("float32")
    extra["season_pitcher_n"] = season_n.astype("float32")
    extra["season_success_reliability"] = season_reliability.astype("float32")
    extra["season_recent_gap"] = (season_est - recent_mean).astype("float32")
    extra["stable_momentum"] = (
        (recent_mean - career_base_arr) / (1 + 5 * recent_std.to_numpy(float))
    ).astype("float32")
    extra["pitcher_success_logit"] = safe_logit(career_rate.to_numpy(float)).astype("float32")
    extra["season_success_logit"] = safe_logit(season_est).astype("float32")

    return hierarchical_base.astype("float32"), extra


def our_cast_for_xgboost(X_transformed, pre):
    X_transformed = X_transformed.copy()
    ordinal_enc = pre.named_transformers_["cat"]
    for col, orig_categories in zip(OUR_CAT_LOW, ordinal_enc.categories_):
        full_col = f"cat__{col}"
        codes = list(range(len(orig_categories)))
        X_transformed[full_col] = pd.Categorical(
            X_transformed[full_col].astype("int64"), categories=codes)
    return X_transformed


def our_cast_for_catboost(X_transformed):
    X_transformed = X_transformed.copy()
    cols = [f"cat__{c}" for c in OUR_CAT_LOW]
    X_transformed[cols] = X_transformed[cols].astype("int64").astype(str)
    return X_transformed


def our_predict_with_pipeline(kind, pipe, X_raw, hierarchical_base):
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    X_t = pre.transform(X_raw)
    if kind == "xgboost":
        X_t = our_cast_for_xgboost(X_t, pre)
    elif kind == "catboost":
        X_t = our_cast_for_catboost(X_t)
    resid_pred = clf.predict(X_t)
    return np.clip(hierarchical_base + resid_pred, 1e-5, 1 - 1e-5)


def predict_ours(test):
    models = []
    for kind, fname in OUR_MODEL_SPECS:
        pipe = joblib.load(os.path.join(MODEL_DIR, fname))
        models.append((kind, pipe))
    snapshots = pd.read_csv(os.path.join(MODEL_DIR, "our_hierarchical_snapshots.csv"))

    test_with_derived = our_add_derived_features(test)
    hierarchical_base, hb_extra = our_build_hierarchical_base(test_with_derived, snapshots)
    test_with_derived = pd.concat([test_with_derived, hb_extra], axis=1)
    missing = [c for c in OUR_FEATURE_COLS if c not in test_with_derived.columns]
    if missing:
        raise ValueError(f"필요한 컬럼이 test 데이터에 없음: {missing}")
    X = test_with_derived[OUR_FEATURE_COLS]

    probs = np.stack(
        [our_predict_with_pipeline(kind, pipe, X, hierarchical_base) for kind, pipe in models],
        axis=0)
    return probs.mean(axis=0)


def main():
    TEST_DIR = "./data"
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    print("Load test data...")
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    sub = pd.read_csv(SAMPLE_SUB_PATH, encoding="utf-8-sig")
    if ID_COL not in test.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(test.columns)[:5]}")
    ids = test[ID_COL].tolist()
    print(f" test={len(test)}  submission={len(sub)}")

    print("\nPredict (조아님 anchor, 원본)...")
    p_anchor = predict_teammate(test)
    print(f" p_anchor: mean={p_anchor.mean():.4f} std={p_anchor.std():.4f}")

    print("\nPredict (우리 모델, 원본 15-seed 앙상블)...")
    p_ours = predict_ours(test)
    print(f" p_ours: mean={p_ours.mean():.4f} std={p_ours.std():.4f}")

    print(f"\nR행 전용 잔차보정 (scale={RESIDUAL_SCALE})...")
    resid_model = CatBoostRegressor()
    resid_model.load_model(os.path.join(MODEL_DIR, "our_model_residual.cbm"))
    is_r = (test["game_type"] == "R").to_numpy()
    correction = np.zeros(len(test), dtype=np.float64)
    if is_r.any():
        feat = pd.DataFrame({"our_model_p": p_ours[is_r]})
        correction[is_r] = resid_model.predict(feat)
    p = np.clip(p_anchor + RESIDUAL_SCALE * correction, 1e-5, 1 - 1e-5)
    print(f" p(최종): mean={p.mean():.4f} std={p.std():.4f}, "
          f"F행({(~is_r).sum()}개) 평균={p[~is_r].mean():.4f}, "
          f"R행({is_r.sum()}개) 평균={p[is_r].mean():.4f}")

    print("\nBuild submission...")
    pred_map = dict(zip(ids, p))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        v = pred_map.get(rid)
        if v is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(v)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values

    os.makedirs(OUT_DIR, exist_ok=True)
    sub.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
