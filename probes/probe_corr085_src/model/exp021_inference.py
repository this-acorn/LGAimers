"""EXP-021 계열 엔드포인트 추론 — 우리 팀 자체 구현.

이 파일은 `docs/EXP021_REIMPL_SPEC.md`(방법론 사양서)만 보고 처음부터 작성했다.
공개 저장소(mk-isos)의 실험 기록에서 **방법론과 하이퍼파라미터를 참고**했으며,
학습 산출물(model/ 아래 JSON·텍스트)은 대회 공식 데이터로 우리가 계산한 값이다.

구조 — 해석식 베이스 위에 네 개의 가산 보정을 순차로 얹는다:

    b_tmp  = 0.7·(투수 시즌 EB30) + 0.3·(타자 시즌 EB30)
    b_grp  = clip(b_tmp + 0.7·group_base + 0.3·group_reverse)
    b_bone = ½·clip(b_grp + 0.75·LGB잔차·1[R]) + ½·clip(b_grp + 1.00·HGB잔차·1[R])
    b_team = clip(b_bone + ½·팀(투수)효과 + ½·팀(타자)효과)
    p      = clip(b_team + 저랭크효과)

행 간 독립성(대회 규칙): 모든 값은 (a) 그 행 자신의 컬럼과 (b) 학습에서 동결한 상수표로만
계산된다. 평가 데이터의 다른 행·순서·분포를 참조하는 연산은 없다.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"
MODEL_DIR = "./model"
DATA_DIR = "./data"
OUTPUT_DIR = "./output"

SHRINK_LEVELS = (10.0, 30.0, 100.0, 300.0)
PLAYER_PRIOR_STRENGTH = 200.0
MULTIRATE_SHRINK = 30.0
BASE_PITCHER_WEIGHT = 0.7
BASE_BATTER_WEIGHT = 0.3
GROUP_BASE_WEIGHT = 0.7
GROUP_REVERSE_WEIGHT = 0.3
LGB_RESIDUAL_WEIGHT = 0.75
HGB_RESIDUAL_WEIGHT = 1.00
TEAM_FAMILY_WEIGHT = 0.5
REVERSE_BIN_WIDTH = 0.05
CORRECTION_SCALE = 0.85   # 해석적 베이스 대비 보정 성분의 배율 (1.0 = 원본 EXP-021 그대로)

# multirate 그룹 정의: (그룹명, id 열, 표본수 열, [(metric, rate 열), ...])
MULTIRATE_GROUPS = (
    ("pitcher_control", "pitcher_id", "asof_pitcher_n", (
        ("success", "asof_pitcher_success_rate"),
        ("reverse", "asof_pitcher_reverse_rate"),
        ("middle", "asof_pitcher_middle_rate"),
        ("ball", "asof_pitcher_ball_rate"),
        ("strike", "asof_pitcher_strike_rate"))),
    ("batter_control", "batter_id", "asof_batter_n", (
        ("success", "asof_batter_success_rate"),
        ("middle", "asof_batter_middle_rate"))),
    ("pitcher_pitchmix", "pitcher_id", "asof_pitcher_pitchmix_n", (
        ("fastball", "asof_pitcher_fastball_rate"),
        ("breaking", "asof_pitcher_breaking_rate"),
        ("offspeed", "asof_pitcher_offspeed_rate"))),
)

TOP_BOTTOM_VOCAB = ("B", "T")
BASE_STATE_VOCAB = ("123", "12_", "1_3", "1__", "_23", "_2_", "__3", "___")


def _read_json(name):
    with open(os.path.join(MODEL_DIR, name), encoding="utf-8") as handle:
        return json.load(handle)


def _f32(values):
    return np.asarray(values, dtype=np.float32)


def _column(frame, name):
    """원본 열을 float64 로 꺼낸다. 결측은 그대로 NaN 으로 남긴다."""
    return frame[name].to_numpy(dtype=np.float64, copy=False)


# ---------------------------------------------------------------- 정적 파생 21열
def add_static_features(frame):
    balls = frame["balls_before"].to_numpy(np.int64)
    strikes = frame["strikes_before"].to_numpy(np.int64)
    outs = frame["outs_before"].to_numpy(np.int64)
    count_index = balls * 4 + strikes

    frame["count_index"] = count_index.astype(np.int8)
    frame["count_out_index"] = (count_index * 3 + outs).astype(np.int8)
    frame["is_full_count"] = ((balls == 3) & (strikes == 2)).astype(np.int8)
    frame["has_two_strikes"] = (strikes == 2).astype(np.int8)
    frame["has_three_balls"] = (balls == 3).astype(np.int8)
    frame["count_advantage"] = (strikes - balls).astype(np.int8)
    frame["runner_in_scoring_position"] = (
        (frame["runner_on_2b"].to_numpy(np.int64) == 1)
        | (frame["runner_on_3b"].to_numpy(np.int64) == 1)).astype(np.int8)
    frame["bases_loaded"] = (
        (frame["runner_on_1b"].to_numpy(np.int64) == 1)
        & (frame["runner_on_2b"].to_numpy(np.int64) == 1)
        & (frame["runner_on_3b"].to_numpy(np.int64) == 1)).astype(np.int8)
    frame["same_hand"] = (
        frame["pitcher_hand"].to_numpy(np.int64)
        == frame["batter_hand"].to_numpy(np.int64)).astype(np.int8)
    frame["late_inning"] = (frame["inning"].to_numpy(np.int64) >= 7).astype(np.int8)

    score_gap = _column(frame, "score_diff_pitcher_team")
    frame["close_game"] = (np.abs(score_gap) <= 1).astype(np.int8)

    # log_li 를 먼저 float32 로 확정한 뒤 score_pressure 에 그 값을 쓴다 (사양 §3.2-13)
    log_li = _f32(np.log1p(np.maximum(_column(frame, "li"), 0.0)))
    frame["log_li"] = log_li
    frame["score_pressure"] = _f32(np.abs(score_gap) * log_li.astype(np.float64))
    frame["win_expectancy_gap"] = _f32(
        _column(frame, "home_win_expectancy") - _column(frame, "away_win_expectancy"))

    # 아래 네 개는 결측을 채우지 않는다 — NaN 이 그대로 GBDT 로 전달돼야 한다
    frame["pitcher_batter_success_gap"] = _f32(
        _column(frame, "asof_pitcher_success_rate") - _column(frame, "asof_batter_success_rate"))
    frame["pitcher_recent_success_delta_1_5"] = _f32(
        _column(frame, "asof_pitcher_prev1_game_success_rate")
        - _column(frame, "asof_pitcher_prev5_game_success_rate"))
    frame["pitcher_recent_success_delta_3_5"] = _f32(
        _column(frame, "asof_pitcher_prev3_game_success_rate")
        - _column(frame, "asof_pitcher_prev5_game_success_rate"))
    frame["pitcher_recent_middle_delta_1_5"] = _f32(
        _column(frame, "asof_pitcher_prev1_game_middle_rate")
        - _column(frame, "asof_pitcher_prev5_game_middle_rate"))

    for out_name, src in (("log_pitcher_n", "asof_pitcher_n"),
                          ("log_batter_n", "asof_batter_n"),
                          ("log_pitchmix_n", "asof_pitcher_pitchmix_n")):
        frame[out_name] = _f32(np.log1p(np.maximum(_column(frame, src), 0.0)))
    return frame


# ---------------------------------------------------------------- temporal 45열
def _prior_lookup(records, id_key, ids, fields):
    """상수표를 id 로 조회. 미등록 id 는 0.0, 존재 플래그는 0."""
    index = {int(rec[id_key]): rec for rec in records}
    out = {name: np.zeros(len(ids), dtype=np.float64) for name in fields}
    exists = np.zeros(len(ids), dtype=np.int8)
    for position, raw_id in enumerate(ids):
        rec = index.get(int(raw_id))
        if rec is None:
            continue
        exists[position] = 1
        for name in fields:
            out[name][position] = float(rec.get(name, 0.0))
    return out, exists


def add_temporal_features(frame, state):
    if int(frame["season"].max()) <= int(state["through_season"]):
        raise ValueError("추론 시즌은 저장된 이력 시즌보다 뒤여야 합니다")
    league = float(state["league_rate"])

    for entity, id_col, n_col, rate_col in (
            ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
            ("batter", "batter_id", "asof_batter_n", "asof_batter_success_rate")):
        priors, exists = _prior_lookup(
            state[entity], id_col, frame[id_col].to_numpy(),
            ("prior_n", "prior_successes"))
        prior_n = priors["prior_n"]
        prior_s = priors["prior_successes"]

        career_n = _column(frame, n_col)                       # 결측을 채우지 않는다
        career_s = np.rint(career_n * np.nan_to_num(_column(frame, rate_col), nan=0.0))

        season_n_raw = career_n - prior_n
        season_s_raw = career_s - prior_s
        if np.any(season_n_raw < -1e-6):
            raise ValueError(f"{entity}: 누적 표본수가 저장된 이력보다 작습니다")
        if np.any(season_s_raw < -0.01) or np.any(season_s_raw - season_n_raw > 0.01):
            raise ValueError(f"{entity}: 복원한 시즌 성공수가 허용 범위를 벗어났습니다")
        season_n = np.maximum(season_n_raw, 0.0)
        season_s = np.clip(season_s_raw, 0.0, season_n)

        prior_rate = np.where(prior_n > 0, prior_s / np.maximum(prior_n, 1e-12), league)
        season_rate = np.where(season_n > 0, season_s / np.maximum(season_n, 1e-12), league)
        player_prior = ((prior_s + PLAYER_PRIOR_STRENGTH * league)
                        / (prior_n + PLAYER_PRIOR_STRENGTH))

        prefix = f"temporal_{entity}_"
        frame[prefix + "prior_exists"] = exists
        frame[prefix + "prior_n"] = _f32(prior_n)
        frame[prefix + "log_prior_n"] = _f32(np.log1p(prior_n))
        frame[prefix + "prior_rate"] = _f32(prior_rate)
        frame[prefix + "prior_rate_shrunk_200"] = _f32(player_prior)
        frame[prefix + "season_n"] = _f32(season_n)
        frame[prefix + "log_season_n"] = _f32(np.log1p(season_n))
        frame[prefix + "season_rate"] = _f32(season_rate)
        frame[prefix + "season_minus_prior_rate"] = _f32(season_rate - prior_rate)
        for k in SHRINK_LEVELS:
            tag = int(k)
            frame[f"{prefix}season_global_{tag}"] = _f32((season_s + k * league) / (season_n + k))
            frame[f"{prefix}season_player_{tag}"] = _f32(
                (season_s + k * player_prior) / (season_n + k))
            frame[f"{prefix}reliability_{tag}"] = _f32(season_n / (season_n + k))

    frame["temporal_prior_league_rate"] = np.full(len(frame), np.float32(league), dtype=np.float32)
    # 이 두 열만은 float32 로 저장된 값끼리 float32 산술로 계산한다 (사양 §3.3.1)
    w_p = np.float32(BASE_PITCHER_WEIGHT)
    w_b = np.float32(BASE_BATTER_WEIGHT)
    for suffix in ("global", "player"):
        frame[f"temporal_base_{suffix}_30"] = (
            w_p * frame[f"temporal_pitcher_season_{suffix}_30"].to_numpy(np.float32)
            + w_b * frame[f"temporal_batter_season_{suffix}_30"].to_numpy(np.float32))
    return frame


# ---------------------------------------------------------------- multirate 69열
def add_multirate_features(frame, state):
    if int(frame["season"].max()) <= int(state["through_season"]):
        raise ValueError("추론 시즌은 저장된 multirate 이력보다 뒤여야 합니다")
    global_rates = state["global_rates"]

    for group, id_col, n_col, metrics in MULTIRATE_GROUPS:
        fields = ["prior_n"] + [f"prior_{metric}_count" for metric, _ in metrics]
        priors, _ = _prior_lookup(state["tables"][group], id_col,
                                  frame[id_col].to_numpy(), fields)
        prior_n = priors["prior_n"]
        career_n = np.nan_to_num(_column(frame, n_col), nan=0.0)   # temporal 과 달리 채운다
        season_n_raw = career_n - prior_n
        if np.any(season_n_raw < -1e-6):
            raise ValueError(f"{group}: 누적 표본수가 저장된 이력보다 작습니다")
        season_n = np.maximum(season_n_raw, 0.0)

        prefix = f"multirate_{group}_"
        frame[prefix + "season_n"] = _f32(season_n)
        frame[prefix + "log_season_n"] = _f32(np.log1p(season_n))
        frame[prefix + "reliability_30"] = _f32(season_n / (season_n + MULTIRATE_SHRINK))

        for metric, rate_col in metrics:
            g = float(global_rates[f"{group}_{metric}"])
            prior_count = priors[f"prior_{metric}_count"]
            rate = np.nan_to_num(_column(frame, rate_col), nan=g)   # 여기서만 전역률로 대치
            career_count = np.rint(career_n * rate)
            season_count = np.clip(career_count - prior_count, 0.0, season_n)

            prior_rate = np.where(prior_n > 0, prior_count / np.maximum(prior_n, 1e-12), g)
            shrunk = ((prior_count + PLAYER_PRIOR_STRENGTH * g)
                      / (prior_n + PLAYER_PRIOR_STRENGTH))
            season_rate = np.where(season_n > 0, season_count / np.maximum(season_n, 1e-12), g)

            mp = f"{prefix}{metric}_"
            frame[mp + "prior_rate"] = _f32(prior_rate)
            frame[mp + "prior_shrunk_200"] = _f32(shrunk)
            frame[mp + "season_rate"] = _f32(season_rate)
            frame[mp + "season_global_30"] = _f32(
                (season_count + MULTIRATE_SHRINK * g) / (season_n + MULTIRATE_SHRINK))
            frame[mp + "season_player_30"] = _f32(
                (season_count + MULTIRATE_SHRINK * shrunk) / (season_n + MULTIRATE_SHRINK))
            frame[mp + "season_minus_prior"] = _f32(season_rate - prior_rate)
    return frame


# ---------------------------------------------------------------- 원-핫 12열
def add_indicator_columns(frame):
    top = frame["top_bottom"]
    top_missing = top.isna().to_numpy()
    top_values = top.astype("string").to_numpy()
    for level in TOP_BOTTOM_VOCAB:
        frame[f"top_bottom_{level}"] = ((top_values == level) & ~top_missing).astype(np.int8)
    frame["top_bottom_nan"] = top_missing.astype(np.int8)

    base = frame["base_state"]
    base_missing = base.isna().to_numpy()
    base_values = base.astype("string").to_numpy()
    for level in BASE_STATE_VOCAB:
        # 값 "___" 의 열 이름은 밑줄 네 개("base_state____")가 된다
        frame[f"base_state_{level}"] = ((base_values == level) & ~base_missing).astype(np.int8)
    frame["base_state_nan"] = base_missing.astype(np.int8)
    return frame


def build_matrix(frame, schema):
    if len(set(schema)) != len(schema):
        raise ValueError("피처 스키마에 중복 이름이 있습니다")
    columns = []
    for name in schema:
        if name in frame.columns:
            columns.append(frame[name].to_numpy(dtype=np.float32, copy=False))
        elif name.startswith("top_bottom_") or name.startswith("base_state_"):
            columns.append(np.zeros(len(frame), dtype=np.float32))
        else:
            raise ValueError(f"피처가 프레임에 없습니다: {name}")
    return np.column_stack(columns).astype(np.float32, copy=False)


# ---------------------------------------------------------------- 가산 보정 3종
def group_effect(frame, effects):
    count_index = frame["count_index"].to_numpy(np.int64)
    p_hand = frame["pitcher_hand"].to_numpy(np.int64)
    b_hand = frame["batter_hand"].to_numpy(np.int64)

    base_map = {(int(r["count_index"]), int(r["pitcher_hand"]), int(r["batter_hand"])):
                float(r["effect"]) for r in effects["base"]}
    reverse_rate = _column(frame, "asof_pitcher_reverse_rate")
    finite = np.isfinite(reverse_rate)
    bins = np.where(finite, np.floor(np.divide(reverse_rate, REVERSE_BIN_WIDTH,
                                               out=np.zeros_like(reverse_rate), where=finite)), -1)
    bins = bins.astype(np.int64)
    reverse_map = {(int(r["count_index"]), int(r["pitcher_hand"]), int(r["batter_hand"]),
                    int(r["reverse_rate_bin"])): float(r["effect"]) for r in effects["reverse"]}

    base_effect = np.fromiter(
        (base_map.get((int(c), int(p), int(b)), 0.0)
         for c, p, b in zip(count_index, p_hand, b_hand)),
        dtype=np.float64, count=len(frame))
    reverse_effect = np.fromiter(
        (reverse_map.get((int(c), int(p), int(b), int(v)), 0.0)
         for c, p, b, v in zip(count_index, p_hand, b_hand, bins)),
        dtype=np.float64, count=len(frame))
    return GROUP_BASE_WEIGHT * base_effect + GROUP_REVERSE_WEIGHT * reverse_effect


def team_effect(frame, effects):
    """4개 소스 시즌의 효과를 균등 평균(분모는 항상 4)한 뒤 두 패밀리를 0.5/0.5 로 합친다."""
    n_sources = len(effects["source_seasons"])
    total = np.zeros(len(frame), dtype=np.float64)
    p_hand = frame["pitcher_hand"].to_numpy(np.int64)
    b_hand = frame["batter_hand"].to_numpy(np.int64)

    for family, team_col in (("pitcher_team", "pitcher_team_id"),
                             ("batter_team", "batter_team_id")):
        merged = {}
        for block in effects[family]:
            for rec in block["records"]:
                key = (int(rec[team_col]), int(rec["pitcher_hand"]), int(rec["batter_hand"]))
                merged[key] = merged.get(key, 0.0) + float(rec["effect"])
        team_ids = frame[team_col].to_numpy(np.int64)
        family_effect = np.fromiter(
            (merged.get((int(t), int(p), int(b)), 0.0)
             for t, p, b in zip(team_ids, p_hand, b_hand)),
            dtype=np.float64, count=len(frame)) / n_sources
        total += TEAM_FAMILY_WEIGHT * family_effect
    return total


def lowrank_effect(frame, effects):
    """투수 × 24문맥 저랭크 행렬을 4개 소스에서 조회해 합한 뒤 4로 나눈다(미등록 투수는 0)."""
    position = {(int(c["count_index"]), int(c["batter_hand"])): int(c["position"])
                for c in effects["contexts"]}
    count_index = frame["count_index"].to_numpy(np.int64)
    b_hand = frame["batter_hand"].to_numpy(np.int64)
    # 문맥 어휘는 12카운트 x 2타자핸드로 닫혀 있으나, 미등록 조합이 와도 죽지 않고 0을 기여한다
    slot = np.fromiter((position.get((int(c), int(h)), -1) for c, h in zip(count_index, b_hand)),
                       dtype=np.int64, count=len(frame))

    sources = effects["sources"]
    accumulated = {}
    for block in sources:
        values = np.asarray(block["values"], dtype=np.float64)
        for row, pitcher in enumerate(block["pitcher_ids"]):
            key = int(pitcher)
            if key in accumulated:
                accumulated[key] = accumulated[key] + values[row]
            else:
                accumulated[key] = values[row].copy()

    zero = np.zeros(len(effects["contexts"]), dtype=np.float64)
    pitchers = frame["pitcher_id"].to_numpy(np.int64)
    out = np.fromiter(
        (0.0 if s < 0 else accumulated.get(int(p), zero)[s] for p, s in zip(pitchers, slot)),
        dtype=np.float64, count=len(frame))
    return out / len(sources)


# ---------------------------------------------------------------- HGB (JSON → numpy)
def hgb_predict(matrix, state):
    if state.get("format") != "numeric_hgb_v1":
        raise ValueError("지원하지 않는 HGB 직렬화 형식입니다")
    if int(state["n_features"]) != matrix.shape[1]:
        raise ValueError("HGB 입력 피처 수가 맞지 않습니다")

    total = np.full(matrix.shape[0], float(state["baseline"]), dtype=np.float64)
    all_rows = np.arange(matrix.shape[0])
    for tree in state["trees"]:
        value = np.asarray(tree["value"], dtype=np.float64)
        feature = np.asarray(tree["feature_idx"], dtype=np.intp)
        threshold = np.asarray(tree["num_threshold"], dtype=np.float64)
        missing_left = np.asarray(tree["missing_go_to_left"], dtype=bool)
        left = np.asarray(tree["left"], dtype=np.intp)
        right = np.asarray(tree["right"], dtype=np.intp)
        is_leaf = np.asarray(tree["is_leaf"], dtype=bool)

        node = np.zeros(matrix.shape[0], dtype=np.intp)
        active = all_rows
        while active.size:
            current = node[active]
            arrived = is_leaf[current]
            if arrived.any():
                landed = active[arrived]
                total[landed] += value[node[landed]]        # 리프에서만 누적
            active = active[~arrived]
            if not active.size:
                break
            current = node[active]
            feature_values = matrix[active, feature[current]]
            go_left = np.where(np.isnan(feature_values),
                               missing_left[current],
                               feature_values <= threshold[current])
            node[active] = np.where(go_left, left[current], right[current])
    return total


# ---------------------------------------------------------------- main
def main():
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig")
    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"), encoding="utf-8-sig")
    if list(sample.columns) != [ID_COL, TARGET_COL]:
        raise ValueError("sample_submission 의 컬럼이 예상과 다릅니다")
    if len(test) != len(sample):
        raise ValueError("test 와 sample_submission 의 행 수가 다릅니다")
    if test[ID_COL].isna().any() or sample[ID_COL].isna().any():
        raise ValueError("row_id 에 결측이 있습니다")
    if test[ID_COL].duplicated().any() or sample[ID_COL].duplicated().any():
        raise ValueError("row_id 가 중복됩니다")
    if set(test[ID_COL]) != set(sample[ID_COL]):
        raise ValueError("test 와 sample_submission 의 row_id 집합이 다릅니다")

    schemas = _read_json("feature_schemas.json")
    frame = test.drop(columns=[ID_COL])
    frame = add_static_features(frame)
    frame = add_temporal_features(frame, _read_json("history_state.json"))
    frame = add_multirate_features(frame, _read_json("multirate_state.json"))
    frame = add_indicator_columns(frame)

    analytic_base = frame["temporal_base_global_30"].to_numpy(np.float32).astype(np.float64)
    base = np.clip(analytic_base + group_effect(frame, _read_json("group_effects.json")), 0.0, 1.0)

    import lightgbm as lgb
    booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, "rfull_lightgbm.txt"))
    if booster.num_feature() != len(schemas["lightgbm"]):
        raise ValueError("LightGBM 입력 폭이 스키마와 다릅니다")
    lgb_matrix = build_matrix(frame, schemas["lightgbm"])
    lgb_residual = np.asarray(booster.predict(lgb_matrix), dtype=np.float64)
    del lgb_matrix

    hgb_state = _read_json("histgradientboosting.json")
    hgb_matrix = build_matrix(frame, schemas["histgradientboosting"])
    hgb_residual = hgb_predict(hgb_matrix, hgb_state)
    del hgb_matrix

    regular = (frame["game_type"].astype(str).to_numpy() == "R")
    lgb_branch = np.where(regular, np.clip(base + LGB_RESIDUAL_WEIGHT * lgb_residual, 0.0, 1.0), base)
    hgb_branch = np.where(regular, np.clip(base + HGB_RESIDUAL_WEIGHT * hgb_residual, 0.0, 1.0), base)
    backbone = 0.5 * lgb_branch + 0.5 * hgb_branch          # 평균 직후에는 클립하지 않는다

    with_team = np.clip(backbone + team_effect(frame, _read_json("team_effects.json")), 0.0, 1.0)
    predictions = np.clip(with_team + lowrank_effect(frame, _read_json("lowrank_effects.json")),
                          0.0, 1.0)

    # 해석적 베이스는 우리 챔피언과 크게 중복되고, 보정들(그룹/GBDT잔차/팀/저랭크)이 새로운
    # 정보다. CORRECTION_SCALE 은 그 둘의 비율만 바꾼다 (1.0 이면 원본과 완전 동일).
    if CORRECTION_SCALE != 1.0:
        predictions = np.clip(
            analytic_base + CORRECTION_SCALE * (predictions - analytic_base), 0.0, 1.0)

    if not np.isfinite(predictions).all():
        raise ValueError("예측에 유한하지 않은 값이 있습니다")
    if predictions.min() < 0.0 or predictions.max() > 1.0:
        raise ValueError("예측이 [0,1] 범위를 벗어났습니다")

    lookup = dict(zip(test[ID_COL].to_numpy(), predictions))
    ordered = np.array([lookup[key] for key in sample[ID_COL].to_numpy()], dtype=np.float64)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame({ID_COL: sample[ID_COL], TARGET_COL: ordered}).to_csv(
        os.path.join(OUTPUT_DIR, "submission.csv"), index=False, encoding="utf-8")
    print(f"  exp021(자체 구현): rows={len(ordered)} mean={ordered.mean():.9f} "
          f"min={ordered.min():.9f} max={ordered.max():.9f}", flush=True)


if __name__ == "__main__":
    main()
