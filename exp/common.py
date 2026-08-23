"""
공통 모듈 — 점수 계산 / 데이터 로드 / 피처 엔지니어링 / 인코딩

⚠️ add_features() 는 제출용 script.py 에도 **똑같이** 들어가야 한다.
   학습 때와 추론 때 피처가 다르면 서버에서 에러가 나고 제출 1회가 날아간다.
   그래서 여기 한 곳에만 두고 양쪽에서 가져다 쓴다.
"""

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

LB_BASELINE = 549.51        # 운영진 베이스라인 실제 리더보드 점수
LB_TOP = 1421.99852         # 현재 1위
CAT = ["top_bottom", "game_type", "base_state"]

_T0 = time.time()


def log(*a):
    print(*a, flush=True)


def tick(msg):
    log(f"  [{time.time()-_T0:6.0f}s] {msg}")


# =====================================================================
# 점수
# =====================================================================
def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def raw_score(p, y):
    """대회 점수. max(0,·)를 적용하지 않는다 (음수도 봐야 비교가 된다)."""
    y = np.asarray(y, float)
    r = y.mean()
    return 100000.0 * (1.0 - brier(p, y) / (r * (1.0 - r)))


def decompose(p, y):
    """
    총점 = 변별력 - 중심오차벌점

    Brier(p) = Brier(중심을 정답에 맞춘 p) + d²   (d = 예측평균 - 실제평균)
    → 개선이 '진짜 예측력'인지 '중심이 우연히 맞은 것'인지 구분해준다.
    """
    p, y = np.asarray(p, float), np.asarray(y, float)
    r = y.mean()
    d = p.mean() - r
    total = raw_score(p, y)
    penalty = 100000.0 * d * d / (r * (1 - r))
    return {"총점": total, "변별력": total + penalty, "중심벌점": penalty, "d": d}


def report(name, p, y, width=30):
    z = decompose(p, y)
    log(f"  {name:{width}s} 총점 {z['총점']:8.1f} = 변별력 {z['변별력']:7.1f} "
        f"- 벌점 {z['중심벌점']:6.1f}   (d={z['d']:+.4f})")
    return z


# =====================================================================
# 데이터
# =====================================================================
def load_train(path="data/train.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.replace("﻿", "").strip() for c in df.columns]
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    return df


# =====================================================================
# 피처 엔지니어링
# =====================================================================
# 그룹별로 묶어둔다 — ablation에서 그룹 단위로 빼고 넣기 위해
FEATURE_GROUPS = {
    "G1 볼카운트": ["f_count_state", "f_count_diff", "f_is_3ball",
                    "f_is_2strike", "f_is_full"],
    "G2 매치업/압박": ["f_same_hand", "f_scoring_pos", "f_any_runner"],
    "G3 asof신뢰도": ["f_log_pn", "f_log_bn", "f_smooth_p", "f_smooth_b"],
    "G4 최근폼": ["f_form_dev1", "f_form_dev3", "f_form_trend"],
    "G5 투타차이": ["f_pb_diff"],
    "G6 결측플래그": ["f_miss_p", "f_miss_prev1"],
}
ALL_ENG = [c for g in FEATURE_GROUPS.values() for c in g]


def add_features(d, prior):
    """
    투구 '직전' 정보만 사용. 다른 행을 참조하지 않으므로 test에도 그대로 적용 가능.

    prior : 베이지안 스무딩의 사전확률. 학습 데이터의 control_success 평균을 넣는다.
            ★ 추론 시에도 학습 때와 '같은 값'을 써야 한다 (모델과 함께 저장할 것)
    """
    d = d.copy()
    b, s = d["balls_before"], d["strikes_before"]

    # G1 볼카운트 — 01번 진단에서 r 폭 0.0345 실측
    d["f_count_state"] = (b * 3 + s).astype("int8")
    d["f_count_diff"] = (b - s).astype("int8")
    d["f_is_3ball"] = (b == 3).astype("int8")
    d["f_is_2strike"] = (s == 2).astype("int8")
    d["f_is_full"] = ((b == 3) & (s == 2)).astype("int8")

    # G2 매치업 / 상황 압박
    d["f_same_hand"] = (d["pitcher_hand"] == d["batter_hand"]).astype("int8")
    d["f_scoring_pos"] = ((d["runner_on_2b"] == 1) | (d["runner_on_3b"] == 1)).astype("int8")
    d["f_any_runner"] = (d["num_runners_on"] > 0).astype("int8")

    # G3 asof 신뢰도 — 표본이 적을수록 rate를 믿으면 안 된다
    pn = d["asof_pitcher_n"].fillna(0)
    bn = d["asof_batter_n"].fillna(0)
    d["f_log_pn"] = np.log1p(pn).astype("float32")
    d["f_log_bn"] = np.log1p(bn).astype("float32")
    a = 200.0
    d["f_smooth_p"] = ((d["asof_pitcher_success_rate"].fillna(prior) * pn + prior * a)
                       / (pn + a)).astype("float32")
    d["f_smooth_b"] = ((d["asof_batter_success_rate"].fillna(prior) * bn + prior * a)
                       / (bn + a)).astype("float32")

    # G4 최근폼 — 통산 rate는 드리프트에 오염됨. 최근 창은 덜하다.
    d["f_form_dev1"] = (d["asof_pitcher_prev1_game_success_rate"]
                        - d["asof_pitcher_success_rate"]).astype("float32")
    d["f_form_dev3"] = (d["asof_pitcher_prev3_game_success_rate"]
                        - d["asof_pitcher_success_rate"]).astype("float32")
    d["f_form_trend"] = (d["asof_pitcher_prev1_game_success_rate"]
                         - d["asof_pitcher_prev5_game_success_rate"]).astype("float32")

    # G5 투수 vs 타자 매치업 차이
    d["f_pb_diff"] = (d["asof_pitcher_success_rate"]
                      - d["asof_batter_success_rate"]).astype("float32")

    # G6 결측 자체가 신호 (신인 / 첫 등판 표시)
    d["f_miss_p"] = d["asof_pitcher_success_rate"].isna().astype("int8")
    d["f_miss_prev1"] = d["asof_pitcher_prev1_game_success_rate"].isna().astype("int8")
    return d


# =====================================================================
# 인코딩
# =====================================================================
def fit_encoder(train_df, feats):
    """범주형 3개만 OrdinalEncoder. 처음 보는 값은 -1로."""
    cats = [c for c in CAT if c in feats]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train_df[cats])
    return enc


def to_matrix(df, feats, enc):
    """DataFrame → 모델 입력 행렬. 결측(NaN)은 GBDT가 알아서 처리한다."""
    cats = [c for c in CAT if c in feats]
    nums = [c for c in feats if c not in cats]
    return np.column_stack([enc.transform(df[cats]).astype("float32"),
                            df[nums].to_numpy("float32")])


# 실험용 GBDT 기본 설정 (속도 우선. 최종 제출 땐 상향 예정)
HGB_FAST = dict(max_iter=120, learning_rate=0.08, max_leaf_nodes=31,
                min_samples_leaf=200, l2_regularization=10.0,
                early_stopping=False, random_state=42)


# =====================================================================
# hand delta (exp/14~15에서 채택, 토이 실행으로 as-of 누수 없음 검증)
# =====================================================================
HAND_A1 = 200.0
HAND_A2 = 200.0


def hand_history_table(src, prior, id_col):
    """(eid, season, sh) → 그 시즌 '이전' 시즌들 기준 delta. exp/07과 동일 로직."""
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
    p_over = (tbl["po_succ"] + prior * HAND_A1) / (tbl["po_n"] + HAND_A1)
    p_cond = (tbl["pc_succ"] + p_over * HAND_A2) / (tbl["pc_n"] + HAND_A2)
    tbl["delta"] = np.where(tbl["po_n"] > 0,
                            p_cond - p_over, np.nan).astype("float32")
    return tbl[["eid", "season", "sh", "delta"]]


def attach_hand_deltas_asof(d, tbl_p, tbl_b):
    """학습·검증 행에 hand delta 2개 부착 (add_features 이후 호출)."""
    d = d.copy()
    for tbl, idc, feat in [(tbl_p, "pitcher_id", "f_hand_delta"),
                           (tbl_b, "batter_id", "f_bhand_delta")]:
        key = pd.DataFrame({"eid": d[idc].to_numpy(),
                            "season": d["season"].to_numpy(),
                            "sh": d["f_same_hand"].to_numpy("int8")})
        d[feat] = key.merge(tbl, on=["eid", "season", "sh"], how="left",
                            validate="m:1")["delta"].to_numpy("float32")
    return d


HAND_ENG = ["f_hand_delta", "f_bhand_delta"]
