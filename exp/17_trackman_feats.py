"""
[17] Trackman '원인측' 피처 실험 — 릴리스 일관성 / 구속 / 회전수 / extension

실행:  PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 py -3.12 -u exp/17_trackman_feats.py

배경:
  exp/16 시리즈로 투수 325명(투구량 71.3%)의 pitcher_id ↔ trackman ID 매칭 확보
  (lab/trackman_match_v3.csv). asof_* 피처는 전부 '결과' 통계인데, trackman은
  제구의 '원인' 쪽 신호를 준다 — 특히 릴리스 포인트 반복 일관성은 제구력의
  대표적 역학 지표다. 모델에 전혀 없는 정보 계열.

피처 5개 (전부 시즌 as-of — 그 시즌 이전 1군 trackman 기록만):
  f_tm_relstd  릴리스 흔들림 = sqrt(var(rel_height)+var(rel_side))   ← 주인공
  f_tm_velo    fastball 평균 구속
  f_tm_spin    fastball 평균 회전수
  f_tm_ext     평균 extension
  f_tm_logn    log1p(누적 trackman 투구수)  (신뢰도 신호)
  표본 50투구 미만이면 NaN. 미매칭 투수도 NaN (GBDT 네이티브).

두 가지 매칭 품질 팔:
  strict = 필링 라운드 1~2 (256명, 61.7%) — 오매칭 위험 낮음
  all    = 전체 325명 (71.3%) — 커버리지 높지만 꼬리에 오매칭 가능

기준은 hand delta 포함 67피처 (채택 완료된 현재 제출 구성과 동일).
판정: 4시드 앙상블, 기준 대비 ±15.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "exp")
from common import (log, tick, decompose, load_train, add_features,
                    fit_encoder, to_matrix, ALL_ENG, HAND_ENG, HGB_FAST,
                    hand_history_table, attach_hand_deltas_asof, LB_TOP)

SEEDS = [42, 7, 123, 2024]
THR = 15.0
MIN_N = 50
FIRST = ["DOO_BEA", "HAN_EAG", "KIA_TIG", "KIW_HER", "KT_WIZ", "LG_TWI",
         "LOT_GIA", "NC_DIN", "SAM_LIO", "SK_WYV", "SSG_LAN"]
TM_BASE = ["relstd", "velo", "spin", "ext", "logn"]

# =====================================================================
# 1. train + hand delta (현재 제출 구성 재현)
# =====================================================================
log("train.csv 로딩 중...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS67 = BASE + ALL_ENG + HAND_ENG

tr_raw = df[df["season"] <= 2023].reset_index(drop=True)
va_raw = df[df["season"] == 2024].reset_index(drop=True)
y_tr = tr_raw["control_success"].to_numpy()
y_va = va_raw["control_success"].to_numpy()
PRIOR = float(y_tr.mean())
tbl_p = hand_history_table(df, PRIOR, "pitcher_id")
tbl_b = hand_history_table(df, PRIOR, "batter_id")
tr = attach_hand_deltas_asof(add_features(tr_raw, PRIOR), tbl_p, tbl_b)
va = attach_hand_deltas_asof(add_features(va_raw, PRIOR), tbl_p, tbl_b)
tick(f"train 준비 — 학습 {len(tr):,} / 검증 {len(va):,} (67피처 기준)")

# =====================================================================
# 2. trackman 시즌 as-of 집계
# =====================================================================
tm = pd.read_csv("data/trackman_history.csv",
                 usecols=["season", "pitcher_trackman_id", "pitcher_team",
                          "pitch_type_group", "rel_speed", "spin_rate",
                          "rel_height", "rel_side", "extension"],
                 encoding="utf-8-sig")
tm.columns = [c.replace("﻿", "").strip() for c in tm.columns]
tm1 = tm[tm["pitcher_team"].isin(FIRST)].copy()
tm1 = tm1.rename(columns={"pitcher_trackman_id": "tid"})
tm1["rh2"] = tm1["rel_height"] ** 2
tm1["rs2"] = tm1["rel_side"] ** 2
is_fb = tm1["pitch_type_group"].astype(str).str.lower() == "fastball"
tm1["v_s"] = tm1["rel_speed"].where(is_fb)
tm1["sp_s"] = tm1["spin_rate"].where(is_fb)
tick(f"trackman 1군 {len(tm1):,}행")

agg = (tm1.groupby(["tid", "season"])
          .agg(n_all=("season", "size"),
               rh_n=("rel_height", "count"), rh_s=("rel_height", "sum"),
               rh_ss=("rh2", "sum"),
               rs_n=("rel_side", "count"), rs_s=("rel_side", "sum"),
               rs_ss=("rs2", "sum"),
               ex_n=("extension", "count"), ex_s=("extension", "sum"),
               v_n=("v_s", "count"), v_sum=("v_s", "sum"),
               sp_n=("sp_s", "count"), sp_sum=("sp_s", "sum"))
          .reset_index().sort_values(["tid", "season"]))
g = agg.groupby("tid")
METRICS = ["n_all", "rh_n", "rh_s", "rh_ss", "rs_n", "rs_s", "rs_ss",
           "ex_n", "ex_s", "v_n", "v_sum", "sp_n", "sp_sum"]
for c in METRICS:
    agg["p_" + c] = g[c].cumsum() - agg[c]     # 자기 시즌 제외 → 과거 시즌만


def ratio(s, n, min_n=MIN_N):
    n_ = np.asarray(n, dtype="float64")
    return np.where(n_ >= min_n, np.asarray(s, dtype="float64") / np.maximum(n_, 1),
                    np.nan)


mu_rh = ratio(agg["p_rh_s"], agg["p_rh_n"])
mu_rs = ratio(agg["p_rs_s"], agg["p_rs_n"])
var_rh = ratio(agg["p_rh_ss"], agg["p_rh_n"]) - mu_rh ** 2
var_rs = ratio(agg["p_rs_ss"], agg["p_rs_n"]) - mu_rs ** 2
feat_tbl = pd.DataFrame({
    "tid": agg["tid"].to_numpy(), "season": agg["season"].to_numpy(),
    "relstd": np.sqrt(np.clip(var_rh, 0, None) + np.clip(var_rs, 0, None)),
    "velo": ratio(agg["p_v_sum"], agg["p_v_n"]),
    "spin": ratio(agg["p_sp_sum"], agg["p_sp_n"]),
    "ext": ratio(agg["p_ex_s"], agg["p_ex_n"]),
    "logn": np.where(agg["p_n_all"] > 0, np.log1p(agg["p_n_all"]), np.nan),
})
tick(f"as-of 집계 완료 — {len(feat_tbl):,} 투수-시즌")

# =====================================================================
# 3. 매칭 테이블 → (pid, season) 피처 테이블 2종
# =====================================================================
match = pd.read_csv("lab/trackman_match_v3.csv")
log(f"  매칭 {len(match)}명 (round<=2: {(match['round'] <= 2).sum()}명)")


def pid_table(match_sub):
    mp = match_sub[["pid", "tid"]]
    t = feat_tbl.merge(mp, on="tid", how="inner")
    return t[["pid", "season"] + TM_BASE]


TBL_STRICT = pid_table(match[match["round"] <= 2])
TBL_ALL = pid_table(match)


def attach_tm(frame, tbl, prefix):
    d = frame.copy()
    key = pd.DataFrame({"pid": d["pitcher_id"].to_numpy(),
                        "season": d["season"].to_numpy()})
    mg = key.merge(tbl.rename(columns={c: prefix + c for c in TM_BASE}),
                   on=["pid", "season"], how="left", validate="m:1")
    for c in TM_BASE:
        d[prefix + c] = mg[prefix + c].to_numpy("float32")
    return d


tr = attach_tm(attach_tm(tr, TBL_STRICT, "s_"), TBL_ALL, "a_")
va = attach_tm(attach_tm(va, TBL_STRICT, "s_"), TBL_ALL, "a_")
for pfx, nm in [("s_", "strict"), ("a_", "all")]:
    log(f"  [{nm}] 커버리지  학습 {tr[pfx+'relstd'].notna().mean()*100:5.1f}%  "
        f"검증 {va[pfx+'relstd'].notna().mean()*100:5.1f}%")

# 중복성 진단: relstd가 이미 아는 성공률과 얼마나 겹치나 (낮을수록 새 정보)
m = va["s_relstd"].notna() & va["asof_pitcher_success_rate"].notna()
corr = np.corrcoef(va.loc[m, "s_relstd"], va.loc[m, "asof_pitcher_success_rate"])[0, 1]
log(f"  진단: corr(릴리스흔들림, asof성공률) = {corr:+.3f}  (검증, n={m.sum():,})\n")

# =====================================================================
# 4. 본 실험
# =====================================================================
def run(name, feats):
    enc = fit_encoder(tr, feats)
    Xtr, Xva = to_matrix(tr, feats, enc), to_matrix(va, feats, enc)
    acc = np.zeros(len(va))
    for i, s in enumerate(SEEDS, 1):
        tick(f"{name} seed={s} ({i}/{len(SEEDS)})...")
        m_ = HistGradientBoostingClassifier(**{**HGB_FAST, "random_state": s}).fit(Xtr, y_tr)
        acc += m_.predict_proba(Xva)[:, 1]
    z = decompose(acc / len(SEEDS), y_va)
    log(f"  {name:26s} 총점 {z['총점']:7.1f}  변별력 {z['변별력']:7.1f}  "
        f"벌점 {z['중심벌점']:5.1f}")
    return z


log("=" * 88)
log("기준(67) / +trackman strict / +trackman all — 4시드 앙상블")
log("=" * 88)
z0 = run("기준(67피처)", FEATS67)
z1 = run("+TM strict(256명)", FEATS67 + ["s_" + c for c in TM_BASE])
z2 = run("+TM all(325명)", FEATS67 + ["a_" + c for c in TM_BASE])

g1, g2 = z1["총점"] - z0["총점"], z2["총점"] - z0["총점"]
log("\n" + "=" * 88)
log("판정")
log("=" * 88)
log(f"  +TM strict  총점 {g1:+7.1f}  변별력 {z1['변별력']-z0['변별력']:+7.1f}")
log(f"  +TM all     총점 {g2:+7.1f}  변별력 {z2['변별력']-z0['변별력']:+7.1f}")
log(f"  임계 {THR:.0f} (4시드)\n")
best = max(g1, g2)
if best >= THR:
    w = "strict" if g1 >= g2 else "all"
    log(f"  [채택 후보] {w} 팔이 임계 초과 — 독립 8시드 재측정으로 확증할 것 (회색지대 절차)")
elif best <= -THR:
    log(f"  [기각] 유해 — trackman 피처 축 폐기")
else:
    log(f"  [보류] 노이즈 범위 — 오매칭 노이즈가 신호를 상쇄했거나 정보가 이미 asof에 흡수됨")
    log(f"     후속 판단: strict가 all보다 뚜렷이 좋으면 매칭 품질 문제 → 매칭 개선 여지")
log(f"  참고: 1위 {LB_TOP:.1f} / hand delta 반영 제출 예정 ~847")
log("=" * 88)
