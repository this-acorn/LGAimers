# -*- coding: utf-8 -*-
"""
[90] 에러 분석 마스터 테이블 — 2024 폴드 253,507행 × (원본 슬라이스 컬럼 + 79피처 + 예측들)

행 순서 = df[df.season==2024].reset_index(drop=True) — 모든 lab/*.npy 와 동일 정렬.
probe_kit/y_valid.npy 와의 일치를 assert 로 검증한다.

저장: lab/90_analysis_2024.parquet (+ 스크래치패드 이중 저장)
실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/90_build_analysis.py  (~3분)
"""

import shutil
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "exp")
from common import log, load_train

SCRATCH = (r"C:\Users\gwonn\AppData\Local\Temp\claude\c--Users-gwonn-Desktop-open"
           r"\c69d5971-c907-4b01-8868-c40d5f87726c\scratchpad")

log("train 로딩...")
df = load_train()

# ---- 5클래스 라벨 복원 (exp/85/87/89와 동일) ----
dd = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index()
pid = dd.pitcher_id.to_numpy()
n = dd.asof_pitcher_n.fillna(0).to_numpy("float64")
nxt = (pid[1:] == pid[:-1]) & (np.abs(n[1:] - n[:-1] - 1) < 1e-6)
for k, col in [("lab_mid", "asof_pitcher_middle_rate"),
               ("lab_rev", "asof_pitcher_reverse_rate"),
               ("lab_fb", "asof_pitcher_fastball_rate"),
               ("lab_brk", "asof_pitcher_breaking_rate"),
               ("lab_ball", "asof_pitcher_ball_rate"),
               ("lab_strike", "asof_pitcher_strike_rate")]:
    S = dd[col].fillna(0).to_numpy("float64") * n
    lab = np.full(len(dd), np.nan)
    lab[:-1] = np.where(nxt, np.round(S[1:] - S[:-1]), np.nan)
    dd[k] = np.where((lab == 0) | (lab == 1), lab, np.nan)
rec = dd.set_index("index").sort_index()
for k in ["lab_mid", "lab_rev", "lab_fb", "lab_brk", "lab_ball", "lab_strike"]:
    df[k] = rec[k]
del dd, rec

y_bin = df.control_success.to_numpy("float64")
cls = np.full(len(df), -1, dtype="int8")
ok = df.lab_mid.notna().to_numpy() & df.lab_rev.notna().to_numpy()
m_ = df.lab_mid.to_numpy() == 1
r_ = df.lab_rev.to_numpy() == 1
cls[ok & (y_bin == 1)] = 0
cls[ok & (y_bin == 0) & m_ & ~r_] = 1
cls[ok & (y_bin == 0) & ~m_ & r_] = 2
cls[ok & (y_bin == 0) & m_ & r_] = 3
cls[ok & (y_bin == 0) & ~m_ & ~r_] = 4
df["y5"] = cls

# ---- 선수 메타 (전체 train 기준, 행 독립 아님 주의 — 분석 전용, 피처 아님) ----
p_first = df.groupby("pitcher_id")["season"].min().rename("p_first_season")
b_first = df.groupby("batter_id")["season"].min().rename("b_first_season")
df = df.merge(p_first, on="pitcher_id", how="left").merge(b_first, on="batter_id", how="left")

# ---- 2024 폴드 추출 ----
va = df[df.season == 2024].reset_index(drop=True)
log(f"2024 폴드 {len(va)} 행")
assert len(va) == 253507

y_ref = np.load("probe_kit/y_valid.npy")
assert np.array_equal(va["control_success"].to_numpy("float64"), y_ref.astype("float64")), \
    "probe_kit y_valid 와 행 정렬 불일치!"
log("정렬 검증 통과 (probe_kit/y_valid)")

KEEP = ["row_id", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
        "balls_before", "strikes_before", "outs_before",
        "run_top_before", "run_bot_before", "run_total_before",
        "score_diff_home", "score_diff_pitcher_team",
        "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state",
        "home_win_expectancy", "away_win_expectancy", "li",
        "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
        "pitcher_team_id", "batter_team_id",
        "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
        "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
        "control_success", "y5",
        "lab_mid", "lab_rev", "lab_fb", "lab_brk", "lab_ball", "lab_strike",
        "p_first_season", "b_first_season"]
KEEP = [c for c in KEEP if c in va.columns]
out = va[KEEP].copy()
out = out.rename(columns={"control_success": "y"})

# ---- 엔지니어링 피처 (probe_kit X_valid — 학습에 실제 들어간 79피처) ----
Xv = pd.read_parquet("probe_kit/X_valid.parquet")
assert len(Xv) == len(out)
eng_cols = [c for c in Xv.columns if c.startswith("f_")]
for c in eng_cols:
    out[c] = Xv[c].to_numpy()
log(f"엔지니어링 피처 {len(eng_cols)}개 부착: {eng_cols}")

# ---- 모델 예측 (전부 (253507,) float32, P(성공)) ----
PREDS = {
    "p_mc":        "lab/53_mc_ens.npy",      # 멀티클래스 cat3 lr0.08 2시드 = 챔피언-팀캣 (837.0)
    "p_mc04":      "lab/66_mc_lr04.npy",     # 멀티클래스 cat3 lr0.04 2시드 (852.3)
    "p_bin04":     "lab/65_d6_lr04.npy",     # 이진 lr0.04 2시드 (839.4)
    "p_cs79bin":   "lab/64_base.npy",        # 이진 lr0.08 2시드 = CS79 기준 (800.9)
    "p_ova04":     "lab/78_ova_lr04.npy",    # OneVsAll lr0.04 (859.8 proj8 기준 무다양성)
    "p_mc7":       "lab/75_mc7_lr04.npy",    # 7클래스 lr0.04 (827.4, 기각)
    "p_w090":      "lab/79_w090.npy",        # MC04+시즌가중 0.9 (실전 -27.6 기각)
    "p_ebm":       "lab/83_ebm_sub.npy",     # EBM 40만행 (439.6, 기각 — 이종 구조 참고용)
    "p_depthwise": "lab/69_depthwise.npy",   # 비대칭 트리 (기각 — 이종 구조 참고용)
    "p_lossguide": "lab/69_lossguide.npy",   # 비대칭 트리 (기각 — 이종 구조 참고용)
    "p_csrev":     "lab/72_cs_rev.npy",      # CS reverse 80피처 (기각 — 피처 변형 참고용)
}
for name, path in PREDS.items():
    out[name] = np.load(path).astype("float32")
log(f"예측 {len(PREDS)}개 부착")

# 챔피언 프록시(cat5 완료 전): p_mc (팀캣만 빠진 동일 구성)
out["res_mc"] = (out["y"] - out["p_mc"]).astype("float32")
out["res_mc04"] = (out["y"] - out["p_mc04"]).astype("float32")

out.to_parquet("lab/90_analysis_2024.parquet", index=False)
shutil.copy("lab/90_analysis_2024.parquet", SCRATCH + r"\90_analysis_2024.parquet")
log(f"저장 완료: lab/90_analysis_2024.parquet ({len(out)}행 × {len(out.columns)}컬럼)")
log(f"컬럼: {list(out.columns)}")

r = out["y"].mean()
DEN = r * (1 - r)
for name in ["p_mc", "p_mc04", "p_bin04", "p_cs79bin"]:
    p = out[name].to_numpy("float64")
    sc = 100000 * (1 - np.mean((p - out['y'].to_numpy('float64'))**2) / DEN)
    log(f"  {name:12s} 점수 재현 {sc:8.1f}")
