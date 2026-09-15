# -*- coding: utf-8 -*-
"""
[91a] 영역 분류법 교차표 — 사용자 제안 3영역 {볼 / 가운데 / 비가운데 스트라이크} 정의 검증
       (순수 진단, 학습 없음, ~3분)

추가(08-28): '볼도 스트라이크도 아닌' 행(ball=0 & strike=0, 약 19%)의 정체를
  다음 투구의 카운트 전이로 직접 확인한다.
    ball=1   → 다음 투구 balls+1 (또는 4볼이면 새 타석)
    strike=1 → 다음 투구 strikes+1 / 2S 파울이면 동일 / 삼진이면 새 타석
    둘 다 0  → 새 타석이면 '인플레이(타격)' — 위치 라벨이 아니라 결과 라벨이라는 증거
"""
import numpy as np, pandas as pd, time
t0 = time.time()
cols = ["pitcher_id", "batter_id", "asof_pitcher_n", "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate", "asof_pitcher_middle_rate", "asof_pitcher_reverse_rate",
        "control_success", "season", "balls_before", "strikes_before", "game_type",
        "batter_hand", "inning"]
df = pd.read_csv("data/train.csv", encoding="utf-8-sig", usecols=cols)
df.columns = [c.replace("﻿", "").strip() for c in df.columns]
d = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index(drop=True)
pid = d.pitcher_id.to_numpy(); n = d.asof_pitcher_n.fillna(0).to_numpy("float64")
nxt = (pid[1:] == pid[:-1]) & (np.abs(n[1:] - n[:-1] - 1) < 1e-6)
L = {}
for k, col in {"ball": "asof_pitcher_ball_rate", "strike": "asof_pitcher_strike_rate",
               "mid": "asof_pitcher_middle_rate", "rev": "asof_pitcher_reverse_rate"}.items():
    S = d[col].fillna(0).to_numpy("float64") * n
    lab = np.full(len(d), np.nan)
    lab[:-1] = np.where(nxt, np.round(S[1:] - S[:-1]), np.nan)
    L[k] = lab
m = np.ones(len(d), bool)
for k in L:
    m &= (L[k] == 0) | (L[k] == 1)
y = d.control_success.to_numpy(int)[m]
b = L["ball"][m].astype(int); s = L["strike"][m].astype(int)
mi = L["mid"][m].astype(int); rv = L["rev"][m].astype(int)
print(f"labeled {m.sum():,}/{len(d):,}  ({time.time()-t0:.0f}s)", flush=True)
print("joint (ball,strike,mid): frac, succ_rate, rev_rate")
for bb in (0, 1):
    for ss in (0, 1):
        for mm in (0, 1):
            q = (b == bb) & (s == ss) & (mi == mm)
            if q.sum():
                print(f"  ball={bb} strike={ss} mid={mm}: {q.mean()*100:6.2f}%  succ={y[q].mean():.3f}  "
                      f"rev={rv[q].mean():.3f}  n={q.sum():,}")

# ---------------- 카운트 전이로 라벨의 정체 확인 ----------------
print("\n=== 다음 투구 카운트 전이 × 라벨 (같은 투수의 바로 다음 투구) ===")
bb_ = d.balls_before.to_numpy(int); ss_ = d.strikes_before.to_numpy(int)
bat = d.batter_id.to_numpy(); inn = d.inning.to_numpy(int)
idx = np.where(m)[0]                       # m 인 행은 정의상 nxt 가 True (다음 행 존재)
i1 = idx + 1
new_pa = ((bb_[i1] == 0) & (ss_[i1] == 0)) | (bat[i1] != bat[idx]) | (inn[i1] != inn[idx])
ball_up = (~new_pa) & (bb_[i1] == bb_[idx] + 1) & (ss_[i1] == ss_[idx])
strike_up = (~new_pa) & (ss_[i1] == ss_[idx] + 1) & (bb_[i1] == bb_[idx])
foul2 = (~new_pa) & (ss_[idx] == 2) & (ss_[i1] == 2) & (bb_[i1] == bb_[idx])
other = ~(new_pa | ball_up | strike_up | foul2)
trans = np.select([new_pa, ball_up, strike_up, foul2], ["new_pa", "ball+1", "strike+1", "foul@2S"], "other")
lab3 = np.select([(b == 1) & (s == 0), (b == 0) & (s == 1), (b == 1) & (s == 1)],
                 ["ball", "strike", "both"], "neither")
ct = pd.crosstab(lab3, trans)
print((ct.div(ct.sum(1), axis=0) * 100).round(1).to_string())
print("row n:", ct.sum(1).to_dict())
q = lab3 == "ball"
print(f"  ball=1 & 4볼 상황(balls_before=3) 중 new_pa 비율: {new_pa[q & (bb_[idx]==3)].mean()*100:.1f}%")
q = lab3 == "strike"
print(f"  strike=1 & 2S 상황 중 new_pa(삼진) {new_pa[q & (ss_[idx]==2)].mean()*100:.1f}% / "
      f"foul@2S {foul2[q & (ss_[idx]==2)].mean()*100:.1f}%")
q = lab3 == "neither"
print(f"  neither 중 mid=1 {mi[q].mean()*100:.1f}%  succ {y[q].mean():.3f}  rev {rv[q].mean():.3f}")
print("  neither 의 카운트 분포 (balls,strikes) 상위:")
print(pd.Series(list(zip(bb_[idx][q], ss_[idx][q]))).value_counts(normalize=True).head(6).round(3).to_string())
print("  전체 카운트 분포 상위:")
print(pd.Series(list(zip(bb_[idx], ss_[idx]))).value_counts(normalize=True).head(6).round(3).to_string())

# ---------------- 3영역 정의 ----------------
reg = np.full(len(y), -1)
reg[mi == 1] = 1
reg[(mi == 0) & (b == 1)] = 0
reg[(mi == 0) & (b == 0) & (s == 1)] = 2
print("\nregion (0 ball, 1 mid, 2 edge, -1 undef):")
for r in (-1, 0, 1, 2):
    q = reg == r
    print(f"  r={r}: {q.mean()*100:5.2f}%  succ={y[q].mean():.3f}  rev={rv[q].mean():.3f}")
print("\namong SUCCESS pitches, region dist [-1,0,1,2]:",
      [f"{np.mean(reg[y==1]==r)*100:.1f}%" for r in (-1, 0, 1, 2)])
print("among FAIL pitches, region dist [-1,0,1,2]:",
      [f"{np.mean(reg[y==0]==r)*100:.1f}%" for r in (-1, 0, 1, 2)])
dd = d[m].copy(); dd["reg"] = reg; dd["y"] = y
sc = dd[(dd.y == 1) & (dd.reg >= 0)]
ct = pd.crosstab([sc.balls_before, sc.strikes_before], sc.reg, normalize="index").round(3)
print("\nP(region | success, count):")
print(ct.to_string())
u = dd.reg == -1
print("\nundefined share by season:",
      dd[u].groupby("season").size().div(dd.groupby("season").size()).round(3).to_dict())
print("undefined share by game_type:",
      dd[u].groupby("game_type").size().div(dd.groupby("game_type").size()).round(3).to_dict())
# 투수 간 의도 분산: 성공 투구의 영역 분포가 투수마다 얼마나 다른가 (카운트 고정)
sc2 = sc[sc.season <= 2023]
g = sc2.groupby(["pitcher_id", "balls_before", "strikes_before"]).reg
cell = g.agg(n="size", p_ball=lambda x: np.mean(x == 0), p_edge=lambda x: np.mean(x == 2)).reset_index()
big = cell[cell.n >= 100]
cnt = big.groupby(["balls_before", "strikes_before"])
print(f"\n셀(투수×카운트, n>=100) {len(big):,}개 — 카운트 고정 시 투수 간 P(볼|성공) 표준편차:")
print(cnt.p_ball.std().round(3).to_string())
print("참고: 이항 표본오차 sqrt(p(1-p)/n) 가 n=100,p=.4 이면 .049")
# 신호 대 잡음: 셀 분산 − 평균 이항분산 = 진짜 투수 간 분산 추정
big = big.copy()
big["var_binom"] = big.p_ball * (1 - big.p_ball) / big.n
sig = big.groupby(["balls_before", "strikes_before"]).apply(
    lambda g: pd.Series({"n_cells": len(g), "var_obs": g.p_ball.var(), "var_binom": g.var_binom.mean()}))
sig["var_true"] = sig.var_obs - sig.var_binom
sig["sd_true"] = np.sqrt(sig.var_true.clip(lower=0))
print("\n카운트별: 관측분산 − 이항분산 = 진짜 투수간 분산 (sd_true)")
print(sig.round(4).to_string())
print(f"총 {time.time()-t0:.0f}s")
