# -*- coding: utf-8 -*-
"""
[52] 장부 감사 1탄 — control_success의 제조 공식 역설계 (순수 진단, 학습 없음)

복원 라벨(볼/스트라이크/미들/reverse — asof 연속행 차분, succ 일치율 100% 검증됨)을
실제 타겟과 교차분석해 타겟의 정확한 구성을 알아낸다:
  공식 정의: 실패 = ① 가운데 부근 ② 크게 벗어남 ③ 포수 요구 반대
  질문: middle/reverse 라벨이 ①③과 정확히 일치하는가? ②는 ball의 부분집합인가?
  ball+strike는 전체를 분할하는가?

성공 시 함의: 실패 유형별 분리 모델링(P(middle|x), P(reverse|x), P(빅미스|x))이
가능해짐 — 유형마다 원인이 다르므로(공격성/의도/제구난조) 분리 학습이 신호를 더 뽑을 수 있다.

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/52_target_anatomy.py  (~3분)
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "exp")
from common import log, tick, load_train

df = load_train()
d = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index(drop=True)
pid = d.pitcher_id.to_numpy()
n = d.asof_pitcher_n.fillna(0).to_numpy("float64")
nxt = (pid[1:] == pid[:-1]) & (np.abs(n[1:] - n[:-1] - 1) < 1e-6)

RATES = {"succ": "asof_pitcher_success_rate", "ball": "asof_pitcher_ball_rate",
         "strike": "asof_pitcher_strike_rate", "middle": "asof_pitcher_middle_rate",
         "reverse": "asof_pitcher_reverse_rate"}
L = {}
for k, col in RATES.items():
    S = d[col].fillna(0).to_numpy("float64") * n
    lab = np.full(len(d), np.nan)
    diff = np.round(S[1:] - S[:-1])
    lab[:-1] = np.where(nxt, diff, np.nan)
    L[k] = lab
mask = ~np.isnan(L["succ"])
for k in L:
    mask &= ~np.isnan(L[k]) & ((L[k] == 0) | (L[k] == 1))
y = d.control_success.to_numpy("float64")
log(f"완전 라벨 행: {mask.sum():,} / {len(d):,} ({mask.mean()*100:.1f}%)")
log(f"succ 라벨 vs 실제 y 일치: {np.mean(L['succ'][mask]==y[mask])*100:.2f}%")

s = {k: L[k][mask].astype(int) for k in L}
yy = y[mask].astype(int)

log("\n" + "=" * 70)
log("A. 기본 분할 구조")
log("=" * 70)
log(f"  ball+strike=1 비율: {np.mean(s['ball']+s['strike']==1)*100:.2f}%  "
    f"(둘다0 {np.mean((s['ball']==0)&(s['strike']==0))*100:.2f}% / "
    f"둘다1 {np.mean((s['ball']==1)&(s['strike']==1))*100:.2f}%)")
log(f"  전체 비율: succ {yy.mean():.4f} / ball {s['ball'].mean():.4f} / "
    f"strike {s['strike'].mean():.4f} / middle {s['middle'].mean():.4f} / "
    f"reverse {s['reverse'].mean():.4f}")

log("\n" + "=" * 70)
log("B. 실패 유형과 타겟의 관계")
log("=" * 70)
log(f"  P(실패 | middle=1)  = {1-yy[s['middle']==1].mean():.4f}   (1.0이면 middle⊂실패)")
log(f"  P(실패 | reverse=1) = {1-yy[s['reverse']==1].mean():.4f}")
log(f"  middle ∩ reverse   = {np.mean((s['middle']==1)&(s['reverse']==1))*100:.2f}%")
fail = yy == 0
covered = (s["middle"] == 1) | (s["reverse"] == 1)
log(f"  실패 중 middle 또는 reverse: {covered[fail].mean()*100:.2f}%")
resid = fail & ~covered
log(f"  잔여 실패(=빅미스 추정): 전체의 {resid.mean()*100:.2f}%")
log(f"    잔여 실패 중 ball=1 비율: {s['ball'][resid].mean()*100:.2f}%  "
    f"(높으면 '크게 벗어남'⊂볼)")
log(f"    잔여 실패 중 strike=1 비율: {s['strike'][resid].mean()*100:.2f}%")

log("\n" + "=" * 70)
log("C. 성공의 구성 (역방향 확인)")
log("=" * 70)
succ = yy == 1
log(f"  성공 중 ball=1:   {s['ball'][succ].mean()*100:.2f}%  (의도된 볼 = 성공 가능?)")
log(f"  성공 중 strike=1: {s['strike'][succ].mean()*100:.2f}%")
log(f"  성공 중 middle=1: {s['middle'][succ].mean()*100:.2f}%  (0이어야 정의와 일치)")
log(f"  성공 중 reverse=1:{s['reverse'][succ].mean()*100:.2f}%")

# 후보 공식 검정
log("\n" + "=" * 70)
log("D. 후보 공식 정확도")
log("=" * 70)
for name, pred in [
    ("실패 = middle ∪ reverse ∪ (잔여?)", None),
    ("succ = ~middle & ~reverse", ((s["middle"] == 0) & (s["reverse"] == 0)).astype(int)),
]:
    if pred is None:
        continue
    acc = np.mean(pred == yy)
    log(f"  {name}: 일치율 {acc*100:.2f}%")
# 빅미스를 식별할 수 있는 조합 탐색: 실패인데 middle/reverse 아닌 행의 ball/strike 분포는 위 B에서.
log("\n  → 다음 단계: 잔여 실패(빅미스)를 별도 클래스로 두고")
log("    4클래스 (성공/미들/리버스/빅미스) 분리 모델링 가치 평가")
