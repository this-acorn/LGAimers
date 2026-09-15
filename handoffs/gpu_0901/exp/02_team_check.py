"""
[02] 이 데이터는 리그 전체인가, 특정 팀 경기만인가?
     그리고 성공률 하락은 전 팀 공통인가?

실행:  py -3.12 exp/02_team_check.py

쟁점:
  주장 1) "이 데이터는 LG에서만 나온 것이다"
  주장 2) "LG가 2025 우승했으니 제구력이 떨어졌을 리 없다"

  주장 1이 맞으면 주장 2가 힘을 얻는다. 그래서 1부터 확정한다.

결정적 테스트:
  '모든 행에 반드시 등장하는 팀'이 있는가?
    있다 → 그 팀 경기만 수집한 데이터
    없다 → 리그 전체 데이터
"""

import numpy as np
import pandas as pd

USE = ["season", "game_month", "pitcher_team_id", "batter_team_id",
       "pitcher_id", "control_success"]

print("train.csv 로딩 중... (6개 컬럼만)")
df = pd.read_csv("data/train.csv", encoding="utf-8-sig", usecols=USE)
print(f"  shape = {df.shape}\n")


# ===============================================================
# A. 결정적 테스트 — 모든 행에 등장하는 팀이 있는가?
# ===============================================================
print("=" * 70)
print("A. '모든 행에 등장하는 팀'이 존재하는가?  (LG 전용 데이터 가설 검증)")
print("=" * 70)

teams = sorted(set(df["pitcher_team_id"]) | set(df["batter_team_id"]))
print(f"  등장하는 팀 ID: {teams}  (총 {len(teams)}개)\n")

print(f"  {'팀ID':>5s} {'투수팀 행수':>12s} {'타자팀 행수':>12s} {'둘중하나 포함율':>16s}")
print("  " + "-" * 50)
rows = []
for t in teams:
    as_p = (df["pitcher_team_id"] == t).sum()
    as_b = (df["batter_team_id"] == t).sum()
    involved = ((df["pitcher_team_id"] == t) | (df["batter_team_id"] == t)).mean()
    rows.append((t, as_p, as_b, involved))
    print(f"  {t:5d} {as_p:12,d} {as_b:12,d} {involved*100:15.2f}%")

max_inv = max(r[3] for r in rows)
print()
if max_inv > 0.999:
    who = [r[0] for r in rows if r[3] > 0.999]
    print(f"  ⇒ 팀 {who} 이(가) 모든 행에 등장 → 그 팀 경기만 모은 데이터")
else:
    print(f"  ⇒ 모든 행에 등장하는 팀 없음 (최대 포함율 {max_inv*100:.2f}%)")
    print(f"     → 리그 전체 데이터로 판정")


# ===============================================================
# B. 팀별 투구 점유율 — 한 팀이 압도적인가?
# ===============================================================
print("\n" + "=" * 70)
print("B. 투수팀 기준 점유율 (한 팀이 몰려있으면 그 팀 중심 데이터)")
print("=" * 70)

share = df["pitcher_team_id"].value_counts(normalize=True).sort_index()
for t, v in share.items():
    bar = "█" * int(v * 200)
    print(f"  팀 {t:2d} : {v*100:6.2f}%  {bar}")
print(f"\n  최대 점유율 = {share.max()*100:.2f}%   (균등하면 {100/len(teams):.1f}% 근처)")


# ===============================================================
# C. ★ 핵심 — 성공률 하락이 전 팀 공통인가?
# ===============================================================
print("\n" + "=" * 70)
print("C. 팀별 x 시즌별 제구 성공률 r  ← 하락이 전 팀 공통인가?")
print("=" * 70)

pv = df.pivot_table(index="pitcher_team_id", columns="season",
                    values="control_success", aggfunc="mean")
pv["19→24"] = pv[2024] - pv[2019]
print(pv.to_string(float_format=lambda x: f"{x:.4f}"))

drops = pv["19→24"].dropna()
n_down = (drops < 0).sum()
print(f"\n  2019→2024 하락한 팀 : {n_down} / {len(drops)}")
print(f"  하락폭 평균          : {drops.mean():+.4f}")
print(f"  하락폭 최소~최대     : {drops.min():+.4f} ~ {drops.max():+.4f}")
if n_down == len(drops):
    print("  ⇒ 전 팀이 예외 없이 하락 → 특정 팀 사정이 아니라 리그 전체 현상")
else:
    print("  ⇒ 일부 팀은 하락하지 않음 → 팀 구성 요인 가능성 있음")


# ===============================================================
# D. 시즌 '안'에서도 떨어지는가?  (추세 vs 계단 판별)
# ===============================================================
print("\n" + "=" * 70)
print("D. 시즌 내 월별 r  ← 추세인가 계단인가")
print("=" * 70)

mv = df.pivot_table(index="season", columns="game_month",
                    values="control_success", aggfunc="mean")
print(mv.to_string(float_format=lambda x: f"{x:.4f}"))

print("\n  시즌별 (시즌내 최대 - 최소):")
for s in mv.index:
    row = mv.loc[s].dropna()
    print(f"    {s} : {row.max() - row.min():.4f}   "
          f"(첫달 {row.iloc[0]:.4f} → 끝달 {row.iloc[-1]:.4f}, "
          f"변화 {row.iloc[-1] - row.iloc[0]:+.4f})")

print("\n  ⇒ 시즌 안에서 계속 내려가면 '추세' → 2025도 더 떨어질 것")
print("     시즌 안은 평평한데 연도끼리만 점프하면 '계단' → 2025는 예측 불가")


# ===============================================================
# E. 같은 투수를 추적하면?  (선수 구성 vs 기준 변화)
# ===============================================================
print("\n" + "=" * 70)
print("E. 2023·2024 모두 던진 '같은 투수'들의 r 변화")
print("=" * 70)

p23 = set(df.loc[df["season"] == 2023, "pitcher_id"])
p24 = set(df.loc[df["season"] == 2024, "pitcher_id"])
both = p23 & p24
print(f"  2023 투수 {len(p23)}명 / 2024 투수 {len(p24)}명 / 공통 {len(both)}명\n")

d23 = df[(df["season"] == 2023) & (df["pitcher_id"].isin(both))]
d24 = df[(df["season"] == 2024) & (df["pitcher_id"].isin(both))]

r23_all = df.loc[df["season"] == 2023, "control_success"].mean()
r24_all = df.loc[df["season"] == 2024, "control_success"].mean()
r23_both = d23["control_success"].mean()
r24_both = d24["control_success"].mean()

print(f"  전체 투수    : 2023 r={r23_all:.6f} → 2024 r={r24_all:.6f}  ({r24_all-r23_all:+.6f})")
print(f"  공통 투수만  : 2023 r={r23_both:.6f} → 2024 r={r24_both:.6f}  ({r24_both-r23_both:+.6f})")

# 투수 단위로 짝지어 비교 (표본 200투구 이상만)
a = d23.groupby("pitcher_id")["control_success"].agg(["size", "mean"])
b = d24.groupby("pitcher_id")["control_success"].agg(["size", "mean"])
m = a.join(b, lsuffix="_23", rsuffix="_24")
m = m[(m["size_23"] >= 200) & (m["size_24"] >= 200)]
m["diff"] = m["mean_24"] - m["mean_23"]

print(f"\n  양 시즌 200투구 이상 던진 투수 {len(m)}명 기준:")
print(f"    성공률 하락한 투수 : {(m['diff'] < 0).sum()}명 ({(m['diff']<0).mean()*100:.1f}%)")
print(f"    성공률 상승한 투수 : {(m['diff'] > 0).sum()}명 ({(m['diff']>0).mean()*100:.1f}%)")
print(f"    평균 변화          : {m['diff'].mean():+.6f}")
print(f"    중앙값 변화        : {m['diff'].median():+.6f}")

print("\n  ⇒ 같은 투수 대다수가 같이 떨어졌다면 → 실력 문제 아님 (기준/환경 변화)")
print("     일부만 떨어지고 평균은 그대로면 → 신인 유입 등 구성 변화")

print("\n" + "=" * 70)
print("A~E 완료.")
print("=" * 70)
