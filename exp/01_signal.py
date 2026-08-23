"""
[01] 신호 크기 진단 — 이 문제가 어떤 문제인지 감 잡기

실행:  py -3.12 exp/01_signal.py       (프로젝트 루트에서)

알아내려는 것:
  A. train의 시즌 구조 + 시즌별 성공률 r 이 얼마나 흔들리는가
  B. 그 흔들림 때문에 몇 점을 손해보는가  (calibration 손실의 실제 크기)
  C. asof 피처 하나만 써도 몇 점인가     (출발점 확인)

특징: 8개 컬럼만 읽으므로 수십 초면 끝난다. (train.csv 전체는 368MB)
"""

import numpy as np
import pandas as pd

DATA = "data/train.csv"

# 리더보드 기준선 (우리가 어디쯤 있는지 계속 비교하기 위해)
LB_BASELINE = 549.51      # 운영진 RandomForest 실제 점수
LB_TOP = 1421.99852       # 현재 1위


# ---------------------------------------------------------------
# 대회 공식 점수 산식
# ---------------------------------------------------------------
def brier(p, y):
    """Brier Score = 예측확률과 실제(0/1)의 제곱오차 평균. 낮을수록 좋다."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def score(p, y):
    """
    대회 점수 = max(0, 100000 * (1 - Brier / (r*(1-r))))

    핵심: r 은 '채점 대상 y의 평균'이다.
          즉 y 평균을 상수로 찍으면 Brier == r(1-r) 이 되어 정확히 0점이 나온다.
          → 0점 = "평균만 아는 수준", 양수 = "평균보다 나은 정보를 가짐"
    """
    y = np.asarray(y, dtype=float)
    r = y.mean()
    return max(0.0, 100000.0 * (1.0 - brier(p, y) / (r * (1.0 - r))))


def bar(s, width=44):
    """점수를 1위(1422) 기준 막대로 그려서 직관적으로 보이게."""
    n = int(max(0.0, min(1.0, s / LB_TOP)) * width)
    return "█" * n + "·" * (width - n)


# ---------------------------------------------------------------
# 로드 — 필요한 8개 컬럼만
# ---------------------------------------------------------------
USE = [
    "season", "control_success",
    "asof_pitcher_success_rate", "asof_pitcher_n",
    "asof_batter_success_rate", "asof_batter_n",
    "balls_before", "strikes_before",
]

print("train.csv 로딩 중... (8개 컬럼만)")
df = pd.read_csv(DATA, encoding="utf-8-sig", usecols=USE)
print(f"  shape = {df.shape}")
print(f"  메모리 = {df.memory_usage(deep=True).sum() / 1024**2:.0f} MB\n")


# ===============================================================
# A. 시즌 구조와 성공률 드리프트
# ===============================================================
print("=" * 68)
print("A. 시즌 구조 / 시즌별 제구 성공률 r")
print("=" * 68)

g = df.groupby("season")["control_success"].agg(["size", "mean"])
g.columns = ["행수", "r"]
g["r(1-r)"] = g["r"] * (1 - g["r"])
print(g.to_string(float_format=lambda x: f"{x:.6f}"))

r_all = df["control_success"].mean()
print(f"\n  train 전체 r      = {r_all:.6f}")
print(f"  시즌별 r 최대-최소 = {g['r'].max() - g['r'].min():.6f}")
print(f"  test(2025) 는 train에 없음: {2025 not in g.index}")


# ===============================================================
# B. 드리프트 때문에 몇 점을 손해보는가
# ===============================================================
print("\n" + "=" * 68)
print("B. calibration 손실 — 상수를 어떻게 잡느냐로 몇 점이 갈리는가")
print("=" * 68)
print("  가정: 2024를 '미래'라 치고, 그걸 맞춘다고 할 때")
print("        상수 c 를 무엇으로 잡느냐에 따른 점수 차이\n")

va = df[df["season"] == 2024]
tr = df[df["season"] < 2024]
y_va = va["control_success"].to_numpy()
r_va = y_va.mean()

cands = {
    "전체 train 평균 (2019~2023)": tr["control_success"].mean(),
    "직전 1시즌 평균 (2023)": df.loc[df["season"] == 2023, "control_success"].mean(),
    "직전 3시즌 평균 (2021~23)": df.loc[df["season"].between(2021, 2023), "control_success"].mean(),
    "★ 정답 (2024 실제 평균)": r_va,
}
print(f"  {'상수 c 선택':32s} {'c값':>10s} {'점수':>9s}   손실")
print("  " + "-" * 62)
best = None
for name, c in cands.items():
    s = score(np.full(len(y_va), c), y_va)
    if best is None:
        best = s
    print(f"  {name:32s} {c:10.6f} {s:9.2f}   {s - 0:+.2f}")

print(f"\n  ※ '정답' 상수가 정확히 0.00점인 게 이 지표의 정의다.")
print(f"     (평균만 맞추는 건 0점. 그보다 나쁘면 음수 → max(0,·)로 잘림)")

delta = abs(tr["control_success"].mean() - r_va)
loss = 100000 * delta**2 / (r_va * (1 - r_va))
print(f"\n  train평균 vs 2024실제 차이 δ = {delta:.6f}")
print(f"  → δ 때문에 잃는 점수 ≈ {loss:.1f}점")
print(f"     (베이스라인 총점이 {LB_BASELINE}점이니 이게 {loss/LB_BASELINE*100:.1f}% 규모)")


# ===============================================================
# C. asof 피처 하나로 어디까지 가는가  (2024 홀드아웃)
# ===============================================================
print("\n" + "=" * 68)
print("C. 단순 규칙들의 2024 점수 — 출발점 확인")
print("=" * 68)

r_tr = tr["control_success"].mean()          # 학습기간 평균 (2024 정보 사용 X)
p_rate = va["asof_pitcher_success_rate"].fillna(r_tr).to_numpy()
p_n = va["asof_pitcher_n"].fillna(0).to_numpy()
b_rate = va["asof_batter_success_rate"].fillna(r_tr).to_numpy()

rules = {}
rules["상수 (train 평균)"] = np.full(len(y_va), r_tr)
rules["asof_pitcher_rate 그대로"] = np.clip(p_rate, 0.01, 0.99)

# shrink: 평균 쪽으로 얼마나 끌어당길 것인가
for w in [0.2, 0.4, 0.6, 0.8, 1.0]:
    rules[f"  ↳ shrink w={w:.1f}"] = r_tr + w * (p_rate - r_tr)

# 베이지안 스무딩: 표본이 적으면 자동으로 평균 쪽으로
for a in [50, 200, 500, 1000]:
    rules[f"베이즈 스무딩 a={a}"] = (p_rate * p_n + r_tr * a) / (p_n + a)

# 투수 + 타자 결합
rules["투수+타자 평균"] = (p_rate + b_rate) / 2
rules["  ↳ shrink w=0.5"] = r_tr + 0.5 * ((p_rate + b_rate) / 2 - r_tr)

rows = []
for name, p in rules.items():
    rows.append({"규칙": name, "Brier": brier(p, y_va), "점수": score(p, y_va)})
res = pd.DataFrame(rows).sort_values("점수", ascending=False)

print(f"\n  {'규칙':30s} {'Brier':>10s} {'점수':>9s}  1위(1422) 대비")
print("  " + "-" * 66)
for _, row in res.iterrows():
    print(f"  {row['규칙']:30s} {row['Brier']:10.6f} {row['점수']:9.2f}  {bar(row['점수'])}")

print(f"\n  [기준선] 베이스라인 RF = {LB_BASELINE:.2f}  {bar(LB_BASELINE)}")
print(f"  [기준선] 현재 1위      = {LB_TOP:.2f}  {bar(LB_TOP)}")


# ===============================================================
# D. 볼카운트 — 웹 클로드가 '가장 강할 것'이라 추측한 피처
# ===============================================================
print("\n" + "=" * 68)
print("D. 볼카운트별 성공률 (추측 말고 실측)")
print("=" * 68)

cnt = (df.groupby(["balls_before", "strikes_before"])["control_success"]
         .agg(["size", "mean"]).reset_index())
cnt.columns = ["B", "S", "행수", "r"]
cnt["전체r대비"] = cnt["r"] - r_all
print(cnt.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# 볼카운트만으로 예측했을 때 2024 점수 (train에서 만든 표를 2024에 적용)
tbl = tr.groupby(["balls_before", "strikes_before"])["control_success"].mean()
key = list(zip(va["balls_before"], va["strikes_before"]))
p_cnt = np.array([tbl.get(k, r_tr) for k in key])
print(f"\n  볼카운트 단독 2024 점수 = {score(p_cnt, y_va):.2f}   {bar(score(p_cnt, y_va))}")
print(f"  (train에서 만든 볼카운트별 평균표를 2024에 적용 — 누수 없음)")

print("\n" + "=" * 68)
print("진단 A~D 완료.")
print("=" * 68)
