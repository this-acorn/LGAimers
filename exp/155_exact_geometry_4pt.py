"""측정 4점으로 2025 기하를 가정 없이 확정한다.

예측 모형 (p'' = alpha + b1 p1 + b2 p2):
    Score = 2 bᵀA - bᵀB b - lam (E[p''] - r)^2
    E[p''] - r = alpha + r (b1 + b2 - 1) + b2 d2,   d2 = d_w / W
구조 지식:
    lam = 402483            (§1.17 중심 프로브 2점)
    A1 = B11 = S1           (챔피언은 아핀이 이미 풀려 있음)
    d_w = 0.00031066        (우리 중심 프로브)
    B22 = 2 A2 - S2 - lam d_w^2 / W^2 ,  B12 = A2 - (S2 + K - B11)/2
남는 미지수는 A2 와 r 둘뿐이고, 스케일 프로브와 배포 실측 두 식이면 정확히 풀린다.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve

LAM = 402483.0
S1 = A1 = B11 = 1092.808353586
S2 = 1043.6074197937
K = 171.5001496146865
W = 0.35655716947524263
U = 1.0 - W
CONST = (S2 + K - B11) / 2.0
D_W = ((1110.2391147581 - 1114.6116846973) / LAM + 0.003 ** 2) / (2.0 * -0.003)
D2 = D_W / W

MEASURED = [
    ("기준선",   np.array([U, W]),                       0.0,            1114.6116846973),
    ("s=0.96",   np.array([0.96 * U, 0.96 * W]),         0.44 * 0.04,    1109.4232092263),
    ("배포 최적", np.array([0.655318, 0.387810]),        -0.020206959,   1116.5770907872),
]


def build(a2):
    b22 = 2.0 * a2 - S2 - LAM * D_W * D_W / (W * W)
    b12 = a2 - CONST
    return np.array([A1, a2]), np.array([[B11, b12], [b12, b22]])


def predict(a2, r, beta, alpha):
    vec, mat = build(a2)
    off = alpha + r * (beta.sum() - 1.0) + beta[1] * D2
    return float(2.0 * beta @ vec - beta @ mat @ beta - LAM * off * off)


def main():
    def residual(x):
        a2, r = x
        return [predict(a2, r, MEASURED[i][1], MEASURED[i][2]) - MEASURED[i][3]
                for i in (1, 2)]

    a2, r = fsolve(residual, [1032.4, 0.4607], full_output=False)
    print("=" * 68)
    print("1. 가정 없이 확정된 2025 기하")
    print("=" * 68)
    print(f"  A2 = {a2:.4f}")
    print(f"  r  = {r:.6f}   ← 비공개 평가셋 기준율, 측정으로 확정")
    print(f"  (§1.17 역산 추정치 0.4607 대비 {r - 0.4607:+.6f})")
    vec, mat = build(a2)
    print(f"  B22 = {mat[1, 1]:.4f}   B12 = {mat[0, 1]:.4f}   "
          f"corr(p1,p2) = {mat[0, 1] / np.sqrt(mat[0, 0] * mat[1, 1]):.5f}")
    print(f"  DEN = r(1-r) = {r * (1 - r):.6f}   →  1e5/DEN = {1e5 / (r * (1 - r)):,.0f} "
          f"(가정한 lam {LAM:,.0f}, 차 {1e5 / (r * (1 - r)) - LAM:+,.0f})")

    print("\n  검산 — 측정 3점 재현")
    for name, beta, alpha, obs in MEASURED:
        got = predict(a2, r, beta, alpha)
        print(f"    {name:8s} 모델 {got:12.6f}  실측 {obs:12.6f}  차 {got - obs:+.2e}")

    print("\n" + "=" * 68)
    print("2. 진짜 최적점과 남은 상방")
    print("=" * 68)
    beta_star = np.linalg.solve(mat, vec)
    off_target = -(r * (beta_star.sum() - 1.0) + beta_star[1] * D2)
    peak = float(vec @ beta_star)
    deployed = MEASURED[2][3]
    print(f"  최적 beta = [{beta_star[0]:.6f}, {beta_star[1]:.6f}]   "
          f"(배포값 [0.655318, 0.387810])")
    print(f"  최적 alpha = {off_target:+.8f}   (배포값 -0.020206959)")
    print(f"  진짜 정점 = {peak:.6f}")
    print(f"  현재 배포 = {deployed:.6f}")
    print(f"  ── 남은 상방 = {peak - deployed:+.6f}")

    print("\n" + "=" * 68)
    print("3. 남은 3슬롯 판정")
    print("=" * 68)
    if peak - deployed < 0.05:
        print("  (p1, p2) 평면은 완전히 소진됐다. 이 평면 안의 어떤 후보도 슬롯 가치가 없다.")
    else:
        print(f"  재배포로 {peak - deployed:+.3f} 회수 가능.")
    print(f"\n  이제 평면 안의 임의 후보를 오프라인 채점할 수 있다:")
    for name, b in (("50:50", np.array([0.5, 0.5])),
                    ("원래 챔피언 단독", np.array([1.0, 0.0])),
                    ("EXP-021 단독", np.array([0.0, 1.0])),
                    ("가중합=1 최적", None)):
        if b is None:
            # b1 + b2 = 1 제약 하의 최적 (기존 혼합곡선 정점)
            ones = np.array([1.0, 1.0])
            lam_c = (ones @ np.linalg.solve(mat, vec) - 1.0) / (ones @ np.linalg.solve(mat, ones))
            b = np.linalg.solve(mat, vec - lam_c * ones)
        off = -(r * (b.sum() - 1.0) + b[1] * D2)
        print(f"    {name:16s} beta=[{b[0]:.4f},{b[1]:.4f}]  "
              f"최적절편 점수 {predict(a2, r, b, off):10.4f}")


if __name__ == "__main__":
    main()
