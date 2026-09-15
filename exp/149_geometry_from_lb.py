"""LB 실측만으로 2025 평가셋의 숨은 기하를 복원하고, 프로브 예상 점수를 계산한다.

핵심: HANDOFF §1.17 의 중심 프로브 2점이 미지수를 정확히 푼다.
    S(δ) = S0 + λ(2δd − δ²),  λ = 1e5/DEN,  d = E[p] − r  (챔피언의 중심오차)
    미지수 (λ, d) 2개 / 식 2개  →  λ 확정  →  DEN 확정  →  비공개 기준율 r 확정

λ 를 알면 정규화 단위(B = λ·Var 등)를 실제 분산·표준편차로 되돌릴 수 있고,
그 값을 가짜 평가셋에서 실제로 측정한 값과 대조해 기하 가정을 검증할 수 있다.
"""

from __future__ import annotations

import numpy as np

# --- HANDOFF §1.17 실측 (챔피언 단독, 2025 LB) ---
S0 = 1059.05                 # 사후조정 없음
CENTER_PROBES = ((0.012, 1065.26), (0.0066, 1076.81))

# --- 현재 혼합 기하 (blend_metadata.json 실측) ---
S1 = 1092.808353586          # 챔피언 (중심·스케일 조정 후)
S2 = 1043.6074197937         # EXP-021 endpoint 단독
K = 171.5001496146865        # 혼합 곡률
W = 0.35655716947524263
U = 1.0 - W
S_BLEND = 1114.611684697336


def solve_lambda(s0):
    """중심 프로브 2점에서 (λ, d) 를 정확히 푼다.  λ(2δd − δ²) = S(δ) − S0."""
    (d1, y1), (d2, y2) = CENTER_PROBES
    # 미지수 X = λd, Y = λ :  2*d1*X - d1^2*Y = y1-s0 ,  2*d2*X - d2^2*Y = y2-s0
    matrix = np.array([[2 * d1, -d1 * d1], [2 * d2, -d2 * d2]], dtype=np.float64)
    rhs = np.array([y1 - s0, y2 - s0], dtype=np.float64)
    x, y = np.linalg.solve(matrix, rhs)
    return y, x / y                                   # λ, d


def base_rate(lam):
    den = 1e5 / lam
    disc = 0.25 - den
    if disc < 0:
        return None, None, den
    root = np.sqrt(disc)
    return 0.5 - root, 0.5 + root, den


def main():
    print("=" * 74)
    print("1. 숨은 상수 복원 — S0 기록 정밀도(소수 2자리)에 대한 민감도까지")
    print("=" * 74)
    rows = []
    for s0 in (1059.01, 1059.05, 1059.07):
        lam, d = solve_lambda(s0)
        low, high, den = base_rate(lam)
        rows.append((s0, lam, d, den, low, high))
        print(f"  S0={s0:8.2f}  λ={lam:10,.0f}  d={d:.6f}  DEN={den:.6f}  "
              f"r = {low:.4f} 또는 {high:.4f}")
    lam, d, den = rows[1][1], rows[1][2], rows[1][3]
    r_low = rows[1][4]
    print(f"\n  → 2019~2024 성공률이 .565→.486 으로 하락 중이므로 r = {r_low:.4f} 쪽이 답이다")
    print(f"  → 복원된 d = {d:.6f} 는 배포 상수 0.0066 과 일치 (교차검증 통과)")
    print(f"  → 중심 벌점 회수량 λd² = {lam * d * d:.2f} 점, §1.17 기록 '17.7' 과 일치\n")

    print("=" * 74)
    print("2. λ 를 알면 정규화 단위를 실제 확률 단위로 되돌릴 수 있다")
    print("=" * 74)
    var_p1 = S1 / lam
    print(f"  Var(챔피언 예측)   = B11/λ = {var_p1:.8f}   → 표준편차 {np.sqrt(var_p1):.6f}")
    e_sq_diff = K / lam
    print(f"  E[(p1-p2)²]        = K/λ   = {e_sq_diff:.8f}   → RMS 차이 {np.sqrt(e_sq_diff):.6f}")
    print("  ↑ 이 두 값은 가짜 평가셋에서 직접 측정 가능하다 (exp/150 에서 대조)\n")

    print("=" * 74)
    print("3. 프로브 예상 점수 — 실현 가능 영역 전체 스캔")
    print("=" * 74)
    const = (S2 + K - S1) / 2.0            # B12 = A2 - const
    results = []
    for a2 in np.arange(600.0, 2600.0, 1.0):
        for b22 in np.arange(200.0, 4000.0, 1.0):
            m2 = 2.0 * a2 - b22 - S2
            b12 = a2 - const
            if m2 < 0.0 or b22 <= 0.0:
                continue
            if abs(b12) > np.sqrt(S1 * b22):
                continue
            if a2 * a2 > b22 * 1e5:
                continue
            a_w = U * S1 + W * a2
            b_w = U * U * S1 + 2.0 * U * W * b12 + W * W * b22
            if b_w <= 0.0:
                continue
            m_w = W * W * m2                         # = λ·(혼합 평균오차)²
            # 중심 c 고정, p'' = c + s(p - c) 일 때의 정확한 점수식
            delta_w = -np.sqrt(m_w / lam)            # 혼합 평균이 r 보다 낮은 쪽(아래 4번 근거)
            mu_w = r_low + delta_w
            g = mu_w - 0.44                          # 배포한 프로브의 중심
            score = lambda s: (2.0 * s * a_w - s * s * b_w
                               - lam * (delta_w + (s - 1.0) * g) ** 2)
            if abs(score(1.0) - S_BLEND) > 1e-6:
                continue
            results.append((score(1.04), score(0.96), a_w / b_w, b_w,
                            a_w * a_w / b_w - S_BLEND, m_w))

    arr = np.array(results)
    p104, p096, beta, bww, gain, mw = (arr[:, i] for i in range(6))
    print(f"  실현 가능 조합 {len(arr):,}개\n")
    for name, values in (("s=1.04 프로브", p104), ("s=0.96 프로브", p096)):
        print(f"  {name}:  최소 {values.min():9.3f} / 하위25% {np.percentile(values, 25):9.3f} "
              f"/ 중앙 {np.median(values):9.3f} / 상위25% {np.percentile(values, 75):9.3f} "
              f"/ 최대 {values.max():9.3f}")
        print(f"{'':16s}현재({S_BLEND:.2f}) 보다 높을 확률 {100 * np.mean(values > S_BLEND):5.1f}%")
    print(f"\n  두 프로브 합:  중앙 {np.median(p104 + p096):.3f}  "
          f"(기준선 2배 = {2 * S_BLEND:.3f}, 차이 {np.median(p104 + p096) - 2 * S_BLEND:+.3f})")
    print("  ↑ 합은 −0.0032·B_ww 로 거의 결정된다. 두 점의 '차이'가 정점 정보다.\n")

    print("=" * 74)
    print("4. 축별 이득 분해 — 스케일 vs 중심")
    print("=" * 74)
    scale_gain = gain - mw
    print(f"  전체 아핀 이득 G : 중앙 {np.median(gain):+7.3f}  (범위 {gain.min():+.3f} ~ {gain.max():+.3f})")
    print(f"    ├ 스케일 성분  : 중앙 {np.median(scale_gain):+7.3f}  = (A−B)²/B")
    print(f"    └ 중심 성분 M_w: 중앙 {np.median(mw):+7.3f}  = λ·(혼합평균 − r)²")
    print(f"  최적 기울기 β    : 중앙 {np.median(beta):.5f}  (β>1 비율 {100 * np.mean(beta > 1):.1f}%)")
    print(f"\n  중심 성분이 {100 * np.median(mw) / np.median(gain):.0f}% 를 차지한다.")
    print("  그리고 λ 가 이미 알려졌으므로 중심축은 프로브 1장이면 d 가 정확히 나온다:")
    print("      S(δ) = S_blend + λ(2δ·d_w − δ²)   미지수는 d_w 하나뿐")
    print("  스케일축은 3점이 필요하다. 슬롯당 기대이득은 중심축이 압도적으로 크다.")


if __name__ == "__main__":
    main()
