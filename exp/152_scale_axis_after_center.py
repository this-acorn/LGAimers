"""중심 프로브 실측 후, 스케일축에 남은 이득이 슬롯 2~3장 값을 하는지 판정한다.

중심 프로브가 M_w = lambda * d_w^2 를 확정했으므로 실현 가능 영역의 자유도가 2개에서 1개로 준다:
    M2   = M_w / W^2                       (확정)
    B22  = 2*A2 - S2 - M2                  (M2 정의에서)
    B12  = A2 - (S2 + K - B11)/2           (K 정의에서)
    → 남은 자유변수는 A2 하나뿐

각 A2 에 대해 A_w, B_ww 를 만들고 스케일 이득 (A_w - B_ww)^2 / B_ww 를 본다.
"""

from __future__ import annotations

import numpy as np

LAMBDA = 402483.0
S1 = B11 = A1 = 1092.808353586
S2 = 1043.6074197937
K = 171.5001496146865
W = 0.35655716947524263
U = 1.0 - W
S_BLEND = 1114.611684697336
D_W = 3.1066e-4                      # 중심 프로브 실측
M_W = LAMBDA * D_W * D_W
M2 = M_W / (W * W)
CONST = (S2 + K - B11) / 2.0


def main() -> None:
    print(f"중심 프로브 확정치:  d_w = {D_W:+.8f}   M_w = {M_W:.4f}   M2 = {M2:.4f}\n")

    rows = []
    for a2 in np.arange(400.0, 4000.0, 0.05):
        b22 = 2.0 * a2 - S2 - M2
        b12 = a2 - CONST
        if b22 <= 0.0:
            continue
        if abs(b12) > np.sqrt(B11 * b22):                 # 코시-슈바르츠
            continue
        if a2 * a2 > b22 * 1e5:                           # corr^2 <= 1
            continue
        a_w = U * A1 + W * a2
        b_w = U * U * B11 + 2.0 * U * W * b12 + W * W * b22
        if b_w <= 0.0:
            continue
        if abs(2.0 * a_w - b_w - M_W - S_BLEND) > 1e-6:   # 혼합 항등식
            continue
        rows.append((a2, b22, b12, a_w, b_w, a_w / b_w,
                     (a_w - b_w) ** 2 / b_w, a2 * a2 / b22))

    arr = np.array(rows)
    a2, b22, b12, a_w, b_w, beta, gain, shape2 = (arr[:, i] for i in range(8))
    print(f"실현 가능한 A2 구간: {a2.min():.1f} ~ {a2.max():.1f}  ({len(arr):,} 격자점)\n")
    print(f"  최적 기울기 β   : 최소 {beta.min():.5f} / 중앙 {np.median(beta):.5f} "
          f"/ 최대 {beta.max():.5f}   (β>1 비율 {100 * np.mean(beta > 1):.1f}%)")
    print(f"  스케일 이득     : 최소 {gain.min():+.3f} / 중앙 {np.median(gain):+.3f} "
          f"/ 최대 {gain.max():+.3f}")
    print(f"  이득 > 1.0 비율 : {100 * np.mean(gain > 1.0):.1f}%")
    print(f"  이득 > 3.0 비율 : {100 * np.mean(gain > 3.0):.1f}%\n")

    print("  A2 별 단면 (EXP-021 endpoint 의 판별력이 클수록 A2 가 크다)")
    print(f"  {'A2':>8s} {'B22':>8s} {'shape(p2)':>10s} {'B_ww':>8s} {'β':>8s} {'스케일이득':>10s}")
    for target in np.percentile(a2, [0, 10, 25, 50, 75, 90, 100]):
        i = int(np.argmin(np.abs(a2 - target)))
        print(f"  {a2[i]:8.1f} {b22[i]:8.1f} {shape2[i]:10.2f} {b_w[i]:8.1f} "
              f"{beta[i]:8.5f} {gain[i]:+10.3f}")

    # p2 단독 shape 을 현실 범위로 제한하면 구간이 훨씬 좁아진다
    tight = arr[(shape2 >= S2) & (shape2 <= S2 + 60.0)]
    if len(tight):
        g = tight[:, 6]
        b = tight[:, 5]
        print(f"\n  [shape(p2) ∈ {S2:.0f}~{S2 + 60:.0f} 제한: {len(tight):,}점]")
        print(f"    β  최소 {b.min():.5f} / 중앙 {np.median(b):.5f} / 최대 {b.max():.5f}")
        print(f"    이득 최소 {g.min():+.3f} / 중앙 {np.median(g):+.3f} / 최대 {g.max():+.3f}")

    print("\n" + "=" * 70)
    print("판정")
    print("=" * 70)
    print(f"  중심축 이득  = {M_W:+.3f}  (실측 확정, 사실상 0)")
    print(f"  스케일축 이득 = 중앙 {np.median(gain):+.3f}  → 프로브 2장 + 배포 1장 = 슬롯 3장")
    print(f"  두 축 합계 상한(shape - 현재) = {np.median(gain) + M_W:+.3f}")


if __name__ == "__main__":
    main()
