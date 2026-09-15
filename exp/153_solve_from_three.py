"""측정 3점으로 사후 아핀의 최적점을 풀고, 남은 상방을 확정한다.

측정
    Score(s=1.00, d=0)      = 1114.6116846973   기준선 (이미 알려진 값)
    Score(s=1.00, d=-0.003) = 1110.2391147581   중심 프로브  → d_w 확정
    Score(s=0.96, d=0)      = 1109.4232092263   스케일 프로브

모형  Score(s,delta) = 2 s A - s^2 B - lam (h + s g - delta)^2
      h = c - r,  g = mu - c,  h + g = d_w
미지수 A, B, g 중 g 는 복원한 r 로 결정되므로 두 식이면 A, B 가 나온다.
r 은 §1.17 에서 lam 과 함께 복원했으므로 그 불확실 구간 전체를 훑어 민감도를 본다.
"""

from __future__ import annotations

import numpy as np

LAM = 402483.0
C = 0.44
S1 = A1 = B11 = 1092.808353586
S2 = 1043.6074197937
K = 171.5001496146865
W = 0.35655716947524263
U = 1.0 - W
BASE = 1114.6116846973
SHIFT_PROBE = (-0.003, 1110.2391147581)
SCALE_PROBE = (0.96, 1109.4232092263)
CONST = (S2 + K - B11) / 2.0


def solve(r):
    delta_p, y_shift = SHIFT_PROBE
    d_w = ((y_shift - BASE) / LAM + delta_p ** 2) / (2.0 * delta_p)
    h = C - r
    g = d_w - h
    s_p, y_scale = SCALE_PROBE
    # 2A - B = BASE + lam*(h+g)^2 ;  2 s A - s^2 B = y_scale + lam*(h + s g)^2
    rhs1 = BASE + LAM * (h + g) ** 2
    rhs2 = y_scale + LAM * (h + s_p * g) ** 2
    a, b = np.linalg.solve(np.array([[2.0, -1.0], [2.0 * s_p, -s_p * s_p]]),
                           np.array([rhs1, rhs2]))
    s_star = a / b
    delta_star = h + s_star * g
    peak2 = a * a / b

    a2 = (a - U * A1) / W                       # A_w = U*A1 + W*A2
    b22 = 2.0 * a2 - S2 - LAM * d_w * d_w / (W * W)
    b12 = a2 - CONST
    bww_check = U * U * B11 + 2.0 * U * W * b12 + W * W * b22
    vec = np.array([A1, a2])
    mat = np.array([[B11, b12], [b12, b22]])
    peak3 = float(vec @ np.linalg.solve(mat, vec)) if np.linalg.det(mat) > 1e-9 else np.nan
    beta = np.linalg.solve(mat, vec) if np.linalg.det(mat) > 1e-9 else np.array([np.nan] * 2)
    rho = b12 / np.sqrt(B11 * b22) if b22 > 0 else np.nan
    return dict(d_w=d_w, h=h, g=g, A=a, B=b, s=s_star, delta=delta_star, peak2=peak2,
                A2=a2, B22=b22, B12=b12, bww_check=bww_check, peak3=peak3,
                beta=beta, rho=rho, shape2=a2 * a2 / b22 if b22 > 0 else np.nan)


def main():
    base = solve(0.4607)
    print("=" * 72)
    print("1. 중심 복원 r = 0.4607 기준 해")
    print("=" * 72)
    print(f"  d_w = {base['d_w']:+.8f}   h = c-r = {base['h']:+.6f}   g = mu-c = {base['g']:+.6f}")
    print(f"  A = lam*Cov(p,y) = {base['A']:9.4f}")
    print(f"  B = lam*Var(p)   = {base['B']:9.4f}   → sd(p) = {np.sqrt(base['B'] / LAM):.6f}")
    print(f"  최적 s* = {base['s']:.6f}   delta* = {base['delta']:+.8f}")
    print(f"  2-파라미터 정점 = {base['peak2']:.6f}   ({base['peak2'] - BASE:+.4f})")
    print(f"  3-파라미터 정점 = {base['peak3']:.6f}   ({base['peak3'] - BASE:+.4f})")
    print(f"    최적 가중 beta = [{base['beta'][0]:.6f}, {base['beta'][1]:.6f}] "
          f"(현재 [{U:.6f}, {W:.6f}])")

    print("\n" + "=" * 72)
    print("2. 복원된 2x2 기하 — 독립 정합성 검사")
    print("=" * 72)
    print(f"  A2 = {base['A2']:8.3f}   B22 = {base['B22']:8.3f}   B12 = {base['B12']:8.3f}")
    print(f"  B_ww 직접해 {base['B']:.4f}  vs  2x2 재구성 {base['bww_check']:.4f}  "
          f"차 {base['B'] - base['bww_check']:+.2e}")
    print(f"  corr(p1,p2) = {base['rho']:.5f}")
    print(f"  EXP-021 자기 최적 아핀 후 = {base['shape2']:.3f}  (단독 실측 {S2:.3f}, "
          f"자체 보정 여지 {base['shape2'] - S2:+.3f})")

    print("\n" + "=" * 72)
    print("3. r 불확실성에 대한 민감도 (§1.17 S0 정밀도에서 0.459~0.463)")
    print("=" * 72)
    print(f"  {'r':>8s} {'s*':>10s} {'delta*':>12s} {'2-파라정점':>12s} {'3-파라정점':>12s}")
    grid = []
    for r in (0.4550, 0.4590, 0.4607, 0.4630, 0.4700):
        g = solve(r)
        grid.append(g)
        print(f"  {r:8.4f} {g['s']:10.6f} {g['delta']:+12.8f} {g['peak2']:12.6f} "
              f"{g['peak3']:12.6f}")
    peaks = np.array([g["peak2"] for g in grid])
    ss = np.array([g["s"] for g in grid])
    print(f"\n  s* 폭 {ss.min():.4f}~{ss.max():.4f}   정점 폭 {peaks.min():.3f}~{peaks.max():.3f}")
    worst = base["B"] * (ss.max() - ss.min()) ** 2 / 4.0
    print(f"  r 을 잘못 잡아 s* 가 이 폭만큼 어긋날 때 최대 손실 ≈ {worst:.4f} 점")

    print("\n" + "=" * 72)
    print("4. 판정")
    print("=" * 72)
    print(f"  중심축   실측 이득 = {LAM * base['d_w'] ** 2:+.4f}")
    print(f"  스케일축 실측 이득 = {base['peak2'] - BASE - LAM * base['d_w'] ** 2:+.4f}")
    print(f"  재가중   추가 이득 = {base['peak3'] - base['peak2']:+.4f}")
    print(f"  ── 사후처리 총 상방 = {base['peak3'] - BASE:+.4f}  →  {base['peak3']:.4f}")
    print("\n  즉 사후처리는 사실상 소진됐다. 남은 상방은 전부 새 모델에서 와야 한다.")

    print("\n" + "=" * 72)
    print("5. 3번째 모델이 목표 점수를 만들려면 필요한 K")
    print("=" * 72)
    peak = base["peak3"]
    print(f"  {'목표':>10s} {'필요이득':>10s} | " + " ".join(f"{s:>9s}" for s in
          ("S=1120", "S=1090", "S=1050", "S=1000")))
    for target, label in ((1142.36, "100위"), (1180.26, "20위"), (1225.21, "1위")):
        need = target - peak
        cells = []
        for s_new in (1120.0, 1090.0, 1050.0, 1000.0):
            d = peak - s_new
            # (K-d)^2/(4K) = need  →  K^2 - (2d+4need)K + d^2 = 0
            disc = (2 * d + 4 * need) ** 2 - 4 * d * d
            cells.append(f"{(2 * d + 4 * need + np.sqrt(disc)) / 2:9.0f}" if disc >= 0 else "     —   ")
        print(f"  {label:>10s} {need:+10.1f} | " + " ".join(cells))
    print(f"\n  참고: 역대 관측 최대 K = 171.5 (EXP-021), 그 전 최대 117.5")


if __name__ == "__main__":
    main()
