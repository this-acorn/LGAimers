"""codex 의 CAT5 anti-blend 를 LB 로 복원한 2025 기하에서 슬롯 없이 채점한다.

anti-blend 변환 (exp/161):
    q            = CAT5 raw (V18=0, scale=1, shift=0)
    blended_new  = blended + w (q - blended) = (1-w)·blended - (-w)·q ,  w = -0.5397
챔피언 정의에서
    p1 = 0.49 + 1.06((q + 0.30 V) - 0.49) - 0.0066 = 1.06 q + 0.318 V - 0.036
    => q = (p1 - 0.318 V + 0.036)/1.06
따라서
    blended_new = (1-w)(U p1 + W p2) - w (p1 - 0.318 V + 0.036)/1.06
                = b1 p1 + b2 p2 + bV V + const
즉 V18 항을 빼면 **우리가 이미 측정으로 최적화한 (b1,b2) 평면 위의 한 점**이다.
그 평면의 기하 (A, B) 는 exp/153 에서 LB 3점으로 복원했으므로 직접 채점된다.

    Score(b) = 2 bᵀA - bᵀB b        (절편은 최적으로 잡아준다 = anti-blend 에 유리한 가정)
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

ANTI_W = -0.5397043598776958
CHAMP_SCALE = 1.06
V18_GAMMA = 0.30


def geometry(r):
    delta_p, y_shift = SHIFT_PROBE
    d_w = ((y_shift - BASE) / LAM + delta_p ** 2) / (2.0 * delta_p)
    h = C - r
    g = d_w - h
    s_p, y_scale = SCALE_PROBE
    a, b = np.linalg.solve(
        np.array([[2.0, -1.0], [2.0 * s_p, -s_p * s_p]]),
        np.array([BASE + LAM * (h + g) ** 2, y_scale + LAM * (h + s_p * g) ** 2]))
    a2 = (a - U * A1) / W
    b22 = 2.0 * a2 - S2 - LAM * d_w * d_w / (W * W)
    b12 = a2 - CONST
    return (np.array([A1, a2]), np.array([[B11, b12], [b12, b22]]), d_w)


def score(beta, vec, mat):
    """절편을 최적으로 잡았을 때의 점수 (중심 벌점 0)."""
    return float(2.0 * beta @ vec - beta @ mat @ beta)


def main():
    # anti-blend 가 함의하는 (b1, b2)
    b1 = (1.0 - ANTI_W) * U + ANTI_W / CHAMP_SCALE
    b2 = (1.0 - ANTI_W) * W
    bv = -ANTI_W * V18_GAMMA                      # 추가로 얹히는 V18 계수
    print("anti-blend 를 (p1, p2, V18) 로 분해")
    print(f"  b1(p1)   = {b1:.6f}")
    print(f"  b2(p2)   = {b2:.6f}     합 {b1 + b2:.6f}")
    print(f"  추가 V18 = {bv:+.6f}  → 혼합 내 V18 실효계수 "
          f"{b1 * CHAMP_SCALE * V18_GAMMA + bv:.4f} (현재 {U * CHAMP_SCALE * V18_GAMMA:.4f}, "
          f"{100 * ((b1 * CHAMP_SCALE * V18_GAMMA + bv) / (U * CHAMP_SCALE * V18_GAMMA) - 1):+.0f}%)\n")

    print(f"{'r':>8s} {'현재혼합':>10s} {'최적β':>10s} {'anti-blend':>12s} {'anti−최적':>10s} "
          f"{'anti−현재':>10s}")
    rows = []
    for r in (0.4550, 0.4590, 0.4607, 0.4630, 0.4700):
        vec, mat, d_w = geometry(r)
        opt = float(vec @ np.linalg.solve(mat, vec))
        cur = score(np.array([U, W]), vec, mat)
        anti = score(np.array([b1, b2]), vec, mat)
        rows.append((r, cur, opt, anti))
        print(f"{r:8.4f} {cur:10.3f} {opt:10.3f} {anti:12.3f} {anti - opt:+10.3f} "
              f"{anti - cur:+10.3f}")

    arr = np.array(rows)
    print(f"\n판정: anti-blend 의 (b1,b2) 는 r 을 어디에 놔도 최적점보다 "
          f"{arr[:, 3].max() - arr[:, 2].min():+.2f} ~ {arr[:, 3].min() - arr[:, 2].max():+.2f} 낮다.")
    print(f"      현재 배포 혼합(절편 최적화 포함)보다도 "
          f"{arr[:, 3].min() - arr[:, 1].max():+.2f} ~ {arr[:, 3].max() - arr[:, 1].min():+.2f} 낮다.")

    # 이 방향으로 얼마나 가야 최적인가 — anti weight 의 최적값
    vec, mat, _ = geometry(0.4607)
    print("\nanti weight w 를 바꿔가며 (V18 항 제외):")
    print(f"  {'w':>9s} {'b1':>9s} {'b2':>9s} {'점수':>11s}")
    best = None
    for w in np.arange(-0.60, 0.65, 0.05):
        bb = np.array([(1.0 - w) * U + w / CHAMP_SCALE, (1.0 - w) * W])
        sc = score(bb, vec, mat)
        if best is None or sc > best[1]:
            best = (w, sc)
        if abs(w % 0.15) < 1e-9 or abs(w - ANTI_W) < 0.03:
            print(f"  {w:9.4f} {bb[0]:9.4f} {bb[1]:9.4f} {sc:11.3f}")
    print(f"\n  이 1-파라미터 족의 최적 w = {best[0]:+.3f} → {best[1]:.3f} "
          f"(codex 값 {ANTI_W:+.4f} 는 반대 방향)")


if __name__ == "__main__":
    main()
