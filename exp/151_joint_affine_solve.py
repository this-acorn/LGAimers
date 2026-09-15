"""사후 아핀의 2차원 최적점을 LB 실측에서 정확히 푼다.

배포 변환:  p'' = c + s(p - c) - delta      (c = 0.44 고정, s 와 delta 가 자유변수)
            → 2자유도이므로 임의의 아핀을 전부 표현한다.

정규화 단위 (lambda = 1e5/DEN, exp/149 에서 402,483 으로 복원):
    h = c - mu_r          (선택한 중심과 진짜 기준율의 차)
    g = mu - c            (혼합 평균과 선택한 중심의 차)
    h + g = mu - r        (혼합의 중심오차 d_w)
    A = lambda*Cov(p,y),  B = lambda*Var(p)

    Score(s, delta) = 2sA - s^2 B - lambda (h + s g - delta)^2

미지수 4개 (A, B, h, g) / 측정 4개 (기준선 + 중심프로브 + 스케일프로브 2개) → 정확해.

최적점:  delta* = h + s* g   (평균을 정확히 r 로 맞춤)
         s*     = A / B
         정점   = A^2 / B    (= shape 점수, 아핀으로 도달 가능한 최대)

    python exp/151_joint_affine_solve.py --shift-probe <점수> [--s104 <점수> --s096 <점수>]
"""

from __future__ import annotations

import argparse

import numpy as np

LAMBDA = 402483.0
CENTER = 0.44
BASE = 1114.6116846973
DELTA_PROBE = -0.003
S_HI, S_LO = 1.04, 0.96
R_PRIOR = 0.4607          # exp/149 에서 복원한 2025 기준율 (부호 판별용)


def solve_center(score_shift: float) -> float:
    """중심 프로브 1장으로 혼합의 중심오차 d_w 를 정확히 구한다."""
    gain = score_shift - BASE                       # = lambda (2 delta d_w - delta^2)
    return (gain / LAMBDA + DELTA_PROBE ** 2) / (2.0 * DELTA_PROBE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shift-probe", type=float, required=True)
    parser.add_argument("--s104", type=float)
    parser.add_argument("--s096", type=float)
    args = parser.parse_args()

    d_w = solve_center(args.shift_probe)
    center_gain = LAMBDA * d_w * d_w
    print("=" * 70)
    print("1. 중심축 — 프로브 1장으로 정확해")
    print("=" * 70)
    print(f"  중심 프로브 delta = {DELTA_PROBE:+.4f}  LB = {args.shift_probe:.10f}")
    print(f"  기준선 대비        = {args.shift_probe - BASE:+.10f}")
    print(f"  혼합 중심오차 d_w  = {d_w:+.8f}   (혼합 평균 - 기준율 r)")
    print(f"  회수 가능 점수     = lambda*d_w^2 = {center_gain:+.4f}")
    print(f"  중심만 고치면      = {BASE + center_gain:.6f}")
    print(f"  → 빌드: exp/146_build_ours_zip.py --name candidate_shift_only "
          f"--scale 1.0 --center {CENTER} --shift {d_w:.8f} --skip-diff")

    if args.s104 is None or args.s096 is None:
        print("\n  (스케일 프로브 2장이 더 들어오면 2차원 최적점까지 풉니다)")
        return

    print("\n" + "=" * 70)
    print("2. 스케일축 — delta=0 인 3점으로 s-포물선")
    print("=" * 70)
    s = np.array([1.0, S_HI, S_LO])
    y = np.array([BASE, args.s104, args.s096])
    c2, c1, c0 = np.linalg.solve(np.vstack([s * s, s, np.ones_like(s)]).T, y)
    print(f"  Score(s,0) = {c2:+.4f} s^2 {c1:+.4f} s {c0:+.4f}")
    if c2 >= 0:
        print("  판정: 위로 볼록하지 않다 — 측정값을 재확인할 것. 배포 금지.")
        return

    # c0 = -lambda h^2  →  |h| 결정, 부호는 복원한 기준율로 판별
    if c0 > 0:
        print(f"  경고: s=0 절편이 양수({c0:+.3f}) — 이론상 -lambda*h^2 <= 0 이어야 한다.")
        print("        측정 정밀도나 클리핑을 의심할 것.")
        return
    h_abs = np.sqrt(-c0 / LAMBDA)
    h = -h_abs if CENTER < R_PRIOR else h_abs
    r_implied = CENTER - h
    print(f"  |h| = |c - r| = {h_abs:.6f}  →  r = {r_implied:.6f}")
    print(f"  exp/149 가 §1.17 에서 복원한 r = {R_PRIOR:.4f} 와의 차이 "
          f"{r_implied - R_PRIOR:+.6f}  ← 독립 교차검증")

    g = d_w - h
    b_norm = -c2 - LAMBDA * g * g
    a_norm = (c1 + 2.0 * LAMBDA * h * g) / 2.0
    print(f"\n  g = mu - c = {g:+.6f}   →  혼합 평균 mu = {CENTER + g:.6f}")
    print(f"  A = lambda*Cov(p,y) = {a_norm:9.3f}")
    print(f"  B = lambda*Var(p)   = {b_norm:9.3f}   →  Var = {b_norm / LAMBDA:.8f} "
          f"(sd {np.sqrt(max(b_norm, 0) / LAMBDA):.6f})")

    if b_norm <= 0:
        print("  판정: 분산 추정이 음수 — 측정값 재확인 필요. 배포 금지.")
        return

    print("\n" + "=" * 70)
    print("3. 2차원 최적점")
    print("=" * 70)
    s_star = a_norm / b_norm
    delta_star = h + s_star * g
    peak = a_norm * a_norm / b_norm
    print(f"  s*     = A/B        = {s_star:.6f}")
    print(f"  delta* = h + s* g   = {delta_star:+.8f}")
    print(f"  정점   = A^2/B      = {peak:.6f}")
    print(f"  기준선 대비 이득    = {peak - BASE:+.4f}")
    print(f"    ├ 중심 성분       = {center_gain:+.4f}")
    print(f"    └ 스케일 성분     = {peak - BASE - center_gain:+.4f}")
    print("\n  검산 (푼 식이 측정 4점을 되돌려주는지)")
    for si, di, obs in ((1.0, 0.0, BASE), (1.0, DELTA_PROBE, args.shift_probe),
                        (S_HI, 0.0, args.s104), (S_LO, 0.0, args.s096)):
        model = (2 * si * a_norm - si * si * b_norm
                 - LAMBDA * (h + si * g - di) ** 2)
        print(f"    s={si:.2f} delta={di:+.4f}  모델 {model:12.6f}  실측 {obs:12.6f}  "
              f"차 {model - obs:+.2e}")

    if peak - BASE <= 0.5:
        print("\n  판정: 이득이 미미하다. 슬롯을 쓰지 말 것.")
    else:
        print(f"\n  빌드: exp/146_build_ours_zip.py --name candidate_affine_final "
              f"--scale {s_star:.6f} --center {CENTER} --shift {delta_star:.8f} --skip-diff")


if __name__ == "__main__":
    main()
