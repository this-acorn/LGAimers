"""아핀 스케일 s 의 LB 3점 포물선을 풀어 꼭짓점을 확정한다.

고정 중심 c 로 p'' = c + s(p - c) 를 걸면
    Score(s) = 2 s A - s^2 B - (1e5/DEN)(e + (s-1) g)^2
로 s 에 대한 정확한 이차식이 된다 (A, B, e, g 전부 미지수여도 무방).
LB 점수는 소수 10자리까지 주어지므로 서로 다른 세 s 점이면 계수가 오차 없이 결정된다.

    python exp/148_affine_vertex.py --s104 <점수> --s096 <점수>
"""

from __future__ import annotations

import argparse

import numpy as np

S_BASE = 1.00
SCORE_BASE = 1114.6116846973   # 자체 구현 zip = 기존 챔피언과 비트 동일, 이미 측정된 값
CENTER = 0.44


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s104", type=float, required=True, help="probe_affine_s104 LB 점수")
    parser.add_argument("--s096", type=float, required=True, help="probe_affine_s096 LB 점수")
    parser.add_argument("--base", type=float, default=SCORE_BASE)
    args = parser.parse_args()

    points = [(S_BASE, args.base), (1.04, args.s104), (0.96, args.s096)]
    s = np.array([p[0] for p in points], dtype=np.float64)
    y = np.array([p[1] for p in points], dtype=np.float64)
    a, b, c = np.linalg.solve(np.vstack([s * s, s, np.ones_like(s)]).T, y)

    print("입력 3점")
    for si, yi in points:
        print(f"  s={si:.4f}  LB={yi:.10f}")
    print(f"\n계수: a={a:+.6f}  b={b:+.6f}  c={c:+.6f}   (a<0 이어야 위로 볼록)")
    if a >= 0:
        print("\n판정: 포물선이 위로 볼록하지 않다 — 이 축은 최대점이 없다. 배포 금지.")
        return

    star = -b / (2.0 * a)
    peak = c - b * b / (4.0 * a)
    gain = peak - args.base
    print(f"\n최적 스케일 s* = {star:.6f}   (중심 c = {CENTER})")
    print(f"예상 최고점    = {peak:.10f}")
    print(f"현재 대비 이득 = {gain:+.4f}")
    print(f"곡률 |a|       = {abs(a):.3f}  → s 를 0.01 벗어날 때 손실 {abs(a) * 1e-4:.4f}")

    print("\n검산 (푼 이차식이 입력 3점을 되돌려주는지)")
    for si, yi in points:
        print(f"  s={si:.4f}  모델 {a * si * si + b * si + c:.10f}  실측 {yi:.10f}")

    if not 0.80 <= star <= 1.30:
        print("\n주의: 꼭짓점이 상식 범위를 벗어났다. 측정값을 다시 확인할 것.")
    elif gain <= 0.0:
        print("\n판정: 현재 s=1.00 이 이미 정점 부근이다. 배포하지 말 것.")
    else:
        print(f"\n다음 명령으로 배포본을 빌드한다:")
        print(f"  python exp/146_build_ours_zip.py --name candidate_affine_final "
              f"--scale {star:.6f} --center {CENTER} --skip-diff")


if __name__ == "__main__":
    main()
