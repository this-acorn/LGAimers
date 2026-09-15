"""혼합 안에서 희석된 V18 계수의 최적점을 대칭 프로브 2장으로 정확히 푼다.

챔피언 내부에서 V18 은 preds 에 gamma 배로 더해지고 그 뒤 affine(scale 1.06) 을 통과한다.
혼합은 다시 beta1 을 곱하므로 최종 예측에 대한 V18 계수는

    c(gamma) = beta1 * 1.06 * (gamma - 0.30)      기준점 gamma=0.30 대비 증분

이며 클리핑이 없으면 정확히 선형이다 (실측 출력 범위 0.2956~0.5444, 여유 충분).
따라서 점수는 c 의 정확한 이차식이다:

    Score(c) = S0 + a*c - b*c^2 ,   b = lam * E[V^2] > 0

gamma 0.42 / 0.18 은 c 가 +-0.0833564 로 대칭이므로

    b  = (2*S0 - S_plus - S_minus) / (2 c^2)
    a  = (S_plus - S_minus) / (2 c)
    c* = a / (2b)          gain = a^2 / (4b) >= 0  (항상 비음수)

    python exp/156_v18_gamma_solve.py --g042 <점수> --g018 <점수>
"""

from __future__ import annotations

import argparse

import numpy as np

S0 = 1116.5770907872          # 현재 배포본 (gamma = 0.30)
BETA1 = 0.655318
CHAMP_SCALE = 1.06
GAMMA_BASE = 0.30
LAM = 402483.0
SLOPE = BETA1 * CHAMP_SCALE   # dc/dgamma
C_PROBE = SLOPE * 0.12        # gamma 0.42 / 0.18 에 해당하는 +-c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g042", type=float, required=True)
    parser.add_argument("--g018", type=float, required=True)
    args = parser.parse_args()

    s_plus, s_minus = args.g042, args.g018
    b = (2.0 * S0 - s_plus - s_minus) / (2.0 * C_PROBE ** 2)
    a = (s_plus - s_minus) / (2.0 * C_PROBE)

    print(f"기준점 gamma=0.30 → {S0:.10f}")
    print(f"  gamma=0.42 (c={+C_PROBE:+.7f}) → {s_plus:.10f}  ({s_plus - S0:+.4f})")
    print(f"  gamma=0.18 (c={-C_PROBE:+.7f}) → {s_minus:.10f}  ({s_minus - S0:+.4f})\n")
    print(f"  a = {a:+.4f}   b = lam*E[V^2] = {b:+.4f}")
    if b <= 0:
        print("\n  판정: b <= 0 이면 위로 볼록이 아니다 — 모형이 깨졌다. 배포 금지.")
        return
    print(f"  → E[V^2] = {b / LAM:.3e}   RMS(V18 효과) = {np.sqrt(b / LAM):.6f}")

    c_star = a / (2.0 * b)
    gamma_star = GAMMA_BASE + c_star / SLOPE
    gain = a * a / (4.0 * b)
    print(f"\n  최적 증분 c* = {c_star:+.7f}")
    print(f"  최적 gamma*  = {gamma_star:.6f}   (현재 0.30)")
    print(f"  혼합 내 V18 실효계수 {SLOPE * GAMMA_BASE:.4f} → {SLOPE * gamma_star:.4f}")
    print(f"  이득 = a^2/(4b) = {gain:+.4f}   →  {S0 + gain:.6f}")

    print("\n  검산 (푼 식이 측정 2점을 되돌려주는지)")
    for tag, c, obs in (("gamma=0.42", +C_PROBE, s_plus), ("gamma=0.18", -C_PROBE, s_minus)):
        print(f"    {tag}  모델 {S0 + a * c - b * c * c:12.7f}  실측 {obs:12.7f}")

    best_probe = max(s_plus, s_minus)
    if S0 + gain <= max(S0, best_probe) + 0.02:
        print(f"\n  판정: 정점이 이미 가진 최고점({max(S0, best_probe):.4f})을 의미 있게 넘지 못한다. "
              f"배포하지 말 것.")
    else:
        print(f"\n  빌드: python exp/146_build_ours_zip.py --name candidate_v18opt "
              f"--w-champ 0.655318 --w-exp021 0.387810 --scale 1.0 --center 0.44 "
              f"--shift 0.020206959 --v18-gamma {gamma_star:.6f} --skip-diff")
        if not 0.0 <= gamma_star <= 1.0:
            print("  경고: gamma* 가 상식 범위를 벗어났다. 측정값을 재확인할 것.")


if __name__ == "__main__":
    main()
