"""EXP-021 의 해석적 베이스 대비 '보정 성분' 배율의 최적점을 대칭 프로브 2장으로 푼다.

EXP-021 = analytic_base + corrections
    corrections = group effect + 0.75·LGB 잔차 / 1.00·HGB 잔차 + team EB + rank-6 low-rank

우리 혼합은 beta2 = 0.387810 을 베이스와 보정에 **똑같이** 걸고 있다. 그런데
  - analytic_base (0.7·투수 시즌EB + 0.3·타자 시즌EB) 는 우리 CatBoost 챔피언과 크게 중복이고
  - corrections 는 구조적으로 새로운 정보다 (외부 기록상 935.81 → 1043.61, +107.8)
따라서 균일 가중은 중복분을 과대, 신규분을 과소 평가한다. 이 비율은 (p1,p2) 평면 **밖**이라
지금까지 한 번도 측정된 적이 없다.

컴포넌트에서 CORRECTION_SCALE = m 이면 (클리핑이 없을 때)
    p_exp021(m) = p + (m-1)·(p - analytic_base)
이므로 최종 예측은 기준점에서 방향 f = beta2·(p - analytic_base) 로 (m-1) 만큼 이동한다.
따라서 점수는 (m-1) 의 정확한 이차식이다.

    m = 1.15 / 0.85 는 (m-1) = +-0.15 로 대칭이므로
        b = (2*S0 - S_plus - S_minus) / (2 * 0.15^2)
        a = (S_plus - S_minus) / (2 * 0.15)
        m* = 1 + a/(2b)          gain = a^2/(4b) >= 0   (항상 비음수)

    python exp/157_corr_scale_solve.py --m115 <점수> --m085 <점수>
"""

from __future__ import annotations

import argparse

S0 = 1116.5770907872      # 현재 배포본 (CORRECTION_SCALE = 1.0)
STEP = 0.15


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m115", type=float, required=True)
    parser.add_argument("--m085", type=float, required=True)
    args = parser.parse_args()

    s_plus, s_minus = args.m115, args.m085
    b = (2.0 * S0 - s_plus - s_minus) / (2.0 * STEP ** 2)
    a = (s_plus - s_minus) / (2.0 * STEP)

    print(f"기준점 m=1.00 → {S0:.10f}")
    print(f"  m=1.15 → {s_plus:.10f}  ({s_plus - S0:+.4f})")
    print(f"  m=0.85 → {s_minus:.10f}  ({s_minus - S0:+.4f})\n")
    print(f"  a = {a:+.4f}   b = {b:+.4f}")
    if b <= 0:
        print("\n  판정: b <= 0 — 위로 볼록이 아니다. 클리핑이나 측정 오류를 의심할 것. 배포 금지.")
        return

    m_star = 1.0 + a / (2.0 * b)
    gain = a * a / (4.0 * b)
    print(f"\n  최적 배율 m* = {m_star:.6f}")
    print(f"  이득 = a^2/(4b) = {gain:+.4f}   →  {S0 + gain:.6f}")
    print("\n  검산")
    for tag, c, obs in (("m=1.15", +STEP, s_plus), ("m=0.85", -STEP, s_minus)):
        print(f"    {tag}  모델 {S0 + a * c - b * c * c:12.7f}  실측 {obs:12.7f}")

    best = max(S0, s_plus, s_minus)
    if S0 + gain <= best + 0.02:
        print(f"\n  판정: 정점이 이미 가진 최고점({best:.4f})을 의미 있게 넘지 못한다. 배포 생략.")
    else:
        print(f"\n  빌드: python exp/146_build_ours_zip.py --name candidate_corr_opt "
              f"--w-champ 0.655318 --w-exp021 0.387810 --scale 1.0 --center 0.44 "
              f"--shift 0.020206959 --corr-scale {m_star:.6f} --skip-diff")
        if not 0.2 <= m_star <= 3.0:
            print("  경고: m* 가 상식 범위를 벗어났다 — 클리핑 발생 가능. 리허설 min/max 를 확인할 것.")


if __name__ == "__main__":
    main()
