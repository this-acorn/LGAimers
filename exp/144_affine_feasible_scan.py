"""혼합 예측에 사후 아핀을 한 번 더 걸었을 때의 이득 G(w)를 LB 4점만으로 구간 추정한다.

정규화 단위 (모두 1e5/DEN 배, DEN = r(1-r)):
    A_i  = Cov(p_i, y),  B_ij = Cov(p_i, p_j),  M_i = (E[p_i] - r)^2
    S(p) = 2A - B - M                      (실제 점수)
    shape(p) = A^2 / B                     (자기 최적 아핀 후 점수)
    G   = shape - S = (A-B)^2/B + M        (아핀으로 얻을 수 있는 이득)

가정 A1: 챔피언 p1 은 이미 LB 3점 곡선으로 아핀을 풀어 배포했으므로 최적 아핀 상태다.
        => A1 = B1 = S1, M1 = 0.  (이 가정 하나만으로 미지수가 4개 -> 2개로 준다)

관측식:
    S2 = 2A2 - B22 - M2
    K  = B11 - 2B12 + B22 + M2     (K = 1e5*E[(p1-p2)^2]/DEN, LB 혼합곡선에서 실측)
=> B12 = A2 - (S1 - K + S2)/2 ... 아래에서 유도 그대로 사용

남은 자유도 (A2, B22) 를 실현 가능 영역 전체에서 훑어 G 와 최적 기울기 beta 의 범위를 본다.
"""

import numpy as np

S1 = 1092.808353586          # 챔피언 단독 LB
S2 = 1043.6074197937         # EXP-021 endpoint 단독 LB
K = 171.5001496146865        # 혼합 곡률 (실측)
W = 0.35655716947524263      # 배포 혼합 가중치 (exp021 쪽)
U = 1.0 - W
S_BLEND = 1114.611684697336  # 배포 혼합 LB

B11 = A1 = S1                # 가정 A1
# K = B11 - 2*B12 + B22 + M2 에 M2 = 2*A2 - B22 - S2 를 대입하면 B22 가 소거된다:
#   K = B11 - 2*B12 + 2*A2 - S2   =>   B12 = A2 - (S2 + K - B11)/2
CONST = (S2 + K - B11) / 2.0  # B12 = A2 - CONST


def blend_check():
    """u*S1 + w*S2 + u*w*K 가 실제 LB 와 맞는지 확인 (기하 일관성 점검)."""
    return U * S1 + W * S2 + U * W * K


def evaluate(a2, b22):
    m2 = 2.0 * a2 - b22 - S2
    b12 = a2 - CONST
    if m2 < -1e-9 or b22 <= 0.0:
        return None
    if abs(b12) > np.sqrt(B11 * b22) + 1e-9:      # 코시-슈바르츠
        return None
    if a2 * a2 > b22 * 100000.0 + 1e-6:           # shape <= 1e5 (corr^2 <= 1)
        return None
    a_w = U * A1 + W * a2
    b_w = U * U * B11 + 2.0 * U * W * b12 + W * W * b22
    if b_w <= 0.0:
        return None
    m_w = W * W * m2                              # d1 = 0 이므로 평균편차는 w*d2
    s_w = 2.0 * a_w - b_w - m_w
    if abs(s_w - S_BLEND) > 1e-6:                 # 혼합 항등식과 모순이면 버림
        return None
    return {"A2": a2, "B22": b22, "M2": m2, "B12": b12,
            "beta": a_w / b_w, "gain": (a_w - b_w) ** 2 / b_w + m_w,
            "shape2": a2 * a2 / b22, "S_blend": s_w}


def main():
    print(f"혼합 항등식 재확인:  u*S1 + w*S2 + u*w*K = {blend_check():.9f}  "
          f"(LB {S_BLEND:.9f}, 차이 {blend_check() - S_BLEND:+.2e})")
    print(f"B11 = A1 = {B11:.6f},  B12 = A2 - {CONST:.6f}\n")

    rows = []
    for a2 in np.arange(600.0, 2600.0, 2.0):
        for b22 in np.arange(200.0, 4000.0, 2.0):
            got = evaluate(a2, b22)
            if got is not None:
                rows.append(got)
    if not rows:
        print("실현 가능한 (A2, B22) 조합이 없습니다.")
        return

    gains = np.array([r["gain"] for r in rows])
    betas = np.array([r["beta"] for r in rows])
    print(f"실현 가능 조합 수: {len(rows)}")
    print(f"  아핀 이득 G   : 최소 {gains.min():+8.3f} / 중앙 {np.median(gains):+8.3f} "
          f"/ 최대 {gains.max():+8.3f}")
    print(f"  최적 기울기 β : 최소 {betas.min():8.5f} / 중앙 {np.median(betas):8.5f} "
          f"/ 최대 {betas.max():8.5f}")
    print(f"  β > 1 인 비율 : {100.0 * np.mean(betas > 1.0):.1f}%")
    print(f"  G > 0.5 비율  : {100.0 * np.mean(gains > 0.5):.1f}%\n")

    order = np.argsort(gains)
    for tag, idx in (("최소이득", order[0]), ("중앙값", order[len(order) // 2]),
                     ("최대이득", order[-1])):
        r = rows[idx]
        print(f"  [{tag}] A2={r['A2']:7.1f} B22={r['B22']:7.1f} M2={r['M2']:7.2f} "
              f"B12={r['B12']:7.1f} shape(p2)={r['shape2']:8.2f} "
              f"β={r['beta']:.5f} G={r['gain']:+.3f}")

    # p2 의 shape 점수가 현실적인 범위(단독 LB 1043.6 ~ 그 +40) 로 제한되는 경우
    tight = [r for r in rows if S2 <= r["shape2"] <= S2 + 40.0]
    if tight:
        g = np.array([r["gain"] for r in tight])
        b = np.array([r["beta"] for r in tight])
        print(f"\n  [shape(p2) ∈ {S2:.1f}~{S2 + 40:.1f} 로 제한: {len(tight)}조합]")
        print(f"    G  최소 {g.min():+.3f} / 중앙 {np.median(g):+.3f} / 최대 {g.max():+.3f}")
        print(f"    β  최소 {b.min():.5f} / 중앙 {np.median(b):.5f} / 최대 {b.max():.5f}")


if __name__ == "__main__":
    main()
