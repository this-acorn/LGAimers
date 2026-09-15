"""
[11] 목적함수 재측정 + cold-start 정량화 + 결측 처리 방식 비교

실행:  PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 py -3.12 -u exp/11_objective_coldstart.py

배경 — 왜 '재'측정인가:
  exp/03에서 분류(695.2) vs 회귀(685.7)를 재고 "분류가 낫다"고 결론냈었다.
  그런데 exp/06에서 노이즈 σ=15.3이 밝혀졌다. 9.5점 차이는 완전히 노이즈 안이었다.
  → 그 결론은 근거가 없었다. 4시드 앙상블로 제대로 다시 잰다.

A. 목적함수      log loss(분류) vs MSE(회귀)   ← Brier=MSE이니 회귀가 지표에 직접 부합?
B. cold-start    asof 표본수 구간별로 점수를 쪼개서, 이게 몇 점짜리 문제인지 확정
C. 결측 처리     NaN 그대로 두기 vs 전체평균 채우기 vs 스무딩값으로 대체

C가 특히 중요하다. GBDT는 NaN을 네이티브로 처리하고 '비어있음' 자체를 신호로 쓴다.
평균으로 채우면 그 신호가 지워질 수 있다 — 추정이 아니라 실측으로 확인한다.
"""

import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

sys.path.insert(0, "exp")
from common import (log, tick, decompose, load_train, add_features,
                    fit_encoder, to_matrix, ALL_ENG, HGB_FAST, LB_TOP)

SEEDS = [42, 7, 123, 2024]
NOISE_SD = 15.3
THR = NOISE_SD          # 4시드 앙상블 기준 대략적 유의 임계 (약 15점)

log("train.csv 로딩 중...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS = BASE + ALL_ENG

tr_raw = df[df["season"] <= 2023].reset_index(drop=True)
va_raw = df[df["season"] == 2024].reset_index(drop=True)
y_tr = tr_raw["control_success"].to_numpy()
y_va = va_raw["control_success"].to_numpy()
PRIOR = float(y_tr.mean())

tr = add_features(tr_raw, PRIOR)
va = add_features(va_raw, PRIOR)
enc = fit_encoder(tr, FEATS)
Xtr, Xva = to_matrix(tr, FEATS, enc), to_matrix(va, FEATS, enc)
tick(f"준비 완료 — 학습 {len(tr):,} / 검증 {len(va):,}")
log(f"  유의 임계 약 {THR:.0f}점 (이보다 작으면 노이즈)\n")


def ensemble(Xa, Xb, kind="clf", **override):
    """4시드 학습 → 앙상블 예측 반환"""
    prm = dict(HGB_FAST, **override)
    acc = np.zeros(Xb.shape[0])
    for s in SEEDS:
        # HGB_FAST 에 이미 random_state 가 있으므로 딕셔너리 병합으로 덮어쓴다
        # (키워드로 또 넘기면 "multiple values for keyword argument" 에러)
        p = {**prm, "random_state": s}
        if kind == "clf":
            m = HistGradientBoostingClassifier(**p).fit(Xa, y_tr)
            acc += m.predict_proba(Xb)[:, 1]
        else:
            m = HistGradientBoostingRegressor(loss="squared_error", **p).fit(Xa, y_tr)
            acc += np.clip(m.predict(Xb), 0.0, 1.0)
    return acc / len(SEEDS)


# =====================================================================
# A. 목적함수
# =====================================================================
log("=" * 90)
log("A. 목적함수 — log loss(분류) vs MSE(회귀)   ※ exp/03 결론 재검증")
log("=" * 90)

t0 = time.time()
tick("분류(log loss) 4시드 학습 중...")
p_clf = ensemble(Xtr, Xva, "clf")
z_clf = decompose(p_clf, y_va)
log(f"  {'분류 (log loss)':24s} 총점 {z_clf['총점']:7.1f}  변별력 {z_clf['변별력']:7.1f}  "
    f"벌점 {z_clf['중심벌점']:5.1f}  [{time.time()-t0:.0f}s]")

t0 = time.time()
tick("회귀(MSE) 4시드 학습 중...")
p_reg = ensemble(Xtr, Xva, "reg")
z_reg = decompose(p_reg, y_va)
log(f"  {'회귀 (MSE)':24s} 총점 {z_reg['총점']:7.1f}  변별력 {z_reg['변별력']:7.1f}  "
    f"벌점 {z_reg['중심벌점']:5.1f}  [{time.time()-t0:.0f}s]")

d = z_reg["총점"] - z_clf["총점"]
log(f"\n  ▶ 회귀 − 분류 = {d:+.1f}점  "
    f"→ {'회귀 승' if d >= THR else ('분류 승' if d <= -THR else '차이 없음 (노이즈)')}")
log(f"     (exp/03 단일학습 결과는 -9.5였고, 그건 노이즈였다)")

# 덤: 둘을 평균내면? 서로 다르게 틀리면 앙상블 이득이 난다
z_mix = decompose((p_clf + p_reg) / 2, y_va)
log(f"  ▶ 분류·회귀 평균: 총점 {z_mix['총점']:7.1f}  "
    f"(분류 대비 {z_mix['총점']-z_clf['총점']:+.1f})")


# =====================================================================
# A-2. 신뢰도 곡선 — exp/10에서 최상위 구간이 무너진 게 실전에도 있나?
# =====================================================================
log("\n" + "=" * 90)
log("A-2. 신뢰도 곡선 — 실전 설정(2019~2023 학습)에서도 상위 구간이 부풀려지나?")
log("=" * 90)
log("  exp/10(2019~2022 학습, 2년 건너뜀)에서는 최상위 10%가")
log("  예측 0.69 vs 실제 0.466 으로 무너져 -2000점을 날렸다.")
log("  2023을 포함한 실전 설정에서도 같은 현상이 있는지 확인한다.\n")

q = pd.qcut(p_clf, 10, labels=False, duplicates="drop")
log(f"  {'구간':>4s} {'행수':>9s} {'예측평균':>10s} {'실제평균':>10s} {'차이':>9s} "
    f"{'점수영향':>10s}")
log("  " + "-" * 58)
ece = 0.0
for b in range(q.max() + 1):
    mb = q == b
    pm, ym = p_clf[mb].mean(), y_va[mb].mean()
    ece += mb.sum() * abs(pm - ym) / len(p_clf)
    # 이 구간을 그 구간 실제평균으로 바꿨을 때 전체 점수 변화 = 이 구간의 보정 여지
    pp = p_clf.copy()
    pp[mb] = ym
    r_ = y_va.mean()
    gain = 100000 * (np.mean((p_clf - y_va) ** 2) - np.mean((pp - y_va) ** 2)) / (r_ * (1 - r_))
    flag = "  ←" if abs(pm - ym) > 0.02 else ""
    log(f"  {b:4d} {mb.sum():9,d} {pm:10.4f} {ym:10.4f} {pm-ym:+9.4f} {gain:+10.1f}{flag}")
log(f"  → ECE(가중평균 절대오차) = {ece:.5f}    (exp/10 극단설정은 0.03490)")
log(f"  → '점수영향' 합계가 크면 보정 여지가 있다는 뜻")


# =====================================================================
# B. cold-start 정량화 — 몇 점짜리 문제인가
# =====================================================================
log("\n" + "=" * 90)
log("B. cold-start — asof 표본수 구간별 분해  (분류 모델 예측 사용)")
log("=" * 90)

n_p = va_raw["asof_pitcher_n"].fillna(0).to_numpy()
bins = [-1, 0, 10, 50, 200, 1000, 10**9]
labels = ["n=0", "1-10", "11-50", "51-200", "201-1000", "1000+"]
grp = pd.cut(n_p, bins=bins, labels=labels)

r_all = y_va.mean()
base_all = r_all * (1 - r_all)
brier_all = float(np.mean((p_clf - y_va) ** 2))

log(f"  {'구간':>10s} {'행수':>9s} {'비율':>7s} {'실제r':>8s} {'예측평균':>9s} "
    f"{'Brier':>9s} {'완벽시 이득':>11s}")
log("  " + "-" * 72)
for lb in labels:
    m = np.asarray(grp == lb)   # pd.cut(ndarray)는 Categorical → 비교 결과가 ndarray
    if m.sum() == 0:
        continue
    # 이 구간을 '정답 그대로' 맞췄다면 전체 점수가 얼마나 오르나 = 잠재 최대 이득
    p_perfect = p_clf.copy()
    p_perfect[m] = y_va[m]
    gain = (100000 * (1 - float(np.mean((p_perfect - y_va) ** 2)) / base_all)
            - 100000 * (1 - brier_all / base_all))
    log(f"  {lb:>10s} {m.sum():9,d} {m.mean()*100:6.2f}% {y_va[m].mean():8.4f} "
        f"{p_clf[m].mean():9.4f} {float(np.mean((p_clf[m]-y_va[m])**2)):9.5f} "
        f"{gain:+11.1f}")

m0 = np.asarray(grp == "n=0")
log(f"\n  ▶ 신인(n=0) 비중 {m0.mean()*100:.2f}% · 완벽히 맞춰도 전체 +{'?':>0}")
log(f"     → 위 '완벽시 이득' 열이 각 구간의 이론상 천장이다.")
log(f"       n=0 구간 천장이 작으면 cold-start는 애초에 큰 전선이 아니다.")


# =====================================================================
# C. 결측 처리 방식
# =====================================================================
log("\n" + "=" * 90)
log("C. 통산 rate를 스무딩값으로 '대체'하면?  ← 구간9 과신 문제를 직접 공략")
log("=" * 90)
log("  ※ '전체평균으로 채우기'는 하지 않는다.")
log("    결측을 평균으로 채우면 신인이 '평균적인 투수'로 위장되어,")
log("    exp/10에서 본 '통산 성적 과신' 오류를 오히려 더 만든다.")
log("    GBDT는 NaN을 네이티브로 처리하며 '비어있음'을 신호로 쓰고 있다.\n")
log("  대신 반대 방향을 시험한다 — 극단적인 통산 rate를 평균 쪽으로 당기기:")
log("    asof_pitcher_success_rate  →  (rate×n + prior×200)/(n+200) 로 교체")
log("    표본이 적을수록 강하게 당겨지므로, 과신이 줄어들 것으로 기대\n")

# 스무딩값으로 '대체' (원본 rate 자리를 f_smooth_p 값으로 갈아끼움)
tr_sm, va_sm = tr.copy(), va.copy()
tr_sm["asof_pitcher_success_rate"] = tr_sm["f_smooth_p"]
va_sm["asof_pitcher_success_rate"] = va_sm["f_smooth_p"]
tr_sm["asof_batter_success_rate"] = tr_sm["f_smooth_b"]
va_sm["asof_batter_success_rate"] = va_sm["f_smooth_b"]

tick("스무딩값 대체 학습 중...")
z_sm = decompose(ensemble(to_matrix(tr_sm, FEATS, enc),
                          to_matrix(va_sm, FEATS, enc), "clf"), y_va)

log(f"\n  {'방식':28s} {'총점':>8s} {'변별력':>8s} {'기준대비':>9s}")
log("  " + "-" * 58)
for nm, z in [("기준: NaN 그대로 (현재)", z_clf),
              ("스무딩값으로 대체", z_sm)]:
    log(f"  {nm:28s} {z['총점']:8.1f} {z['변별력']:8.1f} "
        f"{z['총점']-z_clf['총점']:+9.1f}")

best_c = max([("NaN 유지", z_clf), ("스무딩대체", z_sm)], key=lambda x: x[1]["총점"])
log(f"\n  ▶ 최고: {best_c[0]}")
if abs(z_sm["총점"] - z_clf["총점"]) < THR:
    log(f"     차이가 {THR:.0f}점 미만 → 영향 없음. 현재 방식(NaN 유지) 유지")
else:
    log(f"     유의미한 차이 — 최고 방식을 채택할 것")

log("\n" + "=" * 90)
log("요약")
log("=" * 90)
log(f"  A 목적함수   회귀−분류 {d:+.1f}  →  "
    f"{'회귀 채택' if d >= THR else ('분류 유지' if d <= -THR else '무차별 (분류 유지)')}")
log(f"    분류·회귀 평균 앙상블 {z_mix['총점']-z_clf['총점']:+.1f}")
log(f"  C 스무딩대체 {z_sm['총점']-z_clf['총점']:+.1f}  →  최고 = {best_c[0]}")
log(f"  참고: 1위 {LB_TOP:.1f} / 현재 제출 리더보드 830.32")
log("=" * 90)
