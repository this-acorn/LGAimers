"""
[12] 나무 모양(max_leaf_nodes) 실험

실행:  PYTHONIOENCODING=utf-8 py -3.12 -u exp/12_leaves.py

exp/09에서 확인된 것:
  나무 개수(max_iter)를 120 → 400 으로 3배 늘려도 변별력 +6.9 (노이즈 범위).
  용량 축은 죽었다. 이유: 시드 앙상블이 이미 용량 부족을 메꾸고 있었다.
    나무 120개 → 개별 모델이 제각각(단일 폭 29) → 평균 이득 큼
    나무 400개 → 개별 모델이 비슷(단일 폭 17)   → 평균 이득 적음
    → 결국 앙상블 점수는 같은 곳에 수렴

  ★ 그래서 이번엔 빠른 쪽(max_iter=120)을 기준으로 쓴다. 성능은 같고 5배 빠르다.

이번에 보는 축 — max_leaf_nodes:
  용량이 아니라 '한 그루가 잡아낼 수 있는 상호작용의 복잡도'를 정한다.
    잎 15개   단순 규칙만. 과적합 덜함
    잎 31개   현재 (기준)
    잎 63개   더 깊은 조합
    잎 127개  아주 복잡한 조합까지

  왜 유망한가: ablation에서 G2(매치업 피처)가 +107점으로 압도적이었다.
  이 데이터는 '조합'이 중요하다는 뜻이므로, 잎을 늘리면 트리가 더 깊은
  상호작용을 직접 잡아낼 수 있다.

판정 기준: 4시드 앙상블 기준 유의 임계 약 15점 (단일 σ=15.3 → 4시드 σ≈7.6, 2σ≈15)
"""

import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "exp")
from common import (log, tick, decompose, load_train, add_features,
                    fit_encoder, to_matrix, ALL_ENG, LB_TOP)

SEEDS = [42, 7, 123, 2024]
THR = 15.0                 # 4시드 앙상블 유의 임계
CUR_LOCAL_8SEED = 708.7    # 제출 모델(8시드) 로컬 총점
CUR_LB = 830.32            # 그 모델의 실제 리더보드
TRANSFER = 0.958

# exp/09에서 용량 축이 죽었으므로 빠른 설정을 기준으로 쓴다
BASE_PARAMS = dict(max_iter=120, learning_rate=0.08,
                   min_samples_leaf=200, l2_regularization=10.0,
                   early_stopping=False)

log("train.csv 로딩 중...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS = BASE + ALL_ENG

tr = df[df["season"] <= 2023].reset_index(drop=True)
va = df[df["season"] == 2024].reset_index(drop=True)
y_tr, y_va = tr["control_success"].to_numpy(), va["control_success"].to_numpy()
PRIOR = float(y_tr.mean())
tr, va = add_features(tr, PRIOR), add_features(va, PRIOR)
enc = fit_encoder(tr, FEATS)
Xtr, Xva = to_matrix(tr, FEATS, enc), to_matrix(va, FEATS, enc)
tick(f"준비 완료 — {len(FEATS)}피처, 학습 {len(tr):,} / 검증 {len(va):,}")
log(f"  유의 임계 {THR:.0f}점 (이보다 작으면 노이즈)\n")


def run(name, **override):
    prm = {**BASE_PARAMS, **override}
    t0 = time.time()
    acc = np.zeros(len(va))
    singles = []
    for s in SEEDS:
        m = HistGradientBoostingClassifier(**{**prm, "random_state": s}).fit(Xtr, y_tr)
        p = m.predict_proba(Xva)[:, 1]
        acc += p
        singles.append(decompose(p, y_va)["변별력"])
    z = decompose(acc / len(SEEDS), y_va)
    z["단일폭"] = max(singles) - min(singles)
    log(f"  {name:20s} 총점 {z['총점']:7.1f}  변별력 {z['변별력']:7.1f}  "
        f"(단일 {min(singles):.0f}~{max(singles):.0f}, 폭 {z['단일폭']:.0f})  "
        f"[{time.time()-t0:.0f}s]")
    return z


log("=" * 88)
log("max_leaf_nodes 비교 (max_iter=120, lr=0.08 고정)")
log("=" * 88)

LEAVES = [15, 31, 63, 127]
res = {}
for i, L in enumerate(LEAVES, 1):
    tag = "  ← 현재" if L == 31 else ""
    tick(f"{i}/{len(LEAVES)} 잎 {L}개 학습 중...{tag}")
    res[L] = run(f"잎 {L}개" + tag, max_leaf_nodes=L)


# =====================================================================
log("\n" + "=" * 88)
log("판정")
log("=" * 88)

ref = res[31]["변별력"]
log(f"  {'잎 개수':>8s} {'총점':>8s} {'변별력':>8s} {'31개대비':>9s} {'단일폭':>7s}")
log("  " + "-" * 48)
for L in LEAVES:
    z = res[L]
    log(f"  {L:8d} {z['총점']:8.1f} {z['변별력']:8.1f} "
        f"{z['변별력']-ref:+9.1f} {z['단일폭']:7.0f}")

best = max(res, key=lambda k: res[k]["변별력"])
gain = res[best]["변별력"] - ref
log(f"\n  ▶ 최고: 잎 {best}개  (현재 31개 대비 {gain:+.1f}점)")

if gain >= THR:
    proj_local = CUR_LOCAL_8SEED + gain
    proj_lb = CUR_LB + gain * TRANSFER
    log(f"  ✅ 유의미 — 채택할 것")
    log(f"     예상 로컬(8시드)  {CUR_LOCAL_8SEED:.1f} → {proj_local:.1f}")
    log(f"     예상 리더보드     {CUR_LB:.1f} → {proj_lb:.1f}")
    log(f"     1위 {LB_TOP:.1f} 까지 남는 격차 {LB_TOP-proj_lb:.1f}")
    if best == max(LEAVES):
        log(f"     ⚠️ 최댓값이 이겼다 — 더 큰 값(255)도 시험해볼 가치 있음")
else:
    log(f"  ❌ 노이즈 범위 ({THR:.0f}점 미만) — 나무 모양 축도 죽었다")
    log(f"     용량(exp/09)에 이어 모양까지 무효라면,")
    log(f"     하이퍼파라미터 축 전체가 이미 포화 상태일 가능성이 크다")
    log(f"     → 남은 길: 피처(조합형) / 다른 알고리즘 / 앙상블 다양성")
log("=" * 88)
