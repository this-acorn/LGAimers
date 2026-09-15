"""
[04] 피처 그룹 ablation — 어느 피처가 +84.5점을 만들었나?

실행:  PYTHONIOENCODING=utf-8 py -3.12 -u exp/04_ablation.py

두 방향으로 잰다 (같은 그룹이라도 두 값이 다를 수 있고, 그 차이가 정보다):

  ① LOO (빼기)   : 전체 65개에서 그 그룹만 제거 → 얼마나 떨어지나
                   = "이거 없으면 곤란한가?"  (다른 피처로 대체 가능하면 0에 가까움)

  ② ADD (더하기) : 기본 47개에 그 그룹만 추가 → 얼마나 오르나
                   = "이거 하나로 뭘 벌었나?"  (중복 정보라도 값이 나옴)

  LOO는 낮고 ADD는 높다  → 다른 피처와 정보가 겹침 (혼자선 좋은데 대체 가능)
  LOO도 높고 ADD도 높다  → 고유한 정보. ★ 여기를 더 파야 한다
  둘 다 0에 가깝다       → 쓸모없음. 버려도 됨

총 14회 학습 (전체1 + 기본1 + LOO6 + ADD6). 약 15~20분.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "exp")
from common import (log, tick, decompose, report, load_train, add_features,
                    fit_encoder, to_matrix, FEATURE_GROUPS, ALL_ENG,
                    HGB_FAST, LB_BASELINE, LB_TOP)

log("train.csv 로딩 중...")
df = load_train()
tick(f"로딩 완료 {df.shape}")

BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FULL = BASE + ALL_ENG

tr = df[df["season"] <= 2023].reset_index(drop=True)
va = df[df["season"] == 2024].reset_index(drop=True)
y_tr = tr["control_success"].to_numpy()
y_va = va["control_success"].to_numpy()
PRIOR = float(y_tr.mean())          # 스무딩 사전확률 = 학습기간 평균

tr = add_features(tr, PRIOR)
va = add_features(va, PRIOR)
tick(f"피처 생성 완료 (기본 {len(BASE)} → 전체 {len(FULL)})")
log(f"\n학습 2019~2023 {len(tr):,}행 (r={PRIOR:.6f})  /  검증 2024 {len(va):,}행 "
    f"(r={y_va.mean():.6f})\n")


def run(name, feats):
    """주어진 피처 목록으로 학습 → 2024 점수 반환."""
    enc = fit_encoder(tr, feats)
    Xtr = to_matrix(tr, feats, enc)
    Xva = to_matrix(va, feats, enc)
    m = HistGradientBoostingClassifier(**HGB_FAST).fit(Xtr, y_tr)
    p = m.predict_proba(Xva)[:, 1]
    return decompose(p, y_va)


# =====================================================================
# 기준점 2개
# =====================================================================
log("=" * 88)
log("기준점")
log("=" * 88)

def show(label, z):
    log(f"  {label:28s} 총점 {z['총점']:8.1f} = 변별력 {z['변별력']:7.1f} "
        f"- 벌점 {z['중심벌점']:6.1f}")


tick("전체 65피처 학습 중 (1/14)...")
z_full = run("full", FULL)
show(f"전체 {len(FULL)}피처", z_full)

tick("기본 47피처 학습 중 (2/14)...")
z_base = run("base", BASE)
show(f"기본 {len(BASE)}피처", z_base)
log(f"\n  ▶ 엔지니어링 18개 전체 효과 = 변별력 {z_full['변별력']-z_base['변별력']:+.1f}점")


# =====================================================================
# ① LOO — 전체에서 그룹 하나씩 제거
# =====================================================================
log("\n" + "=" * 88)
log("① LOO (빼기) — 전체 65개에서 그 그룹만 빼면 얼마나 떨어지나")
log("=" * 88)

loo = {}
for i, (gname, gcols) in enumerate(FEATURE_GROUPS.items(), start=3):
    tick(f"{gname} 제거 학습 중 ({i}/14)...")
    feats = [c for c in FULL if c not in gcols]
    z = run(gname, feats)
    drop = z_full["변별력"] - z["변별력"]        # 양수 = 빼니까 나빠짐 = 중요함
    loo[gname] = drop
    log(f"  {gname:16s} ({len(gcols)}개) 제거 → 변별력 {z['변별력']:7.1f}  "
        f"손실 {drop:+7.1f}")


# =====================================================================
# ② ADD — 기본에 그룹 하나씩 추가
# =====================================================================
log("\n" + "=" * 88)
log("② ADD (더하기) — 기본 47개에 그 그룹만 넣으면 얼마나 오르나")
log("=" * 88)

add = {}
for i, (gname, gcols) in enumerate(FEATURE_GROUPS.items(), start=9):
    tick(f"{gname} 단독 추가 학습 중 ({i}/14)...")
    feats = BASE + gcols
    z = run(gname, feats)
    gain = z["변별력"] - z_base["변별력"]
    add[gname] = gain
    log(f"  {gname:16s} ({len(gcols)}개) 추가 → 변별력 {z['변별력']:7.1f}  "
        f"이득 {gain:+7.1f}")


# =====================================================================
# 종합 판정
# =====================================================================
log("\n" + "=" * 88)
log("종합 — 어디를 더 파야 하나")
log("=" * 88)
log(f"  {'그룹':16s} {'개수':>4s} {'LOO손실':>9s} {'ADD이득':>9s}   판정")
log("  " + "-" * 76)

rows = []
for gname, gcols in FEATURE_GROUPS.items():
    l, a = loo[gname], add[gname]
    if l >= 15 and a >= 15:
        verdict = "★ 고유 정보. 더 팔 것"
    elif a >= 15 > l:
        verdict = "○ 정보는 있으나 다른 피처와 중복"
    elif l >= 15 > a:
        verdict = "○ 조합에서만 작동 (상호작용)"
    else:
        verdict = "· 효과 미미. 버려도 됨"
    rows.append((gname, len(gcols), l, a, verdict))

for gname, n, l, a, v in sorted(rows, key=lambda x: -(x[2] + x[3])):
    log(f"  {gname:16s} {n:4d} {l:+9.1f} {a:+9.1f}   {v}")

log(f"\n  기준: 전체 65피처 변별력 {z_full['변별력']:.1f} / 기본 47피처 {z_base['변별력']:.1f}")
log(f"  참고: 리더보드 베이스라인 {LB_BASELINE:.1f}, 1위 {LB_TOP:.1f}")
log("=" * 88)
