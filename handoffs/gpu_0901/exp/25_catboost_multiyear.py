# -*- coding: utf-8 -*-
"""
[25] cb_num 다연도 검증 — CatBoost '모델 효과'만 4개 연도 × 4시드로 확증

배경 (exp/23 프로브, 검증 2023 · 2시드):
    hgb     -1396.2
    cb_num  -1297.8    모델 효과      =  +98.4
    cb_cat  -1458.7    선수 CTR 효과  = -160.9   → 중단
  한 팔로 돌렸다면 합계 -62.5만 보고 CatBoost 축 전체를 폐기했을 것이다. 분해가 살렸다.

★ 프로브 해석에서 바로잡은 것 3가지:
  1) cb_cat -160.9 는 [F]형 유해의 '확증'이 아니다. 2023 한 해 · 2시드이고
     CatBoost 시드 분산이 크다(cb_num 시드간 130점). 강한 단서일 뿐이며,
     추가 검증 비용 대비 기대가 낮아 '실전적으로' 폐기하는 것뿐이다.
  2) +98.4 를 LB 점수로 환산하면 안 된다. 로컬↔LB 전이 공식은 이미 폐기됐다
     (데이터포인트 3개 중 하나가 -40 이탈). 말할 수 있는 것은 "2023 로컬에서 +98.4"
     까지이고 2025 리더보드 변화는 알 수 없다.
  3) cb_num 도 완전한 [S]형은 아니다. 동결 선수통계는 없지만 모델이 과거 연도의
     관계를 학습해 고정하므로 연도 변화에 노출된다. hand delta/선수CTR보다
     위험이 '낮다'는 정도지 없는 게 아니다.

★ exp/22가 예측을 버려서 exp/24에서 재학습해야 했다. 같은 실수를 반복하지 않는다 —
  모든 시드별 예측을 lab/25_preds.npz 에 저장한다.

판정 (단독 점수보다 혼합을 우선해서 본다):
  · 4년 중 3년 이상 개선
  · 2024에서 큰 역행 없음
  · 2021~2023으로 고른 **하나의** 혼합비율이 2024에서도 유지
  · 특정 시드 하나가 전체 이득을 만든 것이 아님 (leave-one-seed-out)

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/25_catboost_multiyear.py
소요 예상: CatBoost 16회(~450s each) + HGB 16회(~40s each) ≈ 2.2시간
"""

import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import (log, tick, raw_score, load_train, add_features,
                    fit_encoder, to_matrix, ALL_ENG, HGB_FAST, CAT)

YEARS = [2021, 2022, 2023, 2024]
SEEDS = [42, 7, 123, 2024]
PICK_YEARS, CONFIRM_YEAR = [2021, 2022, 2023], 2024
W_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# ★ 프로브와 동일 설정 유지 — 바꾸면 +98.4와 비교가 깨진다
CB_PARAMS = dict(iterations=500, learning_rate=0.08, depth=6,
                 l2_leaf_reg=10.0, verbose=False, thread_count=6,
                 allow_writing_files=False)
NPZ_PATH = "lab/25_preds.npz"

log("train.csv 로딩 중...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS65 = BASE + ALL_ENG
NUM_FEATS = [c for c in FEATS65 if c not in CAT]
tick(f"로드 {df.shape} — CatBoost {len(YEARS)*len(SEEDS)}회 + HGB {len(YEARS)*len(SEEDS)}회")

store, dump = {}, {}

for Y in YEARS:
    tr_raw = df[df["season"] <= Y - 1].reset_index(drop=True)
    va_raw = df[df["season"] == Y].reset_index(drop=True)
    y_tr = tr_raw["control_success"].to_numpy()
    y_va = va_raw["control_success"].to_numpy()
    prior = float(y_tr.mean())
    tr, va = add_features(tr_raw, prior), add_features(va_raw, prior)
    log(f"\n{'='*88}")
    log(f"검증연도 {Y}  학습 {len(tr):,} / 검증 {len(va):,}  (실제 r={y_va.mean():.4f})")
    log("=" * 88)

    # ---- HGB (같은 시드로 짝지어야 하므로 여기서 다시 학습) ----
    enc = fit_encoder(tr, FEATS65)
    Xtr, Xva = to_matrix(tr, FEATS65, enc), to_matrix(va, FEATS65, enc)
    hgb_p = []
    for s in SEEDS:
        t0 = time.time()
        m = HistGradientBoostingClassifier(**{**HGB_FAST, "random_state": s}).fit(Xtr, y_tr)
        p = m.predict_proba(Xva)[:, 1]
        hgb_p.append(p)
        dump[f"{Y}_hgb_{s}"] = p.astype("float32")
        tick(f"{Y} hgb seed={s}  {raw_score(p, y_va):8.1f}  ({time.time()-t0:.0f}s)")
    del Xtr, Xva

    # ---- cb_num: 저카디널 범주형만. 선수 ID는 수치로 유지(데뷔순서 신호 r=0.704) ----
    def frame(d):
        out = d[NUM_FEATS].copy()
        for c in CAT:
            out[c] = d[c].astype(str)
        return out

    ptr = Pool(frame(tr), y_tr, cat_features=list(CAT))
    pva = Pool(frame(va), cat_features=list(CAT))
    cb_p = []
    for s in SEEDS:
        t0 = time.time()
        m = CatBoostClassifier(**CB_PARAMS, random_seed=s).fit(ptr)
        p = m.predict_proba(pva)[:, 1]
        cb_p.append(p)
        dump[f"{Y}_cbnum_{s}"] = p.astype("float32")
        tick(f"{Y} cb_num seed={s}  {raw_score(p, y_va):8.1f}  ({time.time()-t0:.0f}s)")
    del ptr, pva

    dump[f"{Y}_y"] = y_va.astype("int8")
    store[Y] = {"y": y_va, "hgb": hgb_p, "cb": cb_p,
                "hgb_ens": np.mean(hgb_p, axis=0), "cb_ens": np.mean(cb_p, axis=0)}
    log(f"    → 앙상블  hgb {raw_score(store[Y]['hgb_ens'], y_va):8.1f}   "
        f"cb_num {raw_score(store[Y]['cb_ens'], y_va):8.1f}")
    del tr, va, tr_raw, va_raw

np.savez_compressed(NPZ_PATH, **dump)
tick(f"예측 저장 완료 → {NPZ_PATH}  ({len(dump)}개 배열)")

# =====================================================================
log("\n" + "=" * 88)
log("A. 시드 짝지음 이득 (같은 시드끼리 cb_num - hgb)")
log("=" * 88)
log(f"  {'연도':>6s} " + " ".join(f"{'seed'+str(s):>10s}" for s in SEEDS)
    + f" {'평균':>9s} {'앙상블':>9s}")
paired, ens_gain = {}, {}
for Y in YEARS:
    d = store[Y]
    pg = [raw_score(c, d["y"]) - raw_score(h, d["y"])
          for c, h in zip(d["cb"], d["hgb"])]
    eg = raw_score(d["cb_ens"], d["y"]) - raw_score(d["hgb_ens"], d["y"])
    paired[Y], ens_gain[Y] = pg, eg
    log(f"  {Y:>6d} " + " ".join(f"{g:>+10.1f}" for g in pg)
        + f" {np.mean(pg):>+9.1f} {eg:>+9.1f}")
a = np.array([ens_gain[Y] for Y in YEARS])
log(f"  {'─'*70}")
log(f"  앙상블 이득: 평균 {a.mean():+.1f} / 최악 {a.min():+.1f} / 양수 {(a>0).sum()}/4")

# =====================================================================
log("\n" + "=" * 88)
log("B. leave-one-seed-out — 특정 시드 하나가 이득을 만들었나")
log("=" * 88)
log(f"  {'연도':>6s} {'전체4시드':>10s} " + " ".join(f"{'-'+str(s):>10s}" for s in SEEDS))
for Y in YEARS:
    d = store[Y]
    row = []
    for i in range(len(SEEDS)):
        keep = [j for j in range(len(SEEDS)) if j != i]
        g = (raw_score(np.mean([d["cb"][j] for j in keep], axis=0), d["y"])
             - raw_score(np.mean([d["hgb"][j] for j in keep], axis=0), d["y"]))
        row.append(g)
    log(f"  {Y:>6d} {ens_gain[Y]:>+10.1f} " + " ".join(f"{g:>+10.1f}" for g in row))
log("\n  해석: 한 시드를 뺐을 때 이득이 급감하면 그 시드가 이득을 만든 것 → 신뢰 낮음")

# =====================================================================
log("\n" + "=" * 88)
log("C. 혼합 곡선 — 비율 선택은 2021~2023, 확인은 2024")
log("=" * 88)
log(f"  {'w':>5s} " + " ".join(f"{Y:>10d}" for Y in YEARS) + f" {'선택평균':>10s}")
best_w, best_v = 0.0, -1e9
curve = {}
for w in W_GRID:
    gains = []
    for Y in YEARS:
        d = store[Y]
        p = (1 - w) * d["hgb_ens"] + w * d["cb_ens"]
        gains.append(raw_score(p, d["y"]) - raw_score(d["hgb_ens"], d["y"]))
    curve[w] = gains
    pick = float(np.mean([gains[YEARS.index(Y)] for Y in PICK_YEARS]))
    log(f"  {w:>5.1f} " + " ".join(f"{g:>+10.1f}" for g in gains) + f" {pick:>+10.1f}")
    if pick > best_v:
        best_w, best_v = w, pick

conf_gain = curve[best_w][YEARS.index(CONFIRM_YEAR)]
log(f"\n  선택된 w = {best_w:.1f}  (2021~2023 평균 {best_v:+.1f})")
log(f"  → {CONFIRM_YEAR} 확인 이득 {conf_gain:+.1f}   "
    f"(cb_num 단독 w=1.0은 {curve[1.0][YEARS.index(CONFIRM_YEAR)]:+.1f})")

# =====================================================================
log("\n" + "=" * 88)
log("판정")
log("=" * 88)
g_at_w = curve[best_w]
c1 = sum(1 for g in g_at_w if g > 0) >= 3
c2 = g_at_w[YEARS.index(CONFIRM_YEAR)] > -15
c3 = conf_gain > 0
log(f"  [{'O' if c1 else 'X'}] 4년 중 3년 이상 개선 (w={best_w:.1f}): "
    f"{sum(1 for g in g_at_w if g>0)}/4")
log(f"  [{'O' if c2 else 'X'}] 2024 큰 역행 없음: {g_at_w[YEARS.index(CONFIRM_YEAR)]:+.1f}")
log(f"  [{'O' if c3 else 'X'}] 선택비율이 확인연도에서 유지: {conf_gain:+.1f}")
log(f"  [ ] 시드 의존성은 B절 표를 직접 볼 것")
log(f"\n  ※ 로컬 이득을 LB 점수로 환산하지 말 것 — 전이 공식은 폐기됐다.")
log(f"     말할 수 있는 것은 '로컬 {len(YEARS)}개 연도에서 이렇게 움직였다'까지다.")
log("=" * 88)
