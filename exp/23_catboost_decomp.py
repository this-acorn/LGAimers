# -*- coding: utf-8 -*-
"""
[23] CatBoost 효과 분해 프로브 — '모델 효과'와 '선수 CTR 효과'를 갈라서 잰다

왜 분해하는가:
  CatBoost를 한 팔로 돌리면 서로 위험등급이 다른 두 효과가 섞인다.
    (1) 모델 효과      oblivious(대칭) 트리 + 다른 정규화. 순수 모델클래스 변경.
                       RF→HGB(+195, 전이 성공)와 같은 [S] 등급. 동결테이블 없음.
    (2) 선수 CTR 효과  고카디널 선수 ID의 ordered target statistics.
                       추론 시 ≤2024 전체 집계로 얼어붙는다 → hand delta(-22.2)와
                       **메커니즘이 동일한** [F] 등급.
  한 팔로 돌려서 이기면 어느 쪽이 이긴 건지 알 수 없다. 그래서 세 팔로 나눈다:
    hgb    기준 (현재 제출, LB 830.322760105)
    cb_num CatBoost + 저카디널 범주형만 (ID는 수치 유지)   → (1)만
    cb_cat cb_num + 선수 ID를 범주형으로 추가 (수치본도 유지) → (1)+(2)
  분해:  모델효과 = cb_num - hgb        선수CTR효과 = cb_cat - cb_num

왜 채택/기각 이분법 대신 혼합 비율을 재는가:
  p_w = (1-w)*p_A + w*p_B 로 섞으면 브라이어 차이는 정확히
      Δ = w*(B_B - B_A)  +  w*(w-1)*D,      D = E[(p_A - p_B)^2]
  · 손실항은 w에 **선형** → hand delta급 실패(-22.2)도 w=0.25면 약 -5.5로 묶인다
  · 다양성 이득항 w(w-1)*D 는 0<w<1에서 항상 음수(=이득), w=0.5에서 최대
  → 단독으로 진 모델도 D가 크면 혼합에서 이긴다. 이분법은 이걸 통째로 버린다.

★ D 가 올바른 불일치 지표인 이유 (이전 리뷰의 '오차상관은 y 때문에 1에 가깝다'는
  경고에 대한 답):  e_A - e_B = (p_A - y) - (p_B - y) = p_A - p_B  로 y가 소거된다.
  즉 D 는 공통 y 오염이 원천적으로 없다. XGB 혼합이 이득 0.0 이었던 건 D가 작아서였다.
  → D를 먼저 재고, XGB 수준이면 조기 종료한다.

이 스크립트는 **프로브**다. 검증연도 1개 × 2시드로 D와 최적 w를 먼저 재고,
D가 의미 있게 클 때만 exp/22 하네스에 4개 연도로 태운다.

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/23_catboost_decomp.py
"""

import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "exp")
from common import (log, tick, raw_score, load_train, add_features,
                    fit_encoder, to_matrix, ALL_ENG, HGB_FAST, CAT)

from catboost import CatBoostClassifier, Pool

PROBE_YEAR = 2023      # 2024는 이미 선택에 소진됨. 프로브는 2023으로.
SEEDS = [42, 7]
CB_PARAMS = dict(iterations=500, learning_rate=0.08, depth=6,
                 l2_leaf_reg=10.0, verbose=False, thread_count=6,
                 allow_writing_files=False)
XGB_REF_NOTE = "XGB 혼합 이득 0.0 (FEATURE_BRIEF §3) — D가 이 수준이면 조기 종료"

# =====================================================================
log("train.csv 로딩 중...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS65 = BASE + ALL_ENG

tr_raw = df[df["season"] <= PROBE_YEAR - 1].reset_index(drop=True)
va_raw = df[df["season"] == PROBE_YEAR].reset_index(drop=True)
y_tr = tr_raw["control_success"].to_numpy()
y_va = va_raw["control_success"].to_numpy()
prior = float(y_tr.mean())
tr = add_features(tr_raw, prior)
va = add_features(va_raw, prior)
tick(f"준비 완료 — 검증 {PROBE_YEAR}: 학습 {len(tr):,} / 검증 {len(va):,}")
del df, tr_raw, va_raw

# =====================================================================
# 1. HGB 기준
# =====================================================================
log("\n" + "=" * 86)
log("1. HGB 기준 (현재 제출 구성)")
log("=" * 86)
enc = fit_encoder(tr, FEATS65)
Xtr, Xva = to_matrix(tr, FEATS65, enc), to_matrix(va, FEATS65, enc)
hgb_preds = []
for s in SEEDS:
    t0 = time.time()
    m = HistGradientBoostingClassifier(**{**HGB_FAST, "random_state": s}).fit(Xtr, y_tr)
    hgb_preds.append(m.predict_proba(Xva)[:, 1])
    tick(f"hgb seed={s}  {raw_score(hgb_preds[-1], y_va):7.1f}  ({time.time()-t0:.0f}s)")
p_hgb = np.mean(hgb_preds, axis=0)
s_hgb = raw_score(p_hgb, y_va)
log(f"  → HGB 앙상블 {s_hgb:.1f}")
del Xtr, Xva

# =====================================================================
# 2. CatBoost 준비 — 저카디널 범주형 / 선수 ID 범주형
# =====================================================================
NUM_FEATS = [c for c in FEATS65 if c not in CAT]
ID_CAT = ["pitcher_id_cat", "batter_id_cat"]


def make_frame(d, with_id_cat):
    """CatBoost 입력. 범주형은 문자열로 넘긴다(정수 캐스팅 사고 방지)."""
    out = d[NUM_FEATS].copy()
    for c in CAT:
        out[c] = d[c].astype(str)
    if with_id_cat:
        # ★ ID를 두 벌로: 수치본(NUM_FEATS에 이미 포함, 데뷔순서 신호 r=0.704 보존)
        #   + 범주본(개별 선수 고유 효과)
        out["pitcher_id_cat"] = d["pitcher_id"].astype(str)
        out["batter_id_cat"] = d["batter_id"].astype(str)
    return out


def run_cb(name, with_id_cat):
    cats = list(CAT) + (ID_CAT if with_id_cat else [])
    Ftr, Fva = make_frame(tr, with_id_cat), make_frame(va, with_id_cat)
    ptr = Pool(Ftr, y_tr, cat_features=cats)
    pva = Pool(Fva, cat_features=cats)
    preds = []
    for s in SEEDS:
        t0 = time.time()
        m = CatBoostClassifier(**CB_PARAMS, random_seed=s).fit(ptr)
        preds.append(m.predict_proba(pva)[:, 1])
        tick(f"{name} seed={s}  {raw_score(preds[-1], y_va):7.1f}  ({time.time()-t0:.0f}s)")
    p = np.mean(preds, axis=0)
    log(f"  → {name} 앙상블 {raw_score(p, y_va):.1f}")
    return p


log("\n" + "=" * 86)
log("2. cb_num — CatBoost + 저카디널 범주형만 (모델 효과만)   [S] 등급")
log("=" * 86)
p_cbnum = run_cb("cb_num", False)
s_cbnum = raw_score(p_cbnum, y_va)

log("\n" + "=" * 86)
log("3. cb_cat — + 선수 ID 범주형 (모델 + 선수CTR)            [F] 등급")
log("=" * 86)
p_cbcat = run_cb("cb_cat", True)
s_cbcat = raw_score(p_cbcat, y_va)

# =====================================================================
# 4. 효과 분해
# =====================================================================
log("\n" + "=" * 86)
log(f"4. 효과 분해  (검증 {PROBE_YEAR}, {len(SEEDS)}시드 앙상블)")
log("=" * 86)
log(f"  hgb    {s_hgb:8.1f}")
log(f"  cb_num {s_cbnum:8.1f}   모델 효과      = {s_cbnum-s_hgb:+8.1f}   [S] 등급")
log(f"  cb_cat {s_cbcat:8.1f}   선수 CTR 효과  = {s_cbcat-s_cbnum:+8.1f}   [F] 등급")
log(f"                     합계(=cb_cat-hgb) = {s_cbcat-s_hgb:+8.1f}")


def analyze(name, p_b):
    """혼합 곡선. D는 y가 소거된 순수 예측 불일치라 공통-y 오염이 없다."""
    d2 = (p_hgb - p_b) ** 2
    D = float(d2.mean())
    # 최적 w = E[(p_A-y)(p_A-p_B)] / E[(p_A-p_B)^2]  (OLS 계수 형태)
    w_star = float(((p_hgb - y_va) * (p_hgb - p_b)).mean() / max(D, 1e-12))
    log(f"\n  ■ {name}")
    log(f"    불일치 D = E[(p_hgb-p_b)^2] = {D:.3e}   (RMS 차이 {np.sqrt(D):.4f})")
    log(f"    닫힌형 최적 w* = {w_star:+.3f}   ※ 이 해는 검증연도에 맞춘 값 —"
        f" 채택용 아님(과적합면)")
    log(f"    {'w':>6s} {'혼합점수':>10s} {'기준대비':>10s}")
    for w in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0]:
        sc = raw_score((1 - w) * p_hgb + w * p_b, y_va)
        log(f"    {w:>6.2f} {sc:>10.1f} {sc-s_hgb:>+10.1f}")
    return D, w_star


log("\n" + "=" * 86)
log("5. 혼합 곡선")
log("=" * 86)
D_num, w_num = analyze("hgb × cb_num  [S]", p_cbnum)
D_cat, w_cat = analyze("hgb × cb_cat  [F]", p_cbcat)

log("\n" + "=" * 86)
log("판정 가이드")
log("=" * 86)
log(f"  {XGB_REF_NOTE}")
log(f"  · D가 충분히 크고 w=0.15~0.30 구간에서 이득이 양수면 → exp/22 하네스에")
log(f"    4개 연도로 태워서 **고정 w**(연도별 argmin 금지)로 재검증")
log(f"  · 모델효과[S]가 주 원인이면 위험 낮음 — RF→HGB와 같은 계열")
log(f"  · 선수CTR효과[F]가 주 원인이면 hand delta와 같은 메커니즘 —")
log(f"    혼합 비율을 w<=0.25로 제한해 최악 손실을 선형으로 묶을 것")
log("=" * 86)
