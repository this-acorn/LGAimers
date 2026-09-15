# -*- coding: utf-8 -*-
"""
[30] tm_cov 하네스 + no_batter 독립시드 확인 — 다연도 paired, 독립 시드

팔 3개 (시드 [99,555,31337,1] — exp/22/24b 선택에 미사용):
  base65     exp/27 npz의 hgb 예측 재사용 (같은 시드·같은 설정, 재학습 0회)
  no_batter  batter_id 제거 — exp/24b 회색지대(4/4 양수, 평균 +9.4)의 독립시드 확인
  tm_cov     base65 + 구종군별 릴리스 공분산 7피처 (아래)

tm_cov 설계 (리뷰 합의 반영):
  · 구종군 분리: fastball/breaking/offspeed 별도 집계 — 의도적 릴리스 차이(투심/커브의
    다른 팔각도)를 '제구 불안'으로 오측정하는 exp/17의 결함 수정
  · 공분산 스무딩: Σ = (n·Σ_투수 + k·Σ_리그,구종군)/(n+k), k=100 — MIN_N 컷 대신
    표본 적은 투수를 리그 평균으로 수축 (커버리지 보존)
  · 시즌 as-of: 행의 season S 기준 <S 트랙맨만. (pid, season) 그리드에 명시적으로
    생성하므로 "그 시즌 트랙맨 행이 있어야 키가 생기는" exp/17 결함(exp/21-D) 없음
  · strict 매칭 256명만 (round<=2). 미매칭 NaN (GBDT 네이티브)
  · 피처 7개:
      f_tm_area_F/B/O   sqrt(det Σ_g) — 릴리스 타원 면적 (구종군별)
      f_tm_maxeig_F     fastball Σ 최대 고유값 (가장 흔들리는 방향의 크기)
      f_tm_eigratio_F   fastball 고유값비 (한 방향 쏠림)
      f_tm_logn         log1p(과거 트랙맨 투구수)
      f_tm_dw           r_F·area_F + r_B·area_B + r_O·area_O — 행의 공식 asof 구종믹스로
                        가중한 '오늘 예상 흔들림'. 곱셈+합산이라 트리가 못 만드는 계산
  ※ det vs logdet 는 단조변환이라 트리에 동일 — det 사용. εI 릿지만 적용.

사전 등록 채택 기준:
  no_batter: ≥3/4 양수 & 평균 ≥ +5 & 최악 ≥ −10   (원측정 +9.4의 재현 확인)
  tm_cov  : ≥3/4 양수 & 평균 ≥ +10 & 최악 ≥ −10   ([S] 표준. 동결-물리형이라
            타겟 레짐 변화 비노출이지만, 채택 시 2025 배포 그리드 필수)

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/30_tmcov_harness.py   (~25분)
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "exp")
from common import (log, tick, raw_score, load_train, add_features,
                    fit_encoder, to_matrix, ALL_ENG, HGB_FAST)

YEARS = [2021, 2022, 2023, 2024]
SEEDS = [99, 555, 31337, 1]
NPZ27 = "lab/27_preds.npz"
K_SHRINK = 100.0
EPS = 1e-6
GROUPS = ["fastball", "breaking", "offspeed"]
RATE_COL = {"fastball": "asof_pitcher_fastball_rate",
            "breaking": "asof_pitcher_breaking_rate",
            "offspeed": "asof_pitcher_offspeed_rate"}
FIRST = ["DOO_BEA", "HAN_EAG", "KIA_TIG", "KIW_HER", "KT_WIZ", "LG_TWI",
         "LOT_GIA", "NC_DIN", "SAM_LIO", "SK_WYV", "SSG_LAN"]
TM_FEATS = ["f_tm_area_F", "f_tm_area_B", "f_tm_area_O",
            "f_tm_maxeig_F", "f_tm_eigratio_F", "f_tm_logn", "f_tm_dw"]

# =====================================================================
# 1. 트랙맨 → (pid, season) 구종군별 as-of 공분산 테이블
# =====================================================================
log("trackman 로딩 중...")
tm = pd.read_csv("data/trackman_history.csv",
                 usecols=["season", "pitcher_trackman_id", "pitcher_team",
                          "pitch_type_group", "rel_height", "rel_side"],
                 encoding="utf-8-sig")
tm.columns = [c.replace("﻿", "").strip() for c in tm.columns]
match = pd.read_csv("lab/trackman_match_v3.csv")
strict = match[match["round"] <= 2]
tid2pid = dict(zip(strict.tid, strict.pid))
tm = tm[tm.pitcher_team.isin(FIRST)
        & tm.pitcher_trackman_id.isin(tid2pid)
        & tm.pitch_type_group.isin(GROUPS)
        & tm.rel_height.notna() & tm.rel_side.notna()].copy()
tm["pid"] = tm.pitcher_trackman_id.map(tid2pid)
tick(f"strict {len(strict)}명 / 유효 트랙맨 {len(tm):,}행")

tm["h2"] = tm.rel_height ** 2
tm["s2"] = tm.rel_side ** 2
tm["hs"] = tm.rel_height * tm.rel_side
agg = (tm.groupby(["pid", "pitch_type_group", "season"])
         .agg(n=("rel_height", "size"), sh=("rel_height", "sum"),
              ss=("rel_side", "sum"), shh=("h2", "sum"),
              sss=("s2", "sum"), shs=("hs", "sum"))
         .reset_index())
lg = (tm.groupby(["pitch_type_group", "season"])
        .agg(n=("rel_height", "size"), sh=("rel_height", "sum"),
             ss=("rel_side", "sum"), shh=("h2", "sum"),
             sss=("s2", "sum"), shs=("hs", "sum"))
        .reset_index())
del tm

SUMS = ["n", "sh", "ss", "shh", "sss", "shs"]


def cov_from_sums(r):
    """합계 → 2x2 공분산 (n, [vh, vs, chs])"""
    n = r["n"]
    if n < 2:
        return None
    mh, ms = r["sh"] / n, r["ss"] / n
    vh = max(r["shh"] / n - mh ** 2, 0.0)
    vs = max(r["sss"] / n - ms ** 2, 0.0)
    chs = r["shs"] / n - mh * ms
    return n, vh, vs, chs


def past_sums(frame, keys, Y):
    """season < Y 합계를 keys 단위로 집계"""
    p = frame[frame.season < Y]
    if len(p) == 0:
        return pd.DataFrame(columns=keys + SUMS)
    return p.groupby(keys)[SUMS].sum().reset_index()


rows = []
for Y in range(2020, 2025):          # 행 season 2020~2024 에 대해 as-of <Y
    lgp = past_sums(lg, ["pitch_type_group"], Y)
    lg_cov = {}
    for _, r in lgp.iterrows():
        c = cov_from_sums(r)
        if c:
            lg_cov[r.pitch_type_group] = c
    pp = past_sums(agg, ["pid", "pitch_type_group"], Y)
    for pid, grp in pp.groupby("pid"):
        feat = {"pid": pid, "season": Y}
        total_n = 0.0
        for _, r in grp.iterrows():
            g = r.pitch_type_group
            c = cov_from_sums(r)
            if c is None or g not in lg_cov:
                continue
            n, vh, vs, chs = c
            _, lvh, lvs, lchs = lg_cov[g]
            w = n / (n + K_SHRINK)
            svh = w * vh + (1 - w) * lvh + EPS
            svs = w * vs + (1 - w) * lvs + EPS
            schs = w * chs + (1 - w) * lchs
            det = max(svh * svs - schs ** 2, 0.0)
            tag = {"fastball": "F", "breaking": "B", "offspeed": "O"}[g]
            feat[f"area_{tag}"] = np.sqrt(det)
            total_n += n
            if g == "fastball":
                t = svh + svs
                disc = max(t * t / 4 - det, 0.0)
                e1 = t / 2 + np.sqrt(disc)
                e2 = max(t / 2 - np.sqrt(disc), EPS)
                feat["maxeig_F"] = e1
                feat["eigratio_F"] = e1 / e2
        if total_n > 0:
            feat["logn"] = np.log1p(total_n)
            rows.append(feat)

tm_tbl = pd.DataFrame(rows)
tick(f"as-of 테이블 {len(tm_tbl):,}행 (투수-시즌) / 컬럼 {list(tm_tbl.columns)}")


def attach_tm(d):
    d = d.copy()
    key = pd.DataFrame({"pid": d["pitcher_id"].to_numpy(),
                        "season": d["season"].to_numpy()})
    mg = key.merge(tm_tbl, on=["pid", "season"], how="left", validate="m:1")
    d["f_tm_area_F"] = mg["area_F"].to_numpy("float32")
    d["f_tm_area_B"] = mg["area_B"].to_numpy("float32")
    d["f_tm_area_O"] = mg["area_O"].to_numpy("float32")
    d["f_tm_maxeig_F"] = mg["maxeig_F"].to_numpy("float32")
    d["f_tm_eigratio_F"] = mg["eigratio_F"].to_numpy("float32")
    d["f_tm_logn"] = mg["logn"].to_numpy("float32")
    dw = np.zeros(len(d))
    ok = np.zeros(len(d), dtype=bool)
    for g, tag in [("fastball", "F"), ("breaking", "B"), ("offspeed", "O")]:
        r = d[RATE_COL[g]].to_numpy("float64")
        a = mg[f"area_{tag}"].to_numpy("float64")
        good = ~np.isnan(r) & ~np.isnan(a)
        dw = np.where(good, dw + np.nan_to_num(r) * np.nan_to_num(a), dw)
        ok |= good
    d["f_tm_dw"] = np.where(ok, dw, np.nan).astype("float32")
    return d


# =====================================================================
# 2. 하네스 — base65는 exp/27 npz 재사용, no_batter/tm_cov만 학습
# =====================================================================
z27 = np.load(NPZ27)
log("\ntrain.csv 로딩 중...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS65 = BASE + ALL_ENG
FEATS_NB = [c for c in FEATS65 if c != "batter_id"]
FEATS_TM = FEATS65 + TM_FEATS

results = {}
for Y in YEARS:
    tr_raw = df[df["season"] <= Y - 1].reset_index(drop=True)
    va_raw = df[df["season"] == Y].reset_index(drop=True)
    y_tr = tr_raw["control_success"].to_numpy()
    y_va = va_raw["control_success"].to_numpy()
    prior = float(y_tr.mean())
    tr = attach_tm(add_features(tr_raw, prior))
    va = attach_tm(add_features(va_raw, prior))
    enc = fit_encoder(tr, FEATS65)
    cov = va["f_tm_area_F"].notna().mean() * 100
    p_base = np.mean([z27[f"{Y}_hgb_{s}"] for s in SEEDS], axis=0)
    base_seeds = {s: z27[f"{Y}_hgb_{s}"] for s in SEEDS}
    s_base = raw_score(p_base, y_va)
    log(f"\n{'='*88}")
    log(f"검증연도 {Y}  학습 {len(tr):,} / 검증 {len(va):,}  tm커버리지 {cov:.1f}%  "
        f"base65(재사용) {s_base:.1f}")
    log("=" * 88)

    res_y = {}
    for arm, feats in [("no_batter", FEATS_NB), ("tm_cov", FEATS_TM)]:
        Xtr, Xva = to_matrix(tr, feats, enc), to_matrix(va, feats, enc)
        pg, preds = [], []
        for s in SEEDS:
            tick(f"{Y} {arm} seed={s}")
            m = HistGradientBoostingClassifier(
                **{**HGB_FAST, "random_state": s}).fit(Xtr, y_tr)
            p = m.predict_proba(Xva)[:, 1]
            preds.append(p)
            pg.append(raw_score(p, y_va) - raw_score(base_seeds[s], y_va))
        eg = raw_score(np.mean(preds, axis=0), y_va) - s_base
        res_y[arm] = (np.mean(pg), eg)
        log(f"    {arm:10s} paired평균 {np.mean(pg):+7.1f}  앙상블이득 {eg:+7.1f}")
        del Xtr, Xva
    results[Y] = res_y
    del tr, va, tr_raw, va_raw

# =====================================================================
log("\n" + "=" * 88)
log("판정 (독립 시드, 앙상블 이득 기준)")
log("=" * 88)
log(f"  {'연도':>6s} {'no_batter':>11s} {'tm_cov':>11s}")
for Y in YEARS:
    log(f"  {Y:>6d} {results[Y]['no_batter'][1]:>+11.1f} {results[Y]['tm_cov'][1]:>+11.1f}")
for arm, (th_mean, note) in [("no_batter", (5.0, "원측정 +9.4 재현 확인")),
                             ("tm_cov", (10.0, "[S] 표준 · 채택 시 2025 그리드 필수"))]:
    a = np.array([results[Y][arm][1] for Y in YEARS])
    ok = (a > 0).sum() >= 3 and a.mean() >= th_mean and a.min() >= -10.0
    log(f"\n  {arm}: 양수 {(a>0).sum()}/4, 평균 {a.mean():+.1f}, 최악 {a.min():+.1f}")
    log(f"    기준: ≥3/4 양수 & 평균 ≥ +{th_mean:.0f} & 최악 ≥ -10  ({note})")
    log(f"    → {'채택 후보' if ok else '기각/보류'}")
log("=" * 88)
