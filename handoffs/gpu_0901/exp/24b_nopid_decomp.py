# -*- coding: utf-8 -*-
"""
[24b] no_pid 효과 분해 — 재실행판 (exp/24가 1182초에 외부 종료됨)

exp/24는 2023 no_pitcher 도중 EXIT_CODE=127로 끊겼고 결과 파일도 외부에서 삭제됐다.
살아남은 값(대화 기록):
    2022  base65 2283.3 / no_pitcher 2284.3 / no_batter 2291.5 / no_both 2285.0
    2023  base65 -1408.4  (no_pitcher 4시드까지 돌고 중단)

24b의 변경점:
  1) 연도가 끝날 때마다 **즉시 디스크에 저장** — 중간에 죽어도 앞부분은 남는다
     · lab/24b_preds.npz   예측 배열 (혼합 스윕 재계산용, 재학습 불필요)
     · lab/24b_result.txt  로그
     · 스크래치패드에 백업 사본도 같이 쓴다 (같은 폴더가 정리당해도 살아남게)
  2) 메모리 절감 — 팔마다 to_matrix()를 다시 부르지 않고 **전체 행렬 1회 생성 후 열 슬라이스**.
     학습 DataFrame은 행렬을 만든 즉시 해제한다 (2024년 tr은 1.22M×67로 수백 MB)
  3) 부분 완료여도 분석이 돌아간다 (끝난 연도만으로 표를 만든다)

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/24b_nopid_decomp.py
"""

import gc
import os
import sys
import shutil
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "exp")
from common import (log, tick, raw_score, load_train, add_features,
                    fit_encoder, to_matrix, ALL_ENG, HGB_FAST, CAT)

YEARS = [2021, 2022, 2023, 2024]
SEEDS = [42, 7, 123, 2024]
PICK_YEARS, CONFIRM_YEAR = [2021, 2022, 2023], 2024
W_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
NPZ = "lab/24b_preds.npz"
BACKUP_DIR = ("C:/Users/gwonn/AppData/Local/Temp/claude/c--Users-gwonn-Desktop-open/"
              "ac9f75b2-20b2-4bd0-a3a5-9f91191c2d24/scratchpad")

log("train.csv 로딩 중...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS65 = BASE + ALL_ENG

ARMS = [
    ("base65", FEATS65),
    ("no_pitcher", [c for c in FEATS65 if c != "pitcher_id"]),
    ("no_batter", [c for c in FEATS65 if c != "batter_id"]),
    ("no_both", [c for c in FEATS65 if c not in ("pitcher_id", "batter_id")]),
]
# to_matrix의 열 순서: CAT 먼저, 그 다음 나머지 (FEATS65 순서)
FULL_COLS = [c for c in CAT if c in FEATS65] + [c for c in FEATS65 if c not in CAT]
COL_IX = {n: i for i, n in enumerate(FULL_COLS)}
tick(f"로드 {df.shape} — 팔 {len(ARMS)} × 연도 {len(YEARS)} × 시드 {len(SEEDS)}")

store, dump = {}, {}


def flush():
    """중간 저장 — 외부에서 지워지거나 죽어도 여기까지는 남는다."""
    np.savez_compressed(NPZ, **dump)
    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        shutil.copy(NPZ, os.path.join(BACKUP_DIR, "24b_preds.npz"))
        shutil.copy("lab/24b_result.txt", os.path.join(BACKUP_DIR, "24b_result.txt"))
    except Exception as e:
        log(f"    (백업 실패, 무시하고 계속: {e})")


for Y in YEARS:
    tr_raw = df[df["season"] <= Y - 1].reset_index(drop=True)
    va_raw = df[df["season"] == Y].reset_index(drop=True)
    y_tr = tr_raw["control_success"].to_numpy()
    y_va = va_raw["control_success"].to_numpy()
    prior = float(y_tr.mean())
    tr, va = add_features(tr_raw, prior), add_features(va_raw, prior)
    enc = fit_encoder(tr, FEATS65)

    seen_p = set(tr_raw["pitcher_id"].unique())
    seen_b = set(tr_raw["batter_id"].unique())
    meta = pd.DataFrame({
        "p_n": va["asof_pitcher_n"].fillna(0).to_numpy("float32"),
        "b_n": va["asof_batter_n"].fillna(0).to_numpy("float32"),
        "p_new": (~va_raw["pitcher_id"].isin(seen_p)).to_numpy(),
        "b_new": (~va_raw["batter_id"].isin(seen_b)).to_numpy(),
    })
    log(f"\n{'='*88}")
    log(f"검증연도 {Y}  학습 {len(tr):,} / 검증 {len(va):,}  "
        f"(신인투수행 {meta.p_new.mean()*100:.1f}% / 신인타자행 {meta.b_new.mean()*100:.1f}%)")
    log("=" * 88)

    # ★ 전체 행렬 1회 생성 후 DataFrame 즉시 해제 (메모리 절감)
    Xtr_full = to_matrix(tr, FEATS65, enc)
    Xva_full = to_matrix(va, FEATS65, enc)
    del tr, va, tr_raw, va_raw
    gc.collect()

    preds = {}
    for arm_name, feats in ARMS:
        fs = set(feats)
        idx = [COL_IX[n] for n in FULL_COLS if n in fs]
        full = len(idx) == len(FULL_COLS)
        Xtr_a = Xtr_full if full else Xtr_full[:, idx]
        Xva_a = Xva_full if full else Xva_full[:, idx]
        acc = np.zeros(len(y_va))
        for s in SEEDS:
            tick(f"{Y} {arm_name} seed={s}")
            m = HistGradientBoostingClassifier(
                **{**HGB_FAST, "random_state": s}).fit(Xtr_a, y_tr)
            acc += m.predict_proba(Xva_a)[:, 1]
            del m
        preds[arm_name] = acc / len(SEEDS)
        dump[f"{Y}_{arm_name}"] = preds[arm_name].astype("float32")
        log(f"    {arm_name:11s} 앙상블 {raw_score(preds[arm_name], y_va):8.1f}")
        if not full:
            del Xtr_a, Xva_a
        gc.collect()

    del Xtr_full, Xva_full
    gc.collect()
    dump[f"{Y}_y"] = y_va.astype("int8")
    for c in meta.columns:
        dump[f"{Y}_meta_{c}"] = meta[c].to_numpy()
    store[Y] = {"y": y_va, "meta": meta, "preds": preds}
    flush()
    log(f"    [저장] {Y}까지 lab/24b_preds.npz 에 기록 (중단돼도 여기까진 보존)")

DONE = [Y for Y in YEARS if Y in store]
log(f"\n완료 연도: {DONE}")

# =====================================================================
log("\n" + "=" * 88)
log("A. 투수 ID / 타자 ID 분리 — 어느 쪽이 문제인가")
log("=" * 88)
log(f"  {'연도':>6s} {'no_pitcher':>12s} {'no_batter':>12s} {'no_both':>12s}")
sep = {a: [] for a, _ in ARMS if a != "base65"}
for Y in DONE:
    s0 = raw_score(store[Y]["preds"]["base65"], store[Y]["y"])
    row = []
    for a in ["no_pitcher", "no_batter", "no_both"]:
        g = raw_score(store[Y]["preds"][a], store[Y]["y"]) - s0
        sep[a].append(g)
        row.append(f"{g:+12.1f}")
    log(f"  {Y:>6d} {''.join(row)}")
log(f"  {'─'*46}")
for a in ["no_pitcher", "no_batter", "no_both"]:
    v = np.array(sep[a])
    log(f"  {a:11s} 평균 {v.mean():+7.1f}  최악 {v.min():+7.1f}  "
        f"양수 {(v>0).sum()}/{len(v)}")
log("\n  해석: no_pitcher + no_batter 합이 no_both와 크게 다르면 두 ID가 상호작용한다.")

# =====================================================================
log("\n" + "=" * 88)
log("B. 구간 분해 — 행별 이득 = base 제곱오차 - no_both 제곱오차 (양수면 제거가 이득)")
log("=" * 88)
for Y in DONE:
    d = store[Y]
    y, m = d["y"], d["meta"]
    gain = (d["preds"]["base65"] - y) ** 2 - (d["preds"]["no_both"] - y) ** 2
    r = y.mean()
    scale = 100000.0 / (r * (1 - r))
    log(f"\n  ── {Y} (총 {sep['no_both'][DONE.index(Y)]:+.1f}) ──")
    log(f"    {'구분':<12s} {'구간':<10s} {'기여(점수)':>12s} {'행수':>10s}")
    pn_bin = pd.cut(m.p_n, [-1, 0, 50, 200, 1000, 1e9],
                    labels=["0", "1-50", "51-200", "201-1000", "1000+"])
    for name, grp in [("투수 asof_n", pn_bin),
                      ("신인투수", m.p_new.map({True: "신인", False: "기존"})),
                      ("신인타자", m.b_new.map({True: "신인", False: "기존"}))]:
        agg = pd.DataFrame({"g": gain, "k": grp}).groupby("k", observed=True)["g"]
        for k, v in agg.agg(["mean", "size"]).iterrows():
            log(f"    {name:<12s} {str(k):<10s} {v['mean']*scale:>12.1f} "
                f"{int(v['size']):>10,}")

# =====================================================================
log("\n" + "=" * 88)
log("C. 혼합 스윕 — 비율 선택 2021~2023 / 확인 2024")
log("=" * 88)
pick_ok = [Y for Y in PICK_YEARS if Y in store]
log(f"  {'w':>5s} " + " ".join(f"{Y:>10d}" for Y in DONE) + f" {'선택평균':>10s}")
best_w, best_v, curve = 0.0, -1e9, {}
for w in W_GRID:
    gains = []
    for Y in DONE:
        d = store[Y]
        p = (1 - w) * d["preds"]["base65"] + w * d["preds"]["no_both"]
        gains.append(raw_score(p, d["y"]) - raw_score(d["preds"]["base65"], d["y"]))
    curve[w] = gains
    pick = float(np.mean([gains[DONE.index(Y)] for Y in pick_ok])) if pick_ok else float("nan")
    log(f"  {w:>5.1f} " + " ".join(f"{g:>+10.1f}" for g in gains) + f" {pick:>+10.1f}")
    if pick_ok and pick > best_v:
        best_w, best_v = w, pick

log(f"\n  선택된 w = {best_w:.1f}  (선택연도 평균 {best_v:+.1f})")
if CONFIRM_YEAR in store:
    cg = curve[best_w][DONE.index(CONFIRM_YEAR)]
    log(f"  → {CONFIRM_YEAR} 확인 이득 {cg:+.1f}   "
        f"(전량제거 w=1.0은 {curve[1.0][DONE.index(CONFIRM_YEAR)]:+.1f})")
    log(f"\n  판정: 확인연도가 0 이상이고 전량제거보다 나으면 혼합이 답이다.")
    log(f"        w가 0/1 끝으로 몰리면 혼합은 의미 없다.")
else:
    log(f"  ⚠ {CONFIRM_YEAR} 미완 — 확인 단계 보류")
log("=" * 88)
