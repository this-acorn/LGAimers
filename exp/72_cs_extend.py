# -*- coding: utf-8 -*-
"""
[72] CS 확장 — reverse / breaking 델타 추가 (이진 d6/it500/lr0.04, 2시드 짝지음)

exp/71 근거:
  · 현행 P_RATES = {succ, ball, strike, middle, fb} — reverse·breaking 누락
  · reverse: 나머지 4종 선형회귀 R²=0.715 → 독립 정보.
             타겟 상관 −0.0795 로 success(+0.0843) 다음으로 크다 (middle의 2배)
  · breaking: fastball 로부터 R²=0.208 → 강하게 독립
  · offspeed: R²=1.000000 완전 종속 → 제외 (fb+brk+off=1 이 100% 성립)
  · 결과 5종은 합=1이 아니다(평균 1.706). 제구축(succ+rev+mid=0.892)과
    판정축(ball+strike=0.814)이 별개 라벨링이기 때문. 즉 reverse는
    다른 축들의 잔차가 아니라 제구축의 독립 성분이다.

🚨 이건 새 축이 아니라 **exp/51 재격리**다. exp/51은 이미 −24.7로 기각했다.
   그래도 다시 재는 이유는 exp/51의 설계에 교란 세 개가 있었기 때문이다:
     1) 4피처 **일괄** 투입 — reverse/brk/**off**/f_form_dev5.
        off 는 오늘 R²=1.000000 으로 완전 종속 확인 → 순수 노이즈 열이다.
        f_form_dev5 는 CS 와 무관한 별개 축이다.
     2) **lr=0.08** — 이후 exp/65 에서 그 지점이 +38.5 만큼 틀렸다고 판명됐다.
        과적합 중인 모델에 노이즈 열을 더하면 실제보다 크게 손해가 난다.
     3) exp/51 헤더가 "통과 시 개별 ablation" 이라 적었고, 통과 못 해서 **격리는
        한 번도 수행되지 않았다**. 즉 reverse 단독은 측정된 적이 없다.
   → off/dev5 를 빼고 lr 을 바로잡아 reverse 를 단독 측정한다. 결과가 어느 쪽이든
     이 축은 이번에 영구히 닫힌다. 사전확률은 낮다 — exp/69 뒤에 돌린다.

CS 복원은 이 프로젝트 최대 이득 축(로컬 +107.9 / LB +92.3, 거의 1:1 전이)이라
격리 비용 1.5시간을 쓸 값은 있다. 단, 기대값을 높게 잡지 마라.

★ 짝지음 기준 = exp/65 d6_lr04 2시드 839.4 (79피처).
  준비 블록·시드[42,7]·thread_count=8 을 exp/65와 완전히 동일하게 유지.
  차이는 P_RATES 확장 하나뿐 — 그것이 곧 측정 대상이다.

팔:
  cs_rev      79 + f_cs_p_rev_d              (80피처)
  cs_rev_brk  79 + f_cs_p_rev_d + f_cs_p_brk_d (81피처)

판정 임계 +15 (시드 노이즈 σ≈15.3). 통과 시 MC04 위에서 독립 확증 후 배포.
실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/72_cs_extend.py   (~1.5시간)
"""

import importlib.util
import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import log, tick, raw_score, load_train

spec = importlib.util.spec_from_file_location("s12", 'submissions/submit12_src/script.py')
s12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s12)

# ★ 유일한 변경점 — attach_cs 는 모듈 전역 P_RATES 를 읽으므로 여기서 갈아끼운다.
#   prep 이 s12.attach_cs / s12.P_RATES 를 쓰기 전에 반드시 먼저 실행돼야 한다.
P_RATES_EXT = dict(s12.P_RATES)
P_RATES_EXT["rev"] = "asof_pitcher_reverse_rate"
P_RATES_EXT["brk"] = "asof_pitcher_breaking_rate"
s12.P_RATES = P_RATES_EXT
NEW_REV, NEW_BRK = "f_cs_p_rev_d", "f_cs_p_brk_d"

SEED = 42
NTHREAD = 8
REF_LR04 = 839.4    # exp/65 d6_lr04 2시드 — 동일 하네스/시드/스레드라 짝지음 유효
SEEDS = [42, 7]
K_MIX = 50.0
BASE_CB = dict(l2_leaf_reg=10.0, verbose=False, thread_count=NTHREAD,
               allow_writing_files=False, random_seed=SEED)

log("train 로딩...")
df = load_train()
dd = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index()
pid = dd.pitcher_id.to_numpy()
nn = dd.asof_pitcher_n.fillna(0).to_numpy("float64")
nxt = (pid[1:] == pid[:-1]) & (np.abs(nn[1:] - nn[:-1] - 1) < 1e-6)
for k, col in [("lab_fb", "asof_pitcher_fastball_rate"),
               ("lab_brk", "asof_pitcher_breaking_rate")]:
    S = dd[col].fillna(0).to_numpy("float64") * nn
    lab = np.full(len(dd), np.nan)
    lab[:-1] = np.where(nxt, np.round(S[1:] - S[:-1]), np.nan)
    dd[k] = np.where((lab == 0) | (lab == 1), lab, np.nan)
rec = dd.set_index("index").sort_index()
df["lab_fb"], df["lab_brk"] = rec["lab_fb"], rec["lab_brk"]
b_, s_ = df.balls_before.to_numpy(), df.strikes_before.to_numpy()
df["_cg"] = np.where(s_ > b_, 2, np.where(b_ > s_, 0, 1)).astype("int8")

# 사전 점검 — reverse/breaking 도 rate×n 이 정수여야 CS 복원(누적합 차분)이 성립한다
_n = df["asof_pitcher_n"].fillna(0).to_numpy("float64")
for _c in ["asof_pitcher_reverse_rate", "asof_pitcher_breaking_rate"]:
    _r = df[_c].to_numpy("float64")
    _ok = ~np.isnan(_r)
    _S = _r[_ok] * _n[_ok]
    _frac = np.isclose(_S, np.round(_S), atol=1e-6).mean() * 100
    log(f"  점검: {_c:30s} ×n 이 정수인 비율 {_frac:6.2f}%  (CS 복원 전제)")
    assert _frac > 99.0, f"{_c} 누적합이 정수가 아니다 — CS 복원 전제 위반"


def build_const(src, id_col, n_col, rates):
    d = src.sort_values(n_col).groupby(id_col).tail(1)
    out = pd.DataFrame({"id": d[id_col].to_numpy()})
    n_last = d[n_col].fillna(0).to_numpy("float64")
    out["N_end"] = n_last + 1
    for k, col in rates.items():
        r = d[col].fillna(0).to_numpy("float64")
        if k == "succ":
            out[f"S_{k}"] = np.round(r * n_last) + d["control_success"].to_numpy("float64")
        else:
            out[f"S_{k}"] = r * (n_last + 1)
    return out


def mix_tables(hist):
    t = hist.dropna(subset=["lab_fb"])
    c = (t.groupby(["pitcher_id", "season", "_cg"])
          .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
          .reset_index().sort_values(["pitcher_id", "_cg", "season"]))
    g = c.groupby(["pitcher_id", "_cg"])
    for col in ["n", "fb", "brk"]:
        c[f"p_{col}"] = g[col].cumsum() - c[col]
    o = (t.groupby(["pitcher_id", "season"])
          .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
          .reset_index().sort_values(["pitcher_id", "season"]))
    go = o.groupby("pitcher_id")
    for col in ["n", "fb", "brk"]:
        o[f"po_{col}"] = go[col].cumsum() - o[col]
    tb = c.merge(o[["pitcher_id", "season", "po_n", "po_fb", "po_brk"]],
                 on=["pitcher_id", "season"])
    ofb = (tb.po_fb + 0.5 * K_MIX) / (tb.po_n + K_MIX)
    obk = (tb.po_brk + 0.3 * K_MIX) / (tb.po_n + K_MIX)
    tb["mix_fb"] = np.where(tb.po_n > 0, (tb.p_fb + K_MIX * ofb) / (tb.p_n + K_MIX),
                            np.nan).astype("float32")
    tb["mix_brk"] = np.where(tb.po_n > 0, (tb.p_brk + K_MIX * obk) / (tb.p_n + K_MIX),
                             np.nan).astype("float32")
    cc = (t.groupby(["pitcher_id", "_cg"])
           .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
           .reset_index())
    oo = (t.groupby("pitcher_id").agg(po_n=("lab_fb", "size"), po_fb=("lab_fb", "sum"),
                                      po_brk=("lab_brk", "sum")).reset_index())
    dep = cc.merge(oo, on="pitcher_id")
    o1 = (dep.po_fb + 0.5 * K_MIX) / (dep.po_n + K_MIX)
    o2 = (dep.po_brk + 0.3 * K_MIX) / (dep.po_n + K_MIX)
    dep["mix_fb"] = ((dep.fb + K_MIX * o1) / (dep.n + K_MIX)).astype("float32")
    dep["mix_brk"] = ((dep.brk + K_MIX * o2) / (dep.n + K_MIX)).astype("float32")
    return tb[["pitcher_id", "season", "_cg", "mix_fb", "mix_brk"]], \
        dep[["pitcher_id", "_cg", "mix_fb", "mix_brk"]]


hist = df[df.season <= 2023]
prior = float(hist["control_success"].mean())
cp = build_const(hist, "pitcher_id", "asof_pitcher_n", s12.P_RATES)
cb_ = build_const(hist, "batter_id", "asof_batter_n", s12.B_RATES)
pr = hist.dropna(subset=["lab_fb"])
pfb = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=False,
                         thread_count=NTHREAD, allow_writing_files=False, random_seed=42)
pfb.fit(pr[s12.PFB_IN].fillna(-999), pr["lab_fb"].astype(int))
mix_tr, mix_dep = mix_tables(hist)

tr_rows = df[df.season <= 2023].reset_index(drop=True)
va_rows = df[df.season == 2024].reset_index(drop=True)
y_tr = tr_rows["control_success"].to_numpy()
y_va = va_rows["control_success"].to_numpy()
parts = []
for S in sorted(tr_rows.season.unique()):
    h = df[df.season <= S - 1]
    rows = tr_rows[tr_rows.season == S]
    if len(h) == 0:
        cpS = pd.DataFrame({"id": [], "N_end": [], **{f"S_{k}": [] for k in s12.P_RATES}})
        cbS = pd.DataFrame({"id": [], "N_end": [], **{f"S_{k}": [] for k in s12.B_RATES}})
    else:
        cpS = build_const(h, "pitcher_id", "asof_pitcher_n", s12.P_RATES)
        cbS = build_const(h, "batter_id", "asof_batter_n", s12.B_RATES)
    parts.append(s12.attach_cs(rows, cpS, cbS))
tr = s12.add_features(pd.concat(parts).sort_index(), prior)
k_ = tr[["pitcher_id", "season", "_cg"]].merge(mix_tr, on=["pitcher_id", "season", "_cg"],
                                               how="left")
tr["f_mixcg_fb"] = k_["mix_fb"].to_numpy("float32")
tr["f_mixcg_brk"] = k_["mix_brk"].to_numpy("float32")
tr["f_pfb"] = pfb.predict_proba(tr[s12.PFB_IN].fillna(-999))[:, 1].astype("float32")
va = s12.attach_pt(s12.add_features(s12.attach_cs(va_rows, cp, cb_), prior), mix_dep, pfb)
del parts

BASE = [c for c in pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1)
        .columns.str.replace("﻿", "").str.strip() if c != "row_id"]
ENG18 = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
         "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
         "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
         "f_pb_diff", "f_miss_p", "f_miss_prev1"]
FEATS79 = BASE + ENG18 + s12.CS_FEATS + s12.PT_FEATS
ARMS = [("cs_rev", FEATS79 + [NEW_REV]),
        ("cs_rev_brk", FEATS79 + [NEW_REV, NEW_BRK])]

for c in (NEW_REV, NEW_BRK):
    cov_t = tr[c].notna().mean() * 100
    cov_v = va[c].notna().mean() * 100
    log(f"  신규 {c:16s} 커버리지 train {cov_t:5.1f}% / valid {cov_v:5.1f}%  "
        f"std {va[c].std():.5f}")

FEATS_ALL = FEATS79 + [NEW_REV, NEW_BRK]
Xtr_all, Xva_all = s12.build_matrix(tr, FEATS_ALL), s12.build_matrix(va, FEATS_ALL)
del tr, va
tick(f"준비 완료 — 학습 {len(Xtr_all):,} / 검증 {len(Xva_all):,}")
log(f"짝지음 기준: exp/65 d6_lr04 2시드 = {REF_LR04} "
    f"(79피처/it500/d6/lr0.04/l2=10, thread={NTHREAD})")
log("")

BASE_PRM = dict(iterations=500, depth=6, learning_rate=0.04)
r_ = float(y_va.mean())
DEN = r_ * (1 - r_)
res = {}

log("=" * 80)
log("CS 확장 측정")
log("=" * 80)
for name, feats in ARMS:
    cols = [c for c in feats if c not in s12.CAT] + list(s12.CAT)
    ptr = Pool(Xtr_all[cols], y_tr, cat_features=list(s12.CAT))
    pva = Pool(Xva_all[cols], cat_features=list(s12.CAT))
    ps = []
    for sd in SEEDS:
        t0 = time.time()
        m = CatBoostClassifier(**{**BASE_CB, "random_seed": sd}, **BASE_PRM).fit(ptr)
        p = m.predict_proba(pva)[:, 1]
        ps.append(p)
        tick(f"{name} seed={sd} {raw_score(p, y_va):8.1f}  ({time.time()-t0:.0f}s)")
        del m
    del ptr, pva
    ens = np.mean(ps, axis=0)
    np.save(f"lab/72_{name}.npy", ens.astype("float32"))
    scv = raw_score(ens, y_va)
    pen = 100000 * (ens.mean() - r_) ** 2 / DEN
    res[name] = (scv, scv + pen, pen, len(feats))
    log(f"  -> {name:11s} {len(feats)}피처 2seed {scv:8.1f} "
        f"(기준 {REF_LR04} 대비 {scv - REF_LR04:+.1f}) "
        f"| 변별력 {scv + pen:.1f} 벌점 {pen:.1f}")

log("")
log("=" * 80)
log("판정 (2024 폴드, 2시드 · 기준 839.4/79피처 · 임계 +15)")
log("=" * 80)
log(f"  {'기준 d6_lr04 (79)':22s} {REF_LR04:8.1f}")
for k, (scv, disc, pen, nf) in sorted(res.items(), key=lambda x: -x[1][0]):
    d = scv - REF_LR04
    mark = "★통과" if d >= 15 else ("      " if d > -15 else "  기각")
    log(f"  {k + f' ({nf})':22s} {scv:8.1f}  ({d:+6.1f}) {mark}  "
        f"변별력 {disc:.1f} / 벌점 {pen:.1f}")
if "cs_rev" in res and "cs_rev_brk" in res:
    log(f"\n  breaking 단독 기여: {res['cs_rev_brk'][0] - res['cs_rev'][0]:+.1f}")
log("")
log("  · CS 축은 지금까지 로컬→LB 전이가 거의 1:1 이었다 (+107.9 → +92.3)")
log("  · 통과 시 script.py P_RATES/CS_FEATS 확장 → MC04 위 독립 확증 → 배포")
log("=" * 80)
