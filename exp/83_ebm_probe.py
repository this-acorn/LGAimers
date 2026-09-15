# -*- coding: utf-8 -*-
"""
[83] EBM 다양성 파트너 프로브 — 40만 행 서브샘플 짝지음 (챔피언 후보 아님)

왜 재는가 (exp/77·78 의 결론에서 직접 도출):
  · 우리는 사실상 단일 모델이다. 혼합 이득의 닫힌 해:
        gain = (d + K)^2 / (4K)   — 파트너는 이길 필요가 없다. K 가 크면 져도 된다.
  · OVA 는 "손실함수만 바꾸면 같은 함수로 수렴"해서 죽었다 (K_local 11.7).
    EBM(GA2M: 피처별 형태함수 + 쌍 상호작용 선형합)은 **함수 클래스 자체가 달라서**
    K 가 진짜로 클 수 있는 유일하게 남은 후보다.
  · 챔피언으로는 가망 없다 — 이 문제는 고차 상호작용이 크다(매치업 +107).
    질문은 오직 "d 가 K 를 살릴 만큼 얕은가"이다.

설계 (전체 학습은 수 시간이라 서브샘플 삼단 논법):
  1) 같은 40만 행에서 CatBoost(it500/d6/lr0.04, thread 8, seed 42)와 EBM 을 나란히 학습
     -> 함수클래스 격차 gap 을 짝지음으로 측정
  2) 같은 설정 CatBoost 의 전체 122만 행 점수는 이미 있다 (exp/65 seed42 = 830.9)
     -> 데이터 3배의 값 scale_gain 을 실측으로 안다
  3) 낙관 가정(EBM 도 데이터 3배에서 CatBoost 만큼 번다 — 가법모델은 보통 더 일찍
     포화하므로 이건 EBM 에 유리한 상한)으로 전체학습 EBM 점수를 추정
     -> 그 상한으로도 MC04 대비 혼합이득 < +15 면 영구 종결

판정:
  · 낙관 시나리오 혼합이득 < +15  ->  기각, 축 영구 종결
  · >= +15  ->  전체 학습 승격. 단 HGB 전례 경고 — 파트너 계열 자체의 2025 전이가
    나쁘면(HGB −68) 로컬 K 가 아무리 커도 무효. 승격해도 소량 가중으로만.

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/83_ebm_probe.py   (~1..1.5시간)
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
FEATS = BASE + ENG18 + s12.CS_FEATS + s12.PT_FEATS
Xtr, Xva = s12.build_matrix(tr, FEATS), s12.build_matrix(va, FEATS)
del tr, va
tick(f"준비 완료 — 학습 {len(Xtr):,} / 검증 {len(Xva):,}, 피처 {len(FEATS)}")

# ===========================================================================
log("")
log("=" * 88)
log("A. 서브샘플 짝지음 — 같은 40만 행에서 CatBoost vs EBM")
log("=" * 88)
N_SUB = 400_000
rng = np.random.default_rng(42)
idx = np.sort(rng.choice(len(Xtr), size=N_SUB, replace=False))
Xs = Xtr.iloc[idx].reset_index(drop=True)
ys = y_tr[idx]
log(f"  서브샘플 {N_SUB:,} / {len(Xtr):,}  성공률 {ys.mean():.4f}")

# ---- 프로브 키트 내보내기 (팀 4인 분산 측정용 — 전원 같은 행·같은 기준으로 잰다) ----
import os as _os
import shutil as _shutil
KIT = "probe_kit"
_os.makedirs(KIT, exist_ok=True)
Xs.to_parquet(f"{KIT}/X_train_sub.parquet")
np.save(f"{KIT}/y_train_sub.npy", ys.astype("int8"))
Xva.to_parquet(f"{KIT}/X_valid.parquet")
np.save(f"{KIT}/y_valid.npy", y_va.astype("int8"))
np.save(f"{KIT}/sub_idx.npy", idx)
_shutil.copy("lab/66_mc_lr04.npy", f"{KIT}/mc04_ref.npy")
_sz = sum(_os.path.getsize(f"{KIT}/{_f}") for _f in _os.listdir(KIT)) / 1024 ** 2
log(f"  ★ 프로브 키트 저장: {KIT}/ ({_sz:.0f} MB) — 팀원에게 통째로 전달, "
    f"채점은 score_probe.py")

r_ = float(y_va.mean())
DEN = r_ * (1 - r_)
pva = Pool(Xva, cat_features=list(s12.CAT))

t0 = time.time()
cb = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.04,
                        l2_leaf_reg=10.0, verbose=False, thread_count=NTHREAD,
                        allow_writing_files=False, random_seed=42)
cb.fit(Pool(Xs, ys, cat_features=list(s12.CAT)))
p_cb = cb.predict_proba(pva)[:, 1]
s_cb_sub = raw_score(p_cb, y_va)
ANCHOR_FULL = 830.9   # exp/65 d6_lr04 seed42 — 전체 122만 행, 같은 설정·같은 thread 8
scale_gain = ANCHOR_FULL - s_cb_sub
tick(f"CatBoost(40만) {s_cb_sub:8.1f}   [전체 122만 동일설정 {ANCHOR_FULL} "
     f"→ 데이터 3배의 값 {scale_gain:+.1f}]  ({time.time()-t0:.0f}s)")

# ---- EBM ----
from interpret.glassbox import ExplainableBoostingClassifier


def ebm_frame(X):
    d = X.copy()
    for c in d.columns:
        if c not in s12.CAT:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(-999.0)
    return d


FT = ["nominal" if c in s12.CAT else "continuous" for c in Xs.columns]
t0 = time.time()
ebm = ExplainableBoostingClassifier(interactions=20, outer_bags=4, max_bins=256,
                                    feature_types=FT, n_jobs=6, random_state=42)
ebm.fit(ebm_frame(Xs), ys)
tick(f"EBM 학습 완료 ({time.time()-t0:.0f}s)")
p_ebm = ebm.predict_proba(ebm_frame(Xva))[:, 1].astype("float64")
s_ebm_sub = raw_score(p_ebm, y_va)
np.save("lab/83_ebm_sub.npy", p_ebm.astype("float32"))
pen_e = 100000 * (p_ebm.mean() - r_) ** 2 / DEN
tick(f"EBM(40만, pairwise 20) {s_ebm_sub:8.1f}   변별력 {s_ebm_sub+pen_e:.1f} "
     f"벌점 {pen_e:.1f}")

gap = s_ebm_sub - s_cb_sub
log(f"")
log(f"  ★ 함수클래스 격차 (같은 40만 행): EBM − CatBoost = {gap:+.1f}")

log("")
log("=" * 88)
log("B. MC04 대비 파트너 값 — d 와 K 를 함께")
log("=" * 88)
MC04 = np.load("lab/66_mc_lr04.npy").astype("float64")
S_MC = raw_score(MC04, y_va)
D = float(np.mean((p_ebm - MC04) ** 2))
K = 100000 * D / DEN
log(f"  MC04 2시드 {S_MC:.1f}   D(EBM, MC04) = {D:.4e}   K_local = {K:.1f}")
log(f"  (참고 K_local: CB65↔CS79 191.3 / 같은 계열끼리 42~50 / OVA 11.7 / 시드간 26.8)")

s_ebm_opt = s_ebm_sub + scale_gain
rows = [("현재 (40만 학습 그대로)", s_ebm_sub - S_MC),
        ("낙관 (전체학습 = CB만큼 증가 가정)", s_ebm_opt - S_MC)]
log(f"")
log(f"  {'시나리오':34s} {'d':>8s} {'w*':>7s} {'로컬 혼합이득':>12s} {'K25 환산이득':>12s}")
best_opt = 0.0
for tag, d in rows:
    if K > 0 and d > -K:
        g = (d + K) ** 2 / (4 * K)
        w = min(max(0.5 + d / (2 * K), 0.0), 1.0)
    else:
        g, w = 0.0, 0.0
    K25 = K * 0.614
    g25 = (d + K25) ** 2 / (4 * K25) if (K25 > 0 and d > -K25) else 0.0
    log(f"  {tag:34s} {d:+8.1f} {w:7.3f} {g:+12.1f} {g25:+12.1f}")
    if "낙관" in tag:
        best_opt = g

log("")
log("=" * 88)
log("판정 (임계: 낙관 시나리오의 로컬 혼합이득 +15)")
log("=" * 88)
if best_opt >= 15:
    log(f"  ★통과 후보 — 낙관 혼합이득 {best_opt:+.1f}. 전체 학습 승격을 검토하라.")
    log("    ⚠ 단 HGB 전례: 파트너 계열의 2025 전이가 −68이면 로컬 K는 무효다.")
    log("    승격해도 가중은 w* 이하 소량으로, 그리고 반드시 별도 제출로 전이를 실측하라.")
else:
    log(f"  기각 — 낙관 상한조차 {best_opt:+.1f} < +15. EBM 축 영구 종결.")
    log("    가법(+쌍) 함수클래스는 이 문제의 고차 상호작용을 못 담는다는 뜻이다.")
    log("    이로써 '다른 함수클래스 파트너' 탐색은 EBM까지 소진:")
    log("    LightGBM(9번째 시드) / HGB(전이 −68) / OVA(같은 함수) / 트리구조(−83) / EBM")
log("=" * 88)
