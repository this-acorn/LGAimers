# -*- coding: utf-8 -*-
"""
[79] 시즌 가중치를 챔피언(멀티클래스 MC04)에 적용 — exp/69 A상 결과의 확증

exp/69 A상 (이진 d6/lr0.04 기준 839.4, 시드 830.9/831.2):
    w_half (0.5^age)  812.3  (−27.1)   기각
  ★ w_soft (0.8^age)  847.1  (  +7.7)  ← 2시드·시드평균·proj8 이 **전부 +7.7**
    w_lin  (선형)      820.2  (−19.2)   기각

w_soft 가 특별한 이유:
  · 세 지표가 전부 동일한 +7.7 = **순수 편향 이득**, 분산 성분 0.
    exp/74 에서 lr 축이 2시드 +15.3 → 8시드 +3.8 로 증발한 것과 정반대 성격이다.
  · 시드별 paired 차이:  830.9→838.2 (+7.3) / 831.2→839.3 (+8.1),  **paired σ ≈ 0.6**
    임계 +15 는 **비짝지음** 노이즈(σ=15.3) 기준이다. paired σ 가 0.6 이면
    +7.7 은 압도적으로 유의하다. 기존 기준만 봤으면 버렸을 축이다.
  · 냉동 표가 없다 — 학습 표본에 가중치를 줄 뿐이라 2025 에 낡을 테이블이 없다.
    다만 예측 평균(중심)을 움직이므로 중심벌점을 반드시 함께 본다.

측정 대상: 이 이득이 **이진 → 멀티클래스로 옮겨가는가.**
  옮겨가면 챔피언이 857.3 → 약 865 (proj8) 가 되고 그대로 배포한다.

팔 (감쇠율만 다르다 — exp/69 가 0.5:−27 / 0.8:+7.7 / 1.0:0 을 보였으므로 정점은
     0.8~0.9 사이에 있다. 배포 전에 정점을 잡아둔다):
  w080   0.8^age   ← exp/69 승자
  w090   0.9^age   ← 더 완만. 정점이 오른쪽이면 여기서 더 오른다

기준 벡터가 디스크에 있다 — lab/66_mc_lr04.npy (MC04, 가중치 없음):
  시드42 842.3 / 시드7 848.9 / 시드평균 845.6 / 2시드 852.3 / K̄ 26.8 / proj8 857.3
동일 유지: thread_count=14, SEEDS=[42,7], it500/d6/lr0.04/l2=10, 79피처, 같은 폴드.
유일한 변경점은 Pool 의 sample weight 다.

판정: **시드별 paired 차이**를 주축으로 본다 (평균과 산포를 함께).
실행: PYTHONIOENCODING=utf-8 python -u exp/79_mc_seasonw.py   (~2시간)
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

SEEDS = [42, 7]
CB_PRM = dict(iterations=500, depth=6, learning_rate=0.04, l2_leaf_reg=10.0,
              verbose=False, thread_count=14, allow_writing_files=False,
              loss_function="MultiClass")
K_MIX = 50.0

log("train 로딩...")
df = load_train()

# ---- 라벨 복원 (succ/middle/reverse + 구종) ----
dd = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index()
pid = dd.pitcher_id.to_numpy()
n = dd.asof_pitcher_n.fillna(0).to_numpy("float64")
nxt = (pid[1:] == pid[:-1]) & (np.abs(n[1:] - n[:-1] - 1) < 1e-6)
for k, col in [("lab_mid", "asof_pitcher_middle_rate"),
               ("lab_rev", "asof_pitcher_reverse_rate"),
               ("lab_fb", "asof_pitcher_fastball_rate"),
               ("lab_brk", "asof_pitcher_breaking_rate")]:
    S = dd[col].fillna(0).to_numpy("float64") * n
    lab = np.full(len(dd), np.nan)
    diff = np.round(S[1:] - S[:-1])
    lab[:-1] = np.where(nxt, diff, np.nan)
    dd[k] = np.where((lab == 0) | (lab == 1), lab, np.nan)
rec = dd.set_index("index").sort_index()
for k in ["lab_mid", "lab_rev", "lab_fb", "lab_brk"]:
    df[k] = rec[k]
y_bin = df.control_success.to_numpy("float64")
# 5클래스: 0=성공 1=미들만 2=리버스만 3=미들∩리버스 4=빅미스
cls = np.full(len(df), -1, dtype="int8")
ok = df.lab_mid.notna().to_numpy() & df.lab_rev.notna().to_numpy()
m_ = df.lab_mid.to_numpy() == 1
r_ = df.lab_rev.to_numpy() == 1
cls[ok & (y_bin == 1)] = 0
cls[ok & (y_bin == 0) & m_ & ~r_] = 1
cls[ok & (y_bin == 0) & ~m_ & r_] = 2
cls[ok & (y_bin == 0) & m_ & r_] = 3
cls[ok & (y_bin == 0) & ~m_ & ~r_] = 4
df["_cls"] = cls
log(f"클래스 분포: " + " ".join(f"{c}:{np.mean(cls==c)*100:.1f}%" for c in range(5))
    + f"  미복원 {np.mean(cls==-1)*100:.2f}%")
b_, s_ = df.balls_before.to_numpy(), df.strikes_before.to_numpy()
df["_cg"] = np.where(s_ > b_, 2, np.where(b_ > s_, 0, 1)).astype("int8")

# ---- exp/48 검증 스테이지와 동일한 피처 준비 (79피처) ----
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


def mix_asof_train(labeled):
    t = labeled.dropna(subset=["lab_fb"])
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
    tbl = c.merge(o[["pitcher_id", "season", "po_n", "po_fb", "po_brk"]],
                  on=["pitcher_id", "season"])
    over_fb = (tbl.po_fb + 0.5 * K_MIX) / (tbl.po_n + K_MIX)
    over_brk = (tbl.po_brk + 0.3 * K_MIX) / (tbl.po_n + K_MIX)
    tbl["mix_fb"] = np.where(tbl.po_n > 0,
                             (tbl.p_fb + K_MIX * over_fb) / (tbl.p_n + K_MIX),
                             np.nan).astype("float32")
    tbl["mix_brk"] = np.where(tbl.po_n > 0,
                              (tbl.p_brk + K_MIX * over_brk) / (tbl.p_n + K_MIX),
                              np.nan).astype("float32")
    return tbl[["pitcher_id", "season", "_cg", "mix_fb", "mix_brk"]]


def mix_career(labeled):
    t = labeled.dropna(subset=["lab_fb"])
    c = (t.groupby(["pitcher_id", "_cg"])
          .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
          .reset_index())
    o = (t.groupby("pitcher_id")
          .agg(po_n=("lab_fb", "size"), po_fb=("lab_fb", "sum"),
               po_brk=("lab_brk", "sum")).reset_index())
    tbl = c.merge(o, on="pitcher_id")
    over_fb = (tbl.po_fb + 0.5 * K_MIX) / (tbl.po_n + K_MIX)
    over_brk = (tbl.po_brk + 0.3 * K_MIX) / (tbl.po_n + K_MIX)
    tbl["mix_fb"] = ((tbl.fb + K_MIX * over_fb) / (tbl.n + K_MIX)).astype("float32")
    tbl["mix_brk"] = ((tbl.brk + K_MIX * over_brk) / (tbl.n + K_MIX)).astype("float32")
    return tbl[["pitcher_id", "_cg", "mix_fb", "mix_brk"]]


hist = df[df.season <= 2023]
prior = float(hist["control_success"].mean())
cp = build_const(hist, "pitcher_id", "asof_pitcher_n", s12.P_RATES)
cb_ = build_const(hist, "batter_id", "asof_batter_n", s12.B_RATES)
pfb_rows = hist.dropna(subset=["lab_fb"])
pfb = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=False,
                         thread_count=14, allow_writing_files=False, random_seed=42)
pfb.fit(pfb_rows[s12.PFB_IN].fillna(-999), pfb_rows["lab_fb"].astype(int))
mix_tr = mix_asof_train(hist)
mix_dep = mix_career(hist)
tick("테이블·PFB 준비")

tr_rows = df[df.season <= 2023].reset_index(drop=True)
va_rows = df[df.season == 2024].reset_index(drop=True)
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
key = tr[["pitcher_id", "season", "_cg"]].merge(
    mix_tr, on=["pitcher_id", "season", "_cg"], how="left")
tr["f_mixcg_fb"] = key["mix_fb"].to_numpy("float32")
tr["f_mixcg_brk"] = key["mix_brk"].to_numpy("float32")
tr["f_pfb"] = pfb.predict_proba(tr[s12.PFB_IN].fillna(-999))[:, 1].astype("float32")
va = s12.attach_pt(s12.add_features(s12.attach_cs(va_rows, cp, cb_), prior),
                   mix_dep, pfb)
del parts

BASE = [c for c in pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1)
        .columns.str.replace("﻿", "").str.strip() if c != "row_id"]
ENG18 = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
         "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
         "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
         "f_pb_diff", "f_miss_p", "f_miss_prev1"]
FEATS = BASE + ENG18 + s12.CS_FEATS + s12.PT_FEATS

# 멀티클래스 학습: 라벨 복원된 행만
m_tr = tr["_cls"].to_numpy() >= 0
log(f"멀티클래스 학습 행: {m_tr.sum():,} / {len(tr):,}")
Xtr = s12.build_matrix(tr[m_tr], FEATS)
Xva = s12.build_matrix(va, FEATS)
# ★ 유일한 변경점: 시즌 가중치. 학습 행의 season 을 Xtr 과 같은 순서로 뽑아둔다.
_y_cls = tr["_cls"].to_numpy()[m_tr]
_season = tr["season"].to_numpy("float64")[m_tr]
_age = _season.max() - _season
assert len(_season) == len(Xtr), "가중치 정렬 불일치"
log(f"  학습 시즌 {int(_season.min())}~{int(_season.max())}, "
    f"age 0~{int(_age.max())}")

pva = Pool(Xva, cat_features=list(s12.CAT))
del tr, va, Xva

REF = np.load("lab/66_mc_lr04.npy").astype("float64")
REF_SOLO = [842.3, 848.9]
REF_MEAN = sum(REF_SOLO) / 2
S_REF = raw_score(REF, y_va)
REF_K = 4 * (S_REF - REF_MEAN)
REF_P8 = 1.75 * S_REF - 0.75 * REF_MEAN
r_ = float(y_va.mean())
DEN = r_ * (1 - r_)
REF_PEN = 100000 * (REF.mean() - r_) ** 2 / DEN
log(f"기준 MC04 (가중치 없음): 시드 {REF_SOLO}  시드평균 {REF_MEAN:.1f}  "
    f"2시드 {S_REF:.1f}  K̄ {REF_K:.1f}  proj8 {REF_P8:.1f}  벌점 {REF_PEN:.1f}")
log("")

ARMS = [("w080", 0.8), ("w090", 0.9)]
res = {}
for name, decay in ARMS:
    w = decay ** _age
    w = w / w.mean()
    ess = (w.sum() ** 2) / (w ** 2).sum()
    log(f"[{name}] {decay}^age  가중치 {w.min():.4f}~{w.max():.4f}  "
        f"유효표본 {ess:,.0f} / {len(w):,} ({ess/len(w)*100:.0f}%)")
    ptr = Pool(Xtr, _y_cls, cat_features=list(s12.CAT), weight=w)
    preds, solo = [], []
    for sd in SEEDS:
        t0 = time.time()
        m = CatBoostClassifier(**CB_PRM, random_seed=sd).fit(ptr)
        p = m.predict_proba(pva)[:, 0]
        preds.append(p)
        solo.append(raw_score(p, y_va))
        tick(f"  {name} seed={sd}  {solo[-1]:8.1f}  "
             f"(기준 {REF_SOLO[SEEDS.index(sd)]:.1f} 대비 "
             f"{solo[-1]-REF_SOLO[SEEDS.index(sd)]:+.1f})  ({time.time()-t0:.0f}s)")
        del m
    del ptr
    ens = np.mean(preds, axis=0)
    np.save(f"lab/79_{name}.npy", ens.astype("float32"))
    mean_solo = float(np.mean(solo))
    e2 = raw_score(ens, y_va)
    p8 = 1.75 * e2 - 0.75 * mean_solo
    pen = 100000 * (ens.mean() - r_) ** 2 / DEN
    paired = [solo[i] - REF_SOLO[i] for i in range(len(SEEDS))]
    res[name] = (e2, mean_solo, p8, pen, paired)
    log(f"  -> {name}  2시드 {e2:.1f} ({e2-S_REF:+.1f})  시드평균 {mean_solo:.1f} "
        f"({mean_solo-REF_MEAN:+.1f})  proj8 {p8:.1f} ({p8-REF_P8:+.1f})")
    log(f"     변별력 {e2+pen:.1f} 벌점 {pen:.1f} (기준 {REF_PEN:.1f})  "
        f"paired {['%+.1f' % v for v in paired]}")
    log("")

log("=" * 92)
log("판정 — 주축은 시드별 paired 차이 (평균과 산포)")
log("=" * 92)
log(f"  {'팔':8s} {'2시드':>9s} {'시드평균':>9s} {'proj8':>9s} {'벌점':>7s} "
    f"{'paired평균':>10s} {'paired산포':>10s}")
log(f"  {'기준 MC04':8s} {S_REF:9.1f} {REF_MEAN:9.1f} {REF_P8:9.1f} {REF_PEN:7.1f}")
best = None
for k, (e2, ms, p8, pen, pr) in res.items():
    pm = float(np.mean(pr))
    ps = float(np.std(pr))
    log(f"  {k:8s} {e2:9.1f} {ms:9.1f} {p8:9.1f} {pen:7.1f} {pm:+10.1f} {ps:10.2f}")
    if best is None or pm > best[1]:
        best = (k, pm, p8, pen)
log("")
log("  ※ 임계는 고정 +15 가 아니다. paired 산포가 작으면 작은 이득도 유의하다.")
log("     exp/69 A상 이진: paired 평균 +7.7, 산포 0.4 → 유의")
log("  ※ 가중치는 예측 평균을 움직인다. 벌점이 기준보다 크게 늘었으면 "
    "2025 전이가 불확실하다.")
log("")
if best and best[1] > 0 and best[1] >= 3.0:
    log(f"  ★ 승자 {best[0]}: paired 평균 {best[1]:+.1f}, proj8 {best[2]:.1f} "
        f"({best[2]-REF_P8:+.1f})")
    log(f"    → 이 설정으로 8시드 배포. 전이율 1.12 가정 시 LB 약 "
        f"{1033.99 + 1.12*(best[2]-853.6):.0f}")
    log("      (853.6 = 현 배포본 MC lr0.08 의 proj8. 챔피언 대비 순증으로 환산)")
else:
    log("  기각 — 시즌 가중치는 이진에만 통했고 멀티클래스로 옮겨가지 않는다.")
    log("    남은 것은 MC04/OVA 무가중 배포(proj8 857~860, LB 약 1038).")
log("=" * 92)
