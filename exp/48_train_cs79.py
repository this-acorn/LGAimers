# -*- coding: utf-8 -*-
"""
[48] CS79 배포 학습 — CS76 + 구종 3피처 (f_mixcg_fb/brk, f_pfb), CB 8시드

근거: exp/45 BOTH +18.6 (2시드, CS76 782.3 → 800.9). 초가법성 확인
  (MIX +1.8, PFB +6.0 단독 대비). 구종 라벨은 asof 차분 복원 — train 내 사용은
  공식 Q&A 명시 허용. 추론 입력은 P(구종|사전정보)와 학습 유래 성향표뿐.

단계: 1) 라벨 복원(전체)  2) 검증 4시드 (게이트: 800.9-30 이상)
      3) 최종 8시드 전체 학습 + 배포 테이블  4) 5행 검증
피처 로직 단일 원본 = submissions/submit12_src/script.py

★ venv311 (10스레드 — exp/47과 병렬):
  <venv311>/Scripts/python.exe -u exp/48_train_cs79.py   (~3.5시간)
"""

import importlib.util
import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier, Pool

T0 = time.time()
SEEDS_VAL = [42, 7, 123, 2024]
SEEDS_FIN = [42, 7, 123, 2024, 99, 555, 31337, 1]
CB_PRM = dict(iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=10.0,
              verbose=False, thread_count=10, allow_writing_files=False)
GATE = 800.9 - 30.0
K_MIX = 50.0

spec = importlib.util.spec_from_file_location("s12", "submissions/submit12_src/script.py")
s12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s12)
add_features, attach_cs, attach_pt = s12.add_features, s12.attach_cs, s12.attach_pt
build_matrix, CS_FEATS, PT_FEATS = s12.build_matrix, s12.CS_FEATS, s12.PT_FEATS
P_RATES, B_RATES, PFB_IN, CAT = s12.P_RATES, s12.B_RATES, s12.PFB_IN, s12.CAT


def log(*a):
    print(*a, flush=True)


def tick(m):
    log(f"  [{time.time()-T0:6.0f}s] {m}")


def raw_score(p, y):
    y = np.asarray(y, float)
    r = y.mean()
    return 100000.0 * (1.0 - float(np.mean((np.asarray(p, float) - y) ** 2)) / (r * (1 - r)))


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


def mix_career_table(labeled):
    """(pid, cg) 통산 성향표 — 배포용/검증용 공통 형식 (mix_tbl)"""
    t = labeled.dropna(subset=["lab_fb"])
    c = (t.groupby(["pitcher_id", "_cg"])
          .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
          .reset_index())
    o = (t.groupby("pitcher_id")
          .agg(po_n=("lab_fb", "size"), po_fb=("lab_fb", "sum"), po_brk=("lab_brk", "sum"))
          .reset_index())
    tbl = c.merge(o, on="pitcher_id")
    over_fb = (tbl.po_fb + 0.5 * K_MIX) / (tbl.po_n + K_MIX)
    over_brk = (tbl.po_brk + 0.3 * K_MIX) / (tbl.po_n + K_MIX)
    tbl["mix_fb"] = ((tbl.fb + K_MIX * over_fb) / (tbl.n + K_MIX)).astype("float32")
    tbl["mix_brk"] = ((tbl.brk + K_MIX * over_brk) / (tbl.n + K_MIX)).astype("float32")
    return tbl[["pitcher_id", "_cg", "mix_fb", "mix_brk"]]


def mix_asof_train(labeled):
    """학습 행용: (pid, season, cg) 시즌 as-of 성향 (exp/45와 동일 로직)"""
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


log(f"환경: python {sys.version.split()[0]} / numpy {np.__version__}")
log("train 로딩...")
df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
df.columns = [c.replace("﻿", "").strip() for c in df.columns]
for c in df.select_dtypes("float64").columns:
    df[c] = df[c].astype("float32")
tick(f"로딩 {df.shape}")

# ---- 1. 라벨 복원 (전체 train) ----
d = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index()
pid = d.pitcher_id.to_numpy()
n = d.asof_pitcher_n.fillna(0).to_numpy("float64")
same_next = (pid[1:] == pid[:-1]) & (np.abs(n[1:] - n[:-1] - 1) < 1e-6)
for k, col in [("lab_fb", "asof_pitcher_fastball_rate"),
               ("lab_brk", "asof_pitcher_breaking_rate")]:
    S = d[col].fillna(0).to_numpy("float64") * n
    lab = np.full(len(d), np.nan)
    lab[:-1] = np.where(same_next, np.round(S[1:] - S[:-1]), np.nan)
    lab = np.where((lab == 0) | (lab == 1), lab, np.nan)
    d[k] = lab
# succ 자체검증
Ss = d["asof_pitcher_success_rate"].fillna(0).to_numpy("float64") * n
ls = np.full(len(d), np.nan)
ls[:-1] = np.where(same_next, np.round(Ss[1:] - Ss[:-1]), np.nan)
m_ = ~np.isnan(ls)
agree = float(np.mean(ls[m_] == d.control_success.to_numpy("float64")[m_]))
log(f"★ 라벨 복원 검증: succ 일치율 {agree*100:.2f}%  (100% 근처여야 함)")
assert agree > 0.999, "복원 정밀도 미달"
rec = d.set_index("index").sort_index()
df["lab_fb"], df["lab_brk"] = rec["lab_fb"], rec["lab_brk"]
b_, s_ = df.balls_before.to_numpy(), df.strikes_before.to_numpy()
df["_cg"] = np.where(s_ > b_, 2, np.where(b_ > s_, 0, 1)).astype("int8")
tick(f"라벨 복원 완료 (커버리지 {df.lab_fb.notna().mean()*100:.1f}%)")

FEATS79 = None  # 아래에서 구성


def prep_stage(upto, va_rows):
    """학습(≤upto)·검증 행에 CS + 구종 피처 부착. 반환: tr, va, pfb_model, 배포용 테이블들"""
    hist = df[df.season <= upto]
    prior = float(hist["control_success"].mean())
    cp = build_const(hist, "pitcher_id", "asof_pitcher_n", P_RATES)
    cb_ = build_const(hist, "batter_id", "asof_batter_n", B_RATES)
    # PFB (≤upto 라벨만)
    pfb_rows = hist.dropna(subset=["lab_fb"])
    pfb = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                             verbose=False, thread_count=10,
                             allow_writing_files=False, random_seed=42)
    pfb.fit(pfb_rows[PFB_IN].fillna(-999), pfb_rows["lab_fb"].astype(int))
    # MIX
    mix_tr_tbl = mix_asof_train(hist)          # 학습 행: 시즌 as-of
    mix_dep_tbl = mix_career_table(hist)       # 검증/배포 행: 통산(≤upto)
    # 학습 행 부착 (시즌별 CS 상수 + 시즌 as-of MIX)
    tr_rows = df[df.season <= upto].reset_index(drop=True)
    parts = []
    for S in sorted(tr_rows.season.unique()):
        h = df[df.season <= S - 1]
        rows = tr_rows[tr_rows.season == S]
        if len(h) == 0:
            cpS = pd.DataFrame({"id": [], "N_end": [], **{f"S_{k}": [] for k in P_RATES}})
            cbS = pd.DataFrame({"id": [], "N_end": [], **{f"S_{k}": [] for k in B_RATES}})
        else:
            cpS = build_const(h, "pitcher_id", "asof_pitcher_n", P_RATES)
            cbS = build_const(h, "batter_id", "asof_batter_n", B_RATES)
        parts.append(attach_cs(rows, cpS, cbS))
    tr = add_features(pd.concat(parts).sort_index(), prior)
    key = tr[["pitcher_id", "season", "_cg"]].merge(
        mix_tr_tbl, on=["pitcher_id", "season", "_cg"], how="left")
    tr["f_mixcg_fb"] = key["mix_fb"].to_numpy("float32")
    tr["f_mixcg_brk"] = key["mix_brk"].to_numpy("float32")
    tr["f_pfb"] = pfb.predict_proba(tr[PFB_IN].fillna(-999))[:, 1].astype("float32")
    va = None
    if va_rows is not None:
        va = attach_pt(attach_cs(add_features(va_rows, prior), cp, cb_)
                       .rename(columns={}), mix_dep_tbl, pfb)
        va = add_features(va, prior) if "f_count_state" not in va.columns else va
    return tr, va, pfb, cp, cb_, mix_dep_tbl, prior


# ---- 2. 검증 (≤2023 → 2024, 4시드) ----
log("\n" + "=" * 80)
log("검증: 2019~2023 → 2024 (4시드, 게이트 " + f"{GATE:.1f})")
log("=" * 80)
va_raw = df[df.season == 2024].reset_index(drop=True)
tr, va, _, _, _, _, _ = prep_stage(2023, va_raw)
y_tr = df[df.season <= 2023]["control_success"].to_numpy()
y_va = va_raw["control_success"].to_numpy()
BASE = [c for c in pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1)
        .columns.str.replace("﻿", "").str.strip() if c != "row_id"]
ENG18 = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
         "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
         "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
         "f_pb_diff", "f_miss_p", "f_miss_prev1"]
FEATS79 = BASE + ENG18 + CS_FEATS + PT_FEATS
log(f"피처 {len(FEATS79)}개")
Xtr = build_matrix(tr, FEATS79)
Xva = build_matrix(va, FEATS79)
ptr = Pool(Xtr, y_tr, cat_features=list(CAT))
pva = Pool(Xva, cat_features=list(CAT))
del tr, va, Xtr, Xva
acc = np.zeros(len(y_va))
for i, sd in enumerate(SEEDS_VAL, 1):
    t0 = time.time()
    m = CatBoostClassifier(**CB_PRM, random_seed=sd).fit(ptr)
    acc += m.predict_proba(pva)[:, 1]
    tick(f"[검증] seed={sd} ({i}/{len(SEEDS_VAL)}, {time.time()-t0:.0f}s)")
s_val = raw_score(acc / len(SEEDS_VAL), y_va)
log(f"\n  4시드 앙상블 = {s_val:.1f}  (exp/45 2시드 800.9)")
if s_val < GATE:
    log("  게이트 미달 — 중단")
    raise SystemExit(1)
log("  게이트 통과")
del ptr, pva

# ---- 3. 최종 (전체, 8시드) + 배포 테이블 ----
log("\n" + "=" * 80)
log("최종: 2019~2024 전체 8시드 + 배포 테이블")
log("=" * 80)
tr, _, pfb_fin, cp_fin, cb_fin, mix_fin, prior_fin = prep_stage(2024, None)
y_all = df["control_success"].to_numpy()
Xtr = build_matrix(tr, FEATS79)
ptr = Pool(Xtr, y_all, cat_features=list(CAT))
del tr, Xtr
models = []
for i, sd in enumerate(SEEDS_FIN, 1):
    t0 = time.time()
    models.append(CatBoostClassifier(**CB_PRM, random_seed=sd).fit(ptr))
    tick(f"[최종] seed={sd} ({i}/8, {time.time()-t0:.0f}s)")
del ptr

bundle = {"cb_models": models, "feats": FEATS79, "prior": prior_fin,
          "cs_const_p": cp_fin, "cs_const_b": cb_fin,
          "mix_tbl": mix_fin, "pfb_model": pfb_fin,
          "seeds": SEEDS_FIN, "version": "cs79"}
os.makedirs("submissions/submit12_src/model", exist_ok=True)
joblib.dump(bundle, "submissions/submit12_src/model/model.pkl", compress=3)
tick(f"저장: submissions/submit12_src/model/model.pkl "
     f"({os.path.getsize('submissions/submit12_src/model/model.pkl')/1024**2:.1f} MB)")

# ---- 4. 5행 검증 ----
b2 = joblib.load("submissions/submit12_src/model/model.pkl")
t5 = pd.read_csv("data/test.csv", encoding="utf-8-sig")
t5.columns = [c.replace("﻿", "").strip() for c in t5.columns]
ft5 = s12.attach_pt(s12.attach_cs(s12.add_features(t5, b2["prior"]),
                                  b2["cs_const_p"], b2["cs_const_b"]),
                    b2["mix_tbl"], b2["pfb_model"])
X5 = s12.build_matrix(ft5, b2["feats"])
acc5 = np.zeros(len(t5))
for m in b2["cb_models"]:
    acc5 += m.predict_proba(X5)[:, 1]
p5 = acc5 / len(b2["cb_models"])
log(f"  5행 예측 = {np.round(p5, 6).tolist()}")
log(f"  f_pfb = {np.round(ft5['f_pfb'].to_numpy(), 3).tolist()}")
log(f"  f_mixcg_fb 커버 {ft5['f_mixcg_fb'].notna().sum()}/5")
log(f"\n총 {time.time()-T0:.0f}초. 다음: exp/49 artifacts/submissions/submit12.zip 빌드")
