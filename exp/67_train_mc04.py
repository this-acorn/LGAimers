# -*- coding: utf-8 -*-
"""
[67] MC04 배포 학습 — 멀티클래스 5클래스 + lr0.04 (exp/65·66 확증), 8시드

근거: exp/52 타겟 해부(성공/미들/리버스/미들∩리버스/빅미스 상호배타 분할) →
      exp/53 2024 폴드 +36.1 (837.0 vs 이진 800.9, 시드 각각 817.6/812.2).
피처·전처리 = exp/48(CS79)와 동일. 타겟만 복원 라벨 5클래스. 제출 확률 = P(클래스0).
검증 스테이지 생략 (exp/53이 동일 코드 경로로 검증) — 전체 학습 8시드만.
피처 로직 단일 원본 = submit16_src/script.py

★ venv311:  <venv311>/Scripts/python.exe -u exp/54_train_mc79.py   (~6.5시간, keep-awake 필수)
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
SEEDS = [42, 7, 123, 2024, 99, 555, 31337, 1]
CB_PRM = dict(iterations=500, depth=6, learning_rate=0.04, l2_leaf_reg=10.0,
              verbose=False, thread_count=14, allow_writing_files=False,
              loss_function="MultiClass")
K_MIX = 50.0

spec = importlib.util.spec_from_file_location("s14", 'submissions/submit16_src/script.py')
s16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s16)
add_features, attach_cs, attach_pt = s16.add_features, s16.attach_cs, s16.attach_pt
build_matrix, CS_FEATS, PT_FEATS = s16.build_matrix, s16.CS_FEATS, s16.PT_FEATS
P_RATES, B_RATES, PFB_IN, CAT = s16.P_RATES, s16.B_RATES, s16.PFB_IN, s16.CAT


def log(*a):
    print(*a, flush=True)


def tick(m):
    log(f"  [{time.time()-T0:6.0f}s] {m}")


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

# ---- 라벨 복원: 구종(fb/brk) + 실패유형(mid/rev) ----
d = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index()
pid = d.pitcher_id.to_numpy()
n = d.asof_pitcher_n.fillna(0).to_numpy("float64")
nxt = (pid[1:] == pid[:-1]) & (np.abs(n[1:] - n[:-1] - 1) < 1e-6)
for k, col in [("lab_fb", "asof_pitcher_fastball_rate"),
               ("lab_brk", "asof_pitcher_breaking_rate"),
               ("lab_mid", "asof_pitcher_middle_rate"),
               ("lab_rev", "asof_pitcher_reverse_rate")]:
    S = d[col].fillna(0).to_numpy("float64") * n
    lab = np.full(len(d), np.nan)
    lab[:-1] = np.where(nxt, np.round(S[1:] - S[:-1]), np.nan)
    d[k] = np.where((lab == 0) | (lab == 1), lab, np.nan)
Ss = d["asof_pitcher_success_rate"].fillna(0).to_numpy("float64") * n
ls = np.full(len(d), np.nan)
ls[:-1] = np.where(nxt, np.round(Ss[1:] - Ss[:-1]), np.nan)
mm = ~np.isnan(ls)
agree = float(np.mean(ls[mm] == d.control_success.to_numpy("float64")[mm]))
log(f"★ 라벨 복원 검증: succ 일치율 {agree*100:.2f}%")
assert agree > 0.999
rec = d.set_index("index").sort_index()
for k in ["lab_fb", "lab_brk", "lab_mid", "lab_rev"]:
    df[k] = rec[k]
y_bin = df.control_success.to_numpy("float64")
ok = df.lab_mid.notna().to_numpy() & df.lab_rev.notna().to_numpy()
m_ = df.lab_mid.to_numpy() == 1
r_ = df.lab_rev.to_numpy() == 1
cls = np.full(len(df), -1, dtype="int8")
cls[ok & (y_bin == 1)] = 0
cls[ok & (y_bin == 0) & m_ & ~r_] = 1
cls[ok & (y_bin == 0) & ~m_ & r_] = 2
cls[ok & (y_bin == 0) & m_ & r_] = 3
cls[ok & (y_bin == 0) & ~m_ & ~r_] = 4
df["_cls"] = cls
log("클래스 분포: " + " ".join(f"{c}:{np.mean(cls==c)*100:.1f}%" for c in range(5))
    + f"  미복원 {np.mean(cls==-1)*100:.2f}%")
b_, s_ = df.balls_before.to_numpy(), df.strikes_before.to_numpy()
df["_cg"] = np.where(s_ > b_, 2, np.where(b_ > s_, 0, 1)).astype("int8")

BASE = [c for c in pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1)
        .columns.str.replace("﻿", "").str.strip() if c != "row_id"]
ENG18 = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
         "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
         "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
         "f_pb_diff", "f_miss_p", "f_miss_prev1"]
FEATS79 = BASE + ENG18 + CS_FEATS + PT_FEATS
log(f"피처 {len(FEATS79)}개")

# ---- 최종: 전체(≤2024) 학습 + 배포 테이블 ----
log("\n" + "=" * 80)
log("최종: 2019~2024 전체, MultiClass 8시드 + 배포 테이블")
log("=" * 80)
prior = float(df["control_success"].mean())
cp_fin = build_const(df, "pitcher_id", "asof_pitcher_n", P_RATES)
cb_fin = build_const(df, "batter_id", "asof_batter_n", B_RATES)
pfb_rows = df.dropna(subset=["lab_fb"])
pfb_fin = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                             verbose=False, thread_count=14,
                             allow_writing_files=False, random_seed=42)
pfb_fin.fit(pfb_rows[PFB_IN].fillna(-999), pfb_rows["lab_fb"].astype(int))
mix_tr = mix_asof_train(df)
mix_fin = mix_career_table(df)
tick("배포 테이블·PFB 준비")

parts = []
for S in sorted(df.season.unique()):
    h = df[df.season <= S - 1]
    rows = df[df.season == S]
    if len(h) == 0:
        cpS = pd.DataFrame({"id": [], "N_end": [], **{f"S_{k}": [] for k in P_RATES}})
        cbS = pd.DataFrame({"id": [], "N_end": [], **{f"S_{k}": [] for k in B_RATES}})
    else:
        cpS = build_const(h, "pitcher_id", "asof_pitcher_n", P_RATES)
        cbS = build_const(h, "batter_id", "asof_batter_n", B_RATES)
    parts.append(attach_cs(rows, cpS, cbS))
tr = add_features(pd.concat(parts).sort_index(), prior)
key = tr[["pitcher_id", "season", "_cg"]].merge(
    mix_tr, on=["pitcher_id", "season", "_cg"], how="left")
tr["f_mixcg_fb"] = key["mix_fb"].to_numpy("float32")
tr["f_mixcg_brk"] = key["mix_brk"].to_numpy("float32")
tr["f_pfb"] = pfb_fin.predict_proba(tr[PFB_IN].fillna(-999))[:, 1].astype("float32")
del parts
msk = tr["_cls"].to_numpy() >= 0
log(f"학습 행 (라벨 복원됨): {msk.sum():,} / {len(tr):,}")
Xtr = build_matrix(tr[msk], FEATS79)
ptr = Pool(Xtr, tr["_cls"].to_numpy()[msk], cat_features=list(CAT))
del tr, Xtr
tick("Pool 생성 완료")

models = []
for i, sd in enumerate(SEEDS, 1):
    t0 = time.time()
    models.append(CatBoostClassifier(**CB_PRM, random_seed=sd).fit(ptr))
    tick(f"[최종] seed={sd} ({i}/8, {time.time()-t0:.0f}s)")
del ptr

bundle = {"cb_models": models, "feats": FEATS79, "prior": prior,
          "cs_const_p": cp_fin, "cs_const_b": cb_fin,
          "mix_tbl": mix_fin, "pfb_model": pfb_fin,
          "seeds": SEEDS, "version": "mc04",
          "classes": "0=성공 1=미들 2=리버스 3=미들∩리버스 4=빅미스"}
os.makedirs('submissions/submit16_src/model', exist_ok=True)
joblib.dump(bundle, 'submissions/submit16_src/model/model.pkl', compress=3)
tick(f"저장: submit16_src/model/model.pkl "
     f"({os.path.getsize('submissions/submit16_src/model/model.pkl')/1024**2:.1f} MB)")

# ---- 5행 검증 ----
b2 = joblib.load('submissions/submit16_src/model/model.pkl')
t5 = pd.read_csv("data/test.csv", encoding="utf-8-sig")
t5.columns = [c.replace("﻿", "").strip() for c in t5.columns]
ft5 = s16.attach_pt(s16.attach_cs(s16.add_features(t5, b2["prior"]),
                                  b2["cs_const_p"], b2["cs_const_b"]),
                    b2["mix_tbl"], b2["pfb_model"])
X5 = s16.build_matrix(ft5, b2["feats"])
pr = np.mean([m.predict_proba(X5) for m in b2["cb_models"]], axis=0)
log(f"  5행 클래스 확률 (성공/미들/리버스/둘다/빅미스):")
for row in np.round(pr, 3):
    log(f"    {row.tolist()}")
log(f"  P(성공) = {np.round(pr[:, 0], 6).tolist()}")
log(f"\n총 {time.time()-T0:.0f}초. 다음: exp/68 submit16.zip 빌드")
