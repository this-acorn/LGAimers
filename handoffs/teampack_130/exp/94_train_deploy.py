# -*- coding: utf-8 -*-
"""
[94] 배포 학습 — 챔피언(cat5 · 5클래스 · lr0.08) + exp/93 채택 팔, 2019~2024 전체 8시드 (venv311)

  --arm A|B|C   : 조합 categorical 추가 (submit17_src/script.py 의 add_combos, cat_features 확장)
  --arm M4      : 조합 없음, 타겟만 4클래스 (클래스3 → lab/92_merge.json 의 병합 대상)
  --arm AM4 등  : A/B/C 와 M4 동시 (조합 + 4클래스)
  --arm none    : 챔피언 그대로 (하민 target5_teamcat 재현용)

★ 반드시 venv311 로 실행 (서버 = python 3.11 / numpy 1.26.4 / pandas 2.0.3, pyarrow 없음):
  PYTHONIOENCODING=utf-8 <venv311>/python.exe -u exp/94_train_deploy.py --arm A     (~5~7시간)
★ 실행 전 사용자 검토·승인 필수 (HANDOFF §1.15). 완료 후 exp/95_build_submit17.py 로 zip 빌드+검증.

피처 로직의 단일 원본 = submit17_src/script.py (여기서 import). 조합·CS·구종 피처는 전부 행 독립.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier, Pool

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="A")
ap.add_argument("--threads", type=int, default=14)
ap.add_argument("--seeds", default="42,7,123,2024,99,555,31337,1")
ap.add_argument("--smoke", action="store_true", help="8만행·20it·2시드로 배포 경로만 검증 (version 에 _smoke 표시)")
ap.add_argument("--iterations", type=int, default=500, help="CatBoost iterations (IT 학습곡선 정점 it=300 배포용 — exp/100 IT)")
ap.add_argument("--min-season", type=int, default=2019, help="이 시즌 이상만 학습 행으로 사용 (최근 레짐 백본 LB 프로브). 상수표는 전체 연도 유지")
ap.add_argument("--out-dir", default="submit17_src", help="번들 저장 폴더 (model/model.pkl). 프로브마다 다른 폴더를 주면 챔피언 번들과 충돌 없음")
args = ap.parse_args()
ARM = args.arm.upper()
SEEDS_FIN = [int(s) for s in args.seeds.split(",")]
if args.smoke:
    SEEDS_FIN = SEEDS_FIN[:2]

T0 = time.time()
assert sys.version_info[:2] == (3, 11) and np.__version__.startswith("1.26") and pd.__version__.startswith("2.0"), \
    f"venv311 로 실행해야 함 (현재 python {sys.version.split()[0]} numpy {np.__version__} pandas {pd.__version__})"
try:
    import pyarrow  # noqa
    raise SystemExit("이 환경에 pyarrow 가 있음 — 서버와 다름. venv311 확인")
except ImportError:
    pass

spec = importlib.util.spec_from_file_location("s17", "submit17_src/script.py")
s17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s17)

COMBO_A = ["c_team_matchup", "c_pteam_role", "c_bteam_role"]
COMBO_B = ["c_count_hand", "c_base_out"]
combo_cats = []
if "C" in ARM:
    combo_cats = COMBO_A + COMBO_B
elif "A" in ARM:
    combo_cats = COMBO_A
elif "B" in ARM:
    combo_cats = COMBO_B
FOUR = "M4" in ARM
MERGE = int(json.load(open("lab/92_merge.json"))["merge_into"]) if FOUR else None
CAT_ALL = list(s17.CAT) + list(s17.TEAM_CAT) + combo_cats
CB_PRM = dict(iterations=20 if args.smoke else args.iterations, depth=6, learning_rate=0.08, l2_leaf_reg=10.0,
              loss_function="MultiClass", verbose=False, thread_count=args.threads,
              allow_writing_files=False)
K_MIX = 50.0
VERSION = f"catcombo_{ARM}" + (f"_it{args.iterations}" if args.iterations != 500 else "") + (f"_from{args.min_season}" if args.min_season > 2019 else "") + ("_smoke" if args.smoke else "")


def log(*a):
    print(*a, flush=True)


def tick(m):
    log(f"  [{time.time()-T0:6.0f}s] {m}")


log(f"환경: python {sys.version.split()[0]} / numpy {np.__version__} / pandas {pd.__version__}")
log(f"arm={ARM}  combo_cats={combo_cats}  4클래스={FOUR}" + (f" (3→{MERGE})" if FOUR else "") + f"  cat_features={CAT_ALL}  threads={args.threads}  seeds={SEEDS_FIN}")


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
    tbl = c.merge(o[["pitcher_id", "season", "po_n", "po_fb", "po_brk"]], on=["pitcher_id", "season"])
    over_fb = (tbl.po_fb + 0.5 * K_MIX) / (tbl.po_n + K_MIX)
    over_brk = (tbl.po_brk + 0.3 * K_MIX) / (tbl.po_n + K_MIX)
    tbl["mix_fb"] = np.where(tbl.po_n > 0, (tbl.p_fb + K_MIX * over_fb) / (tbl.p_n + K_MIX), np.nan).astype("float32")
    tbl["mix_brk"] = np.where(tbl.po_n > 0, (tbl.p_brk + K_MIX * over_brk) / (tbl.p_n + K_MIX), np.nan).astype("float32")
    return tbl[["pitcher_id", "season", "_cg", "mix_fb", "mix_brk"]]


log("train 로딩...")
df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
df.columns = [c.replace("﻿", "").strip() for c in df.columns]
for c in df.select_dtypes("float64").columns:
    df[c] = df[c].astype("float32")
tick(f"로딩 {df.shape}")

# ---- 라벨 복원 (exp/73·89 와 동일 기법) ----
d = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index()
pid = d.pitcher_id.to_numpy()
n = d.asof_pitcher_n.fillna(0).to_numpy("float64")
same_next = (pid[1:] == pid[:-1]) & (np.abs(n[1:] - n[:-1] - 1) < 1e-6)
for k, col in [("lab_fb", "asof_pitcher_fastball_rate"), ("lab_brk", "asof_pitcher_breaking_rate"),
               ("lab_mid", "asof_pitcher_middle_rate"), ("lab_rev", "asof_pitcher_reverse_rate")]:
    S = d[col].fillna(0).to_numpy("float64") * n
    lab = np.full(len(d), np.nan)
    lab[:-1] = np.where(same_next, np.round(S[1:] - S[:-1]), np.nan)
    d[k] = np.where((lab == 0) | (lab == 1), lab, np.nan)
Ss = d["asof_pitcher_success_rate"].fillna(0).to_numpy("float64") * n
ls = np.full(len(d), np.nan)
ls[:-1] = np.where(same_next, np.round(Ss[1:] - Ss[:-1]), np.nan)
m_ = ~np.isnan(ls)
agree = float(np.mean(ls[m_] == d.control_success.to_numpy("float64")[m_]))
log(f"★ 라벨 복원 검증(succ): 일치율 {agree*100:.2f}%")
assert agree > 0.999
rec = d.set_index("index").sort_index()
for k in ["lab_fb", "lab_brk", "lab_mid", "lab_rev"]:
    df[k] = rec[k]
del d, rec
b_, s_ = df.balls_before.to_numpy(), df.strikes_before.to_numpy()
df["_cg"] = np.where(s_ > b_, 2, np.where(b_ > s_, 0, 1)).astype("int8")

y_bin = df["control_success"].to_numpy("float64")
cls = np.full(len(df), -1, dtype="int8")
ok = df.lab_mid.notna().to_numpy() & df.lab_rev.notna().to_numpy()
mm = df.lab_mid.to_numpy() == 1
rr = df.lab_rev.to_numpy() == 1
cls[ok & (y_bin == 1)] = 0
cls[ok & (y_bin == 0) & mm & ~rr] = 1
cls[ok & (y_bin == 0) & ~mm & rr] = 2
cls[ok & (y_bin == 0) & mm & rr] = 3
cls[ok & (y_bin == 0) & ~mm & ~rr] = 4
df["_cls"] = cls
log("클래스 분포(5): " + " ".join(f"{c}:{np.mean(cls==c)*100:.1f}%" for c in range(5)) + f"  미복원 {np.mean(cls==-1)*100:.2f}%")

# ---- 2019~2024 전체 학습 행 준비 (exp/73 prep_stage(2024) 와 동일) ----
hist = df
prior = float(hist["control_success"].mean())
cp = build_const(hist, "pitcher_id", "asof_pitcher_n", s17.P_RATES)
cb_ = build_const(hist, "batter_id", "asof_batter_n", s17.B_RATES)
pfb_rows = hist.dropna(subset=["lab_fb"])
pfb = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=False,
                         thread_count=args.threads, allow_writing_files=False, random_seed=42)
pfb.fit(pfb_rows[s17.PFB_IN].fillna(-999), pfb_rows["lab_fb"].astype(int))
del pfb_rows
mix_dep = mix_career_table(hist)
mix_tr = mix_asof_train(hist)
tr_rows = df.reset_index(drop=True)
parts = []
for S in sorted(tr_rows.season.unique()):
    h = df[df.season <= S - 1]
    rows = tr_rows[tr_rows.season == S]
    if len(h) == 0:
        cpS = pd.DataFrame({"id": [], "N_end": [], **{f"S_{k}": [] for k in s17.P_RATES}})
        cbS = pd.DataFrame({"id": [], "N_end": [], **{f"S_{k}": [] for k in s17.B_RATES}})
    else:
        cpS = build_const(h, "pitcher_id", "asof_pitcher_n", s17.P_RATES)
        cbS = build_const(h, "batter_id", "asof_batter_n", s17.B_RATES)
    parts.append(s17.attach_cs(rows, cpS, cbS))
tr = s17.add_features(pd.concat(parts).sort_index(), prior)
del parts
key = tr[["pitcher_id", "season", "_cg"]].merge(mix_tr, on=["pitcher_id", "season", "_cg"], how="left")
tr["f_mixcg_fb"] = key["mix_fb"].to_numpy("float32")
tr["f_mixcg_brk"] = key["mix_brk"].to_numpy("float32")
tr["f_pfb"] = pfb.predict_proba(tr[s17.PFB_IN].fillna(-999))[:, 1].astype("float32")
del key
tick("피처 부착 완료")

BASE = [c for c in pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1)
        .columns.str.replace("﻿", "").str.strip() if c != "row_id"]
ENG18 = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
         "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
         "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
         "f_pb_diff", "f_miss_p", "f_miss_prev1"]
FEATS = BASE + ENG18 + s17.CS_FEATS + s17.PT_FEATS + combo_cats
log(f"피처 {len(FEATS)}개 (79 + {len(combo_cats)}), cat_features {len(CAT_ALL)}개")

mask = tr["_cls"].to_numpy() >= 0
if args.min_season > 2019:
    # 최근 시즌만 학습 (LB 프로브용). CS 상수표·mix 표·PFB 는 위에서 전체 연도로 이미 만들어졌으므로 그대로 (2025 행의 N_end 는 전체 이력이어야 함)
    mask &= tr["season"].to_numpy() >= args.min_season
    log(f"★ 최근 시즌 필터: season >= {args.min_season} → 학습 행 {mask.sum():,}")
y_all = tr["_cls"].to_numpy()[mask].astype("int64")
if FOUR:
    y_all = np.where(y_all == 3, MERGE, y_all)
    remap = {c: i for i, c in enumerate(sorted(np.unique(y_all)))}
    assert remap[0] == 0 and len(remap) == 4
    y_all = np.array([remap[v] for v in y_all], dtype="int64")
    log(f"4클래스: 3→{MERGE}  remap {remap}  분포 " + " ".join(f"{i}:{np.mean(y_all==i)*100:.1f}%" for i in range(4)))
log(f"학습 행: {len(tr):,} → 미복원 {(~mask).sum()} 드롭 → {mask.sum():,}")
if args.smoke:
    idx = np.where(mask)[0][:80000]
    mask = np.zeros(len(tr), dtype=bool)
    mask[idx] = True
    y_all = y_all[:80000]
    log(f"★ SMOKE: 학습 행 {mask.sum():,} / it {CB_PRM['iterations']} / 시드 {SEEDS_FIN} — 배포 경로 검증 전용")
Xtr = s17.build_matrix(tr.loc[mask], FEATS, combo_cats)
if combo_cats:
    log("  조합 레벨 수: " + ", ".join(f"{c}={Xtr[c].nunique()}" for c in combo_cats))
ptr = Pool(Xtr, y_all, cat_features=CAT_ALL)
del tr, Xtr

models = []
for i, sd in enumerate(SEEDS_FIN, 1):
    t0 = time.time()
    m = CatBoostClassifier(**CB_PRM, random_seed=sd).fit(ptr)
    assert list(m.classes_)[0] == 0
    models.append(m)
    tick(f"[최종] seed={sd} ({i}/{len(SEEDS_FIN)}, {time.time()-t0:.0f}s)")
    # 중간 저장 (사고 대비)
    joblib.dump({"cb_models": models, "feats": FEATS, "prior": prior, "cs_const_p": cp, "cs_const_b": cb_,
                 "mix_tbl": mix_dep, "pfb_model": pfb, "seeds": SEEDS_FIN[:i], "combo_cats": combo_cats,
                 "n_classes": 4 if FOUR else 5, "merge": MERGE, "version": f"{VERSION}_partial{i}"},
                f"{args.out_dir}/model/model_partial.pkl", compress=3)
del ptr

bundle = {"cb_models": models, "feats": FEATS, "prior": prior, "cs_const_p": cp, "cs_const_b": cb_,
          "mix_tbl": mix_dep, "pfb_model": pfb, "seeds": SEEDS_FIN, "combo_cats": combo_cats,
          "n_classes": 4 if FOUR else 5, "merge": MERGE, "version": VERSION}
os.makedirs(f"{args.out_dir}/model", exist_ok=True)
joblib.dump(bundle, f"{args.out_dir}/model/model.pkl", compress=3)
if os.path.exists(f"{args.out_dir}/model/model_partial.pkl"):
    os.remove(f"{args.out_dir}/model/model_partial.pkl")
tick(f"저장: {args.out_dir}/model/model.pkl ({os.path.getsize(f'{args.out_dir}/model/model.pkl')/1024**2:.1f} MB)")

# ---- 5행 검증 (실제 2025 test.csv) ----
b2 = joblib.load(f"{args.out_dir}/model/model.pkl")
t5 = pd.read_csv("data/test.csv", encoding="utf-8-sig")
t5.columns = [c.replace("﻿", "").strip() for c in t5.columns]
ft5 = s17.attach_pt(s17.attach_cs(s17.add_features(t5, b2["prior"]), b2["cs_const_p"], b2["cs_const_b"]),
                    b2["mix_tbl"], b2["pfb_model"])
X5 = s17.build_matrix(ft5, b2["feats"], b2["combo_cats"])
acc5 = np.zeros(len(t5))
for m in b2["cb_models"]:
    acc5 += m.predict_proba(X5)[:, 0]
log(f"  5행 예측(P(성공)) = {np.round(acc5 / len(b2['cb_models']), 6).tolist()}")
if combo_cats:
    log("  5행 조합 예시: " + " | ".join(f"{c}={X5[c].iloc[0]}" for c in combo_cats))
log(f"\n총 {time.time()-T0:.0f}초. 다음: exp/95_build_submit17.py 로 zip 빌드+검증 (사용자 검토 후)")
