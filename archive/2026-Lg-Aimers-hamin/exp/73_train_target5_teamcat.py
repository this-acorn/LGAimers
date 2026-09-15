# -*- coding: utf-8 -*-
"""
[73] target5_teamcat 배포 학습 — 5클래스 타겟 분해(dou, submit14, LB 1033.99) +
     team_cat(우리, LB 1018.1353에서 CS79 대비 +24.51 확인) 콤보, CB 8시드

배경: dou가 자기 컴퓨터에서 진행한 submit14가 CS79(79피처, 3 cat_features) 위에
  control_success를 5클래스{성공(0)/미들(1)/리버스(2)/미들∩리버스(3)/빅미스(4)}로
  분해해 멀티클래스로 학습 → P(class0=성공)을 제출값으로 써서 LB 1033.99를 기록함
  (우리 최고 team_cat 1018.1353 대비 +15.6). 미들/리버스 라벨은 공식 asof 컬럼
  (asof_pitcher_middle_rate/asof_pitcher_reverse_rate, data_description.md에 문서화된
  공식 피처)을 구종 라벨(lab_fb/lab_brk)과 완전히 동일한 방식으로 train 내 차분 복원한
  것으로 추정 — dou의 학습 스크립트 원본은 확보하지 못해(레포 비공개) 이 스크립트가
  그 로직을 동일 기법으로 재구성한 것. 사용자 확인: 로컬 홀드아웃 검증 없이 바로
  8시드 전체 학습 → 제출 진행 (2026-08-26).

  이 스크립트가 새로 추가하는 것: dou의 v6(cat_features 3개, script_v6 계열)에는 없던
  team_cat의 pitcher_team_id/batter_team_id cat_features(로컬 실측 +2.0 → 실전 +24.51,
  team_pipeline_sync_2026-08-25.md 참고)를 더해 5개로 — 두 개선축이 겹치지 않는
  독립적인 축(타겟 재정의 vs cat_features 확장)이라는 가설로 스태킹.

★ submit12_src/script.py는 v8(script_v8_target5_teamcat.py)로 먼저 교체돼 있어야 한다.
  이 학습 스크립트는 그 파일을 단일 원본으로 그대로 import한다.

단계: 1) 라벨 복원(전체) — 구종(fb/brk) + 미들/리버스 4종, succ 자체검증(assert)
      2) target5(5클래스) 조립 — 실패인데 미들/리버스 복원 실패한 행은 드롭(추정 0.1%대)
      3) ★ 로컬 홀드아웃 검증 없이 바로 2019~2024 전체 8시드 학습 (사용자 지시)
      4) model.pkl 저장 + 5행 sanity check

★ venv311 (10스레드):
  python -u exp/73_train_target5_teamcat.py   (~3~4시간, CS79/team_cat과 비슷한 규모 —
  멀티클래스라 클래스당 트리 구조가 늘어 다소 더 걸릴 수 있음)

리더보드 결과: 1059.0501189623 (팀 최고 기록, 2026-08-27 확정)
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
SEEDS_FIN = [42, 7, 123, 2024, 99, 555, 31337, 1]
CB_PRM = dict(iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=10.0,
              loss_function="MultiClass", classes_count=5,
              verbose=False, thread_count=10, allow_writing_files=False)
K_MIX = 50.0

# ★ submit12_src/script.py는 v8(script_v8_target5_teamcat.py, CAT/TEAM_CAT 분리 +
#   멀티클래스 추론 버전)로 먼저 교체해둘 것. 이 학습 스크립트는 그 파일을 단일 원본으로
#   그대로 import한다.
spec = importlib.util.spec_from_file_location("s12", "submit12_src/script.py")
s12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s12)
add_features, attach_cs, attach_pt = s12.add_features, s12.attach_cs, s12.attach_pt
build_matrix, CS_FEATS, PT_FEATS = s12.build_matrix, s12.CS_FEATS, s12.PT_FEATS
P_RATES, B_RATES, PFB_IN, CAT = s12.P_RATES, s12.B_RATES, s12.PFB_IN, s12.CAT
TEAM_CAT = s12.TEAM_CAT
CAT_ALL = list(CAT) + list(TEAM_CAT)
assert TEAM_CAT == ["pitcher_team_id", "batter_team_id"], "s12가 v8(team_cat 포함)로 교체됐는지 확인"


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


log(f"환경: python {sys.version.split()[0]} / numpy {np.__version__}")
log(f"cat_features = {CAT_ALL}")
log("train 로딩...")
df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
df.columns = [c.replace("﻿", "").strip() for c in df.columns]
for c in df.select_dtypes("float64").columns:
    df[c] = df[c].astype("float32")
tick(f"로딩 {df.shape}")

# =====================================================================
# 1. 라벨 복원 (전체 train) — 구종 2종(fb/brk, PT_FEATS용) + 미들/리버스 2종(target5용)
#    전부 동일 기법: (pitcher_id, asof_pitcher_n) 정렬 후 다음 행과의 누적치 차분.
#    미들/리버스는 asof_pitcher_middle_rate / asof_pitcher_reverse_rate — 둘 다
#    data_description.md에 문서화된 공식 asof 피처라 구종 라벨과 같은 근거로 train 내
#    사용이 허용된다 ("학습 데이터에서 추출한 정보를 바탕으로 독립 적용 → 가능").
# =====================================================================
d = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index()
pid = d.pitcher_id.to_numpy()
n = d.asof_pitcher_n.fillna(0).to_numpy("float64")
same_next = (pid[1:] == pid[:-1]) & (np.abs(n[1:] - n[:-1] - 1) < 1e-6)
for k, col in [("lab_fb", "asof_pitcher_fastball_rate"),
               ("lab_brk", "asof_pitcher_breaking_rate"),
               ("lab_middle", "asof_pitcher_middle_rate"),
               ("lab_reverse", "asof_pitcher_reverse_rate")]:
    S = d[col].fillna(0).to_numpy("float64") * n
    lab = np.full(len(d), np.nan)
    lab[:-1] = np.where(same_next, np.round(S[1:] - S[:-1]), np.nan)
    lab = np.where((lab == 0) | (lab == 1), lab, np.nan)
    d[k] = lab

# succ 자체검증 (기존 스크립트들과 동일한 필수 안전장치 — 이 diff 기법 자체가
# 맞는지 확인하는 것이지, 로컬 홀드아웃 성능 검증이 아니므로 유지함)
Ss = d["asof_pitcher_success_rate"].fillna(0).to_numpy("float64") * n
ls = np.full(len(d), np.nan)
ls[:-1] = np.where(same_next, np.round(Ss[1:] - Ss[:-1]), np.nan)
m_ = ~np.isnan(ls)
agree = float(np.mean(ls[m_] == d.control_success.to_numpy("float64")[m_]))
log(f"★ 라벨 복원 검증(succ): 일치율 {agree*100:.2f}%  (100% 근처여야 함)")
assert agree > 0.999, "복원 정밀도 미달 — diff 기법 자체가 깨진 것이니 중단"

for k in ["lab_middle", "lab_reverse"]:
    cov = d[k].notna().mean() * 100
    log(f"  {k} 복원 커버리지 {cov:.2f}%  (참고용 — 구종 라벨과 비슷한 범위(~99.9%)가 정상)")

rec = d.set_index("index").sort_index()
for k in ["lab_fb", "lab_brk", "lab_middle", "lab_reverse"]:
    df[k] = rec[k]
b_, s_ = df.balls_before.to_numpy(), df.strikes_before.to_numpy()
df["_cg"] = np.where(s_ > b_, 2, np.where(b_ > s_, 0, 1)).astype("int8")
tick(f"구종 라벨 복원 완료 (커버리지 {df.lab_fb.notna().mean()*100:.1f}%)")

# =====================================================================
# 2. target5 조립 — 0=성공 1=미들 2=리버스 3=미들∩리버스 4=빅미스
#    성공(0)은 control_success로 직접 확정. 실패 행은 lab_middle/lab_reverse로 분류하고,
#    둘 중 하나라도 복원 실패(NaN)한 실패 행은 세부 클래스를 못 매기므로 학습에서 제외한다
#    (구종 라벨 때와 같은 이유로 극소수일 것으로 예상 — 아래서 실측 비율 출력).
# =====================================================================
y_succ = df["control_success"].to_numpy("float64")
lab_mid = df["lab_middle"].to_numpy("float64")
lab_rev = df["lab_reverse"].to_numpy("float64")

cls = np.full(len(df), np.nan)
cls[y_succ == 1] = 0
fail = (y_succ == 0)
cls[fail & (lab_mid == 1) & (lab_rev == 1)] = 3
cls[fail & (lab_mid == 1) & (lab_rev == 0)] = 1
cls[fail & (lab_mid == 0) & (lab_rev == 1)] = 2
cls[fail & (lab_mid == 0) & (lab_rev == 0)] = 4
df["target5"] = cls

n_drop = int(np.isnan(cls).sum())
log(f"★ target5 조립: 드롭(실패인데 미들/리버스 복원 실패) {n_drop}행 "
    f"({n_drop/len(df)*100:.3f}%)")
vc = pd.Series(cls).value_counts(dropna=False).sort_index()
log(f"  클래스 분포(0=성공/1=미들/2=리버스/3=미들∩리버스/4=빅미스, NaN=드롭):\n{vc}")

FEATS79 = None  # 아래에서 구성 (CS79/team_cat과 동일, 새 컬럼 없음)


def prep_stage(upto):
    """학습(≤upto) 행에 CS + 구종 피처 부착. target5는 이미 df에 붙어있으므로 그대로 이월."""
    hist = df[df.season <= upto]
    prior = float(hist["control_success"].mean())
    cp = build_const(hist, "pitcher_id", "asof_pitcher_n", P_RATES)
    cb_ = build_const(hist, "batter_id", "asof_batter_n", B_RATES)
    pfb_rows = hist.dropna(subset=["lab_fb"])
    pfb = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                             verbose=False, thread_count=10,
                             allow_writing_files=False, random_seed=42)
    pfb.fit(pfb_rows[PFB_IN].fillna(-999), pfb_rows["lab_fb"].astype(int))
    mix_dep_tbl = mix_career_table(hist)
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
    # 시즌 as-of 믹스(구종)는 학습 행에도 필요 — 41/48/64와 동일하게 시즌 단위 as-of 사용
    t_hist = hist.dropna(subset=["lab_fb"])
    c = (t_hist.groupby(["pitcher_id", "season", "_cg"])
         .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
         .reset_index().sort_values(["pitcher_id", "_cg", "season"]))
    g = c.groupby(["pitcher_id", "_cg"])
    for col in ["n", "fb", "brk"]:
        c[f"p_{col}"] = g[col].cumsum() - c[col]
    o = (t_hist.groupby(["pitcher_id", "season"])
         .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum"))
         .reset_index().sort_values(["pitcher_id", "season"]))
    go = o.groupby("pitcher_id")
    for col in ["n", "fb", "brk"]:
        o[f"po_{col}"] = go[col].cumsum() - o[col]
    mix_tr_tbl = c.merge(o[["pitcher_id", "season", "po_n", "po_fb", "po_brk"]],
                          on=["pitcher_id", "season"])
    over_fb = (mix_tr_tbl.po_fb + 0.5 * K_MIX) / (mix_tr_tbl.po_n + K_MIX)
    over_brk = (mix_tr_tbl.po_brk + 0.3 * K_MIX) / (mix_tr_tbl.po_n + K_MIX)
    mix_tr_tbl["mix_fb"] = np.where(mix_tr_tbl.po_n > 0,
                                    (mix_tr_tbl.p_fb + K_MIX * over_fb) / (mix_tr_tbl.p_n + K_MIX),
                                    np.nan).astype("float32")
    mix_tr_tbl["mix_brk"] = np.where(mix_tr_tbl.po_n > 0,
                                     (mix_tr_tbl.p_brk + K_MIX * over_brk) / (mix_tr_tbl.p_n + K_MIX),
                                     np.nan).astype("float32")
    mix_tr_tbl = mix_tr_tbl[["pitcher_id", "season", "_cg", "mix_fb", "mix_brk"]]
    key = tr[["pitcher_id", "season", "_cg"]].merge(
        mix_tr_tbl, on=["pitcher_id", "season", "_cg"], how="left")
    tr["f_mixcg_fb"] = key["mix_fb"].to_numpy("float32")
    tr["f_mixcg_brk"] = key["mix_brk"].to_numpy("float32")
    tr["f_pfb"] = pfb.predict_proba(tr[PFB_IN].fillna(-999))[:, 1].astype("float32")
    tr["target5"] = df.loc[tr.index, "target5"].to_numpy()
    return tr, pfb, cp, cb_, mix_dep_tbl, prior


# =====================================================================
# 3. ★ 로컬 홀드아웃 검증 생략 — 사용자 지시(2026-08-26)로 바로 2019~2024 전체 8시드 학습.
#    dou의 submit14(1033.99)가 이미 실전에서 이 타겟 분해 자체를 검증했고, team_cat의
#    cat_features 추가도 우리가 별도로 실전 검증(+24.51) 완료한 상태라 두 축 모두
#    "로컬 홀드아웃 없이 바로 실전"의 리스크가 CS83/CS93류(로컬조차 안 해본 완전 신규
#    피처)보다는 낮다고 보고 진행. 다만 "이 콤보 자체"는 실전 미검증이니 결과는 반드시
#    이 문서/team_pipeline_sync에 즉시 기록할 것.
# =====================================================================
log("\n" + "=" * 80)
log("최종: 2019~2024 전체 8시드 학습 (target5 멀티클래스 + team_cat, 로컬 검증 생략)")
log("=" * 80)
tr, pfb_fin, cp_fin, cb_fin, mix_fin, prior_fin = prep_stage(2024)

BASE = [c for c in pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1)
        .columns.str.replace("﻿", "").str.strip() if c != "row_id"]
ENG18 = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
         "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
         "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
         "f_pb_diff", "f_miss_p", "f_miss_prev1"]
FEATS79 = BASE + ENG18 + CS_FEATS + PT_FEATS
log(f"피처 {len(FEATS79)}개 (CS79와 동일, 새 컬럼 없음), cat_features {len(CAT_ALL)}개 {CAT_ALL}")

mask = tr["target5"].notna().to_numpy()
log(f"학습 행: {len(tr)} → target5 결측 {(~mask).sum()}개 드롭 → 최종 {mask.sum()}행")
tr_m = tr.loc[mask]
y_all = tr_m["target5"].to_numpy("int64")

Xtr = build_matrix(tr_m, FEATS79)
ptr = Pool(Xtr, y_all, cat_features=CAT_ALL)
del tr, tr_m, Xtr

models = []
for i, sd in enumerate(SEEDS_FIN, 1):
    t0 = time.time()
    models.append(CatBoostClassifier(**CB_PRM, random_seed=sd).fit(ptr))
    tick(f"[최종] seed={sd} ({i}/8, {time.time()-t0:.0f}s)")
del ptr

bundle = {"cb_models": models, "feats": FEATS79, "prior": prior_fin,
          "cs_const_p": cp_fin, "cs_const_b": cb_fin,
          "mix_tbl": mix_fin, "pfb_model": pfb_fin,
          "seeds": SEEDS_FIN, "version": "target5_teamcat"}
os.makedirs("submit12_src/model", exist_ok=True)
joblib.dump(bundle, "submit12_src/model/model.pkl", compress=3)
tick(f"저장: submit12_src/model/model.pkl "
     f"({os.path.getsize('submit12_src/model/model.pkl')/1024**2:.1f} MB)")

# =====================================================================
# 4. 5행 검증 (실제 2025 test.csv)
# =====================================================================
b2 = joblib.load("submit12_src/model/model.pkl")
t5 = pd.read_csv("data/test.csv", encoding="utf-8-sig")
t5.columns = [c.replace("﻿", "").strip() for c in t5.columns]
ft5 = s12.attach_pt(s12.attach_cs(s12.add_features(t5, b2["prior"]),
                                  b2["cs_const_p"], b2["cs_const_b"]),
                    b2["mix_tbl"], b2["pfb_model"])
X5 = s12.build_matrix(ft5, b2["feats"])
acc5 = np.zeros(len(t5))
for m in b2["cb_models"]:
    acc5 += m.predict_proba(X5)[:, 0]  # 클래스0 = 성공
p5 = acc5 / len(b2["cb_models"])
log(f"  5행 예측(P(성공)) = {np.round(p5, 6).tolist()}")
log(f"  f_pfb = {np.round(ft5['f_pfb'].to_numpy(), 3).tolist()}")
log(f"  f_mixcg_fb 커버 {ft5['f_mixcg_fb'].notna().sum()}/5")
log(f"\n총 {time.time()-T0:.0f}초. 다음: exp/74_build_submit_target5_teamcat.py 로 zip 빌드+검증")
