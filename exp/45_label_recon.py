# -*- coding: utf-8 -*-
"""
[45] ★ 라벨 복원 금맥 — asof 연속행 차분으로 train 투구별 결과 라벨 복원 (밤샘)

원리: asof_pitcher_n은 투구마다 1씩 증가하는 통산 카운터. 같은 투수의 연속 행
  (asof_n = j, j+1)에서 S_k = rate_k×n 의 차분은 'j번째 투구의 k-결과'(0/1)다.
  → train의 거의 모든 투구에 대해 볼/스트라이크/미들/reverse/구종군 라벨 복원.
합법성: 운영진 명시 판결 2건 — "학습 데이터 내에서는 제약사항 없습니다"
  (jieuni52 08-09, 2jin1 08-17). test에서는 금지 — 여기서는 train만 사용.

Part A. 복원 + 자체검증:
  success_rate 차분으로 복원한 라벨 vs 실제 control_success 일치율 = 방법의 정밀도 증명.
  (성공 라벨이 ~100% 일치하면 같은 산술인 볼/미들/구종 라벨도 신뢰 가능)

Part B. 파생 피처 2계열 → 2024 폴드 (기준 = exp/41 CS76 2시드 782.3):
  MIX: 투수×카운트군별 구종 성향 (시즌 as-of) — 구종은 타겟 유래가 아니라 행동 특성
       f_mixcg_fb / f_mixcg_brk = 이 투수가 이 카운트군에서 패스트볼/변화구 던지는 비율
  PFB: P(fastball | 상황) 전역 모델 (복원 라벨로 학습한 CB 소형모델의 출력을 피처로)
       — 배포 시 모델을 번들에 동봉, 행 독립 적용
  팔: CS76+MIX / CS76+PFB / CS76+둘다

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/45_label_recon.py  (~1.5시간)
"""

import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import log, tick, raw_score, load_train, add_features, CAT, ALL_ENG

SEEDS = [42, 7]
CB_PRM = dict(iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=10.0,
              verbose=False, thread_count=14, allow_writing_files=False)

# ---- exp/41의 CS 함수 재사용 (submit10 script.py가 단일 원본) ----
import importlib.util
spec = importlib.util.spec_from_file_location("s10", "submissions/submit10_src/script.py")
s10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s10)
attach_cs, CS_FEATS, P_RATES, B_RATES = s10.attach_cs, s10.CS_FEATS, s10.P_RATES, s10.B_RATES


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


log("train 로딩...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS76 = BASE + ALL_ENG + CS_FEATS

# =====================================================================
# Part A. 라벨 복원 + 자체검증
# =====================================================================
log("\n" + "=" * 78)
log("Part A. 투구별 결과 라벨 복원 (asof 연속행 차분)")
log("=" * 78)
d = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index()
pid = d.pitcher_id.to_numpy()
n = d.asof_pitcher_n.fillna(0).to_numpy("float64")
same_next = (pid[1:] == pid[:-1]) & (np.abs(n[1:] - n[:-1] - 1) < 1e-6)
log(f"  연속쌍 (같은 투수, asof_n 정확히 +1): {same_next.sum():,} / {len(d)-1:,}")

REC = {"succ": "asof_pitcher_success_rate", "ball": "asof_pitcher_ball_rate",
       "strike": "asof_pitcher_strike_rate", "middle": "asof_pitcher_middle_rate",
       "fb": "asof_pitcher_fastball_rate", "brk": "asof_pitcher_breaking_rate",
       "off": "asof_pitcher_offspeed_rate"}
labels = {}
for k, col in REC.items():
    S = d[col].fillna(0).to_numpy("float64") * n
    diff = S[1:] - S[:-1]
    lab = np.full(len(d), np.nan)
    lab[:-1] = np.where(same_next, np.round(diff), np.nan)
    valid = ~np.isnan(lab[:-1][same_next])
    v = lab[:-1][same_next]
    frac_ok = np.mean(np.abs((S[1:] - S[:-1])[same_next] - np.round((S[1:] - S[:-1])[same_next])) < 0.05)
    labels[k] = lab
    log(f"  {k:7s}: 0/1 정수성 {frac_ok*100:5.1f}%  분포 0={np.mean(v==0)*100:.1f}% 1={np.mean(v==1)*100:.1f}% 기타={np.mean((v!=0)&(v!=1))*100:.2f}%")

# 결정적 검증: succ 복원 라벨 vs 실제 y
y_true = d.control_success.to_numpy("float64")
mask = np.zeros(len(d), dtype=bool)
mask[:-1] = same_next
agree = np.mean(labels["succ"][mask] == y_true[mask])
log(f"\n  ★ 검증: 복원 succ 라벨 vs 실제 control_success 일치율 = {agree*100:.2f}%")
if agree < 0.99:
    log("  ⚠️ 복원 정밀도 미달 — 이하 결과 신뢰 불가")

d["lab_fb"] = labels["fb"]
d["lab_brk"] = labels["brk"]
d["lab_middle"] = labels["middle"]
recon = d.set_index("index").sort_index()      # 원래 행 순서로 복귀
df["lab_fb"] = recon["lab_fb"]
df["lab_brk"] = recon["lab_brk"]
cov = df.lab_fb.notna().mean()
log(f"  구종 라벨 커버리지: {cov*100:.1f}% (각 투수의 마지막 투구 제외 전부)")

# =====================================================================
# Part B-1. MIX: 투수×카운트군 구종 성향 (시즌 as-of)
# =====================================================================
log("\n" + "=" * 78)
log("Part B. 파생 피처 → 2024 폴드")
log("=" * 78)
b_, s_ = df.balls_before.to_numpy(), df.strikes_before.to_numpy()
df["_cg"] = np.where(s_ > b_, 2, np.where(b_ > s_, 0, 1)).astype("int8")


def mix_asof_table(src):
    """(pid, season, cg) → 그 시즌 이전 패스트볼/변화구 비율 (k=50 수축→본인통산)"""
    t = src.dropna(subset=["lab_fb"])
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
    k = 50.0
    over_fb = (tbl.po_fb + 0.5 * k) / (tbl.po_n + k)
    over_brk = (tbl.po_brk + 0.3 * k) / (tbl.po_n + k)
    tbl["mix_fb"] = np.where(tbl.po_n > 0,
                             (tbl.p_fb + k * over_fb) / (tbl.p_n + k), np.nan).astype("float32")
    tbl["mix_brk"] = np.where(tbl.po_n > 0,
                              (tbl.p_brk + k * over_brk) / (tbl.p_n + k), np.nan).astype("float32")
    return tbl[["pitcher_id", "season", "_cg", "mix_fb", "mix_brk"]]


mix_tbl = mix_asof_table(df)
tick(f"MIX 테이블 {len(mix_tbl):,}행")


def attach_mix(x):
    x = x.copy()
    x["_cg"] = np.where(x.strikes_before > x.balls_before, 2,
                        np.where(x.balls_before > x.strikes_before, 0, 1)).astype("int8")
    k = x[["pitcher_id", "season", "_cg"]].merge(
        mix_tbl, on=["pitcher_id", "season", "_cg"], how="left")
    x["f_mixcg_fb"] = k["mix_fb"].to_numpy("float32")
    x["f_mixcg_brk"] = k["mix_brk"].to_numpy("float32")
    return x.drop(columns=["_cg"])


MIX_FEATS = ["f_mixcg_fb", "f_mixcg_brk"]

# =====================================================================
# Part B-2. PFB: P(fastball|상황) 전역 모델 → 출력 피처
# =====================================================================
PFB_IN = ["balls_before", "strikes_before", "outs_before", "inning", "num_runners_on",
          "runner_on_1b", "runner_on_2b", "runner_on_3b", "pitcher_hand", "batter_hand",
          "score_diff_pitcher_team", "li", "asof_pitcher_fastball_rate",
          "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
pfb_train = df[df.season <= 2023].dropna(subset=["lab_fb"])
tick(f"PFB 학습: {len(pfb_train):,}행 (≤2023, 구종 라벨 보유)")
pfb_model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                               verbose=False, thread_count=14,
                               allow_writing_files=False, random_seed=42)
pfb_model.fit(pfb_train[PFB_IN].fillna(-999), pfb_train["lab_fb"].astype(int))
auc_chk = pfb_model.predict_proba(pfb_train[PFB_IN].fillna(-999).head(100000))[:, 1]
tick(f"PFB 학습 완료 (출력 평균 {auc_chk.mean():.3f})")


def attach_pfb(x):
    x = x.copy()
    x["f_pfb"] = pfb_model.predict_proba(x[PFB_IN].fillna(-999))[:, 1].astype("float32")
    return x


# =====================================================================
# 하네스: 2024 폴드, 기준 = exp/41 CS76 2시드 (lab/41_cs_ens.npy = 782.3)
# =====================================================================
hist = df[df.season <= 2023]
cp = build_const(hist, "pitcher_id", "asof_pitcher_n", P_RATES)
cb_tbl = build_const(hist, "batter_id", "asof_batter_n", B_RATES)
tr_raw = df[df.season <= 2023].reset_index(drop=True)
va_raw = df[df.season == 2024].reset_index(drop=True)
y_tr = tr_raw["control_success"].to_numpy()
y_va = va_raw["control_success"].to_numpy()
prior = float(y_tr.mean())

tick("학습 행 CS 부착 (시즌별 상수표)...")
parts = []
for S in sorted(tr_raw.season.unique()):
    h = df[df.season <= S - 1]
    rows = tr_raw[tr_raw.season == S]
    if len(h) == 0:
        cpS = pd.DataFrame({"id": [], "N_end": [], **{f"S_{k}": [] for k in P_RATES}})
        cbS = pd.DataFrame({"id": [], "N_end": [], **{f"S_{k}": [] for k in B_RATES}})
    else:
        cpS = build_const(h, "pitcher_id", "asof_pitcher_n", P_RATES)
        cbS = build_const(h, "batter_id", "asof_batter_n", B_RATES)
    parts.append(attach_cs(rows, cpS, cbS))
tr = attach_pfb(attach_mix(add_features(pd.concat(parts).sort_index(), prior)))
va = attach_pfb(attach_mix(add_features(attach_cs(va_raw, cp, cb_tbl), prior)))
del parts
tick(f"부착 완료 — 검증 MIX 커버리지 {va.f_mixcg_fb.notna().mean()*100:.0f}% / "
     f"PFB 평균 {va.f_pfb.mean():.3f}")

p_ref = np.load("lab/41_cs_ens.npy").astype("float64")
s_ref = raw_score(p_ref, y_va)
log(f"기준 (CS76 2시드): {s_ref:.1f}")


def run(name, extra):
    feats = [c for c in FEATS76 if c not in CAT] + extra
    def frame(x):
        out = x[feats].copy()
        for c in CAT:
            out[c] = x[c].astype(str)
        return out
    ptr = Pool(frame(tr), y_tr, cat_features=list(CAT))
    pva = Pool(frame(va), cat_features=list(CAT))
    preds = []
    for sd in SEEDS:
        t0 = time.time()
        m = CatBoostClassifier(**CB_PRM, random_seed=sd).fit(ptr)
        preds.append(m.predict_proba(pva)[:, 1])
        tick(f"{name} seed={sd}  {raw_score(preds[-1], y_va):8.1f}  ({time.time()-t0:.0f}s)")
    ens = np.mean(preds, axis=0)
    np.save(f"lab/45_{name}_ens.npy", ens.astype("float32"))
    sc = raw_score(ens, y_va)
    log(f"  → {name:10s} {sc:8.1f}  ({sc-s_ref:+.1f})")
    return sc


r_mix = run("mix", MIX_FEATS)
r_pfb = run("pfb", ["f_pfb"])
r_both = run("both", MIX_FEATS + ["f_pfb"])

log("\n" + "=" * 78)
log(f"판정 (2024 폴드, CS76 기준 {s_ref:.1f})")
log("=" * 78)
log(f"  +MIX(투수×카운트 구종성향)  {r_mix:8.1f}  ({r_mix-s_ref:+.1f})")
log(f"  +PFB(P(구종|상황) 모델)     {r_pfb:8.1f}  ({r_pfb-s_ref:+.1f})")
log(f"  +둘다                       {r_both:8.1f}  ({r_both-s_ref:+.1f})")
log("  +15↑ → 내일 배포 증축 후보. 구종 라벨은 타겟 유래가 아니라 [F-물리]급 행동특성.")
log("=" * 78)
