# -*- coding: utf-8 -*-
"""
[101] GPU MLP 블렌드 프로브 — 챔피언(cat5 · target5 · CS79)과 "다르게 틀리는" 신경망 파트너 탐색

목적: CatBoost 교체가 아니라 혼합 이득. 닫힌 해  gain = (d+K)^2/(4K),  d = S_mlp − S_cb,  K = (1e5/DEN)·E[(p_mlp − p_cb)^2]
기준: lab/89_cat5.npy (cat5 2시드 평균 P(성공), 2024 폴드 253,507행) — 정렬·점수(≈873.9) 검증 후에만 진행.

데이터/누수: 학습 season<=2023, 최종 검증 season==2024 (2024 는 게이트 외 미사용). 내부 5% 검증(고정 RNG 12345, 두 시드 공통)으로
  early stopping. 전처리(중앙값·표준화·one-hot 범주)는 학습 95% 에서만 fit. test.csv·2025·TrackMan 미사용.
입력: 챔피언과 동일한 79피처 순서. cat5(top_bottom, game_type, base_state, pitcher_team_id, batter_team_id)만 one-hot(미지 범주 = 전부 0),
  pitcher_id/batter_id 는 수치 유지, NaN 은 학습 중앙값 대치 + missing indicator, float32.
모델: plain PyTorch residual MLP (256, 블록 3, LayerNorm/SiLU/Dropout .05) → 5 로짓. CrossEntropy(가중 없음). P(성공)=softmax[:,0].
학습: seeds [42,7], AdamW(1e-3, wd 1e-5), batch 8192(OOM 시 4096), ≤20 epoch, patience 3, clip 1.0, autocast(bf16→fp16+GradScaler).
게이트: 시드42 후 조기 FAIL(NaN/Inf, shape<=0, 최적 혼합이득<+5, d<=−K, CPU 실행). 최종 PASS 조건은 §10 (아래 evaluate()).
제한: OMP/MKL 등 스레드 --threads (기본 4), torch 스레드 동일, DataLoader 미사용(전 데이터 GPU 상주). 총 150분 상한.
실행:  PYTHONIOENCODING=utf-8 python -u exp/101_mlp_blend_probe.py [--smoke] [--threads 4] [--seeds 42,7]
산출: lab/101_mlp_log.txt, lab/101_mlp_meta.json, lab/101_valid_row_id.npy, lab/101_y2024.npy, lab/101_mlp_seed{42,7}.npy,
      lab/101_mlp_mean.npy, lab/101_mlp_seed{sd}.pt, lab/101_mlp_preprocessor.joblib, lab/101_blend_grid.csv,
      lab/101_mlp_report.txt, lab/101_mlp_report.json
"""

import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true", help="6만행·2 epoch·1시드로 경로만 검증 (산출물 접두어 101s_)")
ap.add_argument("--threads", type=int, default=4, help="CPU 스레드 상한 (같은 기계에 CatBoost 가 돌면 2)")
ap.add_argument("--seeds", default="42,7")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--batch", type=int, default=8192)
ap.add_argument("--time-limit-min", type=float, default=150.0)
ap.add_argument("--allow-cpu", action="store_true", help="smoke 전용: CUDA 없어도 CPU 로 경로 검증 (본 실행에는 절대 사용 금지)")
args = ap.parse_args()
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[k] = str(args.threads)

import gc                      # noqa: E402
import hashlib                 # noqa: E402
import importlib.util          # noqa: E402
import json                    # noqa: E402
import subprocess              # noqa: E402
import time                    # noqa: E402
import numpy as np             # noqa: E402
import pandas as pd            # noqa: E402
import joblib                  # noqa: E402
import torch                   # noqa: E402
import torch.nn as nn          # noqa: E402
import torch.nn.functional as F  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402  (PFB 피처용 — 챔피언 파이프라인과 동일)

torch.set_num_threads(args.threads)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
sys.path.insert(0, "exp")
from common import raw_score, load_train  # noqa: E402

T0 = time.time()
PFX = "lab/101s" if args.smoke else "lab/101"
SEEDS = [int(s) for s in args.seeds.split(",")]
if args.smoke:
    SEEDS = SEEDS[:1]
LOGF = open(f"{PFX}_mlp_log.txt", "a", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOGF.write(s + "\n")
    LOGF.flush()


def tick(m):
    log(f"  [{time.time()-T0:6.0f}s] {m}")


def elapsed_min():
    return (time.time() - T0) / 60.0


def fail(msg, meta=None):
    log(f"FAIL: {msg}")
    rep = {"status": "FAIL", "reason": msg, "elapsed_min": elapsed_min()}
    if meta:
        rep.update(meta)
    with open(f"{PFX}_mlp_report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1, default=float)
    with open(f"{PFX}_mlp_report.txt", "a", encoding="utf-8") as f:
        f.write(f"FAIL: {msg}\n")
    sys.exit(1)


log(f"=== exp/101 MLP blend probe  smoke={args.smoke} seeds={SEEDS} threads={args.threads} ===")

# =====================================================================
# 2. GPU 확인 (CPU 대체 금지)
# =====================================================================
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"],
                         capture_output=True, text=True, timeout=30).stdout.strip()
except Exception as e:  # noqa: BLE001
    smi = f"(nvidia-smi 실행 불가: {e})"
log("nvidia-smi:", smi)
log(f"torch {torch.__version__}  cuda.is_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    DEV = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    log(f"GPU {gpu_name}  VRAM {vram_total:.1f} GB  used {torch.cuda.memory_allocated()/1e9:.2f} GB")
    BF16 = torch.cuda.is_bf16_supported()
else:
    if not (args.smoke and args.allow_cpu):
        fail("CUDA GPU 없음 — 이 실험은 GPU 전용 (CPU 대체 금지). 랩실 GPU 기계에서 실행하세요.")
    DEV = torch.device("cpu")
    gpu_name = "CPU(smoke only)"
    vram_total = 0.0
    BF16 = False
    log("⚠ smoke + --allow-cpu: CPU 로 경로만 검증 (결과는 무효)")
try:
    import psutil
    vm = psutil.virtual_memory()
    log(f"RAM total {vm.total/1e9:.1f} GB  available {vm.available/1e9:.1f} GB")
except Exception:  # noqa: BLE001
    log("RAM 확인 생략 (psutil 없음)")

# =====================================================================
# 데이터 — 챔피언(exp/89·93·100) 파이프라인 그대로
# =====================================================================
spec = importlib.util.spec_from_file_location("s12", 'submissions/submit12_src/script.py')
s12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s12)
K_MIX = 50.0

log("train 로딩...")
df = load_train()
dd = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index()
pid = dd.pitcher_id.to_numpy()
n = dd.asof_pitcher_n.fillna(0).to_numpy("float64")
nxt = (pid[1:] == pid[:-1]) & (np.abs(n[1:] - n[:-1] - 1) < 1e-6)
for k, col in [("lab_mid", "asof_pitcher_middle_rate"), ("lab_rev", "asof_pitcher_reverse_rate"),
               ("lab_fb", "asof_pitcher_fastball_rate"), ("lab_brk", "asof_pitcher_breaking_rate")]:
    S = dd[col].fillna(0).to_numpy("float64") * n
    lab = np.full(len(dd), np.nan)
    diff = np.round(S[1:] - S[:-1])
    lab[:-1] = np.where(nxt, diff, np.nan)
    dd[k] = np.where((lab == 0) | (lab == 1), lab, np.nan)
rec = dd.set_index("index").sort_index()
for k in ["lab_mid", "lab_rev", "lab_fb", "lab_brk"]:
    df[k] = rec[k]
del dd, rec, pid, n, nxt
gc.collect()

y_bin = df.control_success.to_numpy("float64")
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
log("클래스 분포: " + " ".join(f"{c}:{np.mean(cls==c)*100:.1f}%" for c in range(5)) + f"  미복원 {np.mean(cls==-1)*100:.2f}%")
del ok, m_, r_
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


def mix_career(labeled):
    t = labeled.dropna(subset=["lab_fb"])
    c = (t.groupby(["pitcher_id", "_cg"])
          .agg(n=("lab_fb", "size"), fb=("lab_fb", "sum"), brk=("lab_brk", "sum")).reset_index())
    o = (t.groupby("pitcher_id")
          .agg(po_n=("lab_fb", "size"), po_fb=("lab_fb", "sum"), po_brk=("lab_brk", "sum")).reset_index())
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
                         thread_count=args.threads, allow_writing_files=False, random_seed=42)
pfb.fit(pfb_rows[s12.PFB_IN].fillna(-999), pfb_rows["lab_fb"].astype(int))
del pfb_rows
mix_tr = mix_asof_train(hist)
mix_dep = mix_career(hist)
tick(f"테이블·PFB 준비 (PFB thread_count={args.threads}; lab/89 챔피언은 14 — f_pfb 가 미세하게 다를 수 있음, HANDOFF exp/76)")

tr_rows = df[df.season <= 2023].reset_index(drop=True)
va_rows = df[df.season == 2024].reset_index(drop=True)
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
del parts, tr_rows, hist
gc.collect()
key = tr[["pitcher_id", "season", "_cg"]].merge(mix_tr, on=["pitcher_id", "season", "_cg"], how="left")
tr["f_mixcg_fb"] = key["mix_fb"].to_numpy("float32")
tr["f_mixcg_brk"] = key["mix_brk"].to_numpy("float32")
tr["f_pfb"] = pfb.predict_proba(tr[s12.PFB_IN].fillna(-999))[:, 1].astype("float32")
del key
va = s12.attach_pt(s12.add_features(s12.attach_cs(va_rows, cp, cb_), prior), mix_dep, pfb)
del va_rows, df
gc.collect()
tick("피처 부착 완료")

BASE = [c for c in pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=1)
        .columns.str.replace("﻿", "").str.strip() if c != "row_id"]
ENG18 = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
         "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
         "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
         "f_pb_diff", "f_miss_p", "f_miss_prev1"]
FEATS = BASE + ENG18 + s12.CS_FEATS + s12.PT_FEATS
CAT5 = list(s12.CAT) + ["pitcher_team_id", "batter_team_id"]
assert len(FEATS) == 79 and len(CAT5) == 5
NUM = [c for c in FEATS if c not in CAT5]

# =====================================================================
# 8. 기준 예측 정렬 검증
# =====================================================================
y_va = va["control_success"].to_numpy("float64")
row_id_str = va["row_id"].astype(str).to_numpy()                                   # 'TRAIN_0000001' 형식 문자열
row_id_va = pd.Series(row_id_str).str.extract(r"(\d+)")[0].astype("int64").to_numpy()   # 숫자부만 int64 (sha·npy 용)
r_va = float(y_va.mean())
DEN = r_va * (1 - r_va)
sha = hashlib.sha256(row_id_va.tobytes()).hexdigest()
p_cb = np.load("lab/89_cat5.npy").astype("float64")
p_cb_seed = {}
for sd in [42, 7]:
    fp = f"lab/89_cat5_probs_seed{sd}.npy"
    if os.path.exists(fp):
        p_cb_seed[sd] = np.load(fp).astype("float64")[:, 0]
S_cb = raw_score(p_cb, y_va)
log(f"2024 행 {len(va):,}  row_id sha256 {sha[:16]}  y 길이 {len(y_va):,}  lab/89 길이 {len(p_cb):,}  NaN {int(np.isnan(p_cb).sum())}")
log(f"cat5 기준 재계산 {S_cb:.1f} (문서 873.9)")
if len(p_cb) != len(va) or np.isnan(p_cb).any() or abs(S_cb - 873.9) > 1.5:
    fail(f"기준 정렬/점수 불일치: len {len(p_cb)} vs {len(va)}, S_cb {S_cb:.1f}")
np.save(f"{PFX}_valid_row_id.npy", row_id_va)
np.save(f"{PFX}_y2024.npy", y_va.astype("int8"))

# =====================================================================
# 4~5. 내부 5% 검증 분할(고정) + 전처리(학습 95% 에서만 fit)
# =====================================================================
m_tr = tr["_cls"].to_numpy() >= 0
tr = tr[m_tr].reset_index(drop=True)
y_cls = tr["_cls"].to_numpy().astype("int64")
n_all = len(tr)
rng = np.random.default_rng(12345)
perm = rng.permutation(n_all)
n_val = int(round(n_all * 0.05))
idx_val = np.sort(perm[:n_val])
idx_fit = np.sort(perm[n_val:])
if args.smoke:
    idx_fit = idx_fit[:60000]
    idx_val = idx_val[:6000]
log(f"학습행 {len(idx_fit):,}  내부검증 {len(idx_val):,} (고정 RNG 12345, 두 시드 공통)")


class Prep:
    """수치: 학습 중앙값 대치 + missing indicator(학습에 NaN 있는 열) + 표준화. cat5: 학습 범주 one-hot(미지=전부 0). float32."""

    def fit(self, d):
        X = d[NUM].to_numpy("float64")
        self.median = np.nanmedian(X, axis=0)
        self.median = np.where(np.isnan(self.median), 0.0, self.median)
        self.nan_cols = np.where(np.isnan(X).any(axis=0))[0]
        Xi = np.where(np.isnan(X), self.median, X)
        self.mean = Xi.mean(axis=0)
        self.std = Xi.std(axis=0)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        self.lo, self.hi = np.nanmin(X, axis=0), np.nanmax(X, axis=0)   # 학습 관측 범위 — transform 에서 클립 (season 2024·미지 ID 의 선형 외삽 차단)
        self.lo = np.where(np.isnan(self.lo), -np.inf, self.lo)
        self.hi = np.where(np.isnan(self.hi), np.inf, self.hi)
        self.cats = {}
        for c in CAT5:
            v = self._s(d[c])
            self.cats[c] = sorted(set(v.tolist()))
        self.names = (NUM + [f"{NUM[i]}__isnan" for i in self.nan_cols]
                      + [f"{c}=={lv}" for c in CAT5 for lv in self.cats[c]])
        return self

    @staticmethod
    def _s(v):
        if pd.api.types.is_numeric_dtype(v):
            return v.fillna(-1).astype("int64").astype(str).to_numpy()
        return v.astype(str).to_numpy()

    def transform(self, d):
        X = d[NUM].to_numpy("float64")
        isnan = np.isnan(X)
        Xi = np.where(isnan, self.median, X)
        Xi = np.clip(Xi, self.lo, self.hi)                # 학습 범위 밖 값(season 2024, 신규 pitcher/batter_id)을 경계값으로 고정 — 트리의 '마지막 잎' 과 같은 처리
        Xs = (Xi - self.mean) / self.std
        blocks = [Xs.astype("float32"), isnan[:, self.nan_cols].astype("float32")]
        for c in CAT5:
            v = self._s(d[c])
            lv = self.cats[c]
            code = pd.Categorical(v, categories=lv).codes    # 미지 범주 = -1
            oh = np.zeros((len(v), len(lv)), dtype="float32")
            m = code >= 0
            oh[np.where(m)[0], code[m]] = 1.0
            blocks.append(oh)
        return np.concatenate(blocks, axis=1)


prep = Prep().fit(tr.iloc[idx_fit])
X_fit = prep.transform(tr.iloc[idx_fit])
X_val = prep.transform(tr.iloc[idx_val])
X_va = prep.transform(va)
y_fit = y_cls[idx_fit]
y_val = y_cls[idx_val]
del tr, va
gc.collect()
joblib.dump({"median": prep.median, "nan_cols": prep.nan_cols, "mean": prep.mean, "std": prep.std, "lo": prep.lo, "hi": prep.hi,
             "cats": prep.cats, "names": prep.names, "NUM": NUM, "CAT5": CAT5, "FEATS": FEATS},
            f"{PFX}_mlp_preprocessor.joblib")
IN_DIM = X_fit.shape[1]
assert np.isfinite(X_fit).all() and np.isfinite(X_va).all(), "전처리 후 비유한값"
tick(f"전처리 완료 — 입력 차원 {IN_DIM} (수치 {len(NUM)} + 결측지시 {len(prep.nan_cols)} + one-hot {IN_DIM-len(NUM)-len(prep.nan_cols)})")


# =====================================================================
# 6. 모델
# =====================================================================
class Block(nn.Module):
    def __init__(self, d=256, h=512, p=0.05):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)
        self.dp = nn.Dropout(p)

    def forward(self, x):
        return x + self.fc2(self.dp(F.silu(self.fc1(self.ln(x)))))


class ResMLP(nn.Module):
    def __init__(self, in_dim, d=256, n_blocks=3, n_out=5):
        super().__init__()
        self.inp = nn.Linear(in_dim, d)
        self.blocks = nn.ModuleList([Block(d) for _ in range(n_blocks)])
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_out)

    def forward(self, x):
        x = self.inp(x)
        for b in self.blocks:
            x = b(x)
        return self.out(self.ln(x))


def predict_p0(model, X_t, bs=65536):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X_t), bs):
            xb = X_t[i:i + bs]
            with torch.autocast(device_type=DEV.type, dtype=AMP_DTYPE, enabled=(DEV.type == "cuda")):
                lo = model(xb)
            outs.append(torch.softmax(lo.float(), dim=1)[:, 0].cpu().numpy())
    return np.concatenate(outs).astype("float64")


def train_seed(sd, batch):
    torch.manual_seed(sd)
    np.random.seed(sd)
    model = ResMLP(IN_DIM).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    _scaler_on = (DEV.type == "cuda" and AMP_DTYPE == torch.float16)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):   # torch>=2.3 신 API (2.4+ FutureWarning 회피)
        scaler = torch.amp.GradScaler("cuda", enabled=_scaler_on)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=_scaler_on)
    g = torch.Generator(device="cpu").manual_seed(sd)
    Xf = torch.from_numpy(X_fit).to(DEV)
    yf = torch.from_numpy(y_fit).to(DEV)
    Xv = torch.from_numpy(X_val).to(DEV)
    yv0 = (y_val == 0).astype("float64")
    best = (np.inf, None, -1)
    bad = 0
    hist_ = []
    n_fit = len(Xf)
    for ep in range(1, (2 if args.smoke else args.epochs) + 1):
        model.train()
        perm_t = torch.randperm(n_fit, generator=g).to(DEV)
        tot, nb = 0.0, 0
        for i in range(0, n_fit, batch):
            idx = perm_t[i:i + batch]
            xb, yb = Xf[idx], yf[idx]
            with torch.autocast(device_type=DEV.type, dtype=AMP_DTYPE, enabled=(DEV.type == "cuda")):
                loss = F.cross_entropy(model(xb), yb)
            if not torch.isfinite(loss):
                return None, {"error": "NaN/Inf loss", "epoch": ep}
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += float(loss.detach())
            nb += 1
        pv = predict_p0(model, Xv)
        if not np.isfinite(pv).all():
            return None, {"error": "NaN/Inf in internal-val prediction", "epoch": ep}
        brier = float(np.mean((pv - yv0) ** 2))
        hist_.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_brier_p0": brier})
        log(f"   seed {sd} epoch {ep:2d}  loss {tot/max(nb,1):.4f}  내부검증 Brier(P0) {brier:.5f}" + ("  ★" if brier < best[0] else ""))
        if brier < best[0] - 1e-7:
            best = (brier, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                log(f"   early stop (patience {args.patience}) — best epoch {best[2]}")
                break
        if elapsed_min() > args.time_limit_min - 10:
            log("   시간 상한 근접 — 학습 중단")
            break
    model.load_state_dict(best[1])
    torch.save(best[1], f"{PFX}_mlp_seed{sd}.pt")
    p_va = predict_p0(model, torch.from_numpy(X_va).to(DEV))
    peak = torch.cuda.max_memory_allocated() / 1e9 if DEV.type == "cuda" else 0.0
    del Xf, yf, Xv, model, opt
    gc.collect()
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return p_va, {"best_epoch": best[2], "best_val_brier_p0": best[0], "history": hist_, "peak_vram_gb": peak, "batch": batch}


AMP_DTYPE = torch.bfloat16 if BF16 else torch.float16
log(f"autocast dtype: {AMP_DTYPE}  device: {DEV}")


# =====================================================================
# 9. 혼합 가치 계산
# =====================================================================
WEIGHTS = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def center_pen(p):
    return 100000 * (p.mean() - r_va) ** 2 / DEN


def shape(p):
    return raw_score(p - p.mean() + r_va, y_va)


def blend_stats(p_m, p_c):
    S_m, S_c = raw_score(p_m, y_va), raw_score(p_c, y_va)
    d = S_m - S_c
    D_raw = float(np.mean((p_m - p_c) ** 2))
    K = (100000.0 / DEN) * D_raw
    w_star = float(np.clip(0.5 + d / (2 * K), 0, 1)) if K > 0 else 0.0
    theo = 0.0 if (K <= 0 or d <= -K) else (d if d >= K else (d + K) ** 2 / (4 * K))   # w* 클립(0..1) 반영: d≥K 면 파트너 단독 = d
    grid = {w: raw_score((1 - w) * p_c + w * p_m, y_va) - S_c for w in WEIGHTS}
    w_best = max(grid, key=grid.get)
    return {"S_mlp": S_m, "S_cb": S_c, "d": d, "D_raw": D_raw, "K": K, "w_star": w_star,
            "theoretical_gain": theo, "grid": grid, "w_best": w_best, "actual_best_gain": grid[w_best],
            "shape_mlp": shape(p_m) - shape(p_c), "pen_mlp": center_pen(p_m), "pen_cb": center_pen(p_c)}


SEG = {}
try:
    a90 = pd.read_parquet("lab/90_analysis_2024.parquet", columns=["row_id", "game_type", "p_first_season"])
    if len(a90) == len(y_va) and np.array_equal(a90["row_id"].astype(str).to_numpy(), row_id_str):
        gt = a90["game_type"].astype(str).to_numpy()
        rk = a90["p_first_season"].to_numpy() == 2024
        SEG = {"R": gt == "R", "F": gt == "F", "rookie": rk, "veteran": ~rk}
except Exception:  # noqa: BLE001
    pass


def seg_gain(p_new, p_old):
    out = {}
    for name, m in SEG.items():
        dsse = float(np.sum((p_old[m] - y_va[m]) ** 2) - np.sum((p_new[m] - y_va[m]) ** 2))
        out[name] = 100000 * dsse / (len(y_va) * DEN)
    return out


# =====================================================================
# 학습 실행 (시드42 → 조기 게이트 → 시드7)
# =====================================================================
meta = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": gpu_name, "vram_gb": vram_total, "amp_dtype": str(AMP_DTYPE), "threads": args.threads,
        "n_fit": int(len(idx_fit)), "n_val": int(len(idx_val)), "n_2024": int(len(y_va)), "row_id_sha256": sha,
        "in_dim": int(IN_DIM), "feature_names": prep.names, "FEATS79": FEATS, "cat5": CAT5,
        "hparams": {"width": 256, "blocks": 3, "hidden": 512, "dropout": 0.05, "lr": 1e-3, "wd": 1e-5,
                    "batch": args.batch, "epochs": args.epochs, "patience": args.patience, "clip": 1.0},
        "seeds": {}, "S_cb": S_cb}
preds = {}
for sd in SEEDS:
    if elapsed_min() > args.time_limit_min - 15:
        log("시간 상한 — 남은 시드 생략")
        break
    batch = args.batch
    while True:
        try:
            p_sd, info = train_seed(sd, batch)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= 4096:
                fail("OOM at batch 4096", meta)
            batch = 4096
            log(f"   OOM → batch {batch} 로 재시도")
    if p_sd is None:
        fail(f"seed {sd}: {info}", meta)
    if not np.isfinite(p_sd).all():
        fail(f"seed {sd}: 예측에 NaN/Inf", meta)
    np.save(f"{PFX}_mlp_seed{sd}.npy", p_sd.astype("float32"))
    preds[sd] = p_sd
    st = blend_stats(p_sd, p_cb)
    st_pair = blend_stats(p_sd, p_cb_seed[sd]) if sd in p_cb_seed else None
    meta["seeds"][sd] = {**info, "vs_cb_mean": {k: v for k, v in st.items() if k != "grid"}, "grid_vs_cb_mean": st["grid"],
                         "vs_cb_same_seed": ({k: v for k, v in st_pair.items() if k != "grid"} if st_pair else None)}
    tick(f"seed {sd}: S_mlp {st['S_mlp']:.1f}  d {st['d']:+.1f}  K {st['K']:.1f}  w* {st['w_star']:.3f}  "
         f"이론이득 {st['theoretical_gain']:+.1f}  실제최적 {st['actual_best_gain']:+.1f}@w={st['w_best']}  shape {st['shape_mlp']:+.1f}  "
         f"벌점 {st['pen_mlp']:.1f}  best_epoch {info['best_epoch']}")
    # 7. 첫 시드 조기 중단
    if sd == SEEDS[0] and not args.smoke:
        reasons = []
        if DEV.type != "cuda":
            reasons.append("CPU 실행")
        if st["shape_mlp"] + shape(p_cb) <= 0:
            reasons.append("2024 변별력 ≤ 0")
        if st["actual_best_gain"] < 5:
            reasons.append(f"최적 혼합이득 {st['actual_best_gain']:+.1f} < +5")
        if st["d"] <= -st["K"]:
            reasons.append(f"d {st['d']:+.1f} <= -K {-st['K']:.1f}")
        if reasons:
            with open(f"{PFX}_mlp_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=1, default=float)
            fail("시드42 조기 게이트 실패: " + "; ".join(reasons), {"seed42": meta["seeds"][sd]})

# =====================================================================
# 10. 최종 판정 + 11. 저장물
# =====================================================================
if len(preds) == 0:
    fail("완료된 시드 없음", meta)
p_mean = np.mean([preds[s] for s in preds], axis=0)
np.save(f"{PFX}_mlp_mean.npy", p_mean.astype("float32"))
st = blend_stats(p_mean, p_cb)
w_probe = float(min(max(st["w_star"], 0.10), 0.30))
per_seed_probe = {s: raw_score((1 - w_probe) * p_cb + w_probe * preds[s], y_va) - S_cb for s in preds}
seg = seg_gain((1 - w_probe) * p_cb + w_probe * p_mean, p_cb) if SEG else {}
# 평균 이동만으로 얻는 이득: 혼합의 평균을 cb 평균에 맞춰도 남는 이득
p_bl = (1 - w_probe) * p_cb + w_probe * p_mean
gain_probe = raw_score(p_bl, y_va) - S_cb
gain_probe_meanmatched = raw_score(p_bl - p_bl.mean() + p_cb.mean(), y_va) - S_cb
solo = {s: raw_score(preds[s], y_va) for s in preds}
solo_arr = np.array(list(solo.values()))
# K 시드분산 보정 (HANDOFF §1.9 항등식): 2시드 평균끼리의 D_raw 에는 양쪽 모델의 잔여 시드잡음이 K̄/4 씩 들어 있다.
#   K̄ = (1e5/DEN)·E[(p_seed42 − p_seed7)²]  →  K_adj = K − K̄_cb/4 − K̄_mlp/4  (무한시드 극한의 순수 다양성)
Kbar_cb = ((100000.0 / DEN) * float(np.mean((p_cb_seed[42] - p_cb_seed[7]) ** 2))
           if (42 in p_cb_seed and 7 in p_cb_seed) else 4.0 * (873.9 - 853.8))   # 폴백: lab/89_result.txt
Kbar_mlp = ((100000.0 / DEN) * float(np.mean((preds[SEEDS[0]] - preds[SEEDS[1]]) ** 2))
            if len(preds) == 2 else float("nan"))
K_adj = st["K"] - Kbar_cb / 4.0 - (0.0 if np.isnan(Kbar_mlp) else Kbar_mlp / 4.0)

rows = []
for w in WEIGHTS:
    rec = {"w_mlp": w, "gain_mean": st["grid"][w]}
    for s in preds:
        rec[f"gain_seed{s}"] = raw_score((1 - w) * p_cb + w * preds[s], y_va) - S_cb
    rows.append(rec)
pd.DataFrame(rows).to_csv(f"{PFX}_blend_grid.csv", index=False)

gates = {
    "1_cuda_two_seeds": (DEV.type == "cuda" and len(preds) == 2),
    "2_K>=100": st["K"] >= 100,
    "3_d>-K": st["d"] > -st["K"],
    "4_w_star>=0.10": st["w_star"] >= 0.10,
    "5_actual_gain>=15": st["actual_best_gain"] >= 15,
    "6_shape>0": (st["shape_mlp"] + shape(p_cb)) > 0,
    "7_probe_both_seeds_positive": all(v > 0 for v in per_seed_probe.values()) and len(per_seed_probe) == 2,
    "8_not_only_mean_shift": gain_probe_meanmatched > 0.5 * gain_probe if gain_probe > 0 else False,
}
if seg:
    # 세그먼트 집중도 = (이득 점유율 / 행 점유율). R 88%·베테랑 80% 가 이득 대부분을 갖는 건 정상이므로 행 점유율로 정규화.
    # '단일 세그먼트 의존' = 어떤 세그먼트의 집중도 >= 2 (예: F 11.8% 행이 이득의 23.6% 초과, 신인 19.9% 행이 39.8% 초과)
    seg_conc = ({k: (seg[k] / gain_probe) / float(SEG[k].mean()) for k in seg} if gain_probe > 0 else {})
    gates["8b_no_single_segment"] = (max(seg_conc.values()) < 2.0) if gain_probe > 0 else False
    seg = {**seg, **{f"{k}_conc": v for k, v in seg_conc.items()}}
gates = {k: bool(v) for k, v in gates.items()}          # numpy.bool_ → bool (json 에 true/false 로)
status = "SMOKE" if args.smoke else ("PASS" if all(gates.values()) else "FAIL")

rep = {"status": status, "gpu": gpu_name, "model": "plain PyTorch ResMLP(256x3)", "elapsed_min": elapsed_min(),
       "S_cb": S_cb, "solo": solo, "solo_mean_of_seeds": float(solo_arr.mean()), "solo_std": float(solo_arr.std()),
       "solo_min": float(solo_arr.min()), "solo_max": float(solo_arr.max()),
       "S_mlp_mean_pred": st["S_mlp"], "d": st["d"], "D_raw": st["D_raw"], "K": st["K"],
       "Kbar_cb": Kbar_cb, "Kbar_mlp": Kbar_mlp, "K_seedvar_adjusted": K_adj,
       "w_star": st["w_star"], "theoretical_gain": st["theoretical_gain"], "actual_best_gain": st["actual_best_gain"],
       "w_best": st["w_best"], "w_probe": w_probe, "gain_at_w_probe_mean": gain_probe, "gain_at_w_probe_per_seed": per_seed_probe,
       "gain_at_w_probe_meanmatched": gain_probe_meanmatched, "segments_at_w_probe": seg,
       "shape_mlp_minus_cb": st["shape_mlp"], "pen_mlp": st["pen_mlp"], "pen_cb": st["pen_cb"],
       "pred_stats": {"mean": float(p_mean.mean()), "std": float(p_mean.std()), "min": float(p_mean.min()), "max": float(p_mean.max())},
       "gates": gates, "grid": st["grid"]}
meta["report"] = rep
with open(f"{PFX}_mlp_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1, default=float)
with open(f"{PFX}_mlp_report.json", "w", encoding="utf-8") as f:
    json.dump(rep, f, ensure_ascii=False, indent=1, default=float)

lines = [
    "=" * 78,
    f"1. {status}",
    f"2. 모델 plain PyTorch ResMLP(256×3)  GPU {gpu_name}  amp {AMP_DTYPE}",
    f"3. 총 실행시간 {elapsed_min():.1f} 분",
    f"4. cat5 기준 {S_cb:.1f}",
    "5. MLP " + " / ".join(f"seed{s} {v:.1f}" for s, v in solo.items()) + f" / 2시드평균예측 {st['S_mlp']:.1f}  (시드평균 {solo_arr.mean():.1f} std {solo_arr.std():.1f} min {solo_arr.min():.1f} max {solo_arr.max():.1f})",
    f"6. d {st['d']:+.1f}   D_raw {st['D_raw']:.3e}   K {st['K']:.1f} (시드분산 보정 K−K̄cb/4−K̄mlp/4 = {K_adj:.1f}; K̄cb {Kbar_cb:.1f} K̄mlp {Kbar_mlp:.1f})   w* {st['w_star']:.3f}",
    f"7. 이론 혼합이득 {st['theoretical_gain']:+.1f}   실제 최적 {st['actual_best_gain']:+.1f} @ w={st['w_best']}",
    f"8. w_probe={w_probe:.2f}: " + "  ".join(f"seed{s} {v:+.1f}" for s, v in per_seed_probe.items()) + f"  평균 {gain_probe:+.1f}  (평균맞춤 {gain_probe_meanmatched:+.1f})"
    + (("  세그먼트 " + " ".join(f"{k} {v:+.1f}" for k, v in seg.items())) if seg else ""),
    f"9. shape(MLP−cb) {st['shape_mlp']:+.1f}   중심벌점 MLP {st['pen_mlp']:.1f} vs cb {st['pen_cb']:.1f}   예측 mean {p_mean.mean():.4f} std {p_mean.std():.4f} [{p_mean.min():.3f}, {p_mean.max():.3f}]",
    "10. 게이트 " + "  ".join(f"{k}={'OK' if v else 'X'}" for k, v in gates.items()),
    "11. 파일 " + ", ".join([f"{PFX}_mlp_log.txt", f"{PFX}_mlp_meta.json", f"{PFX}_valid_row_id.npy", f"{PFX}_y2024.npy"]
                          + [f"{PFX}_mlp_seed{s}.npy" for s in preds] + [f"{PFX}_mlp_mean.npy"]
                          + [f"{PFX}_mlp_seed{s}.pt" for s in preds] + [f"{PFX}_mlp_preprocessor.joblib", f"{PFX}_blend_grid.csv", f"{PFX}_mlp_report.txt", f"{PFX}_mlp_report.json"]),
    "그리드 w:gain(mean) " + "  ".join(f"{w}:{g:+.1f}" for w, g in st["grid"].items()),
    "=" * 78,
]
with open(f"{PFX}_mlp_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
for ln in lines:
    log(ln)
LOGF.close()
