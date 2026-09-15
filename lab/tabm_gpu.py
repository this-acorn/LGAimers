# -*- coding: utf-8 -*-
"""
TabM-lite GPU 프로브 — 연구실 머신용 단일 파일 (xgb_gpu.py와 같은 사용법)

목적: 트리와 완전히 다른 정보 경로(NN)가 CatBoost와 '다르게 틀리는가'를 잰다.
  단독 승리가 목표가 아니다 — D(예측 불일치)와 혼합 이득이 판정 기준.
  현재 LB 898.62 = CatBoost 단독. 2025는 CB 친화 해(로컬 2023형)로 실측 확정.

구현: TabM (ICLR 2025, yandex-research)의 핵심인 BatchEnsemble MLP를 자체 구현.
  하나의 공유 가중치 + 멤버별 rank-1 어댑터(r, s, bias)로 k개 MLP를 병렬 학습.
  손실 = 멤버별 BCE의 평균 (각 멤버가 독립적으로 학습되는 것이 핵심).
  추론 = 멤버 확률 평균.

준비물 (이 폴더에 함께 둘 것):
  data/train.csv          (대회 데이터 — 이 머신에 이미 있음)
  lab/tabm_ref_2023.npz   (CatBoost 8시드 참조 예측, git pull로 받음)
실행:
  pip install torch  (이미 있으면 생략)
  python lab/tabm_gpu.py
소요: GPU 기준 시드당 수 분. 결과는 stdout — 마지막 판정 블록을 복사해 올 것.

검증 설계 (누수 없음):
  학습 = season ≤ 2021 / 내부 조기종료 검증 = season 2022 / 채점 = season 2023 (미접촉)
  → 2023은 순수 out-of-fold. 참조 CB도 같은 폴드(train ≤2022)라 공정 비교.
    ※ CB는 2022까지 학습했고 TabM은 2021까지+2022는 조기종료용 — TabM에 약간 불리한
      설정이지만 프로브 목적(다양성 측정)에는 문제없다.
"""

import os
import time
import numpy as np
import pandas as pd

T0 = time.time()
SEEDS = [42, 7]
K = 8                    # BatchEnsemble 멤버 수
HIDDEN = 512
DROPOUT = 0.15
LR = 2e-3
WD = 1e-4
BATCH = 8192
MAX_EPOCHS = 40
PATIENCE = 4
CAT_MAPS = {"top_bottom": ["T", "B"], "game_type": ["R", "F"],
            "base_state": ["___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"]}


def log(m):
    print(f"  [{time.time()-T0:6.0f}s] {m}", flush=True)


def brier_score(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - np.mean((p - y) ** 2) / (r * (1 - r)))


# =====================================================================
# 피처 (submissions/submit/script.py add_features와 동일 — 단일 파일 유지 위해 인라인)
# =====================================================================
def add_features(d, prior):
    d = d.copy()
    b, s = d["balls_before"], d["strikes_before"]
    d["f_count_state"] = (b * 3 + s).astype("int8")
    d["f_count_diff"] = (b - s).astype("int8")
    d["f_is_3ball"] = (b == 3).astype("int8")
    d["f_is_2strike"] = (s == 2).astype("int8")
    d["f_is_full"] = ((b == 3) & (s == 2)).astype("int8")
    d["f_same_hand"] = (d["pitcher_hand"] == d["batter_hand"]).astype("int8")
    d["f_scoring_pos"] = ((d["runner_on_2b"] == 1) | (d["runner_on_3b"] == 1)).astype("int8")
    d["f_any_runner"] = (d["num_runners_on"] > 0).astype("int8")
    pn = d["asof_pitcher_n"].fillna(0)
    bn = d["asof_batter_n"].fillna(0)
    d["f_log_pn"] = np.log1p(pn).astype("float32")
    d["f_log_bn"] = np.log1p(bn).astype("float32")
    a = 200.0
    d["f_smooth_p"] = ((d["asof_pitcher_success_rate"].fillna(prior) * pn + prior * a)
                       / (pn + a)).astype("float32")
    d["f_smooth_b"] = ((d["asof_batter_success_rate"].fillna(prior) * bn + prior * a)
                       / (bn + a)).astype("float32")
    d["f_form_dev1"] = (d["asof_pitcher_prev1_game_success_rate"]
                        - d["asof_pitcher_success_rate"]).astype("float32")
    d["f_form_dev3"] = (d["asof_pitcher_prev3_game_success_rate"]
                        - d["asof_pitcher_success_rate"]).astype("float32")
    d["f_form_trend"] = (d["asof_pitcher_prev1_game_success_rate"]
                         - d["asof_pitcher_prev5_game_success_rate"]).astype("float32")
    d["f_pb_diff"] = (d["asof_pitcher_success_rate"]
                      - d["asof_batter_success_rate"]).astype("float32")
    d["f_miss_p"] = d["asof_pitcher_success_rate"].isna().astype("int8")
    d["f_miss_prev1"] = d["asof_pitcher_prev1_game_success_rate"].isna().astype("int8")
    return d


# =====================================================================
# 데이터 → 행렬 (NN용: 원핫 + 표준화 + 결측 대치·지시자)
# =====================================================================
print("train.csv 로딩...", flush=True)
path = "data/train.csv" if os.path.exists("data/train.csv") else "train.csv"
df = pd.read_csv(path, encoding="utf-8-sig")
df.columns = [c.replace("﻿", "").strip() for c in df.columns]
for c in df.select_dtypes("float64").columns:
    df[c] = df[c].astype("float32")
log(f"로드 {df.shape}")

tr_mask = df["season"] <= 2021
va_mask = df["season"] == 2022
te_mask = df["season"] == 2023
prior = float(df.loc[tr_mask, "control_success"].mean())
ft = add_features(df, prior)
y_all = df["control_success"].to_numpy("float32")

BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
ENG = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
       "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
       "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
       "f_pb_diff", "f_miss_p", "f_miss_prev1"]
NUM = [c for c in BASE + ENG if c not in CAT_MAPS]

blocks, names = [], []
for c, vals in CAT_MAPS.items():
    v = ft[c].astype(str).to_numpy()
    for val in vals:
        blocks.append((v == val).astype("float32")[:, None])
        names.append(f"{c}={val}")
Xn = ft[NUM].to_numpy("float32")
med = np.nanmedian(Xn[tr_mask.to_numpy()], axis=0)
nan_mask = np.isnan(Xn)
has_nan = nan_mask.any(axis=0)
for j in np.where(has_nan)[0]:
    blocks.append(nan_mask[:, j].astype("float32")[:, None])
    names.append(f"isnan:{NUM[j]}")
Xn = np.where(nan_mask, med[None, :], Xn)
mu = Xn[tr_mask.to_numpy()].mean(axis=0)
sd = Xn[tr_mask.to_numpy()].std(axis=0) + 1e-6
Xn = np.clip((Xn - mu) / sd, -5, 5).astype("float32")
X_all = np.concatenate([Xn] + blocks, axis=1)
del Xn, blocks, ft
log(f"행렬 {X_all.shape} (수치 {len(NUM)} + 원핫 12 + 결측지시 {int(has_nan.sum())})")

X_tr, y_tr = X_all[tr_mask.to_numpy()], y_all[tr_mask.to_numpy()]
X_va, y_va = X_all[va_mask.to_numpy()], y_all[va_mask.to_numpy()]
X_te, y_te = X_all[te_mask.to_numpy()], y_all[te_mask.to_numpy()]
del X_all, df
log(f"학습 {len(X_tr):,} (≤2021) / 조기종료 {len(X_va):,} (2022) / 채점 {len(X_te):,} (2023)")

# 참조 예측 정합성 확인
ref = np.load("lab/tabm_ref_2023.npz" if os.path.exists("lab/tabm_ref_2023.npz")
              else "tabm_ref_2023.npz")
assert int(ref["n"]) == len(y_te) and int(ref["y_sum"]) == int(y_te.sum()), \
    f"참조 정합 실패: n {int(ref['n'])} vs {len(y_te)}, y_sum {int(ref['y_sum'])} vs {int(y_te.sum())}"
p_cb = ref["p_cb"].astype("float64")
log(f"참조 정합 OK — CB(8시드 풀) 2023 점수 {brier_score(p_cb, y_te):.1f}")

# =====================================================================
# TabM-lite (BatchEnsemble MLP)
# =====================================================================
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
log(f"장치: {DEV}" + (f" ({torch.cuda.get_device_name(0)})" if DEV == "cuda" else " ← GPU 없음, 느릴 수 있음"))


class BELinear(nn.Module):
    """BatchEnsemble: 공유 W + 멤버별 r(입력 스케일)·s(출력 스케일)·bias"""

    def __init__(self, d_in, d_out, k):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=False)
        # TabM 스타일: ±1 무작위 부호 초기화로 멤버 다양성 확보
        self.r = nn.Parameter(torch.empty(k, d_in).bernoulli_(0.5) * 2 - 1)
        self.s = nn.Parameter(torch.empty(k, d_out).bernoulli_(0.5) * 2 - 1)
        self.b = nn.Parameter(torch.zeros(k, d_out))

    def forward(self, x):            # x: (B, k, d_in)
        return self.lin(x * self.r) * self.s + self.b


class TabM(nn.Module):
    def __init__(self, d_in, k=K, h=HIDDEN, p=DROPOUT):
        super().__init__()
        self.k = k
        self.l1 = BELinear(d_in, h, k)
        self.l2 = BELinear(h, h, k)
        self.drop = nn.Dropout(p)
        self.head = nn.Parameter(torch.zeros(k, h))
        self.head_b = nn.Parameter(torch.zeros(k))
        nn.init.normal_(self.head, std=0.02)

    def forward(self, x):            # x: (B, d) → logits (B, k)
        z = x.unsqueeze(1).expand(-1, self.k, -1)
        z = self.drop(torch.relu(self.l1(z)))
        z = self.drop(torch.relu(self.l2(z)))
        return (z * self.head).sum(-1) + self.head_b


def predict(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 65536):
            xb = torch.from_numpy(X[i:i + 65536]).to(DEV)
            out.append(torch.sigmoid(model(xb)).mean(1).cpu().numpy())
    return np.concatenate(out).astype("float64")


def train_one(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TabM(X_tr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    bce = nn.BCEWithLogitsLoss()
    Xg = torch.from_numpy(X_tr)
    yg = torch.from_numpy(y_tr)
    best, best_state, bad = -1e18, None, 0
    n = len(Xg)
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xg[idx].to(DEV), yg[idx].to(DEV)
            logits = model(xb)                       # (B, k)
            loss = bce(logits, yb.unsqueeze(1).expand(-1, K))   # 멤버별 BCE 평균
            opt.zero_grad()
            loss.backward()
            opt.step()
        s_va = brier_score(predict(model, X_va), y_va)
        mark = ""
        if s_va > best:
            best, bad = s_va, 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
            mark = " *"
        else:
            bad += 1
        log(f"seed{seed} ep{ep:02d}  2022검증 {s_va:8.1f}{mark}")
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return predict(model, X_te)


preds = []
for s in SEEDS:
    t0 = time.time()
    preds.append(train_one(s))
    log(f"seed{s} 완료 — 2023 채점 {brier_score(preds[-1], y_te):.1f} ({time.time()-t0:.0f}s)")
p_nn = np.mean(preds, axis=0)

# =====================================================================
# 판정
# =====================================================================
s_nn, s_cb = brier_score(p_nn, y_te), brier_score(p_cb, y_te)
D = float(np.mean((p_nn - p_cb) ** 2))
D_FLOOR = 1.274e-04      # HGB 시드간 거리 (로컬 실측 참조)
print("\n" + "=" * 72, flush=True)
print("판정 (2023 폴드 = 2025의 실측 프록시)", flush=True)
print("=" * 72, flush=True)
print(f"  TabM {len(SEEDS)}시드 앙상블: {s_nn:8.1f}   /   CB 참조: {s_cb:8.1f}   (차 {s_nn-s_cb:+.1f})", flush=True)
print(f"  D(TabM, CB) = {D:.3e}   (HGB시드간 바닥 {D_FLOOR:.3e}의 {D/D_FLOOR:.1f}배)", flush=True)
print(f"\n  {'w_nn':>6s} {'혼합점수':>10s} {'CB단독대비':>10s}", flush=True)
best_w, best_g = 0.0, 0.0
for w in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
    sc = brier_score((1 - w) * p_cb + w * p_nn, y_te)
    g = sc - s_cb
    if g > best_g:
        best_w, best_g = w, g
    print(f"  {w:>6.2f} {sc:>10.1f} {g:>+10.1f}", flush=True)
print(f"\n  최적 w_nn={best_w:.2f} (+{best_g:.1f})", flush=True)
print("  게이트: D가 바닥의 2배 미만이면 폐기 / 혼합 +15↑면 본검증 승격", flush=True)
print("  이 블록 전체를 복사해서 보고할 것", flush=True)
print("=" * 72, flush=True)
