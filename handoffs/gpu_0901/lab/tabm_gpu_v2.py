# -*- coding: utf-8 -*-
"""
TabM-lite GPU 프로브 v2 — 연구실 머신용 (v1의 3가지 의심 + '소심함 아티팩트' 검증)

v1 결과와 문제의식:
  2023 폴드에서 TabM 2시드가 CB를 +638.7 이김 (D=6.7e-04, 혼합 +386@w0.5 격자끝).
  그러나 CB의 2023 중심벌점은 173뿐 → +638의 대부분은 다른 메커니즘.
  ★유력 가설: 2023은 CB 변별력이 -1104인 '자신감이 벌 받는 레짐 단절 해'라서,
    평균 근처로 움츠러든(소심한) 모델이 자동으로 크게 덜 잃는다.
    2025는 그런 해가 아니다 (CB +898 = 확신이 보상받는 해, 로컬 2024형).
  → 결정적 검증: '정상 해' 2024 폴드에서도 TabM이 혼합 가치를 내는가.

v2 변경:
  · 채점을 2023 + 2024 두 폴드로 (같은 모델: 학습 ≤2021, 조기종료 2022 — 둘 다 미접촉)
    ※ 2024 비교에서 CB 참조는 ≤2023 학습이라 TabM에 2년 핸디캡 — 그런데도 혼합
      이득이 나오면 진짜. 무너지면 2023 결과는 재난의 해 아티팩트.
  · 시드 2 → 4 (v1 시드 간 312점 분산 대응)
  · w 격자 0~1.0 전체 (v1의 0.5 격자끝 문제)
  · 진단 출력: 예측평균 / 변별력·중심벌점 분해 / 예측 표준편차(소심함 지표)
  · 예측 저장: lab/tabm_preds_v2.npz — 이 파일을 커밋해서 본 머신으로 보낼 것

준비물: data/train.csv + lab/tabm_ref_v2.npz (git pull)
실행:   python lab/tabm_gpu_v2.py
"""

import os
import time
import numpy as np
import pandas as pd

T0 = time.time()
SEEDS = [42, 7, 123, 2024]
K = 8
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


def decompose(p, y):
    r = y.mean()
    d = p.mean() - r
    total = 100000.0 * (1.0 - np.mean((p - y) ** 2) / (r * (1 - r)))
    pen = 100000.0 * d * d / (r * (1 - r))
    return total, total + pen, pen, d


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


print("train.csv 로딩...", flush=True)
path = "data/train.csv" if os.path.exists("data/train.csv") else "train.csv"
df = pd.read_csv(path, encoding="utf-8-sig")
df.columns = [c.replace("﻿", "").strip() for c in df.columns]
for c in df.select_dtypes("float64").columns:
    df[c] = df[c].astype("float32")
log(f"로드 {df.shape}")

tr_mask = (df["season"] <= 2021).to_numpy()
va_mask = (df["season"] == 2022).to_numpy()
TEST_YEARS = [2023, 2024]
te_masks = {Y: (df["season"] == Y).to_numpy() for Y in TEST_YEARS}
prior = float(df.loc[tr_mask, "control_success"].mean())
ft = add_features(df, prior)
y_all = df["control_success"].to_numpy("float32")

BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
ENG = ["f_count_state", "f_count_diff", "f_is_3ball", "f_is_2strike", "f_is_full",
       "f_same_hand", "f_scoring_pos", "f_any_runner", "f_log_pn", "f_log_bn",
       "f_smooth_p", "f_smooth_b", "f_form_dev1", "f_form_dev3", "f_form_trend",
       "f_pb_diff", "f_miss_p", "f_miss_prev1"]
NUM = [c for c in BASE + ENG if c not in CAT_MAPS]

blocks = []
for c, vals in CAT_MAPS.items():
    v = ft[c].astype(str).to_numpy()
    for val in vals:
        blocks.append((v == val).astype("float32")[:, None])
Xn = ft[NUM].to_numpy("float32")
med = np.nanmedian(Xn[tr_mask], axis=0)
nan_mask = np.isnan(Xn)
for j in np.where(nan_mask.any(axis=0))[0]:
    blocks.append(nan_mask[:, j].astype("float32")[:, None])
Xn = np.where(nan_mask, med[None, :], Xn)
mu = Xn[tr_mask].mean(axis=0)
sd = Xn[tr_mask].std(axis=0) + 1e-6
Xn = np.clip((Xn - mu) / sd, -5, 5).astype("float32")
X_all = np.concatenate([Xn] + blocks, axis=1)
del Xn, blocks, ft
log(f"행렬 {X_all.shape}")

X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
X_va, y_va = X_all[va_mask], y_all[va_mask]
TE = {Y: (X_all[te_masks[Y]], y_all[te_masks[Y]]) for Y in TEST_YEARS}
del X_all, df
log(f"학습 {len(X_tr):,} (≤2021) / 조기종료 {len(X_va):,} (2022) / "
    f"채점 2023 {len(TE[2023][0]):,} · 2024 {len(TE[2024][0]):,}")

ref = np.load("lab/tabm_ref_v2.npz" if os.path.exists("lab/tabm_ref_v2.npz")
              else "tabm_ref_v2.npz")
P_CB = {}
for Y in TEST_YEARS:
    y_te = TE[Y][1]
    assert int(ref[f"n_{Y}"]) == len(y_te) and int(ref[f"ysum_{Y}"]) == int(y_te.sum()), \
        f"{Y} 참조 정합 실패"
    P_CB[Y] = ref[f"p_cb_{Y}"].astype("float64")
    t, _, _, _ = decompose(P_CB[Y], y_te.astype("float64"))
    log(f"참조 정합 OK — {Y} CB 점수 {t:.1f} (본 머신 계산값과 일치해야 함: "
        f"2023=-1277.0 / 2024=+706.5)")

import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
log(f"장치: {DEV}")


class BELinear(nn.Module):
    def __init__(self, d_in, d_out, k):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=False)
        self.r = nn.Parameter(torch.empty(k, d_in).bernoulli_(0.5) * 2 - 1)
        self.s = nn.Parameter(torch.empty(k, d_out).bernoulli_(0.5) * 2 - 1)
        self.b = nn.Parameter(torch.zeros(k, d_out))

    def forward(self, x):
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

    def forward(self, x):
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


def score(p, y):
    return decompose(p, y.astype("float64"))[0]


def train_one(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TabM(X_tr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    bce = nn.BCEWithLogitsLoss()
    Xg, yg = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    best, best_state, bad = -1e18, None, 0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xg))
        for i in range(0, len(Xg), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xg[idx].to(DEV), yg[idx].to(DEV)
            loss = bce(model(xb), yb.unsqueeze(1).expand(-1, K))
            opt.zero_grad()
            loss.backward()
            opt.step()
        s_va = score(predict(model, X_va), y_va)
        if s_va > best:
            best, bad = s_va, 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
            log(f"seed{seed} ep{ep:02d}  2022검증 {s_va:8.1f} *")
        else:
            bad += 1
            log(f"seed{seed} ep{ep:02d}  2022검증 {s_va:8.1f}")
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return {Y: predict(model, TE[Y][0]) for Y in TEST_YEARS}


all_preds = {Y: [] for Y in TEST_YEARS}
for s in SEEDS:
    t0 = time.time()
    pr = train_one(s)
    for Y in TEST_YEARS:
        all_preds[Y].append(pr[Y])
    log(f"seed{s} 완료 — 2023 {score(pr[2023], TE[2023][1]):.1f} / "
        f"2024 {score(pr[2024], TE[2024][1]):.1f}  ({time.time()-t0:.0f}s)")

np.savez_compressed("lab/tabm_preds_v2.npz" if os.path.isdir("lab") else "tabm_preds_v2.npz",
                    **{f"{Y}_seed{s}": p.astype("float32")
                       for Y in TEST_YEARS for s, p in zip(SEEDS, all_preds[Y])})
log("예측 저장 완료 (tabm_preds_v2.npz) — 이 파일을 커밋/복사해서 본 머신으로 보낼 것")

print("\n" + "=" * 76, flush=True)
print("판정 v2 — 핵심: '정상 해' 2024에서도 혼합 가치가 있는가", flush=True)
print("=" * 76, flush=True)
for Y in TEST_YEARS:
    y_te = TE[Y][1].astype("float64")
    p_nn = np.mean(all_preds[Y], axis=0)
    p_cb = P_CB[Y]
    t_n, disc_n, pen_n, d_n = decompose(p_nn, y_te)
    t_c, disc_c, pen_c, d_c = decompose(p_cb, y_te)
    D = float(np.mean((p_nn - p_cb) ** 2))
    print(f"\n■ {Y} 폴드   (r={y_te.mean():.4f})", flush=True)
    print(f"  {'':8s} {'총점':>9s} {'변별력':>9s} {'중심벌점':>9s} {'예측평균':>9s} {'예측std':>8s}", flush=True)
    print(f"  {'TabM4':8s} {t_n:>9.1f} {disc_n:>9.1f} {pen_n:>9.1f} "
          f"{p_nn.mean():>9.4f} {p_nn.std():>8.4f}", flush=True)
    print(f"  {'CB참조':8s} {t_c:>9.1f} {disc_c:>9.1f} {pen_c:>9.1f} "
          f"{p_cb.mean():>9.4f} {p_cb.std():>8.4f}", flush=True)
    print(f"  D = {D:.3e}", flush=True)
    print(f"  {'w_nn':>6s} {'혼합점수':>10s} {'CB대비':>9s}", flush=True)
    best_w, best_g = 0.0, 0.0
    for w in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        sc = score((1 - w) * p_cb + w * p_nn, TE[Y][1])
        g = sc - t_c
        if g > best_g:
            best_w, best_g = w, g
        print(f"  {w:>6.2f} {sc:>10.1f} {g:>+9.1f}", flush=True)
    print(f"  → {Y} 최적 w_nn={best_w:.2f} ({best_g:+.1f})", flush=True)
print("\n해석 가이드:", flush=True)
print("  · 2024 혼합 이득 +15↑ & 2023과 최적 w 부호 일치 → 진짜 다양성. 본검증 승격", flush=True)
print("  · 2024에서 이득 소멸/음수 → 2023 결과는 '재난의 해에 소심한 모델' 아티팩트", flush=True)
print("  · TabM 예측std가 CB보다 크게 작으면 소심함 가설 지지", flush=True)
print("=" * 76, flush=True)
