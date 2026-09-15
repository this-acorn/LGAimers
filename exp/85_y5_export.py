# -*- coding: utf-8 -*-
"""
[85] 프로브 키트에 5클래스 라벨 추가 — 리뷰 반영 ①

근거: 챔피언의 +36.1이 5클래스 타겟에서 왔으므로, 후보들이 이진으로만 학습하면
타겟 이득을 포기한 채 비교당해 부당 기각될 수 있다. 전 모델 5클래스 학습으로 통일.
(예외: EBM — interpret 라이브러리가 멀티클래스에서 상호작용항을 지원하지 않아
이진 유지. 취합 때 타겟 보정 상한을 별도 표기한다.)

라벨 복원은 exp/53/66과 동일 (연속행 asof 차분, succ 일치율 100% 검증된 방식).
정렬 검증: 복원 라벨의 (클래스==0) 이 기존 이진 y 의 (==1) 과 100% 일치해야 통과.

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/85_y5_export.py   (~3분)
"""

import sys
import numpy as np

sys.path.insert(0, "exp")
from common import log, load_train

df = load_train()
dd = df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index()
pid = dd.pitcher_id.to_numpy()
n = dd.asof_pitcher_n.fillna(0).to_numpy("float64")
nxt = (pid[1:] == pid[:-1]) & (np.abs(n[1:] - n[:-1] - 1) < 1e-6)
for k, col in [("lab_mid", "asof_pitcher_middle_rate"),
               ("lab_rev", "asof_pitcher_reverse_rate")]:
    S = dd[col].fillna(0).to_numpy("float64") * n
    lab = np.full(len(dd), np.nan)
    lab[:-1] = np.where(nxt, np.round(S[1:] - S[:-1]), np.nan)
    dd[k] = np.where((lab == 0) | (lab == 1), lab, np.nan)
rec = dd.set_index("index").sort_index()
df["lab_mid"], df["lab_rev"] = rec["lab_mid"], rec["lab_rev"]

y = df.control_success.to_numpy("float64")
cls = np.full(len(df), -1, dtype="int8")
ok = df.lab_mid.notna().to_numpy() & df.lab_rev.notna().to_numpy()
m_ = df.lab_mid.to_numpy() == 1
r_ = df.lab_rev.to_numpy() == 1
cls[ok & (y == 1)] = 0
cls[ok & (y == 0) & m_ & ~r_] = 1
cls[ok & (y == 0) & ~m_ & r_] = 2
cls[ok & (y == 0) & m_ & r_] = 3
cls[ok & (y == 0) & ~m_ & ~r_] = 4

# 키트의 서브샘플 순서에 정렬 (Xtr = df[season<=2023] 원래 순서, exp/83과 동일)
tr_mask = (df.season <= 2023).to_numpy()
y5_all = cls[tr_mask]
idx = np.load("probe_kit/sub_idx.npy")
y5 = y5_all[idx]
yb = np.load("probe_kit/y_train_sub.npy")

v = y5 >= 0
agree = float(np.mean((y5[v] == 0) == (yb[v] == 1)))
log(f"정렬 검증: (5클래스==성공) vs (이진==1) 일치율 {agree*100:.2f}%  "
    f"(미복원 {np.mean(~v)*100:.2f}%)")
assert agree == 1.0, "정렬 불일치 — 내보내기 중단"

np.save("probe_kit/y5_train_sub.npy", y5)
names = ["성공", "미들만", "리버스만", "미들∩리버스", "빅미스"]
log("클래스 분포: " + "  ".join(f"{names[c]} {np.mean(y5==c)*100:.1f}%"
                               for c in range(5)) + f"  | 미복원(-1) {np.mean(~v)*100:.2f}%")
log("저장: probe_kit/y5_train_sub.npy (int8, -1인 행은 학습에서 제외할 것)")
