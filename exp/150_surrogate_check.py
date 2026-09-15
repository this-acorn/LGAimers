"""가짜 평가셋이 2025 실제 평가셋의 대리표본으로 쓸 만한지 검증한다.

exp/149 에서 λ 를 복원했으므로 LB 는 이제 두 개의 '정답'을 알려준다:
    Var(챔피언 예측) = B11/λ = 0.00271517      (표준편차 0.052107)
    E[(p1-p2)²]      = K/λ   = 0.00042611      (RMS 0.020642)

가짜 평가셋에서 같은 양을 직접 재서 맞으면, 같은 표본에서 잰
    mean(p2) - mean(p1)
도 믿을 수 있다. 챔피언은 중심이 잡혀 있으므로(E[p1] = r) 그 차이가 곧
혼합의 중심오차 d_w = W · (E[p2] - r) 를 준다.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'candidates/candidate_exp021_ours_v001_src'
STAGE = ROOT / "_surrogate150"
PYTHON = ROOT / "venv311" / "Scripts" / "python.exe"

LAMBDA = 402483.0
R_2025 = 0.4607
W = 0.35655716947524263
VAR_P1_TRUE = 1092.808353586 / LAMBDA
MSD_TRUE = 171.5001496146865 / LAMBDA


def run_component(component: str, tag: str) -> np.ndarray:
    work = STAGE / tag
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(STAGE / "base", work)
    result = subprocess.run(
        [str(PYTHON), "-c",
         f"import runpy,warnings;warnings.filterwarnings('ignore');"
         f"runpy.run_path('model/{component}', run_name='c')['main']()"],
        cwd=work, capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if result.returncode != 0:
        print((result.stderr or "")[-2500:], file=sys.stderr)
        raise SystemExit(f"{tag}: 실행 실패")
    values = pd.read_csv(work / "output" / "submission.csv")["control_success"].to_numpy(np.float64)
    shutil.rmtree(work)
    return values


def main() -> None:
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 245789
    sys.path.insert(0, str(ROOT / "exp"))
    build_fake_test = __import__("145_rehearse_reimpl").build_fake_test

    if STAGE.exists():
        shutil.rmtree(STAGE)
    base = STAGE / "base"
    (base / "data").mkdir(parents=True)
    shutil.copytree(SRC / "model", base / "model",
                    ignore=shutil.ignore_patterns("__pycache__"))
    real = pd.read_csv(ROOT / "data" / "test.csv", encoding="utf-8-sig")
    fake = build_fake_test(real, rows, 20260831)
    fake.to_csv(base / "data" / "test.csv", index=False, encoding="utf-8")
    pd.DataFrame({"row_id": fake["row_id"], "control_success": 0.0}).to_csv(
        base / "data" / "sample_submission.csv", index=False, encoding="utf-8")

    p1 = run_component("champion_inference.py", "champ")
    p2 = run_component("exp021_inference.py", "exp021")
    np.save(ROOT / "lab" / "150_fake_champion.npy", p1)
    np.save(ROOT / "lab" / "150_fake_exp021.npy", p2)
    shutil.rmtree(STAGE, ignore_errors=True)

    var1 = float(p1.var())
    msd = float(((p1 - p2) ** 2).mean())
    print(f"가짜 평가셋 {len(p1):,}행\n")
    print("검증 1 — 챔피언 예측 분산")
    print(f"  LB 가 알려준 값 : {VAR_P1_TRUE:.8f}   (sd {np.sqrt(VAR_P1_TRUE):.6f})")
    print(f"  가짜셋 실측     : {var1:.8f}   (sd {np.sqrt(var1):.6f})")
    print(f"  비율            : {var1 / VAR_P1_TRUE:.4f}\n")
    print("검증 2 — 두 컴포넌트 제곱거리")
    print(f"  LB 가 알려준 값 : {MSD_TRUE:.8f}   (RMS {np.sqrt(MSD_TRUE):.6f})")
    print(f"  가짜셋 실측     : {msd:.8f}   (RMS {np.sqrt(msd):.6f})")
    print(f"  비율            : {msd / MSD_TRUE:.4f}\n")

    gap = float(p2.mean() - p1.mean())
    blend = (1.0 - W) * p1 + W * p2
    print("측정 — 두 컴포넌트 평균 차이 (검증이 통과했을 때만 신뢰)")
    print(f"  mean(챔피언) = {p1.mean():.6f}   mean(EXP-021) = {p2.mean():.6f}")
    print(f"  차이 E[p2]-E[p1] = {gap:+.6f}")
    print(f"  혼합 평균 = {blend.mean():.6f}\n")
    print("추정 — 혼합의 중심오차와 회수 가능 점수")
    d_w = W * gap
    print(f"  d_w = W·(E[p2]-E[p1]) = {d_w:+.6f}   (챔피언은 중심이 잡혀 있어 E[p1]=r 가정)")
    print(f"  중심 벌점 λ·d_w²      = {LAMBDA * d_w * d_w:+.2f} 점")
    print(f"  → 회수하려면 혼합에서 {d_w:+.6f} 를 빼면 된다 (SHIFT_DELTA = {d_w:+.6f})")


if __name__ == "__main__":
    main()
