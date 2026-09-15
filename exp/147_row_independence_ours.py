"""자체 구현 zip 의 행 독립성 QA — 대회 규정 4) 방어용.

같은 row_id 의 예측이 (a) 동반 행 구성, (b) 행 순서, (c) 배치 크기와 무관하게 비트 단위로
같은지 확인한다. 기준 배치와 6가지 변형을 비교하고 허용 오차는 0 이다.

  1) 역순        전체를 뒤집어 실행
  2) 무작위순열  전체를 섞어 실행
  3) 분할        절반씩 두 번 나눠 실행
  4) 단일행      임의 행 하나만 담아 실행
  5) 중복        각 행을 두 번씩 담아 실행
  6) 소배치      앞 17행만 담아 실행

실행: python exp/147_row_independence_ours.py --zip candidate_exp021_ours_v001.zip [--n 3000]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "_qa147"
PYTHON = ROOT / "venv311" / "Scripts" / "python.exe"


def run(zip_path: Path, frame: pd.DataFrame, tag: str) -> dict:
    work = WORK / tag
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(work)
    (work / "data").mkdir()
    frame.to_csv(work / "data" / "test.csv", index=False, encoding="utf-8")
    pd.DataFrame({"row_id": frame["row_id"], "control_success": 0.0}).to_csv(
        work / "data" / "sample_submission.csv", index=False, encoding="utf-8")

    result = subprocess.run([str(PYTHON), "script.py"], cwd=work, capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if result.returncode != 0:
        print((result.stderr or "")[-2500:], file=sys.stderr)
        raise SystemExit(f"{tag}: 실행 실패")
    out = pd.read_csv(work / "output" / "submission.csv")
    shutil.rmtree(work)
    return dict(zip(out["row_id"].to_numpy(), out["control_success"].to_numpy(np.float64)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default='artifacts/candidates/candidate_exp021_ours_v001.zip')
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "exp"))
    build_fake_test = __import__("145_rehearse_reimpl").build_fake_test
    real = pd.read_csv(ROOT / "data" / "test.csv", encoding="utf-8-sig")
    base = build_fake_test(real, args.n, args.seed)

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    zip_path = ROOT / args.zip
    print(f"대상: {args.zip}   검사 행수: {len(base):,}\n")

    reference = run(zip_path, base, "reference")
    rng = np.random.default_rng(args.seed)
    half = len(base) // 2

    cases = {
        "역순": lambda: run(zip_path, base.iloc[::-1].reset_index(drop=True), "reversed"),
        "무작위순열": lambda: run(
            zip_path, base.iloc[rng.permutation(len(base))].reset_index(drop=True), "shuffled"),
        "분할": lambda: {**run(zip_path, base.iloc[:half].reset_index(drop=True), "split_a"),
                          **run(zip_path, base.iloc[half:].reset_index(drop=True), "split_b")},
        "단일행": lambda: run(zip_path, base.iloc[[len(base) // 3]].reset_index(drop=True),
                              "single"),
        "중복": lambda: run(zip_path, base.loc[base.index.repeat(2)].assign(
            row_id=lambda d: np.arange(1, 2 * len(base) + 1)).reset_index(drop=True), "dup"),
        "소배치": lambda: run(zip_path, base.iloc[:17].reset_index(drop=True), "small"),
    }

    verdicts = []
    for name, thunk in cases.items():
        got = thunk()
        if name == "중복":
            # 중복 케이스는 row_id 를 새로 매겼으므로 원본 순서로 되짚어 비교한다
            values = np.array([got[i] for i in range(1, 2 * len(base) + 1)])
            pairs = values.reshape(-1, 2)
            inner = float(np.abs(pairs[:, 0] - pairs[:, 1]).max())
            compared = np.array([reference[r] for r in base["row_id"]])
            worst = max(inner, float(np.abs(pairs[:, 0] - compared).max()))
        else:
            shared = [r for r in got if r in reference]
            worst = float(max(abs(got[r] - reference[r]) for r in shared))
        verdicts.append((name, worst, len(got)))
        print(f"  {name:8s} 행수 {len(got):6,d}   최대 절대차 {worst:.3e}   "
              f"{'통과' if worst == 0.0 else '실패'}")

    shutil.rmtree(WORK, ignore_errors=True)
    overall = max(v[1] for v in verdicts)
    print(f"\n종합 최대 절대차: {overall:.3e}")
    print("판정: " + ("전 케이스 통과 — 행 독립성 확인" if overall == 0.0 else "실패 — 배포 불가"))


if __name__ == "__main__":
    main()
