"""자체 구현 EXP-021 컴포넌트를 245,789행 규모에서 기존 산출물과 차분 검증한다.

검증 설계
  - 가드(season / pitcher_id / batter_id / asof_* 원본)는 실제 2025 test 5행의 값을 유지한다.
    이 열들은 동결 이력표와의 정합성 검사를 통과해야 하므로 임의로 흔들 수 없다.
  - 나머지 상황 열(카운트·아웃·이닝·주자·핸드·팀·게임타입·LI 등)은 어휘 전체에서 무작위로 채운다.
    group/team/lowrank 조회표와 두 GBDT의 분기를 폭넓게 태우는 것이 목적이다.
  - 두 구현에 완전히 동일한 입력을 주고 최대 절대차가 0인지 본다. 현실성은 필요 없다.

사용법:  python exp/145_rehearse_reimpl.py [--rows 245789] [--seed 20260831]
"""

from __future__ import annotations

import argparse
import os
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'candidates/candidate_exp021_w0356557_v002_src'
OURS = ROOT / "exp" / "143_exp021_component.py"
STAGE = ROOT / 'archive/scratch/_rehearse_143'
PYTHON = ROOT / "venv311" / "Scripts" / "python.exe"

# 가드가 읽는 열 — 실제 test 값을 그대로 유지한다
FROZEN = {
    "season", "pitcher_id", "batter_id",
    "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n",
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
}


def build_fake_test(real: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    reps = int(np.ceil(rows / len(real)))
    fake = pd.concat([real] * reps, ignore_index=True).iloc[:rows].reset_index(drop=True)

    balls = rng.integers(0, 4, rows)
    strikes = rng.integers(0, 3, rows)
    on1, on2, on3 = (rng.integers(0, 2, rows) for _ in range(3))
    choices = {
        "balls_before": balls, "strikes_before": strikes,
        "outs_before": rng.integers(0, 3, rows),
        "inning": rng.integers(1, 13, rows),
        "runner_on_1b": on1, "runner_on_2b": on2, "runner_on_3b": on3,
        "num_runners_on": on1 + on2 + on3,
        "base_state": np.array(["123", "12_", "1_3", "1__", "_23", "_2_", "__3", "___"],
                               dtype=object)[rng.integers(0, 8, rows)],
        "top_bottom": np.array(["T", "B"], dtype=object)[rng.integers(0, 2, rows)],
        "game_type": np.array(["R", "F"], dtype=object)[rng.integers(0, 2, rows)],
        "pitcher_hand": rng.integers(1, 3, rows),
        "batter_hand": rng.integers(1, 3, rows),
        "game_month": rng.integers(3, 11, rows),
        "game_dayofweek": rng.integers(0, 7, rows),
        "li": np.round(rng.uniform(0.0, 6.0, rows), 4),
        "home_win_expectancy": np.round(rng.uniform(0.0, 1.0, rows), 6),
        "away_win_expectancy": np.round(rng.uniform(0.0, 1.0, rows), 6),
        "run_top_before": rng.integers(0, 15, rows),
        "run_bot_before": rng.integers(0, 15, rows),
        "score_diff_home": rng.integers(-12, 13, rows),
        "score_diff_pitcher_team": rng.integers(-12, 13, rows),
    }
    for column, values in choices.items():
        if column in fake.columns and column not in FROZEN:
            fake[column] = values
    if "run_total_before" in fake.columns:
        fake["run_total_before"] = choices["run_top_before"] + choices["run_bot_before"]

    # 팀 id 는 team_effects 에 실재하는 값 + 미등록 값(효과 0 경로)을 섞는다
    for column in ("pitcher_team_id", "batter_team_id"):
        if column in fake.columns:
            fake[column] = rng.integers(12, 25, rows)   # 12~23 실재 + 24 미등록(효과 0 경로)

    fake["row_id"] = np.arange(1, rows + 1)
    return fake


def run_component(stage: Path, component: Path, tag: str) -> np.ndarray:
    work = stage / tag
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(stage / "base", work)
    shutil.copyfile(component, work / "model" / "component.py")
    started = time.time()
    result = subprocess.run(
        [str(PYTHON), "-c",
         "import runpy,warnings;warnings.filterwarnings('ignore');"
         "runpy.run_path('model/component.py', run_name='c')['main']()"],
        cwd=work, capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    elapsed = time.time() - started
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"{tag}: 컴포넌트 실행 실패")
    print(f"  [{tag}] {elapsed:6.1f}s  {result.stdout.strip().splitlines()[-1]}")
    out = pd.read_csv(work / "output" / "submission.csv")
    return out["control_success"].to_numpy(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=245789)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    if STAGE.exists():
        shutil.rmtree(STAGE)
    base = STAGE / "base"
    (base / "data").mkdir(parents=True)
    shutil.copytree(SRC / "model", base / "model",
                    ignore=shutil.ignore_patterns("__pycache__"))

    real = pd.read_csv(ROOT / "data" / "test.csv", encoding="utf-8-sig")
    fake = build_fake_test(real, args.rows, args.seed)
    fake.to_csv(base / "data" / "test.csv", index=False, encoding="utf-8")
    pd.DataFrame({"row_id": fake["row_id"], "control_success": 0.0}).to_csv(
        base / "data" / "sample_submission.csv", index=False, encoding="utf-8")
    print(f"가짜 평가셋 {len(fake):,}행 생성 (가드 열은 실제 2025 값 고정, 상황 열은 무작위)\n")

    theirs = run_component(STAGE, SRC / "model" / "exp021_inference.py", "reference")
    ours = run_component(STAGE, OURS, "ours")

    if theirs.shape != ours.shape:
        raise SystemExit("행 수가 다릅니다")
    diff = np.abs(theirs - ours)
    print(f"\n최대 절대차 : {diff.max():.3e}")
    print(f"불일치 행수 : {int((diff > 0).sum()):,} / {len(diff):,}")
    print(f"평균        : 기존 {theirs.mean():.12f} / 우리 {ours.mean():.12f}")
    print("판정        : " + ("완전 일치 — 교체 안전" if diff.max() == 0.0
                              else "불일치 — 교체 불가"))


if __name__ == "__main__":
    main()
