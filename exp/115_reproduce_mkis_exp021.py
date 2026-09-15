# -*- coding: utf-8 -*-
"""Reproduce temporal-pipeline OOF predictions and evaluate ensemble diversity.

Run the five prerequisite experiments in an isolated local working directory,
verify fold metrics, and compare blend geometry against frozen multiclass,
hierarchical, and affine-calibrated OOF predictions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "lab" / "115_mkis_exp021_work"
CANONICAL = ROOT / "reference" / "mk_isos_lgaimers9"
LOG = ROOT / "lab" / "115_mkis_exp021_reproduction.log"
OUT_JSON = ROOT / "lab" / "115_mkis_exp021_diversity.json"
OUT_TEXT = ROOT / "lab" / "115_mkis_exp021_diversity.txt"

EXPECTED_ENV = {
    "python": "3.12",
    "numpy": "2.5.1",
    "pandas": "3.0.5",
    "sklearn": "1.9.0",
    "lightgbm": "4.6.0",
}

STAGES = [
    (
        "r_full_residual",
        "train_exp019_r_full_residual.py",
        "artifacts/EXP-019/r_full_residual/rfull_l63_m1000_i300/"
        "predictions_branch_w075_2024.npy",
    ),
    (
        "histgb_residual",
        "train_exp019_histgb_residual.py",
        "artifacts/EXP-019/histgb_residual/hist_l15_d4_m3000_i160/"
        "predictions_branch_w100_2024.npy",
    ),
    (
        "team_eb_ensemble",
        "train_exp019_team_eb_ensemble.py",
        "artifacts/EXP-019/team_eb_ensemble/"
        "predictions_all_prior_s1000_2024.npy",
    ),
    (
        "pitcher_count_reference",
        "train_exp020_pitcher_count_eb_atop_team.py",
        "artifacts/EXP-020/pitcher_count_eb_atop_team/"
        "predictions_team_pc_all_2024.npy",
    ),
    (
        "low_rank_pitcher_context",
        "train_exp020_low_rank_pitcher_context_eb.py",
        "artifacts/EXP-020/low_rank_pitcher_context_eb/"
        "predictions_lowrank_s300_r6_2024.npy",
    ),
]

EXPECTED_SKILL = {
    2022: 1789.5967932082258,
    2023: 907.5416355312283,
    2024: 869.9211702032806,
}


def write_log(message: str) -> None:
    timestamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(timestamped, flush=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(timestamped + "\n")


def check_environment() -> dict[str, str]:
    import lightgbm
    import sklearn

    found = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
    }
    for name, expected in EXPECTED_ENV.items():
        if found[name] != expected:
            raise RuntimeError(
                f"environment mismatch {name}: expected {expected}, found {found[name]}"
            )
    return found


def run_stage(label: str, script: str, expected_output: str) -> dict:
    output_path = WORK / expected_output
    if output_path.is_file():
        write_log(f"SKIP {label}: resume artifact exists: {output_path.relative_to(WORK)}")
        return {"label": label, "status": "resumed", "seconds": 0.0}

    command = [sys.executable, "-u", str(WORK / "experiments" / script)]
    write_log(f"START {label}: {script}")
    started = time.time()
    process = subprocess.Popen(
        command,
        cwd=WORK,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=os.environ.copy(),
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line.rstrip("\r\n"))
        lines.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    reader_finished = False
    while process.poll() is None or not reader_finished:
        try:
            line = lines.get(timeout=30.0)
        except queue.Empty:
            write_log(
                f"HEARTBEAT {label}: elapsed={(time.time() - started) / 60:.1f} min"
            )
            continue
        if line is None:
            reader_finished = True
        elif line:
            write_log(f"{label} | {line}")
    returncode = process.wait()
    seconds = time.time() - started
    if returncode != 0:
        raise RuntimeError(f"stage {label} exited {returncode}")
    if not output_path.is_file():
        raise RuntimeError(f"stage {label} missing expected output {output_path}")
    write_log(f"DONE {label}: {seconds / 60:.2f} min")
    return {"label": label, "status": "completed", "seconds": seconds}


def raw_score(probability: np.ndarray, target: np.ndarray) -> float:
    rate = float(target.mean())
    return 100000.0 * (
        1.0 - float(np.mean((probability - target) ** 2)) / (rate * (1.0 - rate))
    )


def champion_oof(year: int) -> np.ndarray:
    if year == 2024:
        base = np.load(ROOT / "lab" / "89_cat5.npy").astype(np.float64)
        effect = np.load(ROOT / "lab" / "103_v18_effect_2024.npy").astype(
            np.float64
        )
    else:
        seed42 = np.load(
            ROOT / "lab" / f"104_cat5_y{year}_probs_seed42.npy"
        ).astype(np.float64)[:, 0]
        seed7 = np.load(
            ROOT / "lab" / f"104_cat5_y{year}_probs_seed7.npy"
        ).astype(np.float64)[:, 0]
        base = 0.5 * (seed42 + seed7)
        effect = np.load(
            ROOT / "lab" / f"104_v18_effect_y{year}.npy"
        ).astype(np.float64)
    corrected = np.clip(base + 0.30 * effect, 0.0, 1.0)
    return np.clip(0.49 + 1.06 * (corrected - 0.49) - 0.0066, 0.0, 1.0)


def blend_geometry(
    champion: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    rate = float(target.mean())
    denominator = rate * (1.0 - rate)
    champion_score = raw_score(champion, target)
    candidate_score = raw_score(candidate, target)
    d = candidate_score - champion_score
    diversity = (
        100000.0
        * float(np.mean((candidate - champion) ** 2))
        / denominator
    )
    if diversity > 0.0:
        weight = float(np.clip((d + diversity) / (2.0 * diversity), 0.0, 1.0))
    else:
        weight = 0.0
    blended = (1.0 - weight) * champion + weight * candidate
    midpoint = 0.5 * (champion + candidate)
    return {
        "champion_score": champion_score,
        "candidate_score": candidate_score,
        "d": d,
        "K": diversity,
        "w_candidate": weight,
        "analytic_gain": raw_score(blended, target) - champion_score,
        "midpoint_score": raw_score(midpoint, target),
        "midpoint_gain": raw_score(midpoint, target) - champion_score,
        "prediction_correlation": float(np.corrcoef(champion, candidate)[0, 1]),
    }


def evaluate(stages: list[dict], environment: dict[str, str]) -> dict:
    metrics_path = (
        WORK
        / "artifacts"
        / "EXP-020"
        / "low_rank_pitcher_context_eb"
        / "validation_metrics.json"
    )
    reproduced_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    fold_results = {}
    exact_metrics = True
    train = pd.read_csv(
        ROOT / "data" / "train.csv",
        encoding="utf-8-sig",
        usecols=["season", "control_success", "game_type"],
    )
    for year in (2022, 2023, 2024):
        candidate = np.load(
            WORK
            / "artifacts"
            / "EXP-020"
            / "low_rank_pitcher_context_eb"
            / f"predictions_lowrank_s300_r6_{year}.npy"
        ).astype(np.float64)
        saved_target = np.load(
            WORK
            / "artifacts"
            / "EXP-020"
            / "low_rank_pitcher_context_eb"
            / f"targets_{year}.npy"
        ).astype(np.float64)
        year_rows = train[train["season"] == year].reset_index(drop=True)
        target = year_rows["control_success"].to_numpy(np.float64)
        if not np.array_equal(saved_target, target):
            raise RuntimeError(f"target or row-order mismatch for {year}")
        champion = champion_oof(year)
        if not (len(champion) == len(candidate) == len(target)):
            raise RuntimeError(f"prediction length mismatch for {year}")
        published = EXPECTED_SKILL[year]
        reproduced = float(
            reproduced_metrics["folds"][str(year)]["candidates"][
                "lowrank_s300_r6"
            ]["skill_score_unclipped"]
        )
        metric_difference = reproduced - published
        exact_metrics &= abs(metric_difference) <= 1e-9
        geometry = blend_geometry(champion, candidate, target)
        game_type = year_rows["game_type"].astype(str).to_numpy()
        geometry.update(
            {
                "rows": len(target),
                "published_candidate_score": published,
                "reproduced_candidate_score": reproduced,
                "published_metric_difference": metric_difference,
                "target_exact": True,
                "F_rows": int(np.sum(game_type == "F")),
                "R_rows": int(np.sum(game_type == "R")),
            }
        )
        fold_results[str(year)] = geometry

    gains = [fold_results[str(year)]["analytic_gain"] for year in (2022, 2023, 2024)]
    diversities = [fold_results[str(year)]["K"] for year in (2022, 2023, 2024)]
    gate = {
        "published_metrics_exact_to_1e-9": bool(exact_metrics),
        "folds_with_gain_ge_20": int(sum(value >= 20.0 for value in gains)),
        "minimum_K": float(min(diversities)),
        "median_K": float(np.median(diversities)),
    }
    gate["pass_for_lb_probe"] = bool(
        exact_metrics
        and gate["folds_with_gain_ge_20"] >= 2
        and gate["minimum_K"] >= 150.0
    )
    return {
        "experiment": 115,
        "description": "exact public EXP-021 OOF reproduction and champion diversity",
        "environment": environment,
        "stages": stages,
        "folds": fold_results,
        "gate": gate,
        "deployment_or_submission_created": False,
    }


def write_report(payload: dict) -> None:
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "exp/115 — temporal OOF reproduction and baseline comparison",
        f"environment={payload['environment']}",
        "",
    ]
    for year, result in payload["folds"].items():
        lines.append(
            f"{year}: champ={result['champion_score']:.3f} "
            f"EXP021={result['candidate_score']:.3f} d={result['d']:+.3f} "
            f"K={result['K']:.3f} w*={result['w_candidate']:.4f} "
            f"gain={result['analytic_gain']:+.3f} midpoint={result['midpoint_gain']:+.3f} "
            f"corr={result['prediction_correlation']:.6f} "
            f"published_diff={result['published_metric_difference']:+.3e}"
        )
    gate = payload["gate"]
    lines.extend(
        [
            "",
            f"exact_metrics={gate['published_metrics_exact_to_1e-9']}",
            f"folds_gain>=20={gate['folds_with_gain_ge_20']}/3",
            f"K min/median={gate['minimum_K']:.3f}/{gate['median_K']:.3f}",
            f"PASS_FOR_LB_PROBE={gate['pass_for_lb_probe']}",
            "No bundle or submission was created.",
        ]
    )
    OUT_TEXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        write_log(line)


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    if not (WORK / "data" / "train.csv").is_file():
        raise FileNotFoundError("isolated work copy is missing data/train.csv")
    if not CANONICAL.is_dir():
        raise FileNotFoundError("required local experiment source is missing")
    environment = check_environment()
    write_log(f"environment exact: {environment}")
    stages = [run_stage(*stage) for stage in STAGES]
    payload = evaluate(stages, environment)
    write_report(payload)


if __name__ == "__main__":
    main()

