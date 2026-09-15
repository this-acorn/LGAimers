# -*- coding: utf-8 -*-
"""Forward validation of the strongest signed endpoint direction from exp159.

The direction is the saved raw CAT5 forward-OOF probability minus the locked
current blend.  Signed weights are fitted only on earlier OOF years and then
applied unchanged to the next year.  No test data or submission is touched.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "lab" / "160_cat5_antiblend_forward.json"
OUT_TXT = ROOT / "lab" / "160_cat5_antiblend_forward.txt"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("e155_for_anti", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(p: np.ndarray, y: np.ndarray) -> float:
    rate = float(y.mean())
    return float(100000.0 * (1.0 - np.mean((p - y) ** 2) / (rate * (1.0 - rate))))


def optimal_weight(base: np.ndarray, endpoint: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    direction = np.where(mask, endpoint - base, 0.0)
    return float(((y - base) @ direction) / (direction @ direction))


def evaluate(base: np.ndarray, endpoint: np.ndarray, y: np.ndarray, game: np.ndarray,
             weights: dict[str, float]) -> dict[str, float]:
    effect = np.zeros(len(y), dtype=np.float64)
    is_r = game == "R"
    if "ALL" in weights:
        effect = weights["ALL"] * (endpoint - base)
    else:
        effect[is_r] = weights.get("R", 0.0) * (endpoint[is_r] - base[is_r])
        effect[~is_r] = weights.get("F", 0.0) * (endpoint[~is_r] - base[~is_r])
    candidate = np.clip(base + effect, 0.0, 1.0)
    base_score = score(base, y)
    denominator = float(y.mean() * (1.0 - y.mean()))
    row_gain = 100000.0 * ((base - y) ** 2 - (candidate - y) ** 2) / (denominator * len(y))
    return {
        "gain": float(score(candidate, y) - base_score),
        "F": float(row_gain[~is_r].sum()),
        "R": float(row_gain[is_r].sum()),
        "mean_shift": float(candidate.mean() - base.mean()),
        "clipped_rate": float(np.mean((candidate <= 0.0) | (candidate >= 1.0))),
    }


def pooled_weight(items: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], segment: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for base, endpoint, y, game in items:
        mask = np.ones(len(y), dtype=bool) if segment == "ALL" else game == segment
        direction = np.where(mask, endpoint - base, 0.0)
        numerator += float((y - base) @ direction)
        denominator += float(direction @ direction)
    return numerator / denominator


def main() -> None:
    e155 = load_module(ROOT / "exp" / "155_cause_runner_currentblend_gate.py")
    frame = pd.read_csv(
        ROOT / "data" / "train.csv", encoding="utf-8-sig",
        usecols=["season", "game_type", "control_success"], low_memory=False,
    )
    data = {}
    for year in (2022, 2023, 2024):
        rows = frame.loc[frame["season"].eq(year)].reset_index(drop=True)
        y = rows["control_success"].to_numpy(np.float64)
        game = rows["game_type"].astype(str).to_numpy()
        n = len(rows)
        cat = e155.load_probability(e155.CAT5[year], n)
        v18 = e155.load_vector(e155.V18[year], n)
        external = e155.load_probability(e155.EXP021[year], n)
        champion = e155.champion_probability(cat, v18)
        base = e155.CHAMPION_WEIGHT * champion + e155.EXP021_WEIGHT * external
        data[year] = (base, cat, y, game)

    same_fold = {}
    for year, (base, endpoint, y, game) in data.items():
        same_fold[str(year)] = {
            segment: optimal_weight(
                base, endpoint, y,
                np.ones(len(y), dtype=bool) if segment == "ALL" else game == segment,
            )
            for segment in ("ALL", "R", "F")
        }

    # Fit on 2022, transfer to 2023.  Then refit on 2022+2023 and open 2024 once.
    fit22_all = {"ALL": same_fold["2022"]["ALL"]}
    fit22_split = {"R": same_fold["2022"]["R"], "F": same_fold["2022"]["F"]}
    transfer23 = {
        "global": evaluate(*data[2023], fit22_all),
        "split": evaluate(*data[2023], fit22_split),
    }
    history = [data[2022], data[2023]]
    fit_hist_all = {"ALL": pooled_weight(history, "ALL")}
    fit_hist_split = {"R": pooled_weight(history, "R"), "F": pooled_weight(history, "F")}
    confirm24 = {
        "global": evaluate(*data[2024], fit_hist_all),
        "split": evaluate(*data[2024], fit_hist_split),
    }
    report = {
        "analysis_only": True,
        "test_opened": False,
        "same_fold_oracle_weights": same_fold,
        "fit_2022": {"global": fit22_all, "split": fit22_split},
        "transfer_2023": transfer23,
        "fit_2022_2023": {"global": fit_hist_all, "split": fit_hist_split},
        "untouched_2024": confirm24,
        "pass": bool(
            transfer23["global"]["gain"] > 0.0
            and confirm24["global"]["gain"] > 0.0
            and transfer23["global"]["F"] >= 0.0
            and transfer23["global"]["R"] >= 0.0
            and confirm24["global"]["F"] >= 0.0
            and confirm24["global"]["R"] >= 0.0
        ),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "=== exp160 CAT5 anti-blend strict forward ===",
        "NO TEST / NO ZIP / NO SUBMISSION",
        f"same-fold weights={same_fold}",
        f"fit2022 global={fit22_all} split={fit22_split}",
        f"transfer2023={transfer23}",
        f"fit2022+23 global={fit_hist_all} split={fit_hist_split}",
        f"untouched2024={confirm24}",
        f"FINAL_PASS={report['pass']}",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
