# -*- coding: utf-8 -*-
"""Fixed new-axis Tensor-EB transfer on the current clean OOF anchor.

Original implementation from an aggregate public method description.  Reads
only official train rows and our own OOF arrays.  The fixed candidate is an
R-only zero-prior residual lookup on
``pitcher_id x batter_hand x exact_count_state`` with alpha=1500 and
scale=0.20.  F rows are an exact fallback.  No test/LB/external artifact is
read, and 2024 is not used to choose any value.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "lab" / "166_pitcher_hand_count_eb.json"
OUT_TXT = ROOT / "lab" / "166_pitcher_hand_count_eb.txt"
ALPHA = 1500.0
SCALE = 0.20
KEYS = ("pitcher_id", "batter_hand", "_count_state")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    started = time.time()
    e158 = load_module(ROOT / "exp" / "158_robust_conditional_tensor_eb.py", "e158_166")
    e155 = load_module(ROOT / "exp" / "155_cause_runner_currentblend_gate.py", "e155_166")
    columns = [
        "season", "game_month", "game_type", "pitcher_id", "pitcher_hand",
        "batter_hand", "balls_before", "strikes_before", "base_state",
        "control_success",
    ]
    frame = e158.prepare_keys(pd.read_csv(
        ROOT / "data" / "train.csv", encoding="utf-8-sig",
        usecols=columns, low_memory=False,
    ))
    rows = {
        year: frame.loc[frame["season"].eq(year)].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    target = {
        year: rows[year]["control_success"].to_numpy(np.float64) for year in rows
    }
    baseline = {
        year: e158.load_current_baseline(e155, rows[year], year) for year in rows
    }

    effects = {}
    diagnostics = {}
    for source_year, valid_year in ((2022, 2023), (2023, 2024)):
        source_mask = rows[source_year]["game_type"].to_numpy() == "R"
        correction, diagnostic = e158.lookup_eb(
            rows[source_year].loc[source_mask].reset_index(drop=True),
            target[source_year][source_mask] - baseline[source_year][source_mask],
            rows[valid_year], KEYS, ALPHA,
        )
        correction = correction.copy()
        correction[rows[valid_year]["game_type"].to_numpy() != "R"] = 0.0
        effects[valid_year] = correction
        diagnostics[valid_year] = diagnostic

    metrics = {
        year: e155.metrics(
            baseline[year], SCALE * effects[year], target[year],
            rows[year]["game_type"].to_numpy(),
        )
        for year in (2023, 2024)
    }
    candidate24 = np.clip(baseline[2024] + SCALE * effects[2024], 0.0, 1.0)
    month = rows[2024]["game_month"].to_numpy()
    early = e158.subset_gain(e155, baseline[2024], candidate24, target[2024], month <= 6)
    late = e158.subset_gain(e155, baseline[2024], candidate24, target[2024], month > 6)
    bootstrap = e158.cluster_bootstrap(
        baseline[2024], candidate24, target[2024],
        rows[2024]["pitcher_id"].to_numpy(np.int64), draws=5000, seed=166,
    )
    hand = {}
    ph = rows[2024]["pitcher_hand"].to_numpy()
    bh = rows[2024]["batter_hand"].to_numpy()
    for left in sorted(pd.unique(ph).tolist()):
        for right in sorted(pd.unique(bh).tolist()):
            mask = (ph == left) & (bh == right)
            hand[f"{left}|{right}"] = e158.subset_gain(
                e155, baseline[2024], candidate24, target[2024], mask
            )
    passed = bool(
        metrics[2023]["raw_gain"] > 0.0
        and metrics[2024]["raw_gain"] >= 8.0
        and early["gain"] > 0.0
        and late["gain"] > 0.0
        and bootstrap["p025"] > 0.0
        and metrics[2024]["F_contribution"] == 0.0
    )
    status = "PASS_FOR_DEPLOY_ENGINEERING" if passed else "FAIL_NO_SUBMISSION"
    payload = {
        "experiment": 166,
        "status": status,
        "analysis_only": True,
        "external_artifacts": False,
        "reads_test_or_lb": False,
        "keys": KEYS,
        "alpha": ALPHA,
        "scale": SCALE,
        "lookup_diagnostics": diagnostics,
        "metrics": metrics,
        "early_2024": early,
        "late_2024": late,
        "hands_2024": hand,
        "bootstrap_2024": bootstrap,
        "elapsed_seconds": time.time() - started,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "=== exp166 pitcher x batter_hand x count residual EB ===",
        f"fixed keys={KEYS} alpha={ALPHA} scale={SCALE}",
        f"2023={metrics[2023]}",
        f"2024={metrics[2024]}",
        f"early={early['gain']:+.3f} late={late['gain']:+.3f}",
        "hands=" + " ".join(f"{k}:{v['gain']:+.3f}" for k, v in hand.items()),
        f"bootstrap={bootstrap}",
        f"FINAL: {status}",
        f"elapsed={payload['elapsed_seconds']:.2f}s",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
