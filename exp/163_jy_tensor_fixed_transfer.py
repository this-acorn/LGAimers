# -*- coding: utf-8 -*-
"""Forward validation of a fixed Tensor-EB residual configuration.

Use regular-season rows, three zero-prior empirical-Bayes tables, component
weights (0.50, 0.40, 0.10), shrinkage (500, 1000, 500), and scale 0.58.
Compare additional fixed scales as diagnostics; freeze the candidate before
2024 evaluation and require positive transfer in both validation years.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train.csv"
OUT_JSON = ROOT / "lab" / "163_jy_tensor_fixed_transfer.json"
OUT_TXT = ROOT / "lab" / "163_jy_tensor_fixed_transfer.txt"

SPECS = (
    ("pitcher_batter_hand", ("pitcher_id", "batter_hand"), 0.50, 500.0),
    (
        "pitcher_count_base",
        ("pitcher_id", "_count_state", "base_state"),
        0.40,
        1000.0,
    ),
    (
        "hand_count_base",
        ("pitcher_hand", "batter_hand", "_count_state", "base_state"),
        0.10,
        500.0,
    ),
)
FROZEN_SCALE = 0.58
DIAGNOSTIC_SCALES = (0.0, 0.40, 0.55, 0.58, 0.75, 1.0)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_effect(e158, source, target, baseline, validation):
    is_r_source = source["game_type"].to_numpy() == "R"
    source_r = source.loc[is_r_source].reset_index(drop=True)
    residual = target[is_r_source] - baseline[is_r_source]
    effect = np.zeros(len(validation), dtype=np.float64)
    components = {}
    for name, keys, weight, alpha in SPECS:
        correction, diagnostics = e158.lookup_eb(
            source_r, residual, validation, keys, alpha
        )
        effect += weight * correction
        components[name] = {
            "keys": list(keys),
            "weight": weight,
            "alpha": alpha,
            **diagnostics,
        }
    is_r_validation = validation["game_type"].to_numpy() == "R"
    effect[~is_r_validation] = 0.0
    if not np.isfinite(effect).all() or np.any(effect[~is_r_validation] != 0.0):
        raise AssertionError("invalid effect or F fallback")
    return effect, components


def evaluate(e155, baseline, effect, target, rows, scale):
    game_type = rows["game_type"].to_numpy()
    result = e155.metrics(baseline, scale * effect, target, game_type)
    candidate = np.clip(baseline + scale * effect, 0.0, 1.0)
    is_f = game_type == "F"
    if not np.array_equal(candidate[is_f], baseline[is_f]):
        raise AssertionError("F rows changed")
    result["clipped_rate"] = float(
        np.mean((baseline + scale * effect < 0.0) | (baseline + scale * effect > 1.0))
    )
    return result


def main():
    started = time.time()
    e158 = load_module(ROOT / "exp" / "158_robust_conditional_tensor_eb.py", "e158")
    e155 = load_module(ROOT / "exp" / "155_cause_runner_currentblend_gate.py", "e155")
    columns = [
        "season", "game_month", "game_type", "pitcher_id", "pitcher_hand",
        "batter_hand", "balls_before", "strikes_before", "base_state",
        "control_success",
    ]
    frame = e158.prepare_keys(
        pd.read_csv(TRAIN, encoding="utf-8-sig", usecols=columns, low_memory=False)
    )
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

    # For each target year, use every available official season before it.
    source23 = frame.loc[frame["season"].lt(2023)].reset_index(drop=True)
    # Earlier seasons do not have matching current-anchor OOF vectors.  The
    # strict comparable correction therefore uses 2022, the latest fully OOF
    # source, while deployment-style full-history sensitivity is handled only
    # after a successful gate.
    effect23, comp23 = tensor_effect(
        e158, rows[2022], target[2022], baseline[2022], rows[2023]
    )
    del source23
    effect24, comp24 = tensor_effect(
        e158, rows[2023], target[2023], baseline[2023], rows[2024]
    )

    fixed23 = evaluate(
        e155, baseline[2023], effect23, target[2023], rows[2023], FROZEN_SCALE
    )
    fixed24 = evaluate(
        e155, baseline[2024], effect24, target[2024], rows[2024], FROZEN_SCALE
    )
    curve23 = {
        str(scale): evaluate(
            e155, baseline[2023], effect23, target[2023], rows[2023], scale
        )
        for scale in DIAGNOSTIC_SCALES
    }
    curve24 = {
        str(scale): evaluate(
            e155, baseline[2024], effect24, target[2024], rows[2024], scale
        )
        for scale in DIAGNOSTIC_SCALES
    }
    candidate24 = np.clip(baseline[2024] + FROZEN_SCALE * effect24, 0.0, 1.0)
    month24 = pd.to_numeric(rows[2024]["game_month"], errors="coerce").to_numpy()
    early = e158.subset_gain(
        e155, baseline[2024], candidate24, target[2024], month24 <= 6
    )
    late = e158.subset_gain(
        e155, baseline[2024], candidate24, target[2024], month24 > 6
    )
    bootstrap = e158.cluster_bootstrap(
        baseline[2024], candidate24, target[2024],
        rows[2024]["pitcher_id"].to_numpy(np.int64), draws=5000, seed=163,
    )
    passed = bool(
        fixed23["raw_gain"] > 0.0
        and fixed24["raw_gain"] >= 8.0
        and early["gain"] > 0.0
        and late["gain"] > 0.0
        and bootstrap["p025"] > 0.0
        and fixed24["F_contribution"] == 0.0
    )
    status = "PASS_FOR_DEPLOY_ENGINEERING" if passed else "FAIL_NO_SUBMISSION"
    payload = {
        "experiment": 163,
        "status": status,
        "analysis_only": True,
        "reads_test": False,
        "uses_lb": False,
        "third_party_code_models_predictions": False,
        "method_source": "public aggregate prose in issue #6",
        "specs": [
            {"name": n, "keys": list(k), "weight": w, "alpha": a}
            for n, k, w, a in SPECS
        ],
        "frozen_scale": FROZEN_SCALE,
        "fixed_2023": fixed23,
        "fixed_2024": fixed24,
        "curve_2023_diagnostic": curve23,
        "curve_2024_diagnostic": curve24,
        "components_2023": comp23,
        "components_2024": comp24,
        "early_2024": early,
        "late_2024": late,
        "bootstrap_2024": bootstrap,
        "elapsed_seconds": time.time() - started,
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "=== exp163 fixed JY-style Tensor-EB transfer ===",
        "official train + our OOF only; no external code/model/prediction/ZIP",
        f"frozen scale={FROZEN_SCALE:.2f} 2023: {fixed23}",
        f"frozen scale={FROZEN_SCALE:.2f} 2024: {fixed24}",
        "2023 curve: " + " ".join(
            f"s={s}:{v['raw_gain']:+.3f}" for s, v in curve23.items()
        ),
        "2024 curve: " + " ".join(
            f"s={s}:{v['raw_gain']:+.3f}" for s, v in curve24.items()
        ),
        f"2024 early={early['gain']:+.3f} late={late['gain']:+.3f}",
        f"2024 bootstrap={bootstrap}",
        f"FINAL: {status}",
        f"elapsed={payload['elapsed_seconds']:.2f}s",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
