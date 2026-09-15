# -*- coding: utf-8 -*-
"""Read-only component-weight audit for the successful V18-style residual.

The submitted candidate used ``0.30 * (hand + pressure_hand)``.  This probe
asks one narrow follow-up question: should the two already validated residual
components have different fixed weights?

Selection uses only HGB year-wise OOF transitions 2021-2023.  The chosen pair
is then checked, without retuning, on exact CAT5 folds for 2022, 2023 and 2024.
It never reads test.csv, builds a bundle, or submits anything.
"""

from __future__ import annotations

_PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


ROOT = Path(str(_PROJECT_ROOT))
DATA = ROOT / "data/train.csv"
OOF = ROOT / "lab/24b_preds.npz"
OUT_JSON = ROOT / "lab/106_v18_component_probe.json"
OUT_TXT = ROOT / "lab/106_v18_component_probe.txt"
YEARS = (2021, 2022, 2023, 2024)
DISCOVERY = (2021, 2022, 2023)
GRID = np.round(np.arange(0.0, 0.701, 0.05), 2)
REFERENCE = (0.30, 0.30)


def load_v18_module():
    path = ROOT / "exp/103_v18_residual.py"
    spec = importlib.util.spec_from_file_location("v18_probe_103", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v18 = load_v18_module()


def make_components(history: pd.DataFrame, validation: pd.DataFrame):
    root_keys = ["game_type"]
    pitcher_keys = ["game_type", "pitcher_id"]
    hand_keys = ["game_type", "pitcher_id", "batter_hand"]
    detail_keys = ["game_type", "pitcher_id", "_pressure", "batter_hand"]

    root = v18.grouped(history, root_keys)
    pitcher = v18.grouped(history, pitcher_keys)
    hand = v18.grouped(history, hand_keys)
    detail = v18.grouped(history, detail_keys)

    root_n = v18.lookup(root, validation, root_keys, "n")
    root_s = v18.lookup(root, validation, root_keys, "success")
    root_rate = (root_s + v18.K_ROOT * v18.ROOT_PRIOR) / (root_n + v18.K_ROOT)

    pitcher_n = v18.lookup(pitcher, validation, pitcher_keys, "n")
    pitcher_s = v18.lookup(pitcher, validation, pitcher_keys, "success")
    pitcher_rate = (pitcher_s + v18.K_PITCHER * root_rate) / (
        pitcher_n + v18.K_PITCHER
    )

    hand_n = v18.lookup(hand, validation, hand_keys, "n")
    hand_s = v18.lookup(hand, validation, hand_keys, "success")
    hand_rate = (hand_s + v18.K_HAND * pitcher_rate) / (hand_n + v18.K_HAND)
    hand_component = (hand_n / (hand_n + v18.K_HAND)) * (
        hand_rate - pitcher_rate
    )

    detail_n = v18.lookup(detail, validation, detail_keys, "n")
    detail_s = v18.lookup(detail, validation, detail_keys, "success")
    detail_rate = (detail_s + v18.K_PRESSURE_HAND * hand_rate) / (
        detail_n + v18.K_PRESSURE_HAND
    )
    detail_component = (detail_n / (detail_n + v18.K_PRESSURE_HAND)) * (
        detail_rate - hand_rate
    )
    return hand_component, detail_component


def score(probability: np.ndarray, target: np.ndarray) -> float:
    rate = float(target.mean())
    return 100000.0 * (
        1.0 - float(np.mean((probability - target) ** 2)) / (rate * (1.0 - rate))
    )


def gain(base, target, hand, detail, hand_weight, detail_weight):
    candidate = np.clip(
        base + hand_weight * hand + detail_weight * detail, 0.0, 1.0
    )
    return float(score(candidate, target) - score(base, target))


def exact_base(year: int) -> np.ndarray:
    if year == 2024:
        return np.load(ROOT / "lab/89_cat5.npy").astype(np.float64)
    arrays = [
        np.load(ROOT / f"lab/104_cat5_y{year}_probs_seed{seed}.npy").astype(
            np.float64
        )[:, 0]
        for seed in (42, 7)
    ]
    return np.mean(arrays, axis=0)


def main() -> None:
    started = time.time()
    lines = [
        "=== exp/106 V18 residual component-weight pre-gate ===",
        "Selection: HGB OOF 2021-2023 minimax; confirmation: exact CAT5 2022-2024",
    ]
    columns = [
        "season",
        "game_type",
        "pitcher_id",
        "batter_hand",
        "balls_before",
        "strikes_before",
        "control_success",
    ]
    frame = pd.read_csv(DATA, encoding="utf-8-sig", usecols=columns, low_memory=False)
    frame["game_type"] = (
        frame["game_type"].astype("string").fillna("__MISSING__").astype(str)
    )
    frame["batter_hand"] = (
        frame["batter_hand"].astype("string").fillna("__MISSING__").astype(str)
    )
    frame["_pressure"] = v18.pressure_code(frame)
    oof = np.load(OOF)

    records = {}
    for year in YEARS:
        validation = frame[frame["season"] == year]
        target = validation["control_success"].to_numpy(np.float64)
        hand, detail = make_components(frame[frame["season"] < year], validation)
        hgb = oof[f"{year}_base65"].astype(np.float64)
        if len(hgb) != len(target) or not np.array_equal(
            target, oof[f"{year}_y"].astype(np.float64)
        ):
            raise AssertionError(f"OOF alignment failed for {year}")
        records[year] = {
            "target": target,
            "hand": hand,
            "detail": detail,
            "hgb": hgb,
        }
        lines.append(
            f"{year}: rows={len(target):,} hand std={hand.std():.6f} "
            f"detail std={detail.std():.6f} corr={np.corrcoef(hand, detail)[0,1]:+.4f}"
        )

    candidates = []
    for hand_weight in GRID:
        for detail_weight in GRID:
            yearly = {
                str(year): gain(
                    records[year]["hgb"],
                    records[year]["target"],
                    records[year]["hand"],
                    records[year]["detail"],
                    float(hand_weight),
                    float(detail_weight),
                )
                for year in DISCOVERY
            }
            candidates.append(
                {
                    "hand_weight": float(hand_weight),
                    "detail_weight": float(detail_weight),
                    "yearly_gain": yearly,
                    "min_gain": float(min(yearly.values())),
                    "mean_gain": float(np.mean(list(yearly.values()))),
                    "distance": float(hand_weight**2 + detail_weight**2),
                }
            )

    selected = max(
        candidates,
        key=lambda row: (row["min_gain"], row["mean_gain"], -row["distance"]),
    )
    sh = selected["hand_weight"]
    sd = selected["detail_weight"]
    lines.extend(
        [
            "",
            f"LOCKED component weights before confirmation: hand={sh:.2f}, detail={sd:.2f}",
            f"discovery gains={selected['yearly_gain']} min={selected['min_gain']:+.3f} "
            f"mean={selected['mean_gain']:+.3f}",
            "",
            "Exact CAT5 confirmation versus reference (.30,.30)",
            "year   selected   reference   incremental",
        ]
    )

    confirmation = {}
    for year in (2022, 2023, 2024):
        row = records[year]
        base = exact_base(year)
        if len(base) != len(row["target"]):
            raise AssertionError(f"CAT5 alignment failed for {year}")
        chosen_gain = gain(
            base, row["target"], row["hand"], row["detail"], sh, sd
        )
        reference_gain = gain(
            base,
            row["target"],
            row["hand"],
            row["detail"],
            *REFERENCE,
        )
        incremental = chosen_gain - reference_gain
        confirmation[str(year)] = {
            "selected_gain": chosen_gain,
            "reference_gain": reference_gain,
            "incremental_gain": incremental,
        }
        lines.append(
            f"{year} {chosen_gain:+10.3f} {reference_gain:+11.3f} {incremental:+12.3f}"
        )

    increments = [row["incremental_gain"] for row in confirmation.values()]
    gate_pass = bool(min(increments) >= 0.0 and np.mean(increments) >= 3.0)
    lines.extend(
        [
            "",
            f"GATE {'PASS' if gate_pass else 'FAIL'}: every exact-year incremental >=0 "
            f"and mean incremental >=3; min={min(increments):+.3f} "
            f"mean={np.mean(increments):+.3f}",
            "PASS authorizes only a separately verified bundle; this script never builds/submits.",
            f"elapsed={time.time()-started:.1f}s",
        ]
    )
    result = {
        "experiment": 106,
        "selection_years": list(DISCOVERY),
        "selection_rule": "maximize minimum HGB OOF raw gain; tie mean then smaller L2",
        "grid": GRID.tolist(),
        "reference_weights": {"hand": REFERENCE[0], "detail": REFERENCE[1]},
        "selected": selected,
        "confirmation": confirmation,
        "gate_pass": gate_pass,
        "elapsed_seconds": time.time() - started,
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
