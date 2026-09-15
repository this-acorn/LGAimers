# -*- coding: utf-8 -*-
"""Read-only evaluation of five predefined hierarchical residual extensions.
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
EXPERIMENT = 107
DATA = ROOT / "data/train.csv"
OOF_PATH = ROOT / "lab/24b_preds.npz"
OUT_TXT = ROOT / "lab/107_v18_extension_probe.txt"
OUT_JSON = ROOT / "lab/107_v18_extension_probe.json"
DISCOVERY = (2021, 2022, 2023)
CONFIRMATION = (2022, 2023, 2024)
WEIGHTS = np.round(np.arange(0.0, 0.501, 0.05), 2)
BASE_GAMMA = 0.30
BOOTSTRAP_DRAWS = 5000
ALPHA = 0.05

# Predeclared directly from the public V25.5 HIER list.  No k/key/bin search.
AXES = {
    "pitcher_bteam": {
        "keys": ["game_type", "pitcher_id", "batter_team_id"],
        "parent": "pitcher",
        "k": 260.0,
    },
    "pitcher_count": {
        "keys": ["game_type", "pitcher_id", "_count_state"],
        "parent": "pitcher",
        "k": 190.0,
    },
    "pitcher_inning": {
        "keys": ["game_type", "pitcher_id", "_inning_phase"],
        "parent": "pitcher",
        "k": 230.0,
    },
    "pitcher_baseout": {
        "keys": ["game_type", "pitcher_id", "_baseout_state"],
        "parent": "pitcher",
        "k": 280.0,
    },
    "pitcher_batter": {
        "keys": ["game_type", "pitcher_id", "batter_id"],
        "parent": "pitcher_bhand",
        "k": 700.0,
    },
}


def load_v18_module():
    path = ROOT / "exp/103_v18_residual.py"
    spec = importlib.util.spec_from_file_location("v18_probe_103_ext", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v18 = load_v18_module()


def score(probability: np.ndarray, target: np.ndarray) -> float:
    rate = float(target.mean())
    return 100000.0 * (
        1.0 - float(np.mean((probability - target) ** 2)) / (rate * (1.0 - rate))
    )


def shape_gain(base: np.ndarray, candidate: np.ndarray, target: np.ndarray) -> float:
    same_mean = candidate - candidate.mean() + base.mean()
    return score(same_mean, target) - score(base, target)


def contribution(base, candidate, target, mask):
    rate = float(target.mean())
    improvement = (base - target) ** 2 - (candidate - target) ** 2
    return float(
        100000.0 * improvement[mask].sum() / (len(target) * rate * (1.0 - rate))
    )


def lookup(table, rows, keys, column):
    return v18.lookup(table, rows, keys, column)


def hierarchy_effects(history: pd.DataFrame, rows: pd.DataFrame):
    root_keys = ["game_type"]
    pitcher_keys = ["game_type", "pitcher_id"]
    batter_keys = ["game_type", "batter_id"]
    hand_keys = ["game_type", "pitcher_id", "batter_hand"]

    root = v18.grouped(history, root_keys)
    pitcher = v18.grouped(history, pitcher_keys)
    batter = v18.grouped(history, batter_keys)
    hand = v18.grouped(history, hand_keys)

    root_n = lookup(root, rows, root_keys, "n")
    root_s = lookup(root, rows, root_keys, "success")
    root_rate = (root_s + v18.K_ROOT * v18.ROOT_PRIOR) / (root_n + v18.K_ROOT)
    pitcher_n = lookup(pitcher, rows, pitcher_keys, "n")
    pitcher_s = lookup(pitcher, rows, pitcher_keys, "success")
    pitcher_rate = (pitcher_s + v18.K_PITCHER * root_rate) / (
        pitcher_n + v18.K_PITCHER
    )
    batter_n = lookup(batter, rows, batter_keys, "n")
    batter_s = lookup(batter, rows, batter_keys, "success")
    batter_rate = (batter_s + 300.0 * root_rate) / (batter_n + 300.0)
    hand_n = lookup(hand, rows, hand_keys, "n")
    hand_s = lookup(hand, rows, hand_keys, "success")
    hand_rate = (hand_s + v18.K_HAND * pitcher_rate) / (hand_n + v18.K_HAND)

    parents = {
        "pitcher": pitcher_rate,
        "batter": batter_rate,
        "pitcher_bhand": hand_rate,
    }
    effects = {}
    diagnostics = {}
    for name, definition in AXES.items():
        keys = definition["keys"]
        strength = float(definition["k"])
        table = v18.grouped(history, keys)
        n = lookup(table, rows, keys, "n")
        success = lookup(table, rows, keys, "success")
        parent = parents[definition["parent"]]
        posterior = (success + strength * parent) / (n + strength)
        effect = (n / (n + strength)) * (posterior - parent)
        effects[name] = effect
        diagnostics[name] = {
            "coverage": float(np.mean(n > 0)),
            "n_mean": float(n.mean()),
            "mean": float(effect.mean()),
            "std": float(effect.std()),
        }
    return effects, diagnostics


def weighted_candidate(base, current_effect, new_effect, weight):
    q0 = np.clip(base + BASE_GAMMA * current_effect, 0.0, 1.0)
    q1 = np.clip(q0 + float(weight) * new_effect, 0.0, 1.0)
    return q0, q1


def paired(base, current_effect, new_effect, target, weight):
    q0, q1 = weighted_candidate(base, current_effect, new_effect, weight)
    return {
        "raw_gain": float(score(q1, target) - score(q0, target)),
        "shape_gain": float(shape_gain(q0, q1, target)),
        "mean_shift": float(q1.mean() - q0.mean()),
    }


def select_weight(records, axis, years):
    rows = []
    for weight in WEIGHTS:
        yearly = {
            str(year): paired(
                records[year]["hgb"],
                records[year]["current"],
                records[year]["axes"][axis],
                records[year]["target"],
                weight,
            )["raw_gain"]
            for year in years
        }
        rows.append(
            {
                "weight": float(weight),
                "yearly_gain": yearly,
                "min_gain": float(min(yearly.values())),
                "mean_gain": float(np.mean(list(yearly.values()))),
            }
        )
    return max(rows, key=lambda row: (row["min_gain"], row["mean_gain"], -row["weight"]))


def bootstrap(base, candidate, target, pitcher, draws, seed):
    work = pd.DataFrame(
        {
            "pitcher": pitcher,
            "n": np.ones(len(target), dtype=np.int32),
            "y": target,
            "base_se": (base - target) ** 2,
            "candidate_se": (candidate - target) ** 2,
        }
    )
    grouped = work.groupby("pitcher", sort=False, observed=True).agg(
        n=("n", "sum"),
        y=("y", "sum"),
        base_se=("base_se", "sum"),
        candidate_se=("candidate_se", "sum"),
    )
    values = grouped[["n", "y", "base_se", "candidate_se"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    gains = np.empty(draws, dtype=float)
    groups = len(values)
    cursor = 0
    while cursor < draws:
        size = min(250, draws - cursor)
        indices = rng.integers(0, groups, size=(size, groups))
        sampled = values[indices].sum(axis=1)
        n = sampled[:, 0]
        rate = sampled[:, 1] / n
        gains[cursor : cursor + size] = 100000.0 * (
            sampled[:, 2] - sampled[:, 3]
        ) / (n * rate * (1.0 - rate))
        cursor += size
    return {
        "median": float(np.median(gains)),
        "p025": float(np.quantile(gains, 0.025)),
        "p975": float(np.quantile(gains, 0.975)),
        "prob_positive": float(np.mean(gains > 0.0)),
        "p_one_sided": float((1 + np.count_nonzero(gains <= 0.0)) / (draws + 1)),
    }


def exact_seed_bases(year):
    if year == 2024:
        return {
            42: np.load(ROOT / "lab/89_cat5_probs_seed42.npy").astype(float)[:, 0],
            7: np.load(ROOT / "lab/89_cat5_probs_seed7.npy").astype(float)[:, 0],
        }
    return {
        seed: np.load(
            ROOT / f"lab/104_cat5_y{year}_probs_seed{seed}.npy"
        ).astype(float)[:, 0]
        for seed in (42, 7)
    }


def main():
    started = time.time()
    lines = [
        f"=== exp/{EXPERIMENT} strict public-hierarchy extension audit ===",
        f"{len(AXES)} axes fixed before reading results; no k/key/bin/negative-weight search.",
    ]
    columns = [
        "season",
        "game_type",
        "pitcher_id",
        "batter_id",
        "pitcher_hand",
        "batter_hand",
        "pitcher_team_id",
        "batter_team_id",
        "balls_before",
        "strikes_before",
        "inning",
        "outs_before",
        "base_state",
        "control_success",
    ]
    frame = pd.read_csv(DATA, encoding="utf-8-sig", usecols=columns, low_memory=False)
    for column in ("game_type", "batter_hand", "base_state"):
        frame[column] = frame[column].astype("string").fillna("__MISSING__").astype(str)
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0).astype(int)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(0).astype(int)
    inning = pd.to_numeric(frame["inning"], errors="coerce").fillna(1).astype(int)
    outs = pd.to_numeric(frame["outs_before"], errors="coerce").fillna(0).astype(int)
    frame["_count_state"] = balls.astype(str) + "_" + strikes.astype(str)
    frame["_inning_phase"] = np.where(
        inning <= 3, "early", np.where(inning <= 6, "mid", np.where(inning <= 9, "late", "extra"))
    )
    frame["_baseout_state"] = frame["base_state"] + "_" + outs.astype(str)
    frame["_pressure"] = v18.pressure_code(frame)
    oof = np.load(OOF_PATH)

    records = {}
    for year in (2021, 2022, 2023, 2024):
        validation = frame[frame["season"] == year]
        history = frame[frame["season"] < year]
        target = validation["control_success"].to_numpy(float)
        hgb = oof[f"{year}_base65"].astype(float)
        if len(hgb) != len(target) or not np.array_equal(target, oof[f"{year}_y"].astype(float)):
            raise AssertionError(f"OOF alignment failed for {year}")
        current, _ = v18.make_effect(history, validation)
        axes, diagnostics = hierarchy_effects(history, validation)
        records[year] = {
            "target": target,
            "hgb": hgb,
            "current": current,
            "axes": axes,
            "pitcher": validation["pitcher_id"].to_numpy(),
            "game_type": validation["game_type"].to_numpy(),
            "diagnostics": diagnostics,
        }
        lines.append(f"{year}: rows={len(target):,} extension effects ready")

    loo = {}
    for axis in AXES:
        folds = {}
        for heldout in DISCOVERY:
            train_years = tuple(year for year in DISCOVERY if year != heldout)
            selected = select_weight(records, axis, train_years)
            evaluation = paired(
                records[heldout]["hgb"],
                records[heldout]["current"],
                records[heldout]["axes"][axis],
                records[heldout]["target"],
                selected["weight"],
            )
            q0, q1 = weighted_candidate(
                records[heldout]["hgb"],
                records[heldout]["current"],
                records[heldout]["axes"][axis],
                selected["weight"],
            )
            boot = bootstrap(
                q0,
                q1,
                records[heldout]["target"],
                records[heldout]["pitcher"],
                BOOTSTRAP_DRAWS,
                seed=10700 + heldout,
            )
            folds[str(heldout)] = {
                "train_years": list(train_years),
                "selected_weight": selected["weight"],
                **evaluation,
                "bootstrap": boot,
            }
        full = select_weight(records, axis, DISCOVERY)
        preliminary = bool(
            all(row["selected_weight"] > 0 for row in folds.values())
            and all(row["raw_gain"] > 0 and row["shape_gain"] > 0 for row in folds.values())
        )
        # Requiring the worst fold's one-sided p to pass makes the Holm test conservative.
        p_value = float(max(row["bootstrap"]["p_one_sided"] for row in folds.values()))
        loo[axis] = {
            "folds": folds,
            "full_discovery": full,
            "preliminary_pass": preliminary,
            "p_value": p_value,
            "holm_pass": False,
        }

    ordered = sorted(loo, key=lambda axis: loo[axis]["p_value"])
    holm_open = True
    for rank, axis in enumerate(ordered):
        threshold = ALPHA / (len(ordered) - rank)
        passed = bool(holm_open and loo[axis]["p_value"] <= threshold)
        loo[axis]["holm_threshold"] = threshold
        loo[axis]["holm_pass"] = passed
        if not passed:
            holm_open = False

    survivors = [
        axis for axis in AXES if loo[axis]["preliminary_pass"] and loo[axis]["holm_pass"]
    ]
    lines.extend(["", "LOO/Holm results", "axis                 weights       raw heldout       shape heldout      p(max) Holm"])
    for axis in AXES:
        folds = loo[axis]["folds"]
        lines.append(
            f"{axis:20s} "
            f"{'/'.join(f'{folds[str(y)]['selected_weight']:.2f}' for y in DISCOVERY):11s} "
            f"{'/'.join(f'{folds[str(y)]['raw_gain']:+.2f}' for y in DISCOVERY):17s} "
            f"{'/'.join(f'{folds[str(y)]['shape_gain']:+.2f}' for y in DISCOVERY):17s} "
            f"{loo[axis]['p_value']:.4f} {loo[axis]['holm_pass']}"
        )

    winner = None
    confirmation = None
    gate_pass = False
    if survivors:
        winner = max(
            survivors,
            key=lambda axis: (
                loo[axis]["full_discovery"]["min_gain"],
                loo[axis]["full_discovery"]["mean_gain"],
                -loo[axis]["full_discovery"]["weight"],
            ),
        )
        weight = float(loo[winner]["full_discovery"]["weight"])
        lines.extend(["", f"FROZEN WINNER: {winner} weight={weight:.2f}", "Exact CAT5 confirmation"])
        confirmation = {}
        for year in CONFIRMATION:
            row = records[year]
            seed_bases = exact_seed_bases(year)
            seed_results = {}
            for seed, base in seed_bases.items():
                seed_results[str(seed)] = paired(
                    base, row["current"], row["axes"][winner], row["target"], weight
                )
            ensemble = np.mean(list(seed_bases.values()), axis=0)
            result = paired(
                ensemble, row["current"], row["axes"][winner], row["target"], weight
            )
            q0, q1 = weighted_candidate(ensemble, row["current"], row["axes"][winner], weight)
            result["F_contribution"] = contribution(q0, q1, row["target"], row["game_type"] == "F")
            result["R_contribution"] = contribution(q0, q1, row["target"], row["game_type"] == "R")
            result["bootstrap"] = bootstrap(
                q0, q1, row["target"], row["pitcher"], BOOTSTRAP_DRAWS, 107 + year
            )
            result["seed_results"] = seed_results
            confirmation[str(year)] = result
            lines.append(
                f"{year}: raw={result['raw_gain']:+.3f} shape={result['shape_gain']:+.3f} "
                f"seeds={seed_results['42']['raw_gain']:+.3f}/{seed_results['7']['raw_gain']:+.3f} "
                f"F/R={result['F_contribution']:+.3f}/{result['R_contribution']:+.3f} "
                f"boot95=[{result['bootstrap']['p025']:+.3f},{result['bootstrap']['p975']:+.3f}]"
            )
        raws = [confirmation[str(year)]["raw_gain"] for year in CONFIRMATION]
        shapes = [confirmation[str(year)]["shape_gain"] for year in CONFIRMATION]
        seeds_nonnegative = all(
            confirmation[str(year)]["seed_results"][str(seed)]["raw_gain"] >= 0
            for year in CONFIRMATION
            for seed in (42, 7)
        )
        y2024 = confirmation["2024"]
        gate_pass = bool(
            min(raws) >= 3.0
            and np.mean(raws) >= 6.0
            and min(shapes) > 0.0
            and seeds_nonnegative
            and y2024["raw_gain"] >= 8.0
            and y2024["shape_gain"] >= 5.0
            and y2024["F_contribution"] >= 0.0
            and y2024["R_contribution"] >= 0.0
            and y2024["bootstrap"]["p025"] > 0.0
        )
    else:
        lines.extend(["", "No LOO+Holm survivor; exact confirmation skipped by design."])

    lines.extend(
        [
            "",
            f"FINAL GATE {'PASS' if gate_pass else 'FAIL'}",
            "PASS would authorize only a separate audited bundle; this script never builds/submits.",
            f"elapsed={time.time()-started:.1f}s",
        ]
    )
    output = {
        "experiment": EXPERIMENT,
        "baseline_gamma": BASE_GAMMA,
        "axes": AXES,
        "weight_grid": WEIGHTS.tolist(),
        "selection": "2021-2023 leave-one-year-out; conservative worst-fold p; Holm alpha=.05; top1 only",
        "loo": loo,
        "survivors": survivors,
        "winner": winner,
        "confirmation": confirmation,
        "gate_pass": gate_pass,
        "elapsed_seconds": time.time() - started,
    }
    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
