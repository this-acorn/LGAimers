# -*- coding: utf-8 -*-
"""[124] Strict next-season audit of a strongly-shrunk output-residual map.

This is an analysis-only pre-gate.  It does not train a model, touch test.csv,
build a submission, or tune a parameter on 2024.

Question
--------
Does a source-season residual correction indexed by

  * batter_team_id, and
  * pitcher_id x hand_matchup

add information on top of the frozen CAT5 + V18(gamma=.30) local champion?

Protocol
--------
The smoothing constants are frozen before looking at our results:
robust=(team 10000, pitcher-hand 1000), clean=(team 50, pitcher-hand 500).
For each transition, maps are fitted only to the source season's champion
residual and applied to the following season:

  2022 -> 2023
  2023 -> 2024

The source residual is centered within source-season game_type.  Those regime
means are deliberately not propagated.  All prediction arrays are pre-existing
exact two-seed CAT5 outputs; the V18 correction is also pre-existing and fixed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/train.csv"
OUT_TXT = ROOT / "lab/124_iam_residual_map_audit.txt"
OUT_JSON = ROOT / "lab/124_iam_residual_map_audit.json"
GAMMA = 0.30
EPS = 1e-6
# The first pass is a cheap direction gate.  A full cluster bootstrap is only
# justified after both strict transitions are positive.
BOOTSTRAP_DRAWS = 0

VARIANTS = {
    "robust": {"team_alpha": 10000.0, "pitcher_hand_alpha": 1000.0},
    "clean": {"team_alpha": 50.0, "pitcher_hand_alpha": 500.0},
}


def score(prediction: np.ndarray, target: np.ndarray) -> float:
    rate = float(np.mean(target))
    brier = float(np.mean((prediction - target) ** 2))
    return 100000.0 * (1.0 - brier / (rate * (1.0 - rate)))


def equal_mean_shape_gain(base: np.ndarray, candidate: np.ndarray, target: np.ndarray) -> float:
    shifted = candidate - float(np.mean(candidate)) + float(np.mean(base))
    return score(shifted, target) - score(base, target)


def domain_contribution(
    base: np.ndarray, candidate: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> float:
    rate = float(np.mean(target))
    improvement = (base - target) ** 2 - (candidate - target) ** 2
    return float(100000.0 * np.sum(improvement[mask]) / (len(target) * rate * (1.0 - rate)))


def load_cat5(year: int) -> np.ndarray:
    if year == 2024:
        paths = [
            ROOT / "lab/89_cat5_probs_seed42.npy",
            ROOT / "lab/89_cat5_probs_seed7.npy",
        ]
    else:
        paths = [
            ROOT / f"lab/104_cat5_y{year}_probs_seed42.npy",
            ROOT / f"lab/104_cat5_y{year}_probs_seed7.npy",
        ]
    return np.mean([np.load(path).astype(np.float64)[:, 0] for path in paths], axis=0)


def load_v18_effect(year: int) -> np.ndarray:
    if year == 2024:
        path = ROOT / "lab/103_v18_effect_2024.npy"
    else:
        path = ROOT / f"lab/104_v18_effect_y{year}.npy"
    return np.load(path).astype(np.float64)


def champion_prediction(year: int) -> np.ndarray:
    return np.clip(load_cat5(year) + GAMMA * load_v18_effect(year), EPS, 1.0 - EPS)


def attach_hand_matchup(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    pitcher = out["pitcher_hand"].astype("string")
    batter = out["batter_hand"].astype("string")
    unknown = pitcher.isna() | batter.isna()
    out["hand_matchup"] = np.where(pitcher.eq(batter), "same", "opposite")
    out.loc[unknown, "hand_matchup"] = "unknown"
    return out


def fit_maps(
    source: pd.DataFrame,
    prediction: np.ndarray,
    team_alpha: float,
    pitcher_hand_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = attach_hand_matchup(source)
    work["residual"] = work["control_success"].to_numpy(np.float64) - prediction
    game_mean = work.groupby("game_type", observed=True)["residual"].transform("mean")
    work["centered_residual"] = work["residual"] - game_mean

    team = (
        work.groupby("batter_team_id", observed=True, sort=False)["centered_residual"]
        .agg(["size", "sum"])
        .reset_index()
        .rename(columns={"size": "n", "sum": "residual_sum"})
    )
    team["team_correction"] = team["residual_sum"] / (team["n"] + team_alpha)

    pitcher_hand = (
        work.groupby(["pitcher_id", "hand_matchup"], observed=True, sort=False)["centered_residual"]
        .agg(["size", "sum"])
        .reset_index()
        .rename(columns={"size": "n", "sum": "residual_sum"})
    )
    pitcher_hand["pitcher_hand_correction"] = (
        pitcher_hand["residual_sum"] / (pitcher_hand["n"] + pitcher_hand_alpha)
    )
    return team, pitcher_hand


def apply_maps(
    target: pd.DataFrame, team: pd.DataFrame, pitcher_hand: pd.DataFrame
) -> tuple[np.ndarray, dict[str, float]]:
    work = attach_hand_matchup(target)
    work = work.merge(
        team[["batter_team_id", "team_correction"]],
        on="batter_team_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    work = work.merge(
        pitcher_hand[["pitcher_id", "hand_matchup", "pitcher_hand_correction"]],
        on=["pitcher_id", "hand_matchup"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    team_seen = work["team_correction"].notna().to_numpy()
    pitcher_hand_seen = work["pitcher_hand_correction"].notna().to_numpy()
    team_effect = work["team_correction"].fillna(0.0).to_numpy(np.float64)
    pitcher_hand_effect = work["pitcher_hand_correction"].fillna(0.0).to_numpy(np.float64)
    effect = team_effect + pitcher_hand_effect
    diagnostics = {
        "team_coverage": float(np.mean(team_seen)),
        "pitcher_hand_coverage": float(np.mean(pitcher_hand_seen)),
        "team_effect_mean": float(np.mean(team_effect)),
        "team_effect_std": float(np.std(team_effect)),
        "pitcher_hand_effect_mean": float(np.mean(pitcher_hand_effect)),
        "pitcher_hand_effect_std": float(np.std(pitcher_hand_effect)),
        "total_effect_mean": float(np.mean(effect)),
        "total_effect_std": float(np.std(effect)),
    }
    return effect, diagnostics


def cluster_bootstrap(
    base: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    pitcher_id: np.ndarray,
    seed: int,
) -> dict[str, float]:
    if BOOTSTRAP_DRAWS <= 0:
        return {
            "draws": 0,
            "median": None,
            "p025": None,
            "p975": None,
            "prob_positive": None,
        }
    rows = pd.DataFrame(
        {
            "pitcher_id": pitcher_id,
            "n": np.ones(len(target), dtype=np.int32),
            "y": target,
            "base_se": (base - target) ** 2,
            "candidate_se": (candidate - target) ** 2,
        }
    )
    grouped = rows.groupby("pitcher_id", observed=True, sort=False).agg(
        n=("n", "sum"),
        y=("y", "sum"),
        base_se=("base_se", "sum"),
        candidate_se=("candidate_se", "sum"),
    )
    values = grouped[["n", "y", "base_se", "candidate_se"]].to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    gains = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    cursor = 0
    while cursor < BOOTSTRAP_DRAWS:
        batch = min(250, BOOTSTRAP_DRAWS - cursor)
        indices = rng.integers(0, len(values), size=(batch, len(values)))
        sums = values[indices].sum(axis=1)
        rate = sums[:, 1] / sums[:, 0]
        gains[cursor : cursor + batch] = (
            100000.0
            * (sums[:, 2] - sums[:, 3])
            / (sums[:, 0] * rate * (1.0 - rate))
        )
        cursor += batch
    return {
        "draws": int(BOOTSTRAP_DRAWS),
        "median": float(np.median(gains)),
        "p025": float(np.quantile(gains, 0.025)),
        "p975": float(np.quantile(gains, 0.975)),
        "prob_positive": float(np.mean(gains > 0.0)),
    }


def evaluate(
    base: np.ndarray,
    effect: np.ndarray,
    target: np.ndarray,
    game_type: np.ndarray,
    pitcher_id: np.ndarray,
    seed: int,
) -> dict[str, object]:
    candidate = np.clip(base + effect, EPS, 1.0 - EPS)
    f_mask = game_type == "F"
    return {
        "base_score": float(score(base, target)),
        "candidate_score": float(score(candidate, target)),
        "raw_gain": float(score(candidate, target) - score(base, target)),
        "shape_gain": float(equal_mean_shape_gain(base, candidate, target)),
        "mean_shift": float(np.mean(candidate) - np.mean(base)),
        "F_contribution": float(domain_contribution(base, candidate, target, f_mask)),
        "R_contribution": float(domain_contribution(base, candidate, target, ~f_mask)),
        "bootstrap": cluster_bootstrap(base, candidate, target, pitcher_id, seed),
    }


def main() -> None:
    columns = [
        "season",
        "game_type",
        "pitcher_id",
        "batter_team_id",
        "pitcher_hand",
        "batter_hand",
        "control_success",
    ]
    frame = pd.read_csv(DATA, encoding="utf-8-sig", usecols=columns, low_memory=False)
    frame["game_type"] = frame["game_type"].astype("string").fillna("__MISSING__").astype(str)

    years = (2022, 2023, 2024)
    rows = {year: frame.loc[frame["season"].eq(year)].reset_index(drop=True) for year in years}
    predictions = {year: champion_prediction(year) for year in years}
    for year in years:
        expected = rows[year]["control_success"].to_numpy(np.float64)
        if len(predictions[year]) != len(expected):
            raise AssertionError(f"prediction length mismatch for {year}")

    report: dict[str, object] = {
        "experiment": 124,
        "analysis_only": True,
        "test_opened": False,
        "submission_built": False,
        "gamma": GAMMA,
        "variants": {},
    }
    lines = [
        "=== exp/124 strict next-season output-residual map audit ===",
        "base: exact 2-seed CAT5 + frozen V18 gamma=.30",
        "transitions: 2022->2023 and 2023->2024; no training/test/submission",
        "",
    ]

    for variant, params in VARIANTS.items():
        lines.append(
            f"[{variant}] team alpha={params['team_alpha']:.0f}, "
            f"pitcher-hand alpha={params['pitcher_hand_alpha']:.0f}"
        )
        transitions: dict[str, object] = {}
        for source_year, target_year in ((2022, 2023), (2023, 2024)):
            team, pitcher_hand = fit_maps(
                rows[source_year],
                predictions[source_year],
                params["team_alpha"],
                params["pitcher_hand_alpha"],
            )
            effect, diagnostics = apply_maps(rows[target_year], team, pitcher_hand)
            target = rows[target_year]["control_success"].to_numpy(np.float64)
            result = evaluate(
                predictions[target_year],
                effect,
                target,
                rows[target_year]["game_type"].to_numpy(),
                rows[target_year]["pitcher_id"].to_numpy(),
                12400 + target_year,
            )

            # Fixed ablations; diagnostic only, never used to choose the main arm.
            team_effect, _ = apply_maps(
                rows[target_year],
                team,
                pitcher_hand.assign(pitcher_hand_correction=0.0),
            )
            hand_effect, _ = apply_maps(
                rows[target_year],
                team.assign(team_correction=0.0),
                pitcher_hand,
            )
            result["team_only_raw_gain"] = float(
                score(np.clip(predictions[target_year] + team_effect, EPS, 1.0 - EPS), target)
                - score(predictions[target_year], target)
            )
            result["pitcher_hand_only_raw_gain"] = float(
                score(np.clip(predictions[target_year] + hand_effect, EPS, 1.0 - EPS), target)
                - score(predictions[target_year], target)
            )
            result["effect_diagnostics"] = diagnostics
            key = f"{source_year}_to_{target_year}"
            transitions[key] = result
            boot = result["bootstrap"]
            boot_text = (
                f"boot95=[{boot['p025']:+.3f},{boot['p975']:+.3f}]"
                if boot["draws"]
                else "bootstrap=deferred"
            )
            lines.append(
                f"  {source_year}->{target_year}: raw={result['raw_gain']:+.3f} "
                f"shape={result['shape_gain']:+.3f} F/R={result['F_contribution']:+.3f}/"
                f"{result['R_contribution']:+.3f} team/hand={result['team_only_raw_gain']:+.3f}/"
                f"{result['pitcher_hand_only_raw_gain']:+.3f} {boot_text}"
            )
        raw_values = [float(v["raw_gain"]) for v in transitions.values()]
        shape_values = [float(v["shape_gain"]) for v in transitions.values()]
        gate = bool(min(raw_values) > 0.0 and min(shape_values) > 0.0 and np.mean(raw_values) >= 8.0)
        report["variants"][variant] = {
            "parameters": params,
            "transitions": transitions,
            "mean_raw_gain": float(np.mean(raw_values)),
            "mean_shape_gain": float(np.mean(shape_values)),
            "gate_pass": gate,
        }
        lines.append(
            f"  mean raw/shape={np.mean(raw_values):+.3f}/{np.mean(shape_values):+.3f}; "
            f"gate={'PASS' if gate else 'FAIL'}"
        )
        lines.append("")

    robust_pass = bool(report["variants"]["robust"]["gate_pass"])
    report["final_gate_pass"] = robust_pass
    lines.append(f"FINAL ROBUST GATE: {'PASS' if robust_pass else 'FAIL'}")
    lines.append("A pass authorizes only a deployment-mechanics audit, not a submission.")

    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
