# -*- coding: utf-8 -*-
"""Post-hoc three-year audit of saved EXP-170 predictions; no model fitting."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "lab"
TRAIN = ROOT / "data" / "train.csv"
OUT_JSON = LAB / "171_jy_proxy_three_year_audit.json"
OUT_TXT = LAB / "171_jy_proxy_three_year_audit.txt"
EPS = 1e-6


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def score(prediction: np.ndarray, target: np.ndarray) -> float:
    rate = float(target.mean())
    return float(100000.0 * (1.0 - np.mean((prediction - target) ** 2) /
                             (rate * (1.0 - rate))))


def geometry(anchor: np.ndarray, candidate: np.ndarray,
             target: np.ndarray) -> dict[str, float]:
    d = score(candidate, target) - score(anchor, target)
    rate = float(target.mean())
    k = float(100000.0 * np.mean((candidate - anchor) ** 2) /
              (rate * (1.0 - rate)))
    w_raw = float((d + k) / (2.0 * k)) if k > 1e-15 else 0.0
    w = float(np.clip(w_raw, 0.0, 1.0))
    blended = np.clip(anchor + w * (candidate - anchor), EPS, 1.0 - EPS)
    return {
        "anchor_score": score(anchor, target),
        "candidate_score": score(candidate, target),
        "d": d, "K": k, "d_plus_K": d + k,
        "w_unconstrained": w_raw, "w_constrained": w,
        "analytic_gain": float(w * (d + k) - w * w * k),
        "actual_clipped_gain": score(blended, target) - score(anchor, target),
    }


def bootstrap(anchor: np.ndarray, candidate: np.ndarray, target: np.ndarray,
              pitcher: np.ndarray, seed: int, draws: int = 5000):
    rate = float(target.mean())
    row_gain = (100000.0 * ((anchor - target) ** 2 - (candidate - target) ** 2)
                / (len(target) * rate * (1.0 - rate)))
    grouped = pd.DataFrame({"p": pitcher, "g": row_gain}).groupby(
        "p", sort=False, dropna=False
    )["g"].sum().to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, np.float64)
    for start in range(0, draws, 100):
        count = min(100, draws - start)
        indices = rng.integers(0, len(grouped), size=(count, len(grouped)))
        samples[start:start + count] = grouped[indices].sum(axis=1)
    quantile = np.quantile(samples, [0.025, 0.5, 0.975])
    return {"p025": float(quantile[0]), "median": float(quantile[1]),
            "p975": float(quantile[2]),
            "prob_positive": float(np.mean(samples > 0.0)),
            "pitchers": int(len(grouped)), "draws": draws}


def current_anchor(e155, rows: pd.DataFrame, year: int) -> np.ndarray:
    n = len(rows)
    target = rows["control_success"].to_numpy(np.float64)
    if not np.array_equal(e155.load_vector(e155.EXP021_TARGET[year], n), target):
        raise AssertionError(f"target mismatch {year}")
    cat = e155.load_probability(e155.CAT5[year], n)
    v18 = e155.load_vector(e155.V18[year], n)
    endpoint = e155.load_probability(e155.EXP021[year], n)
    return (e155.CHAMPION_WEIGHT * e155.champion_probability(cat, v18)
            + e155.EXP021_WEIGHT * endpoint)


def audit_route(anchor, candidate, target, pitcher, month, year):
    early = month <= 6
    late = ~early
    return {
        "geometry": geometry(anchor, candidate, target),
        "early_gain": score(candidate[early], target[early]) -
                      score(anchor[early], target[early]),
        "late_gain": score(candidate[late], target[late]) -
                     score(anchor[late], target[late]),
        "pitcher_bootstrap": bootstrap(anchor, candidate, target, pitcher,
                                        seed=171000 + year),
    }


def main() -> None:
    e155 = module(ROOT / "exp" / "155_cause_runner_currentblend_gate.py", "e171")
    frame = pd.read_csv(
        TRAIN, encoding="utf-8-sig",
        usecols=["season", "game_month", "game_type", "pitcher_id", "control_success"],
        low_memory=False,
    )
    frame.columns = frame.columns.str.replace("\ufeff", "", regex=False).str.strip()
    results = {}
    lines = ["EXP171 saved EXP170 three-year route audit"]
    for year in (2022, 2023, 2024):
        rows = frame.loc[frame["season"].eq(year)].reset_index(drop=True)
        target = rows["control_success"].to_numpy(np.float64)
        anchor = current_anchor(e155, rows, year)
        team = np.load(LAB / f"170_team_y{year}_ensemble.npy",
                       allow_pickle=False).astype(np.float64)
        if team.shape != anchor.shape or not np.isfinite(team).all():
            raise AssertionError(f"team prediction mismatch {year}")
        is_f = rows["game_type"].astype(str).to_numpy() == "F"
        variants = {
            "standalone": team,
            "F_team_R_anchor": np.where(is_f, team, anchor),
            "R_team_F_anchor": np.where(~is_f, team, anchor),
        }
        results[str(year)] = {
            name: audit_route(
                anchor, candidate, target,
                rows["pitcher_id"].to_numpy(),
                pd.to_numeric(rows["game_month"], errors="raise").to_numpy(), year,
            )
            for name, candidate in variants.items()
        }
        text = results[str(year)]["F_team_R_anchor"]
        geometry_f = text["geometry"]
        boot = text["pitcher_bootstrap"]
        lines.append(
            f"{year} F-route d={geometry_f['d']:+.6f} K={geometry_f['K']:.6f} "
            f"w={geometry_f['w_unconstrained']:+.6f} early={text['early_gain']:+.6f} "
            f"late={text['late_gain']:+.6f} CI=[{boot['p025']:+.6f},{boot['p975']:+.6f}]"
        )
    gate_2024 = results["2024"]["F_team_R_anchor"]
    gate_checks = {
        "raw_gain_ge_20": gate_2024["geometry"]["d"] >= 20.0,
        "early_positive": gate_2024["early_gain"] > 0.0,
        "late_positive": gate_2024["late_gain"] > 0.0,
        "pitcher_p025_positive": gate_2024["pitcher_bootstrap"]["p025"] > 0.0,
    }
    payload = {
        "experiment": 171, "source": "saved official-only EXP170 predictions",
        "no_model_fit": True, "years": results,
        "F_route_deploy_gate_2024": {"checks": gate_checks,
                                     "pass": bool(all(gate_checks.values()))},
        "decision": "PASS_DEPLOY" if all(gate_checks.values())
                    else "FAIL_NO_DEPLOY_NO_ZIP",
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines.append(f"decision={payload['decision']} checks={gate_checks}")
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
