# -*- coding: utf-8 -*-
"""Discovery-only screen for an explicit home-venue residual hierarchy.

The deployed models see pitcher and batter team separately, but do not expose
the deterministic *venue* identity (home team) as one pooled category.  This
screen asks whether a fixed residual correction learned on 2022 transfers to
2023.  It deliberately does not load 2024 or test data unless the 2023
stability gate passes.

All lookups are learned from official train rows and are applied row by row.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "lab" / "171_venue_hierarchy_discovery.json"
OUT_TXT = ROOT / "lab" / "171_venue_hierarchy_discovery.txt"
SCALES = (0.25, 0.50, 0.75, 1.00)
PARENT_ALPHA = 1200.0
CHILD_ALPHA = 2400.0
CHILD_WEIGHT = 0.50


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    top = out["top_bottom"].astype(str).eq("T").to_numpy()
    pteam = pd.to_numeric(out["pitcher_team_id"], errors="coerce").fillna(-1).to_numpy(np.int64)
    bteam = pd.to_numeric(out["batter_team_id"], errors="coerce").fillna(-1).to_numpy(np.int64)
    out["_home_team"] = np.where(top, pteam, bteam).astype(np.int64)
    inning = pd.to_numeric(out["inning"], errors="coerce").fillna(0).to_numpy()
    out["_inning_phase"] = np.where(inning <= 3, "early", np.where(inning <= 6, "middle", "late"))
    for col in ("game_type", "_inning_phase"):
        out[col] = out[col].astype("string").fillna("__MISSING__").astype(str)
    return out


def eb_lookup(source: pd.DataFrame, residual: np.ndarray, valid: pd.DataFrame,
              keys: tuple[str, ...], alpha: float) -> tuple[np.ndarray, dict]:
    work = source.loc[:, list(keys)].copy()
    work["r"] = residual
    tab = work.groupby(list(keys), observed=True, sort=False, dropna=False)["r"].agg(["sum", "size"]).reset_index()
    tab["effect"] = tab["sum"] / (tab["size"] + alpha)
    left = valid.loc[:, list(keys)].copy()
    left["_i"] = np.arange(len(left), dtype=np.int64)
    joined = left.merge(tab[list(keys) + ["effect", "size"]], on=list(keys), how="left", sort=False,
                        validate="m:1").sort_values("_i")
    effect = joined["effect"].fillna(0.0).to_numpy(np.float64)
    return effect, {
        "groups": int(len(tab)),
        "coverage": float(joined["size"].notna().mean()),
        "std": float(effect.std()),
        "mean_abs": float(np.abs(effect).mean()),
    }


def score(p: np.ndarray, y: np.ndarray) -> float:
    r = float(y.mean())
    return 100000.0 * (1.0 - float(np.mean((p - y) ** 2)) / (r * (1.0 - r)))


def gain(base: np.ndarray, cand: np.ndarray, y: np.ndarray, mask=None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    return score(cand[mask], y[mask]) - score(base[mask], y[mask])


def pitcher_bootstrap(base: np.ndarray, cand: np.ndarray, y: np.ndarray,
                      pitcher: np.ndarray, draws: int = 5000) -> dict:
    # Score differences are a positive constant times mean per-row Brier gain;
    # resampling pitcher clusters therefore preserves its sign and CI ordering.
    row = (base - y) ** 2 - (cand - y) ** 2
    f = pd.DataFrame({"p": pitcher, "s": row, "n": 1}).groupby("p", sort=False).agg({"s": "sum", "n": "sum"})
    sums, ns = f["s"].to_numpy(), f["n"].to_numpy()
    rng = np.random.default_rng(171)
    vals = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        idx = rng.integers(0, len(f), size=len(f))
        vals[i] = sums[idx].sum() / ns[idx].sum()
    denom = float(y.mean() * (1.0 - y.mean()))
    vals *= 100000.0 / denom
    return {"clusters": int(len(f)), "draws": draws, "p025": float(np.quantile(vals, .025)),
            "median": float(np.median(vals)), "p975": float(np.quantile(vals, .975)),
            "prob_positive": float(np.mean(vals > 0))}


def main() -> None:
    e158 = load_module(ROOT / "exp" / "158_robust_conditional_tensor_eb.py", "e158_171")
    e155 = load_module(ROOT / "exp" / "155_cause_runner_currentblend_gate.py", "e155_171")
    cols = ["season", "game_month", "inning", "top_bottom", "game_type",
            "pitcher_id", "pitcher_team_id", "batter_team_id", "control_success"]
    # Discovery firewall: only 2022 and 2023 are read initially.
    frame = pd.read_csv(ROOT / "data" / "train.csv", usecols=cols, encoding="utf-8-sig", low_memory=False)
    frame = prepare(frame.loc[frame["season"].isin([2022, 2023])].reset_index(drop=True))
    rows = {year: frame.loc[frame["season"].eq(year)].reset_index(drop=True) for year in (2022, 2023)}
    y = {year: rows[year]["control_success"].to_numpy(np.float64) for year in rows}
    base = {year: e158.load_current_baseline(e155, rows[year], year) for year in rows}

    src_resid = y[2022] - base[2022]
    parent, d_parent = eb_lookup(rows[2022], src_resid, rows[2023],
                                 ("game_type", "_home_team"), PARENT_ALPHA)
    child, d_child = eb_lookup(rows[2022], src_resid, rows[2023],
                               ("game_type", "_home_team", "_inning_phase"), CHILD_ALPHA)
    effect = parent + CHILD_WEIGHT * child

    gt = rows[2023]["game_type"].to_numpy()
    month = rows[2023]["game_month"].to_numpy()
    results = []
    for scale in SCALES:
        cand = np.clip(base[2023] + scale * effect, 0.0, 1.0)
        results.append({
            "scale": scale,
            "gain": gain(base[2023], cand, y[2023]),
            "F": gain(base[2023], cand, y[2023], gt == "F"),
            "R": gain(base[2023], cand, y[2023], gt == "R"),
            "early": gain(base[2023], cand, y[2023], month <= 6),
            "late": gain(base[2023], cand, y[2023], month > 6),
            "mean_shift": float((cand - base[2023]).mean()),
            "effect_std": float((cand - base[2023]).std()),
        })
    best = max(results, key=lambda z: z["gain"])
    best_cand = np.clip(base[2023] + best["scale"] * effect, 0.0, 1.0)
    boot = pitcher_bootstrap(base[2023], best_cand, y[2023], rows[2023]["pitcher_id"].to_numpy())
    passed = bool(best["gain"] >= 8.0 and best["F"] > 0.0 and best["R"] > 0.0
                  and best["early"] > 0.0 and best["late"] > 0.0 and boot["p025"] > 0.0)
    payload = {
        "experiment": 171,
        "hypothesis": "pooled home venue x game_type with venue x inning child",
        "discovery": "2022 residual -> 2023",
        "reads_2024": False,
        "parent_alpha": PARENT_ALPHA,
        "child_alpha": CHILD_ALPHA,
        "child_weight": CHILD_WEIGHT,
        "diagnostics": {"parent": d_parent, "child": d_child},
        "results": results,
        "best": best,
        "bootstrap": boot,
        "gate_pass": passed,
        "status": "PASS_MAY_OPEN_2024" if passed else "FAIL_DO_NOT_OPEN_2024",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["EXP171 explicit venue hierarchy discovery (2022 -> 2023)"]
    for row in results:
        lines.append(f"s={row['scale']:.2f} gain={row['gain']:+.3f} F/R={row['F']:+.3f}/{row['R']:+.3f} "
                     f"early/late={row['early']:+.3f}/{row['late']:+.3f}")
    lines.extend([f"best={best}", f"bootstrap={boot}", f"FINAL: {payload['status']}"])
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
