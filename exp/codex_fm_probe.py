# -*- coding: utf-8 -*-
"""
Low-dimensional pitcher-batter matchup probe (Codex-owned experiment).

This is deliberately not a direct historical pair-rate table and not a
high-cardinality categorical feature.  It learns a regularized low-rank
interaction from chronological OOF residuals:

    residual ~= additive pitcher/batter effects + u_pitcher dot v_batter

Only the centered dot-product is used as the correction.  Unknown players
fall back to zero.  Hyperparameters are selected on 2022 and 2023 only;
2024 is an untouched confirmation fold.

Inputs
------
lab/24b_preds.npz : year-wise OOF HGB predictions for 2021..2024
lab/89_cat5.npy   : current 2024 validation champion
data/train.csv    : row-aligned IDs, season, and target

Outputs (full run)
------------------
lab/codex_fm_result.txt
lab/codex_fm_selection.csv
lab/codex_fm_selected.json
lab/codex_fm_2024_effect.npy
lab/codex_fm_2024_candidate.npy
lab/codex_fm_2024_diagnostics.npz

Run
---
python -u exp/codex_fm_probe.py --smoke
python -u exp/codex_fm_probe.py
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os

# Keep this probe from competing with the concurrently running CatBoost job.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATA_PATH = Path("data/train.csv")
OOF_PATH = Path("lab/24b_preds.npz")
CHAMP_PATH = Path("lab/89_cat5.npy")
YEARS = (2021, 2022, 2023, 2024)
SELECT_FOLDS = (((2021,), 2022), ((2021, 2022), 2023))
FIXED_WEIGHTS = (0.25, 0.50, 0.75, 1.00)
BIAS_REG = 200.0
BIAS_ITERS = 12
MIN_GATE_GAIN = 0.05  # exclude floating-point/no-effect pseudo-passes


def set_below_normal_priority() -> None:
    """Best-effort Windows priority reduction; harmless on other systems."""
    if os.name != "nt":
        return
    try:
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, 0x00004000)
    except Exception:
        pass


def raw_score(pred: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rate = float(y.mean())
    return 100000.0 * (
        1.0 - float(np.mean((pred - y) ** 2)) / (rate * (1.0 - rate))
    )


@dataclass
class YearData:
    year: int
    pitcher: np.ndarray
    batter: np.ndarray
    y: np.ndarray
    base: np.ndarray
    residual: np.ndarray
    source_index: np.ndarray


@dataclass
class PreparedInteractions:
    pitcher_levels: pd.Index
    batter_levels: pd.Index
    pair_pitcher: np.ndarray
    pair_batter: np.ndarray
    pair_weight: np.ndarray
    target: np.ndarray
    seen_pair_codes: np.ndarray
    additive_mu: float
    additive_pitcher: np.ndarray
    additive_batter: np.ndarray
    raw_pair_mse: float
    interaction_pair_mse: float

    @property
    def n_pitchers(self) -> int:
        return len(self.pitcher_levels)

    @property
    def n_batters(self) -> int:
        return len(self.batter_levels)


@dataclass
class FactorModel:
    prepared: PreparedInteractions
    pitcher_factors: np.ndarray
    batter_factors: np.ndarray
    rank: int
    ridge: float
    seed: int
    train_loss: float


class Reporter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def _id_strings(series: pd.Series, missing: str) -> np.ndarray:
    return series.astype("string").fillna(missing).astype(str).to_numpy()


def load_years(smoke: bool, reporter: Reporter) -> tuple[dict[int, YearData], np.ndarray]:
    needed = {"season", "pitcher_id", "batter_id", "control_success"}
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", usecols=list(needed))
    archive = np.load(OOF_PATH)
    champion_full = np.load(CHAMP_PATH).astype(np.float64)
    rng = np.random.default_rng(20260830)
    out: dict[int, YearData] = {}

    for year in YEARS:
        rows = df.loc[df["season"] == year].reset_index(drop=True)
        y_file = archive[f"{year}_y"].astype(np.float64)
        base_file = archive[f"{year}_base65"].astype(np.float64)
        if len(rows) != len(y_file) or len(base_file) != len(y_file):
            raise ValueError(
                f"{year} row mismatch: csv={len(rows)}, y={len(y_file)}, base={len(base_file)}"
            )
        y_csv = rows["control_success"].to_numpy(dtype=np.float64)
        if not np.array_equal(y_csv, y_file):
            bad = int(np.count_nonzero(y_csv != y_file))
            raise ValueError(f"{year} label alignment failed ({bad:,} mismatches)")

        source_index = np.arange(len(rows), dtype=np.int64)
        if smoke and len(rows) > 30000:
            source_index = np.sort(rng.choice(len(rows), size=30000, replace=False))
            rows = rows.iloc[source_index].reset_index(drop=True)
            y_file = y_file[source_index]
            base_file = base_file[source_index]

        residual = y_file - base_file
        residual = residual - residual.mean()  # do not let year calibration become matchup
        out[year] = YearData(
            year=year,
            pitcher=_id_strings(rows["pitcher_id"], "__MISSING_PITCHER__"),
            batter=_id_strings(rows["batter_id"], "__MISSING_BATTER__"),
            y=y_file,
            base=base_file,
            residual=residual,
            source_index=source_index,
        )
        reporter.log(
            f"load {year}: n={len(rows):,}, y_rate={y_file.mean():.6f}, "
            f"base={raw_score(base_file, y_file):.2f}, residual_mean={residual.mean():+.3e}"
        )

    if len(champion_full) != len(archive["2024_y"]):
        raise ValueError(
            f"champion row mismatch: champ={len(champion_full)}, y2024={len(archive['2024_y'])}"
        )
    champion = champion_full[out[2024].source_index] if smoke else champion_full
    if len(champion) != len(out[2024].y):
        raise ValueError("champion/subsample alignment failed")
    reporter.log(f"load 2024 champion: n={len(champion):,}, score={raw_score(champion, out[2024].y):.2f}")
    return out, champion


def concatenate_training(
    data: dict[int, YearData], years: Iterable[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = tuple(years)
    return (
        np.concatenate([data[y].pitcher for y in years]),
        np.concatenate([data[y].batter for y in years]),
        np.concatenate([data[y].residual for y in years]),
    )


def prepare_interactions(
    pitcher: np.ndarray,
    batter: np.ndarray,
    residual: np.ndarray,
    bias_reg: float = BIAS_REG,
    bias_iters: int = BIAS_ITERS,
) -> PreparedInteractions:
    """Aggregate rows, remove regularized additive player effects, retain interaction."""
    p_code, p_levels = pd.factorize(pitcher, sort=True)
    b_code, b_levels = pd.factorize(batter, sort=True)
    if np.any(p_code < 0) or np.any(b_code < 0):
        raise ValueError("unexpected missing ID after sentinel conversion")
    n_p, n_b = len(p_levels), len(b_levels)
    row_pair_code = p_code.astype(np.int64) * n_b + b_code.astype(np.int64)
    pair_codes, inverse = np.unique(row_pair_code, return_inverse=True)
    weight = np.bincount(inverse).astype(np.float64)
    target = np.bincount(inverse, weights=residual).astype(np.float64) / weight
    pair_p = (pair_codes // n_b).astype(np.int32)
    pair_b = (pair_codes % n_b).astype(np.int32)

    p_exposure = np.bincount(pair_p, weights=weight, minlength=n_p)
    b_exposure = np.bincount(pair_b, weights=weight, minlength=n_b)
    p_bias = np.zeros(n_p, dtype=np.float64)
    b_bias = np.zeros(n_b, dtype=np.float64)
    mu = float(np.average(target, weights=weight))
    for _ in range(bias_iters):
        p_num = np.bincount(
            pair_p,
            weights=weight * (target - mu - b_bias[pair_b]),
            minlength=n_p,
        )
        p_bias = p_num / (p_exposure + bias_reg)
        b_num = np.bincount(
            pair_b,
            weights=weight * (target - mu - p_bias[pair_p]),
            minlength=n_b,
        )
        b_bias = b_num / (b_exposure + bias_reg)
        mu = float(np.average(target - p_bias[pair_p] - b_bias[pair_b], weights=weight))

    interaction = target - mu - p_bias[pair_p] - b_bias[pair_b]
    raw_mse = float(np.average(target**2, weights=weight))
    int_mse = float(np.average(interaction**2, weights=weight))
    return PreparedInteractions(
        pitcher_levels=pd.Index(p_levels),
        batter_levels=pd.Index(b_levels),
        pair_pitcher=pair_p,
        pair_batter=pair_b,
        pair_weight=weight,
        target=interaction,
        seen_pair_codes=pair_codes,
        additive_mu=mu,
        additive_pitcher=p_bias,
        additive_batter=b_bias,
        raw_pair_mse=raw_mse,
        interaction_pair_mse=int_mse,
    )


def _group_slices(code: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(code, kind="stable")
    counts = np.bincount(code, minlength=size)
    edges = np.empty(size + 1, dtype=np.int64)
    edges[0] = 0
    np.cumsum(counts, out=edges[1:])
    return order, edges


def fit_factors(
    prepared: PreparedInteractions,
    rank: int,
    ridge: float,
    seed: int,
    iterations: int,
) -> FactorModel:
    """Weighted explicit-feedback ALS on observed aggregated pairs."""
    pp = prepared.pair_pitcher
    pb = prepared.pair_batter
    weight = prepared.pair_weight
    target = prepared.target
    n_p, n_b = prepared.n_pitchers, prepared.n_batters
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 0.08, size=(n_p, rank))
    v = rng.normal(0.0, 0.08, size=(n_b, rank))
    order_p, edge_p = _group_slices(pp, n_p)
    order_b, edge_b = _group_slices(pb, n_b)
    eye = np.eye(rank, dtype=np.float64)
    p_exposure = np.bincount(pp, weights=weight, minlength=n_p)
    b_exposure = np.bincount(pb, weights=weight, minlength=n_b)

    for _ in range(iterations):
        for i in range(n_p):
            idx = order_p[edge_p[i] : edge_p[i + 1]]
            if not len(idx):
                continue
            other = v[pb[idx]]
            w = weight[idx]
            lhs = other.T @ (other * w[:, None]) + ridge * eye
            rhs = other.T @ (w * target[idx])
            u[i] = np.linalg.solve(lhs, rhs)
        u -= np.average(u, axis=0, weights=p_exposure)

        for j in range(n_b):
            idx = order_b[edge_b[j] : edge_b[j + 1]]
            if not len(idx):
                continue
            other = u[pp[idx]]
            w = weight[idx]
            lhs = other.T @ (other * w[:, None]) + ridge * eye
            rhs = other.T @ (w * target[idx])
            v[j] = np.linalg.solve(lhs, rhs)
        v -= np.average(v, axis=0, weights=b_exposure)

    fitted = np.einsum("ij,ij->i", u[pp], v[pb])
    loss = float(np.average((target - fitted) ** 2, weights=weight))
    return FactorModel(prepared, u, v, rank, ridge, seed, loss)


def predict_effect(
    model: FactorModel, pitcher: np.ndarray, batter: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prep = model.prepared
    p = prep.pitcher_levels.get_indexer(pitcher)
    b = prep.batter_levels.get_indexer(batter)
    known = (p >= 0) & (b >= 0)
    effect = np.zeros(len(pitcher), dtype=np.float64)
    effect[known] = np.einsum(
        "ij,ij->i", model.pitcher_factors[p[known]], model.batter_factors[b[known]]
    )
    seen_pair = np.zeros(len(pitcher), dtype=bool)
    pair_code = p[known].astype(np.int64) * prep.n_batters + b[known].astype(np.int64)
    seen_pair[known] = np.isin(pair_code, prep.seen_pair_codes, assume_unique=False)
    return effect, known, seen_pair


def ensemble_effect(
    prepared: PreparedInteractions,
    rank: int,
    ridge: float,
    seeds: tuple[int, ...],
    iterations: int,
    validation: YearData,
    reporter: Reporter,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[FactorModel]]:
    effects = []
    models = []
    known = seen = None
    for seed in seeds:
        model = fit_factors(prepared, rank, ridge, seed, iterations)
        effect, known_i, seen_i = predict_effect(model, validation.pitcher, validation.batter)
        effects.append(effect)
        models.append(model)
        known, seen = known_i, seen_i
        reporter.log(
            f"      seed={seed}: interaction train pair-MSE={model.train_loss:.8f}, "
            f"effect_std={effect.std():.9f}"
        )
    return np.mean(effects, axis=0), known, seen, models


def gain_for_effect(base: np.ndarray, y: np.ndarray, effect: np.ndarray, gamma: float) -> float:
    candidate = np.clip(base + gamma * effect, 0.0, 1.0)
    return raw_score(candidate, y) - raw_score(base, y)


def unconstrained_effect_optimum(
    base: np.ndarray, y: np.ndarray, effect: np.ndarray
) -> tuple[float, float, float]:
    """Quadratic optimum before probability clipping; diagnostic, never used on 2024."""
    denom = float(np.mean(effect**2))
    if denom <= 1e-20:
        return 0.0, 0.0, 0.0
    numer = float(np.mean((y - base) * effect))
    beta = numer / denom
    rate = float(y.mean())
    scale = 100000.0 / (rate * (1.0 - rate))
    signed_gain = scale * (2.0 * beta * numer - beta * beta * denom)
    corr = float(np.corrcoef(y - base, effect)[0, 1]) if effect.std() > 0 else 0.0
    return beta, signed_gain, corr


def quadratic_mix(
    base: np.ndarray, candidate: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float, dict[float, float]]:
    delta = candidate - base
    rate = float(y.mean())
    scale = 100000.0 / (rate * (1.0 - rate))
    d = raw_score(candidate, y) - raw_score(base, y)
    curvature = scale * float(np.mean(delta**2))
    if curvature <= 1e-15:
        w_star = 0.0
    else:
        w_star = float(np.clip((d + curvature) / (2.0 * curvature), 0.0, 1.0))
    gain_star = d * w_star + curvature * w_star * (1.0 - w_star)
    fixed = {
        w: raw_score(base + w * delta, y) - raw_score(base, y) for w in FIXED_WEIGHTS
    }
    return d, curvature, w_star, gain_star, fixed


def pitcher_bootstrap(
    base: np.ndarray,
    candidate: np.ndarray,
    y: np.ndarray,
    pitcher: np.ndarray,
    draws: int,
    seed: int = 20260830,
) -> np.ndarray:
    _, code = np.unique(pitcher, return_inverse=True)
    n_clusters = int(code.max()) + 1
    count = np.bincount(code).astype(np.float64)
    y_sum = np.bincount(code, weights=y).astype(np.float64)
    e0 = np.bincount(code, weights=(base - y) ** 2).astype(np.float64)
    e1 = np.bincount(code, weights=(candidate - y) ** 2).astype(np.float64)
    rng = np.random.default_rng(seed)
    gains = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        n = count[sampled].sum()
        rate = y_sum[sampled].sum() / n
        gains[draw] = (
            100000.0
            * ((e0[sampled].sum() - e1[sampled].sum()) / n)
            / (rate * (1.0 - rate))
        )
    return gains


def subgroup_diagnostics(
    base: np.ndarray,
    candidate: np.ndarray,
    y: np.ndarray,
    known: np.ndarray,
    seen: np.ndarray,
    reporter: Reporter,
) -> None:
    rate = float(y.mean())
    full_scale = 100000.0 / (rate * (1.0 - rate))
    row_gain = (base - y) ** 2 - (candidate - y) ** 2
    groups = (
        ("known+seen_pair", known & seen),
        ("known+unseen_pair", known & ~seen),
        ("unknown_player", ~known),
    )
    reporter.log("    subgroup coverage and gain:")
    for name, mask in groups:
        n = int(mask.sum())
        contribution = full_scale * float(np.mean(row_gain * mask))
        local = float("nan")
        if n and 0.0 < y[mask].mean() < 1.0:
            local = raw_score(candidate[mask], y[mask]) - raw_score(base[mask], y[mask])
        reporter.log(
            f"      {name:19s} n={n:8,d} ({mask.mean()*100:6.2f}%)  "
            f"global_contrib={contribution:+8.3f}  local_gain={local:+8.3f}"
        )


def evaluate_2024(
    name: str,
    base: np.ndarray,
    y: np.ndarray,
    effect: np.ndarray,
    gamma: float,
    pitcher: np.ndarray,
    known: np.ndarray,
    seen: np.ndarray,
    draws: int,
    reporter: Reporter,
) -> np.ndarray:
    candidate = np.clip(base + gamma * effect, 0.0, 1.0)
    d, curvature, w_star, gain_star, fixed = quadratic_mix(base, candidate, y)
    reporter.log(f"\n  [{name}] base={raw_score(base, y):.3f}, frozen candidate={raw_score(candidate, y):.3f}")
    reporter.log(
        f"    frozen d={d:+.3f}, K={curvature:.3f}, analytic w*={w_star:.4f}, "
        f"oracle mix gain={gain_star:+.3f}"
    )
    reporter.log(
        "    fixed candidate-weight gains: "
        + "  ".join(f"w={w:.2f} {gain:+.3f}" for w, gain in fixed.items())
    )
    boot = pitcher_bootstrap(base, candidate, y, pitcher, draws=draws)
    lo, med, hi = np.quantile(boot, [0.025, 0.5, 0.975])
    reporter.log(
        f"    pitcher bootstrap frozen d: median={med:+.3f}, 95% [{lo:+.3f},{hi:+.3f}], "
        f"P(d>0)={(boot > 0).mean():.3f} ({draws:,} draws)"
    )
    subgroup_diagnostics(base, candidate, y, known, seen, reporter)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="small sampled structural test; not a verdict")
    args = parser.parse_args()
    set_below_normal_priority()

    suffix = "_smoke" if args.smoke else ""
    result_path = Path(f"lab/codex_fm_result{suffix}.txt")
    selection_path = Path(f"lab/codex_fm_selection{suffix}.csv")
    selected_path = Path(f"lab/codex_fm_selected{suffix}.json")
    reporter = Reporter(result_path)
    reporter.log("=" * 96)
    reporter.log("CODEX low-rank pitcher-batter residual matchup probe")
    reporter.log(
        f"mode={'SMOKE (not a verdict)' if args.smoke else 'FULL'}, threads=1, process_priority=BelowNormal"
    )
    reporter.log("selection: 2021->2022 and 2021~22->2023 only; 2024 untouched confirmation")
    reporter.log("correction: pure centered u_pitcher dot v_batter; additive player effects are discarded")
    reporter.log("=" * 96)

    data, champion = load_years(args.smoke, reporter)
    ranks = (2, 4) if args.smoke else (2, 4, 8)
    ridges = (1.0, 10.0) if args.smoke else (0.3, 1.0, 3.0, 10.0, 30.0)
    gammas = (
        (0.005, 0.02, 0.05, 0.10)
        if args.smoke
        else (0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
    )
    seeds = (42,) if args.smoke else (42, 7)
    iterations = 3 if args.smoke else 12
    bootstrap_draws = 200 if args.smoke else 5000

    prepared_by_fold: dict[tuple[int, ...], PreparedInteractions] = {}
    for train_years, _ in SELECT_FOLDS:
        p, b, r = concatenate_training(data, train_years)
        prep = prepare_interactions(p, b, r)
        prepared_by_fold[train_years] = prep
        reporter.log(
            f"prepare {train_years}: rows={len(r):,}, pitchers={prep.n_pitchers:,}, "
            f"batters={prep.n_batters:,}, pairs={len(prep.pair_weight):,}, "
            f"pair-MSE raw={prep.raw_pair_mse:.8f} -> after additive={prep.interaction_pair_mse:.8f}"
        )

    records: list[dict[str, float | int | bool]] = []
    reporter.log("\n" + "=" * 96)
    reporter.log("HYPERPARAMETER SELECTION (2024 is not read here)")
    reporter.log("=" * 96)
    for rank in ranks:
        for ridge in ridges:
            fold_effect: dict[int, np.ndarray] = {}
            fold_known: dict[int, np.ndarray] = {}
            fold_seen: dict[int, np.ndarray] = {}
            reporter.log(f"\n  rank={rank}, ridge={ridge:g}")
            for train_years, val_year in SELECT_FOLDS:
                effect, known, seen, _ = ensemble_effect(
                    prepared_by_fold[train_years],
                    rank,
                    ridge,
                    seeds,
                    iterations,
                    data[val_year],
                    reporter,
                )
                fold_effect[val_year] = effect
                fold_known[val_year] = known
                fold_seen[val_year] = seen
                reporter.log(
                    f"      validation {val_year}: known_both={known.mean()*100:.2f}%, "
                    f"unseen_pair_among_known={(known & ~seen).sum()/max(known.sum(),1)*100:.2f}%"
                )
                beta_opt, gain_opt, corr = unconstrained_effect_optimum(
                    data[val_year].base, data[val_year].y, effect
                )
                reporter.log(
                    f"      validation {val_year}: residual/effect corr={corr:+.6f}, "
                    f"unconstrained beta*={beta_opt:+.6f}, gain*={gain_opt:+.3f}"
                )

            for gamma in gammas:
                gain22 = gain_for_effect(data[2022].base, data[2022].y, fold_effect[2022], gamma)
                gain23 = gain_for_effect(data[2023].base, data[2023].y, fold_effect[2023], gamma)
                stable = bool(gain22 > MIN_GATE_GAIN and gain23 > MIN_GATE_GAIN)
                record = {
                    "rank": rank,
                    "ridge": ridge,
                    "gamma": gamma,
                    "gain_2022": gain22,
                    "gain_2023": gain23,
                    "mean_gain": (gain22 + gain23) / 2.0,
                    "min_gain": min(gain22, gain23),
                    "stable_positive": stable,
                    "known_2022": float(fold_known[2022].mean()),
                    "known_2023": float(fold_known[2023].mean()),
                    "unseen_2022": float((fold_known[2022] & ~fold_seen[2022]).mean()),
                    "unseen_2023": float((fold_known[2023] & ~fold_seen[2023]).mean()),
                }
                records.append(record)
                reporter.log(
                    f"      gamma={gamma:6.4f}: gain22={gain22:+8.3f}, "
                    f"gain23={gain23:+8.3f}, mean={record['mean_gain']:+8.3f}, stable={stable}"
                )

    table = pd.DataFrame(records)
    table = table.sort_values(
        ["stable_positive", "mean_gain", "min_gain", "rank", "ridge"],
        ascending=[False, False, False, True, False],
    ).reset_index(drop=True)
    table.to_csv(selection_path, index=False, encoding="utf-8-sig")
    stable = table.loc[table["stable_positive"]]
    gate_pass = len(stable) > 0
    reporter.log("\nTop selection rows:")
    reporter.log(table.head(12).to_string(index=False))
    reporter.log(
        f"\nHistorical stability gate (each year > {MIN_GATE_GAIN:.2f}): "
        f"{'PASS' if gate_pass else 'FAIL'}"
    )

    if not gate_pass and not args.smoke:
        payload = {
            "status": "rejected_before_2024",
            "reason": "No rank/ridge/gamma improved both 2022 and 2023.",
            "selection_file": str(selection_path),
        }
        selected_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        reporter.log("2024 remains untouched. Stop: no CatBoost retraining, deployment, or submission.")
        return

    chosen = (stable.iloc[0] if gate_pass else table.iloc[0]).to_dict()
    rank = int(chosen["rank"])
    ridge = float(chosen["ridge"])
    gamma = float(chosen["gamma"])
    reporter.log(
        f"\nSELECTED (without 2024): rank={rank}, ridge={ridge:g}, gamma={gamma:g}, "
        f"gain22={chosen['gain_2022']:+.3f}, gain23={chosen['gain_2023']:+.3f}"
    )

    # Untouched 2024 confirmation.  In smoke mode this is only a sampled code-path test.
    p, b, r = concatenate_training(data, (2021, 2022, 2023))
    final_prep = prepare_interactions(p, b, r)
    reporter.log(
        f"\nprepare final 2021~23: rows={len(r):,}, pitchers={final_prep.n_pitchers:,}, "
        f"batters={final_prep.n_batters:,}, pairs={len(final_prep.pair_weight):,}"
    )
    effect, known, seen, _ = ensemble_effect(
        final_prep, rank, ridge, seeds, iterations, data[2024], reporter
    )
    reporter.log("\n" + "=" * 96)
    reporter.log("UNTOUCHED 2024 CONFIRMATION" + (" (SMOKE ONLY)" if args.smoke else ""))
    reporter.log("=" * 96)
    reporter.log(
        f"  effect mean={effect.mean():+.7f}, std={effect.std():.7f}, "
        f"q01/q50/q99={np.quantile(effect,[.01,.5,.99])}"
    )

    base_candidate = evaluate_2024(
        "HGB base65 + frozen FM",
        data[2024].base,
        data[2024].y,
        effect,
        gamma,
        data[2024].pitcher,
        known,
        seen,
        bootstrap_draws,
        reporter,
    )
    champ_candidate = evaluate_2024(
        "CAT5 champion + frozen FM",
        champion,
        data[2024].y,
        effect,
        gamma,
        data[2024].pitcher,
        known,
        seen,
        bootstrap_draws,
        reporter,
    )

    prefix = f"lab/codex_fm{suffix}"
    np.save(prefix + "_2024_effect.npy", effect.astype(np.float32))
    np.save(prefix + "_2024_candidate.npy", champ_candidate.astype(np.float32))
    np.savez_compressed(
        prefix + "_2024_diagnostics.npz",
        y=data[2024].y.astype(np.int8),
        base65=data[2024].base.astype(np.float32),
        champion=champion.astype(np.float32),
        effect=effect.astype(np.float32),
        base65_candidate=base_candidate.astype(np.float32),
        champion_candidate=champ_candidate.astype(np.float32),
        known_both=known,
        seen_pair=seen,
        source_index=data[2024].source_index,
    )
    payload = {
        "status": "smoke_only" if args.smoke else "confirmed_2024",
        "historical_gate_pass": bool(gate_pass),
        "rank": rank,
        "ridge": ridge,
        "gamma": gamma,
        "seeds": list(seeds),
        "iterations": iterations,
        "gain_2022": float(chosen["gain_2022"]),
        "gain_2023": float(chosen["gain_2023"]),
        "selection_file": str(selection_path),
        "result_file": str(result_path),
    }
    selected_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    reporter.log("\nSaved Codex-prefixed artifacts. No CatBoost training or submission was started.")


if __name__ == "__main__":
    main()
