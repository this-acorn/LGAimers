# -*- coding: utf-8 -*-
"""Strict forward diagnostic for a robust conditional Tensor-EB residual.

This is an original implementation from the method specification published in
calico-cat17/LG-Aimers-9th issue #5.  It does not import or copy third-party
source, model files, or predictions.  It reads only official train rows and
the frozen OOF arrays already present in this workspace.

Protocol (fixed before opening 2024 results):
  * learn three zero-prior EB residual tables on 2022 R rows and apply to 2023;
  * learn a pitcher reliability map from the resulting 2023 row losses;
  * relearn the same EB tables on 2023 R rows and apply the frozen reliability
    map to 2024 R rows;
  * F rows remain exactly equal to the current blended OOF baseline.

No test data, deployable model, ZIP, leaderboard value, or submission is read
or produced by this script.
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
OUT_JSON = ROOT / "lab" / "158_robust_conditional_tensor_eb.json"
OUT_TXT = ROOT / "lab" / "158_robust_conditional_tensor_eb.txt"

TENSOR_SPECS = (
    ("pitcher_batter_hand", ("pitcher_id", "batter_hand"), 0.25, 500.0),
    (
        "pitcher_count_base",
        ("pitcher_id", "_count_state", "base_state"),
        0.75,
        1000.0,
    ),
    (
        "hand_count_base",
        ("pitcher_hand", "batter_hand", "_count_state", "base_state"),
        0.05,
        500.0,
    ),
)
RELIABILITY_ALPHA = 500.0
RELIABLE_WEIGHT = 1.0
UNSTABLE_WEIGHT = 0.25
UNSEEN_WEIGHT = 0.50
PRIMARY_SCALE = 1.0
SCALE_GRID = (0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 2.0)
BOOTSTRAP_DRAWS = 5000


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["pitcher_id"] = (
        pd.to_numeric(out["pitcher_id"], errors="coerce").fillna(-1).astype("int64")
    )
    for column in ("pitcher_hand", "batter_hand", "base_state", "game_type"):
        out[column] = out[column].astype("string").fillna("__MISSING__").astype(str)
    balls = pd.to_numeric(out["balls_before"], errors="coerce").fillna(-1).astype("int16")
    strikes = (
        pd.to_numeric(out["strikes_before"], errors="coerce").fillna(-1).astype("int16")
    )
    out["_count_state"] = (balls * 3 + strikes).astype("int16")
    return out


def load_current_baseline(e155, rows: pd.DataFrame, year: int) -> np.ndarray:
    n = len(rows)
    target = rows["control_success"].to_numpy(np.float64)
    saved_target = e155.load_vector(e155.EXP021_TARGET[year], n)
    if not np.array_equal(saved_target, target):
        raise AssertionError(f"target/order mismatch for {year}")
    cat = e155.load_probability(e155.CAT5[year], n)
    v18 = e155.load_vector(e155.V18[year], n)
    endpoint = e155.load_probability(e155.EXP021[year], n)
    champion = e155.champion_probability(cat, v18)
    baseline = e155.CHAMPION_WEIGHT * champion + e155.EXP021_WEIGHT * endpoint
    if baseline.shape != (n,) or not np.isfinite(baseline).all():
        raise AssertionError(f"invalid baseline for {year}")
    return baseline


def lookup_eb(
    source: pd.DataFrame,
    source_residual: np.ndarray,
    validation: pd.DataFrame,
    keys: tuple[str, ...],
    alpha: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if len(source) != len(source_residual):
        raise AssertionError("source residual length mismatch")
    work = source.loc[:, list(keys)].copy()
    work["_residual"] = source_residual
    table = (
        work.groupby(list(keys), sort=False, observed=True, dropna=False)["_residual"]
        .agg(["sum", "size"])
        .reset_index()
    )
    table["_correction"] = table["sum"] / (table["size"] + alpha)
    left = validation.loc[:, list(keys)].copy()
    left["_lookup_row"] = np.arange(len(left), dtype=np.int64)
    joined = left.merge(
        table.loc[:, list(keys) + ["_correction", "size"]],
        on=list(keys),
        how="left",
        sort=False,
        validate="m:1",
    ).sort_values("_lookup_row")
    if not np.array_equal(joined["_lookup_row"].to_numpy(), np.arange(len(left))):
        raise AssertionError("lookup changed validation row order")
    correction = joined["_correction"].fillna(0.0).to_numpy(np.float64)
    matched = joined["size"].notna().to_numpy()
    return correction, {
        "groups": int(len(table)),
        "coverage": float(matched.mean()),
        "correction_mean": float(correction.mean()),
        "correction_std": float(correction.std()),
        "correction_mean_abs": float(np.abs(correction).mean()),
        "correction_max_abs": float(np.abs(correction).max(initial=0.0)),
    }


def tensor_effect(
    source: pd.DataFrame,
    source_target: np.ndarray,
    source_baseline: np.ndarray,
    validation: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, object]]:
    source_r_mask = source["game_type"].to_numpy() == "R"
    source_r = source.loc[source_r_mask].reset_index(drop=True)
    residual = source_target[source_r_mask] - source_baseline[source_r_mask]
    effect = np.zeros(len(validation), dtype=np.float64)
    components: dict[str, object] = {}
    for name, keys, weight, alpha in TENSOR_SPECS:
        correction, diagnostic = lookup_eb(
            source_r, residual, validation, keys, alpha
        )
        effect += weight * correction
        components[name] = {
            "keys": list(keys),
            "weight": weight,
            "alpha": alpha,
            **diagnostic,
        }
    is_r = validation["game_type"].to_numpy() == "R"
    effect[~is_r] = 0.0
    if not np.isfinite(effect).all() or np.any(effect[~is_r] != 0.0):
        raise AssertionError("tensor effect is invalid or altered an F row")
    return effect, {
        "source_R_rows": int(source_r_mask.sum()),
        "validation_R_rows": int(is_r.sum()),
        "components": components,
        "tensor_mean_R": float(effect[is_r].mean()),
        "tensor_std_R": float(effect[is_r].std()),
        "tensor_mean_abs_R": float(np.abs(effect[is_r]).mean()),
        "tensor_max_abs_R": float(np.abs(effect[is_r]).max(initial=0.0)),
    }


def learn_pitcher_reliability(
    rows: pd.DataFrame,
    target: np.ndarray,
    baseline: np.ndarray,
    effect: np.ndarray,
) -> tuple[dict[int, float], dict[str, object]]:
    is_r = rows["game_type"].to_numpy() == "R"
    candidate = np.clip(baseline + effect, 0.0, 1.0)
    row_gain = np.square(baseline - target) - np.square(candidate - target)
    global_mean_gain = float(row_gain[is_r].mean())
    work = pd.DataFrame(
        {
            "pitcher_id": rows.loc[is_r, "pitcher_id"].to_numpy(np.int64),
            "gain": row_gain[is_r],
        }
    )
    grouped = work.groupby("pitcher_id", sort=False, observed=True)["gain"].agg(
        ["sum", "size"]
    )
    grouped["shrunk_gain"] = (
        grouped["sum"] + RELIABILITY_ALPHA * global_mean_gain
    ) / (grouped["size"] + RELIABILITY_ALPHA)
    grouped["gate"] = np.where(
        grouped["shrunk_gain"] > 0.0, RELIABLE_WEIGHT, UNSTABLE_WEIGHT
    )
    mapping = {int(index): float(value) for index, value in grouped["gate"].items()}
    return mapping, {
        "source_pitchers": int(len(grouped)),
        "global_mean_row_loss_gain": global_mean_gain,
        "reliable_pitchers": int((grouped["gate"] == RELIABLE_WEIGHT).sum()),
        "unstable_pitchers": int((grouped["gate"] == UNSTABLE_WEIGHT).sum()),
        "reliable_row_share": float(
            grouped.loc[grouped["gate"] == RELIABLE_WEIGHT, "size"].sum()
            / grouped["size"].sum()
        ),
        "shrunk_gain_min": float(grouped["shrunk_gain"].min()),
        "shrunk_gain_median": float(grouped["shrunk_gain"].median()),
        "shrunk_gain_max": float(grouped["shrunk_gain"].max()),
    }


def apply_reliability(
    rows: pd.DataFrame, effect: np.ndarray, mapping: dict[int, float]
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pitcher = rows["pitcher_id"].to_numpy(np.int64)
    gate = np.fromiter(
        (mapping.get(int(value), UNSEEN_WEIGHT) for value in pitcher),
        dtype=np.float64,
        count=len(rows),
    )
    is_r = rows["game_type"].to_numpy() == "R"
    gated = effect * gate
    gated[~is_r] = 0.0
    if np.any(gated[~is_r] != 0.0):
        raise AssertionError("reliability gate altered an F row")
    return gated, gate, {
        "weight_1_row_share_R": float(np.mean(gate[is_r] == RELIABLE_WEIGHT)),
        "weight_025_row_share_R": float(np.mean(gate[is_r] == UNSTABLE_WEIGHT)),
        "weight_05_unseen_row_share_R": float(np.mean(gate[is_r] == UNSEEN_WEIGHT)),
        "mean_gate_R": float(gate[is_r].mean()),
    }


def subset_gain(
    e155, baseline: np.ndarray, candidate: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    if int(mask.sum()) < 2:
        return {"rows": int(mask.sum()), "gain": float("nan")}
    return {
        "rows": int(mask.sum()),
        "target_rate": float(target[mask].mean()),
        "baseline_score": e155.score(baseline[mask], target[mask]),
        "candidate_score": e155.score(candidate[mask], target[mask]),
        "gain": float(
            e155.score(candidate[mask], target[mask])
            - e155.score(baseline[mask], target[mask])
        ),
    }


def cluster_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    pitcher: np.ndarray,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = 158,
) -> dict[str, float | int]:
    work = pd.DataFrame(
        {
            "pitcher": pitcher,
            "n": np.ones(len(target), dtype=np.int32),
            "y": target,
            "base_se": np.square(baseline - target),
            "candidate_se": np.square(candidate - target),
        }
    )
    grouped = work.groupby("pitcher", sort=False, observed=True).agg(
        n=("n", "sum"),
        y=("y", "sum"),
        base_se=("base_se", "sum"),
        candidate_se=("candidate_se", "sum"),
    )
    values = grouped[["n", "y", "base_se", "candidate_se"]].to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    gains = np.empty(draws, dtype=np.float64)
    cursor = 0
    while cursor < draws:
        width = min(200, draws - cursor)
        indices = rng.integers(0, len(values), size=(width, len(values)))
        sampled = values[indices].sum(axis=1)
        rate = sampled[:, 1] / sampled[:, 0]
        denominator = sampled[:, 0] * rate * (1.0 - rate)
        gains[cursor : cursor + width] = (
            100000.0 * (sampled[:, 2] - sampled[:, 3]) / denominator
        )
        cursor += width
    return {
        "draws": int(draws),
        "pitchers": int(len(values)),
        "median": float(np.median(gains)),
        "p025": float(np.quantile(gains, 0.025)),
        "p975": float(np.quantile(gains, 0.975)),
        "prob_positive": float(np.mean(gains > 0.0)),
    }


def main() -> None:
    started = time.time()
    e155 = load_module(ROOT / "exp" / "155_cause_runner_currentblend_gate.py", "e155")
    columns = [
        "season",
        "game_month",
        "game_type",
        "pitcher_id",
        "pitcher_hand",
        "batter_hand",
        "balls_before",
        "strikes_before",
        "base_state",
        "control_success",
    ]
    frame = prepare_keys(
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
        year: load_current_baseline(e155, rows[year], year) for year in rows
    }

    effect23, tensor23 = tensor_effect(
        rows[2022], target[2022], baseline[2022], rows[2023]
    )
    discovery = e155.metrics(
        baseline[2023], effect23, target[2023], rows[2023]["game_type"].to_numpy()
    )
    reliability_map, reliability = learn_pitcher_reliability(
        rows[2023], target[2023], baseline[2023], effect23
    )

    effect24, tensor24 = tensor_effect(
        rows[2023], target[2023], baseline[2023], rows[2024]
    )
    gated24, gate24, gate_diagnostic = apply_reliability(
        rows[2024], effect24, reliability_map
    )
    candidate24 = np.clip(baseline[2024] + PRIMARY_SCALE * gated24, 0.0, 1.0)
    game_type24 = rows[2024]["game_type"].to_numpy()
    confirmation = e155.metrics(
        baseline[2024], PRIMARY_SCALE * gated24, target[2024], game_type24
    )
    is_f = game_type24 == "F"
    if not np.array_equal(candidate24[is_f], baseline[2024][is_f]):
        raise AssertionError("F fallback is not exact")

    month = pd.to_numeric(rows[2024]["game_month"], errors="coerce").to_numpy()
    periods = {
        "early_month_le_6": subset_gain(
            e155, baseline[2024], candidate24, target[2024], month <= 6
        ),
        "late_month_gt_6": subset_gain(
            e155, baseline[2024], candidate24, target[2024], month > 6
        ),
    }
    hands: dict[str, object] = {}
    pitcher_hand = rows[2024]["pitcher_hand"].to_numpy()
    batter_hand = rows[2024]["batter_hand"].to_numpy()
    for ph in sorted(pd.unique(pitcher_hand).tolist()):
        for bh in sorted(pd.unique(batter_hand).tolist()):
            mask = (pitcher_hand == ph) & (batter_hand == bh)
            hands[f"pitcher={ph}|batter={bh}"] = subset_gain(
                e155, baseline[2024], candidate24, target[2024], mask
            )

    bootstrap = cluster_bootstrap(
        baseline[2024],
        candidate24,
        target[2024],
        rows[2024]["pitcher_id"].to_numpy(np.int64),
    )
    scale_curve: list[dict[str, float]] = []
    for scale in SCALE_GRID:
        row = e155.metrics(
            baseline[2024], scale * gated24, target[2024], game_type24
        )
        scale_curve.append({"scale": scale, **row})
    oracle = e155.scalar_oracle(baseline[2024], gated24, target[2024])
    all_hands_positive = bool(
        all(float(value["gain"]) > 0.0 for value in hands.values())
    )
    passed = bool(
        confirmation["raw_gain"] > 0.0
        and confirmation["equal_mean_shape_gain"] > 0.0
        and float(periods["early_month_le_6"]["gain"]) > 0.0
        and float(periods["late_month_gt_6"]["gain"]) > 0.0
        and all_hands_positive
        and float(bootstrap["p025"]) > 0.0
        and confirmation["F_contribution"] == 0.0
    )
    status = "PASS_FOR_DEPLOY_ENGINEERING" if passed else "FAIL_NO_SUBMISSION"
    payload = {
        "experiment": 158,
        "status": status,
        "analysis_only": True,
        "reads_test": False,
        "uses_lb": False,
        "third_party_code_or_artifacts_read": False,
        "method_source": "calico-cat17/LG-Aimers-9th issue #5 prose specification",
        "protocol": {
            "tensor_specs": [
                {"name": name, "keys": list(keys), "weight": weight, "alpha": alpha}
                for name, keys, weight, alpha in TENSOR_SPECS
            ],
            "source_rows": "immediately previous season, game_type R only",
            "residual": "control_success - current blended OOF",
            "shrinkage_prior": 0.0,
            "reliability_alpha": RELIABILITY_ALPHA,
            "reliable_weight": RELIABLE_WEIGHT,
            "unstable_weight": UNSTABLE_WEIGHT,
            "unseen_weight": UNSEEN_WEIGHT,
            "primary_scale_fixed_before_2024": PRIMARY_SCALE,
            "scale_curve_and_oracle_are_2024_diagnostics_only": True,
            "early_late_definition": "game_month <= 6 versus > 6",
        },
        "alignment": {
            str(year): {
                "rows": int(len(rows[year])),
                "baseline_score": e155.score(baseline[year], target[year]),
                "target_rate": float(target[year].mean()),
            }
            for year in rows
        },
        "tensor_2022_to_2023": tensor23,
        "discovery_2023_ungated": discovery,
        "reliability_fit_2023": reliability,
        "tensor_2023_to_2024": tensor24,
        "gate_application_2024": gate_diagnostic,
        "confirmation_2024_fixed_scale": confirmation,
        "periods_2024": periods,
        "hands_2024": hands,
        "all_hands_positive": all_hands_positive,
        "pitcher_cluster_bootstrap_2024": bootstrap,
        "scale_curve_2024_diagnostic_only": scale_curve,
        "scalar_oracle_2024_diagnostic_only": oracle,
        "f_rows_exact_fallback": True,
        "elapsed_seconds": float(time.time() - started),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    hand_text = ", ".join(
        f"{name}:{float(value['gain']):+.3f}" for name, value in hands.items()
    )
    curve_text = " ".join(
        f"s={row['scale']:.2f}:{row['raw_gain']:+.3f}/{row['equal_mean_shape_gain']:+.3f}"
        for row in scale_curve
    )
    lines = [
        "=== exp158 robust conditional Tensor-EB on current blended OOF ===",
        "DIAGNOSTIC ONLY: official train + saved OOF; no test/model/ZIP/LB",
        "Independent implementation from Issue #5 prose; no third-party code/artifact read",
        f"2023 ungated discovery: {discovery}",
        f"2023 reliability fit: {reliability}",
        f"2024 gate application: {gate_diagnostic}",
        f"2024 fixed scale=1.0 confirmation: {confirmation}",
        f"2024 early/late: {periods}",
        f"2024 hands gains: {hand_text}",
        f"2024 pitcher bootstrap: {bootstrap}",
        f"2024 scale curve raw/equal-mean (diagnostic only): {curve_text}",
        f"2024 scalar oracle (diagnostic only): {oracle}",
        "F rows exact fallback: True",
        f"FINAL: {status}",
        f"elapsed={payload['elapsed_seconds']:.2f}s",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
