# -*- coding: utf-8 -*-
"""[122] Strict temporal CatBoost continuation gate.

This experiment asks one narrow question that earlier season weighting and
recent-window scratch models did not answer: can we keep the exact current
CAT5 function and append a few recent-season gradient trees with ``init_model``?

Selection protocol
------------------
1. Discovery only: train<=2022, re-expose 2022, validate 2023.
2. The append strength is fixed before looking: 50 trees, lr=.04, depth=6,
   l2=10.  Three fits are made: all-history control, recent-all, recent-F.
   The F model is routed only on validation F rows; R is bitwise the base.
3. Only if a recent branch passes every declared discovery gate is its branch
   identity locked and the 2024 confirmation opened exactly once.

No test row, submission bundle, or public score is read or written here.
The script is resumable through saved CBM files and JSON output.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "exp")
from common import load_train, raw_score


ROOT = Path(".")
EXP104 = ROOT / "exp" / "104_cat5_yearfold.py"
OUT_JSON = ROOT / "lab" / "122_temporal_continuation.json"
OUT_TXT = ROOT / "lab" / "122_temporal_continuation.txt"
LIVE_TXT = ROOT / "lab" / "122_temporal_continuation_live.txt"

BASE_PARAMS = dict(
    iterations=500,
    depth=6,
    learning_rate=0.08,
    l2_leaf_reg=10.0,
    verbose=False,
    allow_writing_files=False,
    loss_function="MultiClass",
)
APPEND_PARAMS = dict(
    iterations=50,
    depth=6,
    learning_rate=0.04,
    l2_leaf_reg=10.0,
    verbose=False,
    allow_writing_files=False,
    loss_function="MultiClass",
)
AFFINE_CENTER = 0.49
AFFINE_SCALE = 1.06
AFFINE_SHIFT = 0.0066
V18_GAMMA = 0.30
BRANCHES = ("ALL_HISTORY_CONTROL", "RECENT_ALL", "RECENT_F")

DISCOVERY_GATE = {
    "raw_gain_min": 0.0,
    "oracle_shape_gain_min": 15.0,
    "equal_mean_gain_min": 10.0,
    "shape_over_control_min": 5.0,
    "analytic_blend_gain_min": 20.0,
    "bootstrap_p025_min": 0.0,
}
CONFIRM_GATE = {
    "raw_gain_min": 0.0,
    "oracle_shape_gain_min": 15.0,
    "equal_mean_gain_min": 10.0,
    "shape_over_control_min": 5.0,
    "analytic_blend_gain_min": 20.0,
    "bootstrap_p025_min": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--discovery-only",
        action="store_true",
        help="stop even if discovery passes; useful for controlled scheduling",
    )
    return parser.parse_args()


ARGS = parse_args()
STARTED = time.time()
LINES: list[str] = []


def log(message: str = "") -> None:
    line = str(message)
    print(line, flush=True)
    LINES.append(line)
    LIVE_TXT.parent.mkdir(parents=True, exist_ok=True)
    with LIVE_TXT.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def tick(message: str) -> None:
    log(f"  [{time.time() - STARTED:8.1f}s] {message}")


def load_exp104():
    """Import exp/104 without letting its module-level argparse see our CLI."""

    saved_argv = sys.argv[:]
    try:
        sys.argv = [str(EXP104)]
        spec = importlib.util.spec_from_file_location("exp104_for_122", EXP104)
        if spec is None or spec.loader is None:
            raise ImportError(EXP104)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved_argv
    module.NTHREAD = int(ARGS.threads)
    return module


FOLD = load_exp104()


def score(probability: np.ndarray, target: np.ndarray) -> float:
    return float(raw_score(np.asarray(probability, np.float64), target))


def oracle_shape(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Same-fold unconstrained affine oracle; diagnostic ceiling, not deploy score."""

    p = np.asarray(probability, np.float64)
    y = np.asarray(target, np.float64)
    variance = float(np.var(p))
    if variance <= 0.0 or not np.isfinite(variance):
        return {"score": float("-inf"), "corr": float("nan"), "a": float("nan"), "b": float("nan")}
    covariance = float(np.mean((p - p.mean()) * (y - y.mean())))
    slope = covariance / variance
    intercept = float(y.mean() - slope * p.mean())
    correlation = float(covariance / math.sqrt(variance * float(np.var(y))))
    return {
        "score": float(100000.0 * correlation * correlation),
        "corr": correlation,
        "a": intercept,
        "b": slope,
    }


def champion_space(probability: np.ndarray, effect: np.ndarray) -> np.ndarray:
    corrected = np.clip(
        np.asarray(probability, np.float64) + V18_GAMMA * np.asarray(effect, np.float64),
        0.0,
        1.0,
    )
    return np.clip(
        AFFINE_CENTER + AFFINE_SCALE * (corrected - AFFINE_CENTER) - AFFINE_SHIFT,
        0.0,
        1.0,
    )


def equal_mean_gain(base: np.ndarray, candidate: np.ndarray, target: np.ndarray) -> float:
    same_mean = candidate - candidate.mean() + base.mean()
    return score(same_mean, target) - score(base, target)


def domain_contribution(
    base: np.ndarray, candidate: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> float:
    rate = float(target.mean())
    row_gain = (base - target) ** 2 - (candidate - target) ** 2
    return float(100000.0 * row_gain[mask].sum() / (len(target) * rate * (1.0 - rate)))


def blend_math(base: np.ndarray, candidate: np.ndarray, target: np.ndarray) -> dict[str, float]:
    base_score = score(base, target)
    candidate_score = score(candidate, target)
    d_value = candidate_score - base_score
    rate = float(target.mean())
    k_value = float(
        100000.0 * np.mean((candidate - base) ** 2) / (rate * (1.0 - rate))
    )
    if k_value <= 0.0:
        weight = 0.0
    else:
        weight = float(np.clip((d_value + k_value) / (2.0 * k_value), 0.0, 1.0))
    blended = (1.0 - weight) * base + weight * candidate
    measured_gain = score(blended, target) - base_score
    formula_gain = d_value * weight + k_value * weight * (1.0 - weight)
    if not np.isclose(measured_gain, formula_gain, atol=1e-7, rtol=1e-9):
        raise AssertionError((measured_gain, formula_gain))
    return {
        "d": d_value,
        "K": k_value,
        "optimal_weight": weight,
        "analytic_gain": float(measured_gain),
    }


def cluster_bootstrap(
    base: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    pitcher: np.ndarray,
    seed: int,
    draws: int = 3000,
) -> dict[str, float]:
    work = pd.DataFrame(
        {
            "pitcher": pitcher,
            "n": np.ones(len(target), np.int32),
            "y": target,
            "base_se": (base - target) ** 2,
            "cand_se": (candidate - target) ** 2,
        }
    )
    grouped = work.groupby("pitcher", sort=False, observed=True).agg(
        n=("n", "sum"), y=("y", "sum"), base_se=("base_se", "sum"), cand_se=("cand_se", "sum")
    )
    values = grouped[["n", "y", "base_se", "cand_se"]].to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    gains = np.empty(draws, np.float64)
    cursor = 0
    while cursor < draws:
        width = min(200, draws - cursor)
        indices = rng.integers(0, len(values), size=(width, len(values)))
        sampled = values[indices].sum(axis=1)
        rate = sampled[:, 1] / sampled[:, 0]
        gains[cursor : cursor + width] = 100000.0 * (
            sampled[:, 2] - sampled[:, 3]
        ) / (sampled[:, 0] * rate * (1.0 - rate))
        cursor += width
    return {
        "draws": int(draws),
        "pitchers": int(len(values)),
        "median": float(np.median(gains)),
        "p025": float(np.quantile(gains, 0.025)),
        "p975": float(np.quantile(gains, 0.975)),
        "prob_positive": float(np.mean(gains > 0.0)),
    }


def empty_const(rates: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({"id": [], "N_end": [], **{f"S_{name}": [] for name in rates}})


def prepare_fold(year: int) -> dict[str, object]:
    tick(f"year={year}: load and restore labels")
    frame = FOLD.recover_auxiliary_labels(load_train())
    # exp/104 stringifies batter_hand only for V18 lookup keys.  If that frame
    # is then passed straight to add_features, numeric pitcher_hand is compared
    # with string batter_hand and f_same_hand becomes identically zero.  Compute
    # the V18 effect with those stable string keys first, then restore the raw
    # numeric hand representation before building the exact exp/89 features.
    effect_history = frame[frame["season"] <= year - 1]
    effect_validation = frame[frame["season"] == year].reset_index(drop=True)
    effect, effect_diagnostics = FOLD.v18.make_effect(effect_history, effect_validation)
    frame["batter_hand"] = pd.to_numeric(frame["batter_hand"], errors="coerce")
    history = frame[frame["season"] <= year - 1]
    validation_rows = frame[frame["season"] == year].reset_index(drop=True)
    target = validation_rows["control_success"].to_numpy(np.float64)
    game_type = validation_rows["game_type"].astype(str).to_numpy()
    pitcher = validation_rows["pitcher_id"].to_numpy()
    del effect_history, effect_validation

    prior = float(history["control_success"].mean())
    pitcher_const = FOLD.build_const(
        history, "pitcher_id", "asof_pitcher_n", FOLD.s12.P_RATES
    )
    batter_const = FOLD.build_const(
        history, "batter_id", "asof_batter_n", FOLD.s12.B_RATES
    )
    pfb_rows = history.dropna(subset=["lab_fb"])
    if ARGS.smoke and len(pfb_rows) > 50000:
        pfb_rows = pfb_rows.sample(50000, random_state=122)
    pfb = CatBoostClassifier(
        iterations=3 if ARGS.smoke else 300,
        depth=6,
        learning_rate=0.1,
        verbose=False,
        thread_count=ARGS.threads,
        allow_writing_files=False,
        random_seed=42,
    )
    pfb.fit(pfb_rows[FOLD.s12.PFB_IN].fillna(-999), pfb_rows["lab_fb"].astype(int))
    del pfb_rows
    mix_train = FOLD.mix_asof_train(history)
    mix_validation = FOLD.mix_career(history)
    tick(f"year={year}: constants, pitchmix, PFB ready")

    train_rows = frame[frame["season"] <= year - 1].reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    for season in sorted(train_rows["season"].unique()):
        earlier = frame[frame["season"] <= season - 1]
        rows = train_rows[train_rows["season"] == season]
        if len(earlier):
            pitcher_season = FOLD.build_const(
                earlier, "pitcher_id", "asof_pitcher_n", FOLD.s12.P_RATES
            )
            batter_season = FOLD.build_const(
                earlier, "batter_id", "asof_batter_n", FOLD.s12.B_RATES
            )
        else:
            pitcher_season = empty_const(FOLD.s12.P_RATES)
            batter_season = empty_const(FOLD.s12.B_RATES)
        parts.append(FOLD.s12.attach_cs(rows, pitcher_season, batter_season))
    train = FOLD.s12.add_features(pd.concat(parts).sort_index(), prior)
    del parts, train_rows
    key = train[["pitcher_id", "season", "_cg"]].merge(
        mix_train, on=["pitcher_id", "season", "_cg"], how="left"
    )
    train["f_mixcg_fb"] = key["mix_fb"].to_numpy(np.float32)
    train["f_mixcg_brk"] = key["mix_brk"].to_numpy(np.float32)
    train["f_pfb"] = pfb.predict_proba(train[FOLD.s12.PFB_IN].fillna(-999))[:, 1].astype(np.float32)
    del key, mix_train

    validation = FOLD.s12.attach_pt(
        FOLD.s12.add_features(
            FOLD.s12.attach_cs(validation_rows, pitcher_const, batter_const), prior
        ),
        mix_validation,
        pfb,
    )
    feature_diagnostics = {
        "train_same_hand_rate": float(train["f_same_hand"].mean()),
        "valid_same_hand_rate": float(validation["f_same_hand"].mean()),
    }
    for name, value in feature_diagnostics.items():
        if not 0.30 <= value <= 0.75:
            raise AssertionError(f"{name} implausible: {value}")
    tick(f"year={year}: hand QA {feature_diagnostics}")
    del history, frame, pitcher_const, batter_const, mix_validation, pfb
    gc.collect()
    tick(f"year={year}: exact 79 features ready")

    train_mask = train["_cls"].to_numpy() >= 0
    labeled = train.loc[train_mask]
    labels = labeled["_cls"].to_numpy(np.int8)
    seasons = labeled["season"].to_numpy()
    domains = labeled["game_type"].astype(str).to_numpy()
    train_matrix = FOLD.build_features(labeled)
    valid_matrix = FOLD.build_features(validation)
    del train, labeled, validation, validation_rows
    gc.collect()

    if ARGS.smoke:
        rng = np.random.default_rng(122 + year)
        indices: list[np.ndarray] = []
        for season in np.unique(seasons):
            candidates = np.flatnonzero(seasons == season)
            indices.append(rng.choice(candidates, size=min(10000, len(candidates)), replace=False))
        keep = np.sort(np.concatenate(indices))
        train_matrix = train_matrix.iloc[keep].reset_index(drop=True)
        labels = labels[keep]
        seasons = seasons[keep]
        domains = domains[keep]
        valid_matrix = valid_matrix.iloc[:20000].reset_index(drop=True)
        target = target[:20000]
        game_type = game_type[:20000]
        pitcher = pitcher[:20000]
        effect = effect[:20000]

    recent = seasons == year - 1
    recent_f = recent & (domains == "F")
    if recent.sum() == 0 or recent_f.sum() == 0:
        raise AssertionError(f"empty continuation subset year={year}")
    base_pool = Pool(train_matrix, labels, cat_features=FOLD.CAT5)
    recent_matrix = train_matrix.loc[recent]
    recent_labels = labels[recent]
    recent_pool = Pool(recent_matrix, recent_labels, cat_features=FOLD.CAT5)
    # A literal F-only Pool can omit one of the five labels.  CatBoost then
    # creates a four-dimensional continuation head that cannot be summed with
    # the five-class base.  Keep every recent row in the Pool so the implicit
    # class metadata is identical, but give R rows zero training weight.  The
    # deployed prediction is still routed only on F; R remains the base exactly.
    recent_domains = domains[recent]
    recent_f_weight = (recent_domains == "F").astype(np.float64)
    recent_f_pool = Pool(
        recent_matrix,
        recent_labels,
        cat_features=FOLD.CAT5,
        weight=recent_f_weight,
    )
    valid_pool = Pool(valid_matrix, cat_features=FOLD.CAT5)
    counts = {
        "train": int(len(labels)),
        "recent": int(recent.sum()),
        "recent_F": int(recent_f.sum()),
        "valid": int(len(target)),
    }
    del train_matrix, valid_matrix, labels, seasons, domains, recent, recent_f
    del recent_matrix, recent_labels, recent_domains, recent_f_weight
    gc.collect()
    tick(f"year={year}: pools ready {counts}")
    return {
        "base_pool": base_pool,
        "recent_pool": recent_pool,
        "recent_f_pool": recent_f_pool,
        "valid_pool": valid_pool,
        "target": target,
        "game_type": game_type,
        "pitcher": pitcher,
        "effect": effect,
        "effect_diagnostics": effect_diagnostics,
        "feature_diagnostics": feature_diagnostics,
        "counts": counts,
    }


def model_path(year: int, branch: str) -> Path:
    suffix = "_smoke" if ARGS.smoke else ""
    safe = branch.lower() + "_exacthand_v3"
    return ROOT / "lab" / f"122_y{year}_seed{ARGS.seed}_{safe}{suffix}.cbm"


def load_or_train_base(year: int, pool: Pool) -> CatBoostClassifier:
    path = model_path(year, "base")
    if path.is_file():
        model = CatBoostClassifier()
        model.load_model(path)
        if list(model.feature_names_) != list(FOLD.FEATURES):
            raise AssertionError("cached base feature schema/order mismatch")
        tick(f"year={year}: loaded {path.name}, trees={model.tree_count_}")
        return model
    params = dict(BASE_PARAMS)
    params["iterations"] = 3 if ARGS.smoke else BASE_PARAMS["iterations"]
    model = CatBoostClassifier(
        **params, thread_count=ARGS.threads, random_seed=ARGS.seed
    ).fit(pool)
    if list(model.feature_names_) != list(FOLD.FEATURES):
        raise AssertionError("trained base feature schema/order mismatch")
    model.save_model(path)
    tick(f"year={year}: base trained and saved, trees={model.tree_count_}")
    return model


def load_or_continue(
    year: int, branch: str, pool: Pool, base_model: CatBoostClassifier
) -> CatBoostClassifier:
    path = model_path(year, branch)
    if path.is_file():
        model = CatBoostClassifier()
        model.load_model(path)
        if list(model.feature_names_) != list(base_model.feature_names_):
            raise AssertionError(f"cached {branch} feature schema/order mismatch")
        tick(f"year={year}: loaded {path.name}, trees={model.tree_count_}")
        return model
    params = dict(APPEND_PARAMS)
    params["iterations"] = 2 if ARGS.smoke else APPEND_PARAMS["iterations"]
    model = CatBoostClassifier(
        **params, thread_count=ARGS.threads, random_seed=ARGS.seed
    ).fit(pool, init_model=base_model)
    if list(model.feature_names_) != list(base_model.feature_names_):
        raise AssertionError(f"continued {branch} feature schema/order mismatch")
    expected = int(base_model.tree_count_ + params["iterations"])
    if model.tree_count_ != expected:
        raise AssertionError(f"continuation tree count {model.tree_count_} != {expected}")
    model.save_model(path)
    tick(f"year={year}: {branch} continued and saved, trees={model.tree_count_}")
    return model


def cached_probability_path(year: int) -> Path | None:
    if year == 2024:
        return ROOT / "lab" / f"89_cat5_probs_seed{ARGS.seed}.npy"
    # exp/104 historical caches contain the f_same_hand dtype bug described in
    # prepare_fold and therefore are not an exactness oracle for current CAT5.
    return None


def evaluate_branch(
    name: str,
    base: np.ndarray,
    candidate: np.ndarray,
    control: np.ndarray,
    target: np.ndarray,
    game_type: np.ndarray,
    pitcher: np.ndarray,
    bootstrap_seed: int,
) -> dict[str, object]:
    base_shape = oracle_shape(base, target)
    candidate_shape = oracle_shape(candidate, target)
    control_shape = oracle_shape(control, target)
    f_mask = game_type == "F"
    bootstrap = cluster_bootstrap(
        base, candidate, target, pitcher, seed=bootstrap_seed, draws=300 if ARGS.smoke else 3000
    )
    result: dict[str, object] = {
        "name": name,
        "base_score": score(base, target),
        "candidate_score": score(candidate, target),
        "raw_gain": score(candidate, target) - score(base, target),
        "base_oracle_shape": base_shape,
        "candidate_oracle_shape": candidate_shape,
        "control_oracle_shape": control_shape,
        "oracle_shape_gain": float(candidate_shape["score"] - base_shape["score"]),
        "shape_over_control": float(candidate_shape["score"] - control_shape["score"]),
        "equal_mean_gain": float(equal_mean_gain(base, candidate, target)),
        "mean_shift": float(candidate.mean() - base.mean()),
        "F_contribution": domain_contribution(base, candidate, target, f_mask),
        "R_contribution": domain_contribution(base, candidate, target, ~f_mask),
        "blend": blend_math(base, candidate, target),
        "bootstrap": bootstrap,
    }
    return result


def gate(result: dict[str, object], thresholds: dict[str, float]) -> tuple[bool, dict[str, bool]]:
    checks = {
        "raw_gain": float(result["raw_gain"]) >= thresholds["raw_gain_min"],
        "oracle_shape_gain": float(result["oracle_shape_gain"]) >= thresholds["oracle_shape_gain_min"],
        "equal_mean_gain": float(result["equal_mean_gain"]) >= thresholds["equal_mean_gain_min"],
        "shape_over_control": float(result["shape_over_control"]) >= thresholds["shape_over_control_min"],
        "analytic_blend_gain": float(result["blend"]["analytic_gain"]) >= thresholds["analytic_blend_gain_min"],
        "bootstrap_p025": float(result["bootstrap"]["p025"]) > thresholds["bootstrap_p025_min"],
    }
    return bool(all(checks.values())), checks


def report_branch(result: dict[str, object], passed: bool, checks: dict[str, bool]) -> None:
    blend = result["blend"]
    boot = result["bootstrap"]
    log(
        f"  {result['name']:19s} raw={result['raw_gain']:+8.2f} "
        f"oracleShape={result['oracle_shape_gain']:+8.2f} "
        f"equalMean={result['equal_mean_gain']:+8.2f} "
        f"overCtl={result['shape_over_control']:+8.2f} "
        f"F/R={result['F_contribution']:+7.2f}/{result['R_contribution']:+7.2f}"
    )
    log(
        f"      d={blend['d']:+8.2f} K={blend['K']:8.2f} w*={blend['optimal_weight']:.4f} "
        f"blendGain={blend['analytic_gain']:+7.2f} boot95=[{boot['p025']:+.2f},{boot['p975']:+.2f}] "
        f"GATE={'PASS' if passed else 'FAIL'} {checks}"
    )


def save_summary(summary: dict[str, object]) -> None:
    summary["elapsed_seconds"] = float(time.time() - STARTED)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_TXT.write_text("\n".join(LINES) + "\n", encoding="utf-8")


def run_fold(year: int, selected_branch: str | None = None) -> dict[str, object]:
    log("")
    log("=" * 100)
    log(f"YEAR {year}: train<={year-1}, re-expose {year-1}, valid={year}, seed={ARGS.seed}")
    log("=" * 100)
    data = prepare_fold(year)
    base_model = load_or_train_base(year, data["base_pool"])
    base_probability = base_model.predict_proba(data["valid_pool"])

    cached = cached_probability_path(year)
    qa: dict[str, object]
    if not ARGS.smoke and cached is not None and cached.is_file():
        reference = np.load(cached).astype(np.float64)
        if reference.shape != base_probability.shape:
            raise AssertionError((reference.shape, base_probability.shape))
        maximum_difference = float(np.max(np.abs(reference - base_probability)))
        qa = {"cached": str(cached), "max_abs_diff": maximum_difference, "pass": maximum_difference <= 1e-6}
        log(f"  exact base QA vs {cached.name}: max_abs_diff={maximum_difference:.3e}")
        if not qa["pass"]:
            raise AssertionError(f"base exactness QA failed: {maximum_difference}")
        del reference
    else:
        qa = {
            "cached": None if cached is None else str(cached),
            "max_abs_diff": None,
            "pass": True,
            "reason": "smoke or no trustworthy pre-2024 exact cache",
        }

    control_model = load_or_continue(
        year, "ALL_HISTORY_CONTROL", data["base_pool"], base_model
    )
    control_probability = control_model.predict_proba(data["valid_pool"])

    branches_to_run = [selected_branch] if selected_branch else ["RECENT_ALL", "RECENT_F"]
    branch_probabilities: dict[str, np.ndarray] = {}
    for branch in branches_to_run:
        if branch == "RECENT_ALL":
            pool = data["recent_pool"]
        elif branch == "RECENT_F":
            pool = data["recent_f_pool"]
        else:
            raise ValueError(branch)
        model = load_or_continue(year, branch, pool, base_model)
        branch_probabilities[branch] = model.predict_proba(data["valid_pool"])
        del model
        gc.collect()

    effect = data["effect"]
    target = data["target"]
    game_type = data["game_type"]
    pitcher = data["pitcher"]
    base_final = champion_space(base_probability[:, 0], effect)
    control_final = champion_space(control_probability[:, 0], effect)
    np.save(
        ROOT / "lab" / f"122_y{year}_seed{ARGS.seed}_base_final{'_smoke' if ARGS.smoke else ''}.npy",
        base_final.astype(np.float32),
    )
    np.save(
        ROOT / "lab" / f"122_y{year}_seed{ARGS.seed}_control_final{'_smoke' if ARGS.smoke else ''}.npy",
        control_final.astype(np.float32),
    )

    results: dict[str, object] = {}
    thresholds = DISCOVERY_GATE if year == 2023 else CONFIRM_GATE
    for offset, (branch, probability) in enumerate(branch_probabilities.items()):
        routed = probability[:, 0].copy()
        if branch == "RECENT_F":
            routed[game_type != "F"] = base_probability[game_type != "F", 0]
            if not np.array_equal(routed[game_type != "F"], base_probability[game_type != "F", 0]):
                raise AssertionError("RECENT_F altered an R row before postprocessing")
        candidate_final = champion_space(routed, effect)
        if branch == "RECENT_F" and not np.array_equal(
            candidate_final[game_type != "F"], base_final[game_type != "F"]
        ):
            raise AssertionError("RECENT_F altered an R row after postprocessing")
        result = evaluate_branch(
            branch,
            base_final,
            candidate_final,
            control_final,
            target,
            game_type,
            pitcher,
            bootstrap_seed=122000 + year * 10 + offset,
        )
        passed, checks = gate(result, thresholds)
        result["gate_pass"] = passed
        result["gate_checks"] = checks
        results[branch] = result
        report_branch(result, passed, checks)
        np.save(
            ROOT / "lab" / f"122_y{year}_seed{ARGS.seed}_{branch.lower()}_final{'_smoke' if ARGS.smoke else ''}.npy",
            candidate_final.astype(np.float32),
        )

    del base_model, control_model, base_probability, control_probability, branch_probabilities
    gc.collect()
    return {
        "year": year,
        "counts": data["counts"],
        "effect_diagnostics": data["effect_diagnostics"],
        "feature_diagnostics": data["feature_diagnostics"],
        "base_exactness_qa": qa,
        "branches": results,
    }


def main() -> None:
    LIVE_TXT.parent.mkdir(parents=True, exist_ok=True)
    LIVE_TXT.write_text("", encoding="utf-8")
    log("=== exp/122 strict temporal CatBoost continuation ===")
    log(f"mode={'SMOKE' if ARGS.smoke else 'FULL'} threads={ARGS.threads} seed={ARGS.seed}")
    log(f"base={BASE_PARAMS}")
    log(f"append={APPEND_PARAMS}")
    log(f"discovery_gate={DISCOVERY_GATE}")
    log("shape=1e5*corr^2 is reported only as same-fold unconstrained-affine oracle")

    summary: dict[str, object] = {
        "experiment": 122,
        "smoke": bool(ARGS.smoke),
        "seed": int(ARGS.seed),
        "threads": int(ARGS.threads),
        "base_params": BASE_PARAMS,
        "append_params": APPEND_PARAMS,
        "discovery_gate": DISCOVERY_GATE,
        "confirmation_gate": CONFIRM_GATE,
        "folds": {},
        "reads_test_or_public_lb": False,
    }

    discovery = run_fold(2023)
    summary["folds"]["2023"] = discovery
    eligible = [
        result
        for result in discovery["branches"].values()
        if bool(result["gate_pass"])
    ]
    if eligible:
        # Locked selection rule: analytic blend gain, then oracle shape gain,
        # then lexical branch name.  Confirmation cannot change the choice.
        selected = max(
            eligible,
            key=lambda result: (
                float(result["blend"]["analytic_gain"]),
                float(result["oracle_shape_gain"]),
                str(result["name"]),
            ),
        )["name"]
        summary["selected_branch"] = selected
        summary["discovery_pass"] = True
        log(f"DISCOVERY PASS: locked branch={selected}")
    else:
        summary["selected_branch"] = None
        summary["discovery_pass"] = False
        summary["confirmation_opened"] = False
        log("DISCOVERY FAIL: 2024 remains unopened; experiment stops")
        save_summary(summary)
        return

    if ARGS.smoke or ARGS.discovery_only:
        summary["confirmation_opened"] = False
        log("confirmation not opened by run mode")
        save_summary(summary)
        return

    summary["confirmation_opened"] = True
    confirmation = run_fold(2024, selected_branch=str(selected))
    summary["folds"]["2024"] = confirmation
    confirmed = confirmation["branches"][str(selected)]
    summary["confirmation_pass"] = bool(confirmed["gate_pass"])
    blend = confirmed["blend"]
    discovery_selected = discovery["branches"][str(selected)]
    summary["moonshot_track"] = bool(
        confirmed["gate_pass"]
        and float(discovery_selected["raw_gain"]) > 0.0
        and float(discovery_selected["oracle_shape_gain"]) > 0.0
        and (
            float(confirmed["raw_gain"]) >= 50.0
            or float(blend["analytic_gain"]) >= 50.0
        )
    )
    log(
        "CONFIRMATION "
        + ("PASS" if summary["confirmation_pass"] else "FAIL")
        + f"; moonshot_track={summary['moonshot_track']}"
    )
    log("No bundle or submission was generated.")
    save_summary(summary)


if __name__ == "__main__":
    main()
