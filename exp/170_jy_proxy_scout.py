# -*- coding: utf-8 -*-
"""EXP-170: official-data-only JY-style heterogeneous residual scout.

This is an independent implementation from public method descriptions.  It
does not read third-party code, predictions, weights, models, or archives.

Discovery is strictly staged:

* train <=2021 -> predict 2022, then fit the one-probability R residual head;
* train <=2022 -> predict 2023 and evaluate the frozen head at scale 0.15.

Confirmation rebuilds the same feature contract for 2024, trains through
2023, and evaluates once.  The heterogeneous base is also evaluated as a
standalone endpoint against the frozen current-clean OOF, including exact
d/K analytic blend geometry.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "lab"
TRAIN = ROOT / "data" / "train.csv"
CACHE23 = LAB / "127_y2023_full_features.joblib"
CACHE24 = LAB / "170_y2024_features.joblib"
DISCOVERY_JSON = LAB / "170_jy_proxy_discovery.json"
CONFIRM_JSON = LAB / "170_jy_proxy_confirmation.json"
LIVE_LOG = LAB / "170_jy_proxy_live.log"
EPS = 1e-6
THREADS_DEFAULT = 6
SEED = 42
HEAD_SCALE = 0.15

# Public phase-state parameters.  Unpublished training details are fixed here
# explicitly so this experiment is reproducible and cannot drift after 2024.
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "lgb": {
        "n_estimators": 136,
        "learning_rate": 0.0438430751494052,
        "num_leaves": 114,
        "min_child_samples": 548,
        "subsample": 0.8462157153779587,
        "colsample_bytree": 0.6446835238410527,
        "reg_lambda": 0.0790298703310911,
    },
    "cat": {
        "iterations": 198,
        "learning_rate": 0.0639592615506268,
        "depth": 8,
        "l2_leaf_reg": 18.971156962243573,
    },
    "xgb": {
        "n_estimators": 255,
        "learning_rate": 0.017895810269671696,
        "max_depth": 8,
        "subsample": 0.6777809131457516,
        "colsample_bytree": 0.6313501720730692,
        "min_child_weight": 72,
        "reg_lambda": 1.5091241008455867,
    },
}
HEAD_SPEC = {
    "iterations": 150,
    "depth": 4,
    "learning_rate": 0.05,
    "l2_leaf_reg": 10.0,
    "loss_function": "RMSE",
    "random_seed": SEED,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("discover", "confirm"))
    parser.add_argument(
        "--families", default="lgb,cat",
        help="comma separated subset of lgb,cat,xgb",
    )
    parser.add_argument("--threads", type=int, default=THREADS_DEFAULT)
    parser.add_argument("--rebuild-2024-features", action="store_true")
    return parser.parse_args()


ARGS = parse_args()
FAMILIES = tuple(value.strip() for value in ARGS.families.split(",") if value.strip())
if not FAMILIES or any(value not in MODEL_SPECS for value in FAMILIES):
    raise ValueError(f"invalid families: {FAMILIES}")
if ARGS.threads < 1 or ARGS.threads > 6:
    raise ValueError("thread budget must be in [1, 6]")
STARTED = time.time()


def log(message: str) -> None:
    line = f"[{time.time() - STARTED:8.1f}s] {message}"
    print(line, flush=True)
    with LIVE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_exp127():
    original = sys.argv[:]
    try:
        sys.argv = [str(ROOT / "exp" / "127_dynamic_hier_residual.py"), "--full",
                    "--threads", str(ARGS.threads)]
        return load_module(ROOT / "exp" / "127_dynamic_hier_residual.py", "exp170_e127")
    finally:
        sys.argv = original


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_rows() -> pd.DataFrame:
    columns = ["row_id", "season", "game_month", "game_type", "pitcher_id",
               "control_success"]
    frame = pd.read_csv(TRAIN, encoding="utf-8-sig", usecols=columns, low_memory=False)
    frame.columns = frame.columns.str.replace("\ufeff", "", regex=False).str.strip()
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    if not frame["season"].is_monotonic_increasing or frame["row_id"].duplicated().any():
        raise AssertionError("official row ordering/identity contract failed")
    return frame


def load_current_anchor(rows: pd.DataFrame, year: int) -> np.ndarray:
    e155 = load_module(ROOT / "exp" / "155_cause_runner_currentblend_gate.py",
                       f"exp170_e155_{year}")
    n = len(rows)
    target = rows["control_success"].to_numpy(np.float64)
    saved_target = e155.load_vector(e155.EXP021_TARGET[year], n)
    if not np.array_equal(target, saved_target):
        raise AssertionError(f"current anchor target/order mismatch for {year}")
    cat = e155.load_probability(e155.CAT5[year], n)
    v18 = e155.load_vector(e155.V18[year], n)
    endpoint = e155.load_probability(e155.EXP021[year], n)
    champion = e155.champion_probability(cat, v18)
    anchor = e155.CHAMPION_WEIGHT * champion + e155.EXP021_WEIGHT * endpoint
    if anchor.shape != (n,) or not np.isfinite(anchor).all():
        raise AssertionError(f"bad current anchor {year}")
    return anchor


def numeric_pair(train: pd.DataFrame, valid: pd.DataFrame,
                 categorical: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic train-defined coding for numeric-only tree families."""
    train_matrix = np.empty((len(train), train.shape[1]), dtype=np.float32)
    valid_matrix = np.empty((len(valid), valid.shape[1]), dtype=np.float32)
    categorical_set = set(categorical)
    for index, column in enumerate(train.columns):
        if column in categorical_set:
            categories = pd.Index(train[column].astype(str).unique())
            train_matrix[:, index] = pd.Categorical(
                train[column].astype(str), categories=categories
            ).codes.astype(np.float32)
            valid_matrix[:, index] = pd.Categorical(
                valid[column].astype(str), categories=categories
            ).codes.astype(np.float32)
        else:
            train_matrix[:, index] = pd.to_numeric(
                train[column], errors="coerce"
            ).to_numpy(np.float32)
            valid_matrix[:, index] = pd.to_numeric(
                valid[column], errors="coerce"
            ).to_numpy(np.float32)
    train_matrix[~np.isfinite(train_matrix)] = np.nan
    valid_matrix[~np.isfinite(valid_matrix)] = np.nan
    return train_matrix, valid_matrix


def fit_family(name: str, train_x: pd.DataFrame, train_residual: np.ndarray,
               valid_x: pd.DataFrame, categorical: list[str]) -> np.ndarray:
    spec = MODEL_SPECS[name]
    log(f"fit {name}: train={len(train_x):,} valid={len(valid_x):,}")
    if name == "cat":
        model = CatBoostRegressor(
            **spec, loss_function="RMSE", random_seed=SEED,
            thread_count=ARGS.threads, verbose=False, allow_writing_files=False,
        )
        model.fit(Pool(train_x, train_residual, cat_features=categorical))
        prediction = model.predict(Pool(valid_x, cat_features=categorical))
    else:
        train_numeric, valid_numeric = numeric_pair(train_x, valid_x, categorical)
        if name == "lgb":
            model = LGBMRegressor(
                **spec, objective="regression", random_state=SEED,
                n_jobs=ARGS.threads, verbosity=-1,
                bagging_seed=SEED, feature_fraction_seed=SEED, data_random_seed=SEED,
            )
        else:
            try:
                from xgboost import XGBRegressor
            except ImportError as error:
                raise RuntimeError("xgboost is not installed; run lgb,cat first") from error
            model = XGBRegressor(
                **spec, objective="reg:squarederror", random_state=SEED,
                n_jobs=ARGS.threads, tree_method="hist", verbosity=0,
            )
        model.fit(train_numeric, train_residual)
        prediction = model.predict(valid_numeric)
        del train_numeric, valid_numeric
    del model
    gc.collect()
    prediction = np.asarray(prediction, np.float64)
    if prediction.shape != (len(valid_x),) or not np.isfinite(prediction).all():
        raise AssertionError(f"bad {name} prediction")
    return prediction


def base_ensemble(train_x: pd.DataFrame, train_y: np.ndarray, train_p0: np.ndarray,
                  valid_x: pd.DataFrame, valid_p0: np.ndarray,
                  categorical: list[str], label: str) -> tuple[np.ndarray, dict[str, Any]]:
    residual = np.asarray(train_y, np.float64) - np.asarray(train_p0, np.float64)
    member_predictions: list[np.ndarray] = []
    diagnostics: dict[str, Any] = {}
    for family in FAMILIES:
        effect = fit_family(family, train_x, residual, valid_x, categorical)
        prediction = np.clip(np.asarray(valid_p0, np.float64) + effect, EPS, 1.0 - EPS)
        path = LAB / f"170_{label}_{family}.npy"
        np.save(path, prediction.astype(np.float32))
        diagnostics[family] = {
            "path": str(path), "mean": float(prediction.mean()),
            "std": float(prediction.std()), "sha256": sha256_file(path),
        }
        member_predictions.append(prediction)
    ensemble = np.mean(member_predictions, axis=0)
    if not np.isfinite(ensemble).all():
        raise AssertionError("non-finite ensemble")
    path = LAB / f"170_{label}_ensemble.npy"
    np.save(path, ensemble.astype(np.float32))
    diagnostics["ensemble"] = {
        "path": str(path), "mean": float(ensemble.mean()),
        "std": float(ensemble.std()), "sha256": sha256_file(path),
    }
    return ensemble, diagnostics


def score(prediction: np.ndarray, target: np.ndarray) -> float:
    rate = float(np.mean(target))
    return float(100000.0 * (1.0 - np.mean(np.square(prediction - target)) /
                             (rate * (1.0 - rate))))


def geometry(anchor: np.ndarray, candidate: np.ndarray,
             target: np.ndarray) -> dict[str, float]:
    base_score = score(anchor, target)
    candidate_score = score(candidate, target)
    rate = float(np.mean(target))
    k = float(100000.0 * np.mean(np.square(candidate - anchor)) /
              (rate * (1.0 - rate)))
    d = candidate_score - base_score
    w_raw = float((d + k) / (2.0 * k)) if k > 1e-15 else 0.0
    w = float(np.clip(w_raw, 0.0, 1.0))
    blended_unclipped = anchor + w * (candidate - anchor)
    blended = np.clip(blended_unclipped, EPS, 1.0 - EPS)
    return {
        "anchor_score": base_score, "candidate_score": candidate_score,
        "d": d, "K": k, "d_plus_K": d + k,
        "optimal_weight_candidate_unconstrained": w_raw,
        "optimal_weight_candidate_constrained": w,
        "analytic_gain_unclipped_quadratic": float(w * (d + k) - w * w * k),
        "actual_clipped_gain": score(blended, target) - base_score,
        "blend_mean": float(blended.mean()),
    }


def contribution(anchor: np.ndarray, candidate: np.ndarray, target: np.ndarray,
                 mask: np.ndarray) -> float:
    rate = float(target.mean())
    gain = np.square(anchor - target) - np.square(candidate - target)
    return float(100000.0 * gain[np.asarray(mask, bool)].sum() /
                 (len(target) * rate * (1.0 - rate)))


def subset_gain(anchor: np.ndarray, candidate: np.ndarray, target: np.ndarray,
                mask: np.ndarray) -> float:
    mask = np.asarray(mask, bool)
    return score(candidate[mask], target[mask]) - score(anchor[mask], target[mask])


def pitcher_bootstrap(anchor: np.ndarray, candidate: np.ndarray,
                      target: np.ndarray, pitcher: np.ndarray,
                      draws: int = 3000) -> dict[str, float]:
    rate = float(target.mean())
    row_gain = (
        100000.0
        * (np.square(anchor - target) - np.square(candidate - target))
        / (len(target) * rate * (1.0 - rate))
    )
    grouped = pd.DataFrame({"pitcher": pitcher, "gain": row_gain}).groupby(
        "pitcher", sort=False, dropna=False
    )["gain"].sum().to_numpy(np.float64)
    rng = np.random.default_rng(170)
    samples = np.empty(draws, np.float64)
    for start in range(0, draws, 100):
        size = min(100, draws - start)
        indices = rng.integers(0, len(grouped), size=(size, len(grouped)))
        samples[start:start + size] = grouped[indices].sum(axis=1)
    q = np.quantile(samples, [0.025, 0.5, 0.975])
    return {"pitchers": int(len(grouped)), "draws": draws,
            "p025": float(q[0]), "median": float(q[1]), "p975": float(q[2]),
            "prob_positive": float(np.mean(samples > 0.0))}


def residual_head(source_team: np.ndarray, source_anchor: np.ndarray,
                  source_target: np.ndarray, source_game: np.ndarray,
                  valid_team: np.ndarray, valid_anchor: np.ndarray,
                  valid_game: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_r = np.asarray(source_game).astype(str) == "R"
    valid_r = np.asarray(valid_game).astype(str) == "R"
    model = CatBoostRegressor(
        **HEAD_SPEC, thread_count=ARGS.threads, verbose=False,
        allow_writing_files=False,
    )
    model.fit(source_team[source_r, None],
              (source_target - source_anchor)[source_r])
    effect = np.zeros(len(valid_team), np.float64)
    effect[valid_r] = model.predict(valid_team[valid_r, None])
    candidate = np.clip(valid_anchor + HEAD_SCALE * effect, EPS, 1.0 - EPS)
    if not np.array_equal(candidate[~valid_r], valid_anchor[~valid_r]):
        raise AssertionError("R-only head altered F rows")
    return candidate, effect


def evaluation(rows: pd.DataFrame, anchor: np.ndarray, team: np.ndarray,
               head_candidate: np.ndarray) -> dict[str, Any]:
    target = rows["control_success"].to_numpy(np.float64)
    game = rows["game_type"].astype(str).to_numpy()
    is_r = game == "R"
    month = pd.to_numeric(rows["game_month"], errors="raise").to_numpy()
    routed_team = np.where(is_r, team, anchor)
    f_routed_team = np.where(~is_r, team, anchor)
    head_gain = score(head_candidate, target) - score(anchor, target)
    bootstrap = pitcher_bootstrap(
        anchor, head_candidate, target, rows["pitcher_id"].to_numpy()
    )
    checks = {
        "raw_gain_ge_8": head_gain >= 8.0,
        "F_exact_zero": np.array_equal(head_candidate[~is_r], anchor[~is_r]),
        "R_positive": contribution(anchor, head_candidate, target, is_r) > 0.0,
        "early_nonnegative": subset_gain(anchor, head_candidate, target, month <= 6) >= 0.0,
        "late_nonnegative": subset_gain(anchor, head_candidate, target, month > 6) >= 0.0,
        "pitcher_p025_nonnegative": bootstrap["p025"] >= 0.0,
    }
    return {
        "rows": int(len(rows)), "target_rate": float(target.mean()),
        "anchor_score": score(anchor, target),
        "standalone_team_geometry": geometry(anchor, team, target),
        "R_routed_team_geometry": geometry(anchor, routed_team, target),
        "F_routed_team_geometry": geometry(anchor, f_routed_team, target),
        "F_segment_geometry": geometry(anchor[~is_r], team[~is_r], target[~is_r]),
        "R_segment_geometry": geometry(anchor[is_r], team[is_r], target[is_r]),
        "head": {
            "score": score(head_candidate, target), "raw_gain": head_gain,
            "F_contribution": contribution(anchor, head_candidate, target, ~is_r),
            "R_contribution": contribution(anchor, head_candidate, target, is_r),
            "early_gain": subset_gain(anchor, head_candidate, target, month <= 6),
            "late_gain": subset_gain(anchor, head_candidate, target, month > 6),
            "pitcher_bootstrap": bootstrap, "checks": checks,
            "pass": bool(all(checks.values())),
        },
    }


def load_cache23() -> dict[str, Any]:
    if not CACHE23.is_file():
        raise FileNotFoundError(CACHE23)
    log("load frozen 2023 101-feature cache")
    payload = joblib.load(CACHE23)
    if len(payload["feature_names"]) != 101:
        raise AssertionError("unexpected feature contract")
    return payload


def build_or_load_2024_features(feature_names: list[str], categorical: list[str]):
    if CACHE24.is_file() and not ARGS.rebuild_2024_features:
        log("load cached 2024 feature matrix")
        payload = joblib.load(CACHE24)
        if payload["feature_names"] != feature_names:
            raise AssertionError("2024 cache feature schema mismatch")
        return payload

    log("build official-data-only 2024 feature matrix")
    m = load_exp127()
    frame = pd.read_csv(TRAIN, encoding="utf-8-sig", low_memory=False)
    frame.columns = frame.columns.str.replace("\ufeff", "", regex=False).str.strip()
    base_columns = [column for column in frame.columns
                    if column not in ("row_id", "control_success")]
    if len(base_columns) != 47:
        raise AssertionError("base feature contract changed")
    history = m.recover_pitch_labels(frame.loc[frame["season"].le(2023)].copy())
    valid_raw = m.add_count_group(frame.loc[frame["season"].eq(2024)].copy())
    for local in (history, valid_raw):
        local["pitcher_hand"] = pd.to_numeric(local["pitcher_hand"], errors="coerce")
        local["batter_hand"] = pd.to_numeric(local["batter_hand"], errors="coerce")
    volatility, missing = m.row_volatility(history)
    finite = volatility[(~missing) & np.isfinite(volatility)]
    volatility_scale = max(float(np.quantile(finite, 0.75)), 0.01)

    # attach_existing_features also fits its train-only pitch-type helper.  A
    # deterministic stratified 5k history sample is sufficient for the
    # discarded training return; the validation features still use all history.
    dummy = pd.concat(
        [
            part.sample(min(1000, len(part)), random_state=170)
            for _, part in frame.loc[frame["season"].le(2023)].groupby(
                "season", sort=True
            )
        ],
        axis=0,
    )
    dummy = dummy.sort_index(kind="stable")
    dummy = dummy.loc[history.index.intersection(dummy.index)].copy()
    dummy = history.loc[dummy.index].copy()
    _, valid_enriched, existing_diag = m.attach_existing_features(
        dummy, valid_raw, history, base_columns
    )
    hierarchy, hierarchy_diag = m.attach_hierarchical(
        valid_raw, history, volatility_scale, "exp170-valid-2024"
    )
    hierarchy.index = valid_enriched.index
    for name in m.HIER_FEATURES:
        valid_enriched[name] = hierarchy[name].to_numpy()
    valid_x = m.model_frame(valid_enriched, feature_names)
    valid_p0 = hierarchy["h_base"].to_numpy(np.float64)
    output = {
        "feature_names": feature_names, "cat_features": categorical,
        "valid_x": valid_x, "valid_p0": valid_p0,
        "volatility_scale": volatility_scale,
        "existing_diagnostics": existing_diag,
        "hierarchy_diagnostics": hierarchy_diag,
        "row_ids": valid_raw["row_id"].astype(str).to_numpy(),
    }
    temporary = CACHE24.with_suffix(".joblib.tmp")
    joblib.dump(output, temporary, compress=3)
    os.replace(temporary, CACHE24)
    log(f"saved 2024 feature cache: {CACHE24}")
    return output


def discover() -> None:
    cache = load_cache23()
    rows_all = official_rows()
    train_positions = np.asarray(cache["train_positions"], np.int64)
    train_rows = rows_all.iloc[train_positions].reset_index(drop=True)
    rows22 = train_rows.loc[train_rows["season"].eq(2022)].reset_index(drop=True)
    rows23 = rows_all.loc[rows_all["season"].eq(2023)].reset_index(drop=True)
    mask21 = train_rows["season"].to_numpy() <= 2021
    mask22 = train_rows["season"].to_numpy() == 2022
    if not np.array_equal(cache["valid_y"], rows23["control_success"].to_numpy()):
        raise AssertionError("cache/official 2023 target mismatch")

    team22, diag22 = base_ensemble(
        cache["train_x"].loc[mask21].reset_index(drop=True),
        np.asarray(cache["train_y"])[mask21], np.asarray(cache["train_p0"])[mask21],
        cache["train_x"].loc[mask22].reset_index(drop=True),
        np.asarray(cache["train_p0"])[mask22], cache["cat_features"], "team_y2022",
    )
    team23, diag23 = base_ensemble(
        cache["train_x"], np.asarray(cache["train_y"]), np.asarray(cache["train_p0"]),
        cache["valid_x"], np.asarray(cache["valid_p0"]),
        cache["cat_features"], "team_y2023",
    )
    anchor22 = load_current_anchor(rows22, 2022)
    anchor23 = load_current_anchor(rows23, 2023)
    head23, effect23 = residual_head(
        team22, anchor22, rows22["control_success"].to_numpy(np.float64),
        rows22["game_type"].to_numpy(), team23, anchor23,
        rows23["game_type"].to_numpy(),
    )
    np.save(LAB / "170_head_effect_y2023.npy", effect23.astype(np.float32))
    np.save(LAB / "170_head_candidate_y2023.npy", head23.astype(np.float32))
    result = {
        "experiment": 170, "phase": "strict_discovery_2022_to_2023",
        "status": "PASS_SCOUT" if evaluation(rows23, anchor23, team23, head23)["head"]["pass"]
                  else "HEAD_FAIL_STANDALONE_REVIEW",
        "data_boundary": "official train + own frozen current-clean OOF/cache only",
        "families": list(FAMILIES), "seed": SEED, "threads": ARGS.threads,
        "feature_contract": {"count": len(cache["feature_names"]),
                             "names": cache["feature_names"],
                             "categorical": cache["cat_features"]},
        "model_specs": {name: MODEL_SPECS[name] for name in FAMILIES},
        "head_spec": {**HEAD_SPEC, "scale": HEAD_SCALE,
                      "feature": "single p_team probability", "domain": "R only"},
        "members_2022": diag22, "members_2023": diag23,
        "evaluation_2023": evaluation(rows23, anchor23, team23, head23),
        "source_sha256": {str(TRAIN): sha256_file(TRAIN), str(CACHE23): sha256_file(CACHE23)},
        "elapsed_seconds": time.time() - STARTED,
    }
    write_json(DISCOVERY_JSON, result)
    log(f"discovery={result['status']} result={DISCOVERY_JSON}")
    log(json.dumps(result["evaluation_2023"], ensure_ascii=False))


def confirm() -> None:
    if not DISCOVERY_JSON.is_file():
        raise FileNotFoundError("run discover first")
    discovery = json.loads(DISCOVERY_JSON.read_text(encoding="utf-8"))
    if tuple(discovery["families"]) != FAMILIES:
        raise AssertionError("confirmation family set differs from discovery lock")
    cache = load_cache23()
    cache24 = build_or_load_2024_features(cache["feature_names"], cache["cat_features"])
    rows_all = official_rows()
    rows23 = rows_all.loc[rows_all["season"].eq(2023)].reset_index(drop=True)
    rows24 = rows_all.loc[rows_all["season"].eq(2024)].reset_index(drop=True)
    if not np.array_equal(rows24["row_id"].astype(str), cache24["row_ids"]):
        raise AssertionError("2024 feature row ordering mismatch")
    train_x = pd.concat([cache["train_x"], cache["valid_x"]], ignore_index=True)
    train_y = np.concatenate([cache["train_y"], cache["valid_y"]])
    train_p0 = np.concatenate([cache["train_p0"], cache["valid_p0"]])
    team24, diag24 = base_ensemble(
        train_x, train_y, train_p0, cache24["valid_x"], cache24["valid_p0"],
        cache["cat_features"], "team_y2024",
    )
    team23 = np.load(LAB / "170_team_y2023_ensemble.npy", allow_pickle=False).astype(np.float64)
    anchor23 = load_current_anchor(rows23, 2023)
    anchor24 = load_current_anchor(rows24, 2024)
    head24, effect24 = residual_head(
        team23, anchor23, rows23["control_success"].to_numpy(np.float64),
        rows23["game_type"].to_numpy(), team24, anchor24,
        rows24["game_type"].to_numpy(),
    )
    np.save(LAB / "170_head_effect_y2024.npy", effect24.astype(np.float32))
    np.save(LAB / "170_head_candidate_y2024.npy", head24.astype(np.float32))
    evaluated = evaluation(rows24, anchor24, team24, head24)
    result = {
        "experiment": 170, "phase": "one_shot_confirmation_2023_to_2024",
        "status": "PASS_EXPAND_TO_15" if evaluated["head"]["pass"]
                  else "HEAD_FAIL_STANDALONE_REVIEW",
        "locked_discovery_sha256": sha256_file(DISCOVERY_JSON),
        "families": list(FAMILIES), "seed": SEED, "threads": ARGS.threads,
        "model_specs": {name: MODEL_SPECS[name] for name in FAMILIES},
        "head_spec": {**HEAD_SPEC, "scale": HEAD_SCALE,
                      "feature": "single p_team probability", "domain": "R only"},
        "members_2024": diag24, "evaluation_2024": evaluated,
        "feature_cache_2024": str(CACHE24),
        "post_2024_tuning": False,
        "elapsed_seconds": time.time() - STARTED,
    }
    write_json(CONFIRM_JSON, result)
    log(f"confirmation={result['status']} result={CONFIRM_JSON}")
    log(json.dumps(evaluated, ensure_ascii=False))


if __name__ == "__main__":
    if ARGS.phase == "discover":
        discover()
    else:
        confirm()
