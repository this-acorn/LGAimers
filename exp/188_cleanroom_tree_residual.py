# -*- coding: utf-8 -*-
"""EXP-188: official-data heterogeneous tree residual ensemble.

This is an independent implementation built from our own repository components.
It never reads or extracts another team's source, model, prediction, weight, or
archive.  The only learning rows are official ``data/train.csv`` rows, and the
residual target uses our own season-aligned current-affine OOF probabilities.

The default two-family path is intentionally modest enough to run on CPU, while
``--device auto`` will use an available GPU and fall back to CPU per family.
Models trained on GPU are exported in formats that support CPU inference.

Strict workflow::

    python exp/188_cleanroom_tree_residual.py discover --smoke --device cpu
    python exp/188_cleanroom_tree_residual.py confirm  --smoke --device cpu

Full workflow::

    python exp/188_cleanroom_tree_residual.py all --device auto

``discover`` fits 2022 residuals and locks family weights plus one conservative
scale on 2023.  ``confirm`` refits the locked model family on 2022+2023 and opens
2024 once.  ``build`` is allowed only after a full strict PASS, then refits on
2022--2024 and wraps our current affine package.  The model sees our two endpoint
probabilities and their disagreement, with only count, game type, and experience
as small interaction contexts.  Raw IDs, teams, score, runner state, and the rest
of the original feature table are deliberately excluded.  Every test feature is
a deterministic function of the current row and frozen models; no test-row
aggregate, order, frequency, or distribution is used.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"
SAMPLE_PATH = ROOT / "data" / "sample_submission.csv"
BASE_ZIP = ROOT / 'artifacts/candidates/candidate_affine_opt.zip'
DEFAULT_WORK = ROOT / "lab" / "188_tree_residual"

ID_COLUMN = "row_id"
TARGET_COLUMN = "control_success"
SEED = 188
EPSILON = 1.0e-6
CORRECTION_CLIP = 0.06
YEAR_WEIGHTS = {2022: 0.49, 2023: 0.70, 2024: 1.00}
DISCOVERY_SCALE_GRID = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)
WEIGHT_GRID_UNITS = 4
SMOKE_ROWS_PER_YEAR = 800

# Exact constants used by our current affine package.  These reconstruct the
# same transform on our season-aligned OOF components without reading test data.
AFFINE_CHAMPION_WEIGHT = 0.655318
AFFINE_SECOND_WEIGHT = 0.38781
AFFINE_SHIFT = 0.020206959

MODEL_FEATURES = (
    "_component_primary",
    "_component_secondary",
    "_anchor_probability",
    "_component_gap",
    "_component_abs_gap",
    "_component_gap_sq",
    "_primary_from_anchor",
    "_secondary_from_anchor",
    "_game_type_code",
    "_count_state",
    "_log_pitcher_n",
    "_log_batter_n",
    "_gap_by_game_type",
    "_gap_by_count",
    "_gap_by_pitcher_experience",
    "_gap_by_batter_experience",
)

MODEL_CATEGORICAL = (
    "_game_type_code",
    "_count_state",
)

TRAIN_COLUMNS = (
    ID_COLUMN,
    "season",
    TARGET_COLUMN,
    "game_type",
    "game_month",
    "pitcher_id",
    "balls_before",
    "strikes_before",
    "asof_pitcher_n",
    "asof_batter_n",
)


MODEL_SPECS: dict[str, dict[str, Any]] = {
    # Rounded, deliberately conservative settings chosen for this implementation.
    "lgb": {
        "num_boost_round": 180,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_data_in_leaf": 800,
        "lambda_l2": 30.0,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.85,
    },
    "cat": {
        "iterations": 200,
        "learning_rate": 0.04,
        "depth": 7,
        "l2_leaf_reg": 25.0,
        "random_strength": 0.25,
        "bagging_temperature": 0.5,
    },
    "xgb": {
        "n_estimators": 220,
        "learning_rate": 0.035,
        "max_depth": 6,
        "min_child_weight": 64.0,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_lambda": 12.0,
        "max_bin": 256,
    },
}

SMOKE_OVERRIDES: dict[str, dict[str, Any]] = {
    "lgb": {"num_boost_round": 4, "min_data_in_leaf": 50},
    "cat": {"iterations": 4},
    "xgb": {"n_estimators": 4},
}


RUNTIME_SOURCE = r'''# -*- coding: utf-8 -*-
"""Row-local affine anchor plus frozen official-data tree correction."""

from __future__ import annotations

import gc
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLUMN = "row_id"
TARGET_COLUMN = "control_success"
MODEL_DIR = Path("./model")
TEST_PATH = Path("./data/test.csv")
SAMPLE_PATH = Path("./data/sample_submission.csv")
OUTPUT_DIR = Path("./output")
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    value = pd.to_numeric(frame[column], errors="coerce")
    if np.isfinite(default):
        value = value.fillna(default)
    return value


def add_component_features(frame: pd.DataFrame, primary: np.ndarray,
                           secondary: np.ndarray, anchor: np.ndarray) -> pd.DataFrame:
    """Endpoint-disagreement features with only minimal row-local context."""
    out = pd.DataFrame(index=frame.index)
    primary = np.asarray(primary, dtype=np.float64)
    secondary = np.asarray(secondary, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    if any(value.shape != (len(frame),) for value in (primary, secondary, anchor)):
        raise RuntimeError("component length mismatch")
    gap = secondary - primary
    balls = _numeric(frame, "balls_before", -1).astype("int16")
    strikes = _numeric(frame, "strikes_before", -1).astype("int16")
    count_state = (balls * 3 + strikes).astype("int16")
    game_code = (
        frame["game_type"].astype("string").fillna("__MISSING__")
        .map({"R": 0, "F": 1}).fillna(2).astype("int8")
    )
    log_pitcher_n = np.log1p(
        _numeric(frame, "asof_pitcher_n", 0).clip(lower=0)
    ).astype("float32")
    log_batter_n = np.log1p(
        _numeric(frame, "asof_batter_n", 0).clip(lower=0)
    ).astype("float32")
    out["_component_primary"] = primary.astype("float32")
    out["_component_secondary"] = secondary.astype("float32")
    out["_anchor_probability"] = anchor.astype("float32")
    out["_component_gap"] = gap.astype("float32")
    out["_component_abs_gap"] = np.abs(gap).astype("float32")
    out["_component_gap_sq"] = np.square(gap).astype("float32")
    out["_primary_from_anchor"] = (primary - anchor).astype("float32")
    out["_secondary_from_anchor"] = (secondary - anchor).astype("float32")
    out["_game_type_code"] = game_code
    out["_count_state"] = count_state
    out["_log_pitcher_n"] = log_pitcher_n
    out["_log_batter_n"] = log_batter_n
    out["_gap_by_game_type"] = (gap * game_code.to_numpy()).astype("float32")
    out["_gap_by_count"] = (gap * count_state.to_numpy()).astype("float32")
    out["_gap_by_pitcher_experience"] = (
        gap * log_pitcher_n.to_numpy()
    ).astype("float32")
    out["_gap_by_batter_experience"] = (
        gap * log_batter_n.to_numpy()
    ).astype("float32")
    return out


def transform(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    matrix = pd.DataFrame(index=frame.index)
    categorical = set(state["categorical_features"])
    mappings = state["category_mappings"]
    for column in state["feature_names"]:
        if column in categorical:
            text = frame[column].astype("string").fillna("__MISSING__").astype(str)
            matrix[column] = text.map(mappings[column]).fillna(-1).astype("int32")
        else:
            matrix[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    return matrix


def member_prediction(member: dict, matrix: pd.DataFrame,
                      categorical: list[str]) -> np.ndarray:
    family = member["family"]
    path = MODEL_DIR / member["file"]
    if family == "lgb":
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(path))
        value = model.predict(matrix, num_threads=member["inference_threads"])
    elif family == "cat":
        from catboost import CatBoostRegressor, Pool
        model = CatBoostRegressor()
        model.load_model(str(path))
        value = model.predict(Pool(matrix, cat_features=categorical))
    elif family == "xgb":
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.load_model(str(path))
        value = model.predict(matrix)
    else:
        raise RuntimeError(f"unsupported family: {family}")
    del model
    gc.collect()
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (len(matrix),) or not np.isfinite(value).all():
        raise RuntimeError(f"invalid {family} prediction")
    return value


def run_component(filename: str, namespace: str) -> pd.DataFrame:
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
    module = runpy.run_path(str(MODEL_DIR / filename), run_name=namespace)
    component_main = module.get("main")
    if not callable(component_main):
        raise RuntimeError(f"component has no callable main: {filename}")
    component_main()
    result = pd.read_csv(OUTPUT_PATH, encoding="utf-8-sig")
    if list(result.columns) != [ID_COLUMN, TARGET_COLUMN]:
        raise RuntimeError(f"invalid component schema: {filename}")
    if result[ID_COLUMN].duplicated().any():
        raise RuntimeError(f"duplicate component row_id: {filename}")
    del component_main, module
    gc.collect()
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    primary_frame = run_component("champion_inference.py", "primary_component")
    primary_ids = primary_frame[ID_COLUMN].to_numpy(copy=True)
    primary = primary_frame[TARGET_COLUMN].to_numpy(np.float64, copy=True)
    del primary_frame
    secondary_frame = run_component("exp021_inference.py", "secondary_component")
    secondary_ids = secondary_frame[ID_COLUMN].to_numpy(copy=True)
    secondary = secondary_frame[TARGET_COLUMN].to_numpy(np.float64, copy=True)
    del secondary_frame
    if not np.array_equal(primary_ids, secondary_ids):
        raise RuntimeError("component row order mismatch")
    gc.collect()

    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig", low_memory=False)
    test.columns = [str(column).replace("\ufeff", "").strip() for column in test.columns]
    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
    sample.columns = [str(column).replace("\ufeff", "").strip() for column in sample.columns]
    state = json.loads((MODEL_DIR / "tree_residual_state.json").read_text(encoding="utf-8"))
    transform_state = state["anchor_transform"]
    anchor = np.clip(
        float(transform_state["primary_weight"]) * primary
        + float(transform_state["secondary_weight"]) * secondary
        - float(transform_state["shift"]),
        0.0,
        1.0,
    )
    if len(test) != len(anchor) or not np.array_equal(test[ID_COLUMN].to_numpy(), primary_ids):
        raise RuntimeError("test and component row order differ")
    if len(sample) != len(anchor) or not np.array_equal(sample[ID_COLUMN].to_numpy(), primary_ids):
        raise RuntimeError("sample and component row order differ")
    enriched = add_component_features(test, primary, secondary, anchor)
    matrix = transform(enriched, state)
    effect = np.zeros(len(test), dtype=np.float64)
    for member in state["members"]:
        raw = member_prediction(member, matrix, state["categorical_features"])
        raw = np.clip(raw, -state["correction_clip"], state["correction_clip"])
        effect += float(member["weight"]) * raw
    if state["route"] == "R":
        is_r = test["game_type"].astype(str).to_numpy() == "R"
        effect = np.where(is_r, effect, 0.0)
    elif state["route"] != "all":
        raise RuntimeError(f"invalid route: {state['route']}")
    correction = float(state["scale"]) * effect
    prediction = np.clip(anchor + correction, 0.0, 1.0)
    if not np.isfinite(prediction).all():
        raise RuntimeError("non-finite final prediction")
    if np.any((prediction < 0.0) | (prediction > 1.0)):
        raise RuntimeError("final prediction outside [0,1]")

    pd.DataFrame({ID_COLUMN: primary_ids, TARGET_COLUMN: prediction}).to_csv(
        OUTPUT_PATH, index=False, encoding="utf-8"
    )
    print(
        f"Saved: {OUTPUT_PATH} | rows={len(prediction)} | "
        f"families={','.join(member['family'] for member in state['members'])} | "
        f"scale={state['scale']:.6f} | correction_mean={correction.mean():+.9f} | "
        f"correction_std={correction.std():.9f} | mean={prediction.mean():.9f} | "
        f"min={prediction.min():.9f} | max={prediction.max():.9f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("discover", "confirm", "build", "all"))
    parser.add_argument(
        "--families",
        default="lgb,cat",
        help="comma-separated subset of lgb,cat,xgb; default is lgb,cat",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--output-zip", type=Path)
    parser.add_argument("--qa-timeout", type=int, default=300)
    args = parser.parse_args()
    args.route = "all"
    args.families = tuple(
        value.strip() for value in str(args.families).split(",") if value.strip()
    )
    if not args.families or len(set(args.families)) != len(args.families):
        parser.error("families must be a non-empty unique list")
    unknown = sorted(set(args.families) - set(MODEL_SPECS))
    if unknown:
        parser.error(f"unknown families: {unknown}")
    if not 1 <= args.threads <= 8:
        parser.error("threads must be between 1 and 8")
    if args.qa_timeout < 30:
        parser.error("qa-timeout must be at least 30 seconds")
    args.work_dir = args.work_dir.resolve()
    return args


STARTED = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - STARTED:8.1f}s] {message}", flush=True)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(native(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def phase_dir(args: argparse.Namespace) -> Path:
    return args.work_dir / ("smoke" if args.smoke else "full")


def discovery_path(args: argparse.Namespace) -> Path:
    return phase_dir(args) / "discovery_lock.json"


def confirmation_path(args: argparse.Namespace) -> Path:
    return phase_dir(args) / "confirmation.json"


def build_path(args: argparse.Namespace) -> Path:
    return phase_dir(args) / "build.json"


def effective_specs(smoke: bool) -> dict[str, dict[str, Any]]:
    output = {family: dict(spec) for family, spec in MODEL_SPECS.items()}
    if smoke:
        for family, values in SMOKE_OVERRIDES.items():
            output[family].update(values)
    return output


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    value = pd.to_numeric(frame[column], errors="coerce")
    if np.isfinite(default):
        value = value.fillna(default)
    return value


def add_component_features(
    frame: pd.DataFrame,
    primary: np.ndarray,
    secondary: np.ndarray,
    anchor: np.ndarray,
) -> pd.DataFrame:
    """Build the narrow nonlinear endpoint-disagreement contract."""
    primary = np.asarray(primary, np.float64)
    secondary = np.asarray(secondary, np.float64)
    anchor = np.asarray(anchor, np.float64)
    if any(value.shape != (len(frame),) for value in (primary, secondary, anchor)):
        raise ValueError("component length mismatch")
    gap = secondary - primary
    balls = _numeric(frame, "balls_before", -1).astype("int16")
    strikes = _numeric(frame, "strikes_before", -1).astype("int16")
    count_state = (balls * 3 + strikes).astype("int16")
    game_code = (
        frame["game_type"].astype("string").fillna("__MISSING__")
        .map({"R": 0, "F": 1}).fillna(2).astype("int8")
    )
    log_pitcher_n = np.log1p(
        _numeric(frame, "asof_pitcher_n", 0).clip(lower=0)
    ).astype("float32")
    log_batter_n = np.log1p(
        _numeric(frame, "asof_batter_n", 0).clip(lower=0)
    ).astype("float32")
    out = pd.DataFrame(index=frame.index)
    out["_component_primary"] = primary.astype("float32")
    out["_component_secondary"] = secondary.astype("float32")
    out["_anchor_probability"] = anchor.astype("float32")
    out["_component_gap"] = gap.astype("float32")
    out["_component_abs_gap"] = np.abs(gap).astype("float32")
    out["_component_gap_sq"] = np.square(gap).astype("float32")
    out["_primary_from_anchor"] = (primary - anchor).astype("float32")
    out["_secondary_from_anchor"] = (secondary - anchor).astype("float32")
    out["_game_type_code"] = game_code
    out["_count_state"] = count_state
    out["_log_pitcher_n"] = log_pitcher_n
    out["_log_batter_n"] = log_batter_n
    out["_gap_by_game_type"] = (gap * game_code.to_numpy()).astype("float32")
    out["_gap_by_count"] = (gap * count_state.to_numpy()).astype("float32")
    out["_gap_by_pitcher_experience"] = (
        gap * log_pitcher_n.to_numpy()
    ).astype("float32")
    out["_gap_by_batter_experience"] = (
        gap * log_batter_n.to_numpy()
    ).astype("float32")
    if tuple(out.columns) != MODEL_FEATURES:
        raise AssertionError("component feature order drift")
    return out


def feature_contract(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    features = list(MODEL_FEATURES)
    categorical = list(MODEL_CATEGORICAL)
    if list(frame.columns) != features:
        raise AssertionError("unexpected columns outside narrow component contract")
    return features, categorical


def category_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("__MISSING__").astype(str)


def fit_category_schema(
    frame: pd.DataFrame, categorical: Iterable[str]
) -> dict[str, dict[str, int]]:
    mappings: dict[str, dict[str, int]] = {}
    for column in categorical:
        values = sorted(category_text(frame[column]).unique().tolist())
        mappings[column] = {value: index for index, value in enumerate(values)}
    return mappings


def transform_features(
    frame: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    mappings: dict[str, dict[str, int]],
) -> pd.DataFrame:
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    categorical_set = set(categorical)
    matrix = pd.DataFrame(index=frame.index)
    for column in features:
        if column in categorical_set:
            matrix[column] = (
                category_text(frame[column]).map(mappings[column]).fillna(-1).astype("int32")
            )
        else:
            matrix[column] = pd.to_numeric(frame[column], errors="coerce").astype(
                "float32"
            )
    if list(matrix.columns) != features:
        raise AssertionError("feature matrix order drift")
    return matrix


def load_current_affine_oof(e155, rows: pd.DataFrame, year: int) -> dict[str, np.ndarray]:
    target = rows[TARGET_COLUMN].to_numpy(np.float64)
    n_rows = len(rows)
    saved_target = e155.load_vector(e155.EXP021_TARGET[year], n_rows)
    if not np.array_equal(saved_target, target):
        raise AssertionError(f"OOF target/order mismatch for {year}")
    first = e155.load_probability(e155.CAT5[year], n_rows)
    first_effect = e155.load_vector(e155.V18[year], n_rows)
    second = e155.load_probability(e155.EXP021[year], n_rows)
    primary = e155.champion_probability(first, first_effect)
    secondary = second
    anchor = np.clip(
        AFFINE_CHAMPION_WEIGHT * primary
        + AFFINE_SECOND_WEIGHT * second
        - AFFINE_SHIFT,
        0.0,
        1.0,
    )
    if any(
        value.shape != (n_rows,) or not np.isfinite(value).all()
        for value in (primary, secondary, anchor)
    ):
        raise AssertionError(f"invalid current affine OOF for {year}")
    return {"primary": primary, "secondary": secondary, "anchor": anchor}


def deterministic_smoke_positions(rows: pd.DataFrame, year: int) -> np.ndarray:
    count = min(SMOKE_ROWS_PER_YEAR, len(rows))
    if count == len(rows):
        return np.arange(len(rows), dtype=np.int64)
    # Stable, stratified sampling retains both game regimes and target classes.
    selected: list[np.ndarray] = []
    group_count = rows.groupby(["game_type", TARGET_COLUMN], dropna=False).ngroups
    per_group = max(1, count // max(group_count, 1))
    for group_index, (_, group) in enumerate(
        rows.groupby(["game_type", TARGET_COLUMN], sort=True, dropna=False)
    ):
        take = min(per_group, len(group))
        rng = np.random.default_rng(SEED * 10000 + year * 10 + group_index)
        selected.append(rng.choice(group.index.to_numpy(), size=take, replace=False))
    positions = np.unique(np.concatenate(selected))
    if len(positions) < count:
        remaining = np.setdiff1d(np.arange(len(rows)), positions, assume_unique=True)
        rng = np.random.default_rng(SEED * 10000 + year)
        extra = rng.choice(remaining, size=count - len(positions), replace=False)
        positions = np.concatenate([positions, extra])
    return np.sort(positions[:count]).astype(np.int64)


def load_year_blocks(
    years: Iterable[int], smoke: bool
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(TRAIN_PATH)
    e155 = load_module(
        ROOT / "exp" / "155_cause_runner_currentblend_gate.py", "exp188_anchor"
    )
    frame = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=list(TRAIN_COLUMNS),
        low_memory=False,
    )
    frame.columns = [str(column).replace("\ufeff", "").strip() for column in frame.columns]
    if frame[ID_COLUMN].duplicated().any():
        raise AssertionError("official train has duplicate row_id")
    blocks: dict[int, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {
        "official_train_rows": int(len(frame)),
        "official_train_columns": int(frame.shape[1]),
        "years": {},
    }
    for year in years:
        full_rows = frame.loc[frame["season"].eq(year)].reset_index(drop=True)
        if full_rows.empty:
            raise ValueError(f"official train has no season {year}")
        full_components = load_current_affine_oof(e155, full_rows, year)
        if smoke:
            positions = deterministic_smoke_positions(full_rows, year)
            rows = full_rows.iloc[positions].reset_index(drop=True)
            primary = full_components["primary"][positions]
            secondary = full_components["secondary"][positions]
            anchor = full_components["anchor"][positions]
        else:
            positions = np.arange(len(full_rows), dtype=np.int64)
            rows = full_rows
            primary = full_components["primary"]
            secondary = full_components["secondary"]
            anchor = full_components["anchor"]
        target = rows[TARGET_COLUMN].to_numpy(np.float64)
        blocks[year] = {
            "year": year,
            "rows": rows,
            "primary": primary,
            "secondary": secondary,
            "anchor": anchor,
            "target": target,
            "positions": positions,
        }
        diagnostics["years"][str(year)] = {
            "full_rows": int(len(full_rows)),
            "used_rows": int(len(rows)),
            "target_rate": float(target.mean()),
            "anchor_mean": float(anchor.mean()),
            "anchor_std": float(anchor.std()),
            "component_gap_mean": float((secondary - primary).mean()),
            "component_gap_std": float((secondary - primary).std()),
        }
    del frame
    gc.collect()
    return blocks, diagnostics


def gpu_is_visible() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return False
    return result.returncode == 0 and "GPU" in (result.stdout or "")


def backend_order(requested: str) -> tuple[str, ...]:
    if requested == "cpu":
        return ("cpu",)
    if requested == "gpu":
        return ("gpu",)
    return ("gpu", "cpu") if gpu_is_visible() else ("cpu",)


def fit_family(
    family: str,
    train_matrix: pd.DataFrame,
    train_residual: np.ndarray,
    train_weight: np.ndarray,
    valid_matrix: pd.DataFrame | None,
    categorical: list[str],
    args: argparse.Namespace,
):
    spec = effective_specs(args.smoke)[family]
    failures: list[dict[str, str]] = []
    for backend in backend_order(args.device):
        started = time.time()
        try:
            log(
                f"fit {family} backend={backend} train={len(train_matrix):,} "
                f"valid={0 if valid_matrix is None else len(valid_matrix):,}"
            )
            if family == "lgb":
                import lightgbm as lgb

                params = {
                    "objective": "regression_l2",
                    "metric": "l2",
                    "learning_rate": spec["learning_rate"],
                    "num_leaves": spec["num_leaves"],
                    "min_data_in_leaf": spec["min_data_in_leaf"],
                    "lambda_l2": spec["lambda_l2"],
                    "feature_fraction": spec["feature_fraction"],
                    "bagging_fraction": spec["bagging_fraction"],
                    "bagging_freq": 1,
                    "max_bin": 63 if backend == "gpu" else 127,
                    "verbosity": -1,
                    "seed": SEED,
                    "feature_fraction_seed": SEED,
                    "bagging_seed": SEED,
                    "data_random_seed": SEED,
                    "num_threads": args.threads,
                    "force_col_wise": backend == "cpu",
                    "device_type": backend,
                }
                dataset = lgb.Dataset(
                    train_matrix,
                    label=np.asarray(train_residual, np.float32),
                    weight=np.asarray(train_weight, np.float32),
                    categorical_feature=categorical,
                    free_raw_data=True,
                )
                model = lgb.train(
                    params, dataset, num_boost_round=int(spec["num_boost_round"])
                )
                prediction = (
                    None
                    if valid_matrix is None
                    else model.predict(valid_matrix, num_threads=args.threads)
                )
                del dataset
            elif family == "cat":
                from catboost import CatBoostRegressor, Pool

                model = CatBoostRegressor(
                    iterations=int(spec["iterations"]),
                    learning_rate=float(spec["learning_rate"]),
                    depth=int(spec["depth"]),
                    l2_leaf_reg=float(spec["l2_leaf_reg"]),
                    random_strength=float(spec["random_strength"]),
                    bagging_temperature=float(spec["bagging_temperature"]),
                    loss_function="RMSE",
                    random_seed=SEED,
                    thread_count=args.threads,
                    task_type="GPU" if backend == "gpu" else "CPU",
                    devices="0" if backend == "gpu" else None,
                    verbose=False,
                    allow_writing_files=False,
                )
                train_pool = Pool(
                    train_matrix,
                    label=np.asarray(train_residual, np.float64),
                    weight=np.asarray(train_weight, np.float64),
                    cat_features=categorical,
                )
                model.fit(train_pool)
                prediction = (
                    None
                    if valid_matrix is None
                    else model.predict(Pool(valid_matrix, cat_features=categorical))
                )
                del train_pool
            elif family == "xgb":
                from xgboost import XGBRegressor

                parameters = {
                    **spec,
                    "objective": "reg:squarederror",
                    "random_state": SEED,
                    "n_jobs": args.threads,
                    "tree_method": "hist",
                    "verbosity": 0,
                }
                if backend == "gpu":
                    parameters["device"] = "cuda"
                model = XGBRegressor(**parameters)
                model.fit(
                    train_matrix,
                    train_residual,
                    sample_weight=train_weight,
                    verbose=False,
                )
                prediction = None if valid_matrix is None else model.predict(valid_matrix)
            else:
                raise ValueError(f"unsupported family: {family}")

            if prediction is not None:
                prediction = np.asarray(prediction, np.float64)
                if prediction.shape != (len(valid_matrix),) or not np.isfinite(
                    prediction
                ).all():
                    raise AssertionError(f"invalid {family} validation prediction")
                prediction = np.clip(prediction, -CORRECTION_CLIP, CORRECTION_CLIP)
            diagnostic = {
                "family": family,
                "backend": backend,
                "fit_seconds": float(time.time() - started),
                "fallback_failures": failures,
                "spec": spec,
            }
            return model, prediction, diagnostic
        except Exception as error:
            failures.append(
                {
                    "backend": backend,
                    "type": type(error).__name__,
                    "message": str(error)[-1200:],
                }
            )
            log(f"{family} backend={backend} failed: {type(error).__name__}: {error}")
            gc.collect()
            if args.device == "gpu":
                break
    raise RuntimeError(f"all {family} backends failed: {failures}")


def concatenate_blocks(
    blocks: dict[int, dict[str, Any]], years: Iterable[int], final_fit: bool
) -> tuple[
    pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    frames: list[pd.DataFrame] = []
    primary_values: list[np.ndarray] = []
    secondary_values: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    years = tuple(years)
    newest = max(years)
    for year in years:
        block = blocks[year]
        frames.append(block["rows"])
        primary_values.append(block["primary"])
        secondary_values.append(block["secondary"])
        anchors.append(block["anchor"])
        targets.append(block["target"])
        if final_fit:
            weight = YEAR_WEIGHTS[year]
        else:
            weight = 1.0 if year == newest else YEAR_WEIGHTS[year] / YEAR_WEIGHTS[newest]
        weights.append(np.full(len(block["rows"]), weight, np.float64))
    return (
        pd.concat(frames, ignore_index=True),
        np.concatenate(primary_values),
        np.concatenate(secondary_values),
        np.concatenate(anchors),
        np.concatenate(targets),
        np.concatenate(weights),
    )


def fit_member_set(
    blocks: dict[int, dict[str, Any]],
    train_years: Iterable[int],
    valid_year: int | None,
    args: argparse.Namespace,
    final_fit: bool = False,
) -> dict[str, Any]:
    (
        train_rows,
        train_primary,
        train_secondary,
        train_anchor,
        train_target,
        train_weight,
    ) = concatenate_blocks(blocks, train_years, final_fit=final_fit)
    train_enriched = add_component_features(
        train_rows, train_primary, train_secondary, train_anchor
    )
    features, categorical = feature_contract(train_enriched)
    mappings = fit_category_schema(train_enriched, categorical)
    train_matrix_all = transform_features(
        train_enriched, features, categorical, mappings
    )
    raw_residual = train_target - train_anchor
    # The prior endpoint-contrast audit showed that global calibration transfer
    # is unstable.  Remove each source-year/game-type intercept so the trees can
    # only learn conditional disagreement shape, never a seasonal offset.
    residual_frame = pd.DataFrame(
        {
            "season": train_rows["season"].to_numpy(),
            "game_type": train_rows["game_type"].astype(str).to_numpy(),
            "residual": raw_residual,
        }
    )
    source_centers = (
        residual_frame.groupby(["season", "game_type"], sort=True)["residual"]
        .mean()
        .to_dict()
    )
    center_vector = residual_frame.groupby(
        ["season", "game_type"], sort=False
    )["residual"].transform("mean").to_numpy(np.float64)
    train_residual_all = raw_residual - center_vector
    if args.route == "R":
        domain = train_rows["game_type"].astype(str).to_numpy() == "R"
    else:
        domain = np.ones(len(train_rows), dtype=bool)
    train_matrix = train_matrix_all.loc[domain].reset_index(drop=True)
    train_residual = train_residual_all[domain]
    train_weight = train_weight[domain]
    residual_mean = float(np.average(train_residual, weights=train_weight))
    if not np.isfinite(train_residual).all() or abs(residual_mean) > 0.25:
        raise AssertionError(f"residual target is invalid: mean={residual_mean:+.6f}")
    log(f"residual target mean={residual_mean:+.6f} std={train_residual.std():.6f}")

    valid_rows = None
    valid_matrix = None
    if valid_year is not None:
        block = blocks[valid_year]
        valid_rows = block["rows"]
        valid_enriched = add_component_features(
            valid_rows, block["primary"], block["secondary"], block["anchor"]
        )
        valid_features, valid_categorical = feature_contract(valid_enriched)
        if valid_features != features or valid_categorical != categorical:
            raise AssertionError("train/validation feature contract drift")
        valid_matrix = transform_features(
            valid_enriched, features, categorical, mappings
        )

    models: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    for family in args.families:
        model, prediction, diagnostic = fit_family(
            family,
            train_matrix,
            train_residual,
            train_weight,
            valid_matrix,
            categorical,
            args,
        )
        models[family] = model
        diagnostics[family] = diagnostic
        if prediction is not None:
            if args.route == "R":
                is_r = valid_rows["game_type"].astype(str).to_numpy() == "R"
                prediction = np.where(is_r, prediction, 0.0)
            predictions[family] = prediction

    return {
        "models": models,
        "predictions": predictions,
        "feature_names": features,
        "categorical_features": categorical,
        "category_mappings": mappings,
        "diagnostics": diagnostics,
        "train_rows": int(len(train_rows)),
        "train_domain_rows": int(domain.sum()),
        "train_residual_mean": float(np.average(train_residual, weights=train_weight)),
        "train_residual_std": float(np.std(train_residual)),
        "removed_source_intercepts": {
            f"{int(year)}|{game}": float(value)
            for (year, game), value in source_centers.items()
        },
    }


def normalized_score(prediction: np.ndarray, target: np.ndarray) -> float:
    rate = float(np.mean(target))
    denominator = rate * (1.0 - rate)
    if denominator <= 0.0:
        raise ValueError("degenerate target rate")
    return float(
        100000.0
        * (1.0 - float(np.mean(np.square(prediction - target))) / denominator)
    )


def evaluation_metrics(
    baseline: np.ndarray,
    effect: np.ndarray,
    target: np.ndarray,
    rows: pd.DataFrame,
    scale: float,
) -> dict[str, float]:
    correction = float(scale) * np.asarray(effect, np.float64)
    candidate = np.clip(baseline + correction, 0.0, 1.0)
    equal_mean = np.clip(candidate + float(baseline.mean() - candidate.mean()), 0.0, 1.0)
    base_score = normalized_score(baseline, target)
    denominator = float(target.mean() * (1.0 - target.mean()))
    row_gain = (
        100000.0
        * (np.square(baseline - target) - np.square(candidate - target))
        / (len(target) * denominator)
    )
    is_f = rows["game_type"].astype(str).to_numpy() == "F"
    output = {
        "scale": float(scale),
        "baseline_score": base_score,
        "candidate_score": normalized_score(candidate, target),
        "raw_gain": normalized_score(candidate, target) - base_score,
        "equal_mean_shape_gain": normalized_score(equal_mean, target) - base_score,
        "F_contribution": float(row_gain[is_f].sum()),
        "R_contribution": float(row_gain[~is_f].sum()),
        "mean_shift": float(candidate.mean() - baseline.mean()),
        "effect_mean": float(effect.mean()),
        "effect_std": float(effect.std()),
        "effect_max_abs": float(np.max(np.abs(effect))),
    }
    if abs(output["raw_gain"] - output["F_contribution"] - output["R_contribution"]) > 1e-6:
        raise AssertionError("segment contribution mismatch")
    return output


def subset_gain(
    baseline: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("empty score subset")
    return normalized_score(candidate[mask], target[mask]) - normalized_score(
        baseline[mask], target[mask]
    )


def pitcher_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    pitcher: np.ndarray,
    draws: int,
) -> dict[str, float]:
    rate = float(target.mean())
    row_gain = (
        100000.0
        * (np.square(baseline - target) - np.square(candidate - target))
        / (len(target) * rate * (1.0 - rate))
    )
    grouped = (
        pd.DataFrame({"pitcher": pitcher, "gain": row_gain})
        .groupby("pitcher", sort=False, dropna=False)["gain"]
        .sum()
        .to_numpy(np.float64)
    )
    rng = np.random.default_rng(SEED)
    samples = np.empty(draws, np.float64)
    for start in range(0, draws, 100):
        count = min(100, draws - start)
        indices = rng.integers(0, len(grouped), size=(count, len(grouped)))
        samples[start : start + count] = grouped[indices].sum(axis=1)
    quantiles = np.quantile(samples, [0.025, 0.5, 0.975])
    return {
        "pitchers": int(len(grouped)),
        "draws": int(draws),
        "p025": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p975": float(quantiles[2]),
        "probability_positive": float(np.mean(samples > 0.0)),
    }


def simplex_weights(families: tuple[str, ...]) -> Iterable[dict[str, float]]:
    units = WEIGHT_GRID_UNITS
    for counts in itertools.product(range(units + 1), repeat=len(families)):
        if sum(counts) != units:
            continue
        yield {
            family: float(count / units) for family, count in zip(families, counts)
        }


def blend_effects(
    predictions: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    first = next(iter(predictions.values()))
    output = np.zeros_like(first, dtype=np.float64)
    for family, weight in weights.items():
        output += float(weight) * predictions[family]
    if not np.isfinite(output).all():
        raise AssertionError("non-finite ensemble effect")
    return output


def config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "families": list(args.families),
        "device_request": args.device,
        "threads": args.threads,
        "route": args.route,
        "smoke": args.smoke,
        "seed": SEED,
        "correction_clip": CORRECTION_CLIP,
        "scale_grid": list(DISCOVERY_SCALE_GRID),
        "weight_grid_units": WEIGHT_GRID_UNITS,
        "model_specs": {
            family: effective_specs(args.smoke)[family] for family in args.families
        },
    }


def assert_lock_matches(lock: dict[str, Any], args: argparse.Namespace) -> None:
    expected = config_payload(args)
    actual = lock["config"]
    for key in (
        "families",
        "device_request",
        "threads",
        "route",
        "smoke",
        "seed",
        "correction_clip",
        "scale_grid",
        "weight_grid_units",
        "model_specs",
    ):
        if actual[key] != expected[key]:
            raise AssertionError(f"lock/config mismatch for {key}")


def assert_backend_lock(reference: dict[str, Any], fitted: dict[str, Any]) -> None:
    for family, prior in reference.items():
        current = fitted["diagnostics"][family]
        if current["backend"] != prior["backend"]:
            raise AssertionError(
                f"backend drift for {family}: {prior['backend']} -> {current['backend']}"
            )


def run_discovery(args: argparse.Namespace) -> dict[str, Any]:
    blocks, data_diagnostics = load_year_blocks((2022, 2023), args.smoke)
    fitted = fit_member_set(blocks, (2022,), 2023, args)
    valid = blocks[2023]
    alternatives: list[dict[str, Any]] = []
    for weights in simplex_weights(args.families):
        effect = blend_effects(fitted["predictions"], weights)
        for scale in DISCOVERY_SCALE_GRID:
            metrics = evaluation_metrics(
                valid["anchor"], effect, valid["target"], valid["rows"], scale
            )
            alternatives.append({"weights": weights, "metrics": metrics})
    selected = max(
        alternatives,
        key=lambda row: (
            row["metrics"]["raw_gain"],
            row["metrics"]["equal_mean_shape_gain"],
            -row["metrics"]["scale"],
            -max(row["weights"].values()),
        ),
    )
    individual = {
        family: evaluation_metrics(
            valid["anchor"],
            prediction,
            valid["target"],
            valid["rows"],
            1.0,
        )
        for family, prediction in fitted["predictions"].items()
    }
    payload = {
        "experiment": 188,
        "phase": "discovery_2022_to_2023",
        "status": "SMOKE_LOCK" if args.smoke else "FULL_LOCK",
        "clean_implementation": {
            "learning_data": "official data/train.csv seasons 2022 only",
            "residual_reference": "our season-aligned current-affine OOF",
            "third_party_source_model_prediction_or_weight_read": False,
            "test_read": False,
            "raw_feature_policy": (
                "endpoint components/gap only; row context limited to game type, "
                "count state, and log experience"
            ),
            "exact_novelty_vs_prior": (
                "nonlinear disagreement-by-context surface; EXP169 tested only "
                "F/R scalar endpoint contrast and EXP166 tested scalar/linear ridge"
            ),
            "source_intercept_policy": (
                "remove source-year/game-type residual means before fitting"
            ),
        },
        "config": config_payload(args),
        "data": data_diagnostics,
        "feature_contract": {
            "count": len(fitted["feature_names"]),
            "names": fitted["feature_names"],
            "categorical": fitted["categorical_features"],
        },
        "fit": {
            "train_rows": fitted["train_rows"],
            "train_domain_rows": fitted["train_domain_rows"],
            "residual_mean": fitted["train_residual_mean"],
            "residual_std": fitted["train_residual_std"],
            "removed_source_intercepts": fitted["removed_source_intercepts"],
            "members": fitted["diagnostics"],
        },
        "individual_scale_1": individual,
        "alternatives": alternatives,
        "selected_weights": selected["weights"],
        "locked_scale": selected["metrics"]["scale"],
        "selected_2023": selected["metrics"],
        "source": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "train_size_bytes": TRAIN_PATH.stat().st_size,
        },
        "elapsed_seconds": time.time() - STARTED,
    }
    write_json(discovery_path(args), payload, args.overwrite)
    log(
        f"discovery locked weights={payload['selected_weights']} "
        f"scale={payload['locked_scale']:.3f} "
        f"gain={payload['selected_2023']['raw_gain']:+.4f}"
    )
    return payload


def run_confirmation(args: argparse.Namespace) -> dict[str, Any]:
    path = discovery_path(args)
    if not path.is_file():
        raise FileNotFoundError(f"run discover first: {path}")
    lock = json.loads(path.read_text(encoding="utf-8"))
    assert_lock_matches(lock, args)
    blocks, data_diagnostics = load_year_blocks((2022, 2023, 2024), args.smoke)
    fitted = fit_member_set(blocks, (2022, 2023), 2024, args)
    assert_backend_lock(lock["fit"]["members"], fitted)
    effect = blend_effects(fitted["predictions"], lock["selected_weights"])
    scale = float(lock["locked_scale"])
    valid = blocks[2024]
    metrics = evaluation_metrics(
        valid["anchor"], effect, valid["target"], valid["rows"], scale
    )
    candidate = np.clip(valid["anchor"] + scale * effect, 0.0, 1.0)
    month = pd.to_numeric(valid["rows"]["game_month"], errors="raise").to_numpy()
    early_gain = subset_gain(
        valid["anchor"], candidate, valid["target"], month <= 6
    )
    late_gain = subset_gain(
        valid["anchor"], candidate, valid["target"], month > 6
    )
    bootstrap = pitcher_bootstrap(
        valid["anchor"],
        candidate,
        valid["target"],
        valid["rows"]["pitcher_id"].to_numpy(),
        draws=200 if args.smoke else 3000,
    )
    gates = {
        "discovery_positive_nonzero": bool(
            lock["selected_2023"]["raw_gain"] > 0.0 and scale > 0.0
        ),
        "confirmation_raw_gain_ge_8": bool(metrics["raw_gain"] >= 8.0),
        "confirmation_shape_gain_ge_4": bool(metrics["equal_mean_shape_gain"] >= 4.0),
        "early_positive": bool(early_gain > 0.0),
        "late_positive": bool(late_gain > 0.0),
        "pitcher_cluster_p025_positive": bool(bootstrap["p025"] > 0.0),
    }
    statistical_pass = bool(all(gates.values()))
    deployable = bool(statistical_pass and not args.smoke)
    payload = {
        "experiment": 188,
        "phase": "confirmation_2022_2023_to_2024",
        "status": (
            "SMOKE_COMPLETE_NOT_DEPLOYABLE"
            if args.smoke
            else "PASS_FOR_BUILD"
            if deployable
            else "FAIL_NO_DEFAULT_BUILD"
        ),
        "deployable": deployable,
        "statistical_pass": statistical_pass,
        "post_2024_tuning": False,
        "test_read": False,
        "discovery_lock_sha256": sha256(path),
        "config": config_payload(args),
        "data": data_diagnostics,
        "locked_weights": lock["selected_weights"],
        "locked_scale": scale,
        "confirmation_2024": metrics,
        "early_gain": early_gain,
        "late_gain": late_gain,
        "pitcher_cluster_bootstrap": bootstrap,
        "gates": gates,
        "fit": {
            "train_rows": fitted["train_rows"],
            "train_domain_rows": fitted["train_domain_rows"],
            "residual_mean": fitted["train_residual_mean"],
            "residual_std": fitted["train_residual_std"],
            "removed_source_intercepts": fitted["removed_source_intercepts"],
            "members": fitted["diagnostics"],
        },
        "elapsed_seconds": time.time() - STARTED,
    }
    write_json(confirmation_path(args), payload, args.overwrite)
    log(
        f"confirmation status={payload['status']} raw={metrics['raw_gain']:+.4f} "
        f"shape={metrics['equal_mean_shape_gain']:+.4f} "
        f"early={early_gain:+.4f} late={late_gain:+.4f}"
    )
    return payload


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"unsafe ZIP member: {member.filename}")
    archive.extractall(destination)


def save_family_model(family: str, model: Any, model_dir: Path) -> str:
    if family == "lgb":
        filename = "tree_residual_lgb.txt"
        model.save_model(str(model_dir / filename))
    elif family == "cat":
        filename = "tree_residual_cat.cbm"
        model.save_model(str(model_dir / filename))
    elif family == "xgb":
        filename = "tree_residual_xgb.json"
        model.save_model(str(model_dir / filename))
    else:
        raise ValueError(family)
    return filename


def package_bundle(
    args: argparse.Namespace,
    fitted: dict[str, Any],
    lock: dict[str, Any],
    confirmation: dict[str, Any],
    output_zip: Path,
) -> dict[str, Any]:
    if not BASE_ZIP.is_file():
        raise FileNotFoundError(BASE_ZIP)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="exp188_package_") as temporary_name:
        source = Path(temporary_name) / "source"
        source.mkdir(parents=True)
        with zipfile.ZipFile(BASE_ZIP, "r") as archive:
            safe_extract(archive, source)
        base_script = source / "script.py"
        model_dir = source / "model"
        if not base_script.is_file() or not model_dir.is_dir():
            raise RuntimeError("current affine base package schema changed")
        base_script.replace(model_dir / "affine_anchor.py")
        base_script.write_text(RUNTIME_SOURCE, encoding="utf-8")

        members = []
        for family in args.families:
            filename = save_family_model(family, fitted["models"][family], model_dir)
            members.append(
                {
                    "family": family,
                    "file": filename,
                    "weight": float(lock["selected_weights"][family]),
                    "inference_threads": args.threads,
                    "trained_backend": fitted["diagnostics"][family]["backend"],
                }
            )
        state = {
            "schema_version": 1,
            "experiment": 188,
            "status": "PASSED_STRICT_CONFIRMATION",
            "feature_names": fitted["feature_names"],
            "categorical_features": fitted["categorical_features"],
            "category_mappings": fitted["category_mappings"],
            "members": members,
            "scale": float(lock["locked_scale"]),
            "correction_clip": CORRECTION_CLIP,
            "route": args.route,
            "anchor_transform": {
                "primary_weight": AFFINE_CHAMPION_WEIGHT,
                "secondary_weight": AFFINE_SECOND_WEIGHT,
                "shift": AFFINE_SHIFT,
            },
            "row_independence": "current row plus frozen training state only",
            "fit_data": "official train seasons 2022-2024 plus our own affine OOF",
            "novel_question": (
                "nonlinear component disagreement conditioned only on game type, "
                "count state, and experience"
            ),
            "third_party_artifact_used": False,
            "confirmation_sha256": sha256(confirmation_path(args)),
        }
        (model_dir / "tree_residual_state.json").write_text(
            json.dumps(native(state), ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

        requirements = source / "requirements.txt"
        requirement_text = requirements.read_text(encoding="utf-8")
        if "xgb" in args.families and "xgboost" not in requirement_text.lower():
            import xgboost

            requirement_text = requirement_text.rstrip() + f"\nxgboost=={xgboost.__version__}\n"
            requirements.write_text(requirement_text, encoding="utf-8")

        temporary_zip = output_zip.with_suffix(output_zip.suffix + ".tmp")
        with zipfile.ZipFile(temporary_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(source.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(source).as_posix())
        with zipfile.ZipFile(temporary_zip, "r") as check:
            corrupt = check.testzip()
            roots = sorted({name.split("/")[0] for name in check.namelist()})
        if corrupt is not None:
            raise RuntimeError(f"corrupt package member: {corrupt}")
        if roots != ["model", "requirements.txt", "script.py"]:
            raise RuntimeError(f"invalid package roots: {roots}")
        os.replace(temporary_zip, output_zip)
    return {
        "path": str(output_zip),
        "name": output_zip.name,
        "size_bytes": output_zip.stat().st_size,
        "sha256": sha256(output_zip),
        "roots": ["model", "requirements.txt", "script.py"],
        "hold": False,
    }


def run_packaged_rows(
    work: Path, rows: pd.DataFrame, timeout: int
) -> tuple[pd.DataFrame, str, float]:
    data_dir = work / "data"
    output_dir = work / "output"
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    rows.to_csv(data_dir / "test.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        {ID_COLUMN: rows[ID_COLUMN].to_numpy(), TARGET_COLUMN: np.zeros(len(rows))}
    ).to_csv(data_dir / "sample_submission.csv", index=False, encoding="utf-8")
    output_path = output_dir / "submission.csv"
    if output_path.exists():
        output_path.unlink()
    started = time.time()
    result = subprocess.run(
        [sys.executable, "script.py"],
        cwd=work,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=timeout,
        check=False,
    )
    elapsed = time.time() - started
    if result.returncode != 0:
        raise RuntimeError(
            "packaged inference failed\n"
            + (result.stdout or "")[-3000:]
            + "\n"
            + (result.stderr or "")[-5000:]
        )
    output = pd.read_csv(output_path, encoding="utf-8-sig")
    if list(output.columns) != [ID_COLUMN, TARGET_COLUMN] or len(output) != len(rows):
        raise RuntimeError("packaged output schema/row count mismatch")
    if not np.array_equal(output[ID_COLUMN].to_numpy(), rows[ID_COLUMN].to_numpy()):
        raise RuntimeError("packaged output row order mismatch")
    values = output[TARGET_COLUMN].to_numpy(np.float64)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise RuntimeError("packaged output probabilities invalid")
    stdout_tail = (result.stdout or "").strip().splitlines()[-1]
    return output, stdout_tail, elapsed


def package_qa(output_zip: Path, timeout: int) -> dict[str, Any]:
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig", low_memory=False)
    test.columns = [str(column).replace("\ufeff", "").strip() for column in test.columns]
    if len(test) < 2 or test[ID_COLUMN].duplicated().any():
        raise RuntimeError("official format test is unsuitable for row QA")
    with tempfile.TemporaryDirectory(prefix="exp188_qa_") as temporary_name:
        work = Path(temporary_name)
        with zipfile.ZipFile(output_zip, "r") as archive:
            safe_extract(archive, work)
        original, stdout_tail, original_seconds = run_packaged_rows(work, test, timeout)
        reversed_rows = test.iloc[::-1].reset_index(drop=True)
        reversed_output, _, reversed_seconds = run_packaged_rows(
            work, reversed_rows, timeout
        )
        original_map = original.set_index(ID_COLUMN)[TARGET_COLUMN]
        reversed_map = reversed_output.set_index(ID_COLUMN)[TARGET_COLUMN]
        reversed_difference = float(
            np.max(np.abs(original_map.sort_index() - reversed_map.sort_index()))
        )
        singleton_differences = []
        singleton_seconds = 0.0
        for index in range(min(3, len(test))):
            singleton = test.iloc[[index]].reset_index(drop=True)
            singleton_output, _, elapsed = run_packaged_rows(work, singleton, timeout)
            singleton_seconds += elapsed
            row_id = singleton.iloc[0][ID_COLUMN]
            singleton_differences.append(
                abs(
                    float(singleton_output.iloc[0][TARGET_COLUMN])
                    - float(original_map.loc[row_id])
                )
            )
        max_difference = max([reversed_difference, *singleton_differences])
        if max_difference > 1e-10:
            raise RuntimeError(f"row-independence QA failed: {max_difference:.3e}")
        values = original[TARGET_COLUMN].to_numpy(np.float64)
        return {
            "format_rows": int(len(test)),
            "original_seconds": original_seconds,
            "reversed_seconds": reversed_seconds,
            "singleton_seconds_total": singleton_seconds,
            "max_absolute_difference": max_difference,
            "tolerance": 1e-10,
            "row_order_ok": True,
            "singleton_ok": True,
            "finite_range_ok": True,
            "output_mean": float(values.mean()),
            "output_std": float(values.std()),
            "stdout_tail": stdout_tail,
        }


def default_output_zip(args: argparse.Namespace) -> Path:
    if args.output_zip is not None:
        return args.output_zip.resolve()
    return ROOT / 'artifacts/candidates/candidate_exp188_tree_residual.zip'


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    lock_file = discovery_path(args)
    confirm_file = confirmation_path(args)
    if not lock_file.is_file() or not confirm_file.is_file():
        raise FileNotFoundError("run matching discover and confirm phases first")
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    confirmation = json.loads(confirm_file.read_text(encoding="utf-8"))
    assert_lock_matches(lock, args)
    if confirmation["config"] != config_payload(args):
        raise AssertionError("confirmation/config mismatch")
    approved = bool(confirmation.get("deployable", False))
    if args.smoke or not approved:
        raise RuntimeError(
            "deployment is PASS-only: smoke or failed confirmation cannot build a ZIP"
        )

    blocks, data_diagnostics = load_year_blocks((2022, 2023, 2024), args.smoke)
    fitted = fit_member_set(
        blocks, (2022, 2023, 2024), None, args, final_fit=True
    )
    assert_backend_lock(confirmation["fit"]["members"], fitted)
    output_zip = default_output_zip(args)
    artifact = package_bundle(
        args, fitted, lock, confirmation, output_zip
    )
    qa = package_qa(output_zip, args.qa_timeout)
    payload = {
        "experiment": 188,
        "phase": "deployment_refit_and_package",
        "status": "BUILT_AFTER_STRICT_PASS",
        "submitted": False,
        "config": config_payload(args),
        "clean_implementation": {
            "learning_data": "official data/train.csv seasons 2022-2024",
            "residual_reference": "our season-aligned current-affine OOF",
            "base_package": BASE_ZIP.name,
            "base_package_sha256": sha256(BASE_ZIP),
            "third_party_source_model_prediction_or_weight_read": False,
        },
        "data": data_diagnostics,
        "locked_weights": lock["selected_weights"],
        "locked_scale": lock["locked_scale"],
        "confirmation_status": confirmation["status"],
        "confirmation_sha256": sha256(confirm_file),
        "fit": {
            "train_rows": fitted["train_rows"],
            "train_domain_rows": fitted["train_domain_rows"],
            "residual_mean": fitted["train_residual_mean"],
            "residual_std": fitted["train_residual_std"],
            "removed_source_intercepts": fitted["removed_source_intercepts"],
            "members": fitted["diagnostics"],
            "feature_count": len(fitted["feature_names"]),
        },
        "artifact": artifact,
        "qa": qa,
        "elapsed_seconds": time.time() - STARTED,
    }
    write_json(build_path(args), payload, args.overwrite)
    log(
        f"build status={payload['status']} zip={artifact['name']} "
        f"sha256={artifact['sha256']} qa_max={qa['max_absolute_difference']:.3e}"
    )
    return payload


def main() -> None:
    args = parse_args()
    log(
        f"EXP188 phase={args.phase} families={args.families} device={args.device} "
        f"route={args.route} smoke={args.smoke}"
    )
    if args.phase == "discover":
        run_discovery(args)
    elif args.phase == "confirm":
        run_confirmation(args)
    elif args.phase == "build":
        run_build(args)
    else:
        run_discovery(args)
        confirmation = run_confirmation(args)
        if confirmation.get("deployable", False):
            run_build(args)
        else:
            log("PASS-only build skipped because confirmation is not deployable")
    log(f"done elapsed_seconds={time.time() - STARTED:.2f}")


if __name__ == "__main__":
    main()
