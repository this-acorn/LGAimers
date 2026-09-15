"""Immutable, official-train-only repair of the exp/125 seed-42 screen.

This deliberately does not reuse the f_same_hand-corrupted exp/104 arrays.
It creates a new row-keyed baseline lane, trains exactly one corrected 2022
CAT5 seed-42 fold, reproduces the saved corrected 2023 seed-42 fold from its
CBM, fingerprints the existing valid 2024 seed-42 OOF, and runs the fixed
raw/cutoff/combined two-transition ridge gate.  No test row is read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train.csv"
LANE = ROOT / "lab" / "125_exact_cs_seed42_repair_v2"
MODEL_2023 = ROOT / "lab" / "122_y2023_seed42_base_exacthand_v3.cbm"
SAVED_2023 = ROOT / "lab" / "122_y2023_seed42_base_final.npy"
P2024 = ROOT / "lab" / "89_cat5_probs_seed42.npy"
E2024 = ROOT / "lab" / "103_v18_effect_2024.npy"
AFFINE_CENTER, AFFINE_SCALE, AFFINE_SHIFT = 0.49, 1.06, 0.0066
GAMMA = 0.30
THREADS = 14
SEED = 42
MAX_SECONDS = 60 * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--max-minutes", type=float, default=60.0)
    return parser.parse_args()


ARGS = parse_args()
DEADLINE = time.monotonic() + ARGS.max_minutes * 60.0
LINES: list[str] = []


def log(message: str = "") -> None:
    line = str(message)
    print(line, flush=True)
    LINES.append(line)
    if LANE.is_dir():
        (LANE / "live.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


def left_seconds() -> float:
    return DEADLINE - time.monotonic()


def require_time(stage: str) -> None:
    if left_seconds() <= 0:
        raise TimeoutError(f"60-minute wall-clock timebox expired before {stage}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(value)).hexdigest()


def row_id_hash(ids: np.ndarray) -> str:
    """Content hash, not the unstable object-array pointer representation."""

    digest = hashlib.sha256()
    for value in ids.astype(str, copy=False):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    paths = [
        DATA,
        ROOT / "exp" / "103_v18_residual.py",
        ROOT / "exp" / "104_cat5_yearfold.py",
        ROOT / "exp" / "122_temporal_continuation.py",
        ROOT / "exp" / "125_exact_cs_raw_gate.py",
        Path(__file__),
        MODEL_2023,
        SAVED_2023,
        P2024,
        E2024,
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def load_module(name: str, path: Path, argv: list[str]) -> ModuleType:
    saved = sys.argv[:]
    try:
        sys.argv = argv
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = saved


def quiet_exp122() -> ModuleType:
    module = load_module(
        "exp122_for_129",
        ROOT / "exp" / "122_temporal_continuation.py",
        ["exp/122_temporal_continuation.py", "--threads", str(ARGS.threads), "--seed", "42"],
    )

    # prepare_fold is reused only for its corrected feature construction.  Its
    # ordinary logger writes lab/122_temporal_continuation_live.txt, which this
    # immutable repair must not touch.
    module.log = lambda message="": print(str(message), flush=True)
    return module


def champion_space(probability: np.ndarray, effect: np.ndarray) -> np.ndarray:
    corrected = np.clip(
        np.asarray(probability, np.float64) + GAMMA * np.asarray(effect, np.float64),
        0.0,
        1.0,
    )
    return np.clip(
        AFFINE_CENTER + AFFINE_SCALE * (corrected - AFFINE_CENTER) - AFFINE_SHIFT,
        0.0,
        1.0,
    )


class DeadlineCallback:
    def after_iteration(self, info: Any) -> bool:
        return left_seconds() > 0


def canonical_rows() -> dict[int, np.ndarray]:
    require_time("canonical row-id read")
    frame = pd.read_csv(DATA, encoding="utf-8-sig", usecols=["row_id", "season"], low_memory=False)
    if frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
        raise AssertionError("train row_id must be present and globally unique")
    if not frame["season"].is_monotonic_increasing:
        raise AssertionError("train.csv must be season-sorted")
    result: dict[int, np.ndarray] = {}
    for year in (2022, 2023, 2024):
        rows = frame.loc[frame["season"].eq(year), "row_id"].astype(str).to_numpy()
        if len(rows) == 0:
            raise AssertionError(f"no canonical rows for {year}")
        result[year] = rows
        log(f"canonical {year}: rows={len(rows):,} row_id_sha256={row_id_hash(rows)}")
    return result


def save_baseline(
    year: int,
    row_ids: np.ndarray,
    final: np.ndarray,
    raw_probability: np.ndarray,
    effect: np.ndarray,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if len(row_ids) != len(final) or len(final) != len(raw_probability) or len(final) != len(effect):
        raise AssertionError(f"baseline length mismatch for {year}")
    if not np.isfinite(final).all() or np.any(final < 0.0) or np.any(final > 1.0):
        raise AssertionError(f"invalid final endpoint values for {year}")
    path = LANE / f"baseline_y{year}_seed42.npz"
    if path.exists():
        raise FileExistsError(f"immutable baseline already exists: {path}")
    np.savez_compressed(
        path,
        row_id=row_ids.astype(str),
        final=np.asarray(final, np.float64),
        raw_probability=np.asarray(raw_probability, np.float64),
        v18_effect=np.asarray(effect, np.float64),
    )
    record = {
        "year": year,
        "rows": int(len(row_ids)),
        "row_id_sha256": row_id_hash(row_ids),
        "final_sha256": sha256_array(np.asarray(final, np.float64)),
        "raw_probability_sha256": sha256_array(np.asarray(raw_probability, np.float64)),
        "v18_effect_sha256": sha256_array(np.asarray(effect, np.float64)),
        "endpoint": "clip(.49+1.06*(clip(p+.30*v18,0,1)-.49)-.0066,0,1)",
        "provenance": provenance,
    }
    (LANE / f"baseline_y{year}_seed42.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return record


def train_2022(exp122: ModuleType, row_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    require_time("corrected 2022 feature build")
    log("building corrected exact-hand 2022 fold")
    data = exp122.prepare_fold(2022)
    if len(data["target"]) != len(row_ids):
        raise AssertionError("2022 canonical row count differs from corrected validation fold")
    feature_qa = data["feature_diagnostics"]
    if not all(0.30 <= float(value) <= 0.75 for value in feature_qa.values()):
        raise AssertionError(f"2022 same-hand QA failed: {feature_qa}")
    require_time("corrected 2022 CAT5 fit")
    started = time.monotonic()
    model = CatBoostClassifier(
        **exp122.BASE_PARAMS, thread_count=ARGS.threads, random_seed=SEED
    ).fit(data["base_pool"], callbacks=[DeadlineCallback()])
    elapsed = time.monotonic() - started
    if left_seconds() <= 0 or model.tree_count_ != int(exp122.BASE_PARAMS["iterations"]):
        raise TimeoutError(
            f"2022 CAT5 did not complete the required 500 trees within timebox; trees={model.tree_count_}"
        )
    if list(model.feature_names_) != list(exp122.FOLD.FEATURES):
        raise AssertionError("2022 model feature schema/order mismatch")
    model_path = LANE / "y2022_seed42_base_exacthand.cbm"
    if model_path.exists():
        raise FileExistsError(model_path)
    model.save_model(model_path)
    full = model.predict_proba(data["valid_pool"])
    raw = full[:, 0]
    effect = np.asarray(data["effect"], np.float64)
    final = champion_space(raw, effect)
    provenance = {
        "operation": "trained corrected exact-hand base",
        "model_path": str(model_path.relative_to(ROOT)),
        "model_sha256": sha256_file(model_path),
        "fit_seconds": elapsed,
        "feature_diagnostics": feature_qa,
        "effect_diagnostics": data["effect_diagnostics"],
        "counts": data["counts"],
        "catboost_params": dict(exp122.BASE_PARAMS, thread_count=ARGS.threads, random_seed=SEED),
    }
    log(f"2022 corrected seed42 complete: fit_seconds={elapsed:.1f}")
    return final, raw, effect, provenance


def reproduce_2023(exp122: ModuleType, row_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    require_time("corrected 2023 feature build")
    log("reproducing saved corrected exact-hand 2023 seed42 CBM")
    data = exp122.prepare_fold(2023)
    if len(data["target"]) != len(row_ids):
        raise AssertionError("2023 canonical row count differs from corrected validation fold")
    feature_qa = data["feature_diagnostics"]
    if not all(0.30 <= float(value) <= 0.75 for value in feature_qa.values()):
        raise AssertionError(f"2023 same-hand QA failed: {feature_qa}")
    model = CatBoostClassifier()
    model.load_model(MODEL_2023)
    if list(model.feature_names_) != list(exp122.FOLD.FEATURES):
        raise AssertionError("saved 2023 exact-hand CBM feature schema/order mismatch")
    raw = model.predict_proba(data["valid_pool"])[:, 0]
    effect = np.asarray(data["effect"], np.float64)
    final = champion_space(raw, effect)
    saved = np.load(SAVED_2023).astype(np.float64)
    if saved.shape != final.shape:
        raise AssertionError((saved.shape, final.shape))
    maximum_difference = float(np.max(np.abs(saved - final)))
    if maximum_difference > 1e-6:
        raise AssertionError(f"2023 saved endpoint parity failed: {maximum_difference}")
    provenance = {
        "operation": "reproduced corrected exact-hand CBM",
        "model_path": str(MODEL_2023.relative_to(ROOT)),
        "model_sha256": sha256_file(MODEL_2023),
        "saved_final_path": str(SAVED_2023.relative_to(ROOT)),
        "saved_final_max_abs_diff": maximum_difference,
        "feature_diagnostics": feature_qa,
        "effect_diagnostics": data["effect_diagnostics"],
        "counts": data["counts"],
    }
    log(f"2023 corrected endpoint parity: max_abs_diff={maximum_difference:.3e}")
    return final, raw, effect, provenance


def existing_2024(row_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    require_time("2024 endpoint materialisation")
    probability = np.load(P2024).astype(np.float64)
    effect = np.load(E2024).astype(np.float64)
    if probability.shape != (len(row_ids), 5) or effect.shape != (len(row_ids),):
        raise AssertionError(f"2024 source array shape mismatch: {probability.shape}/{effect.shape}")
    raw = probability[:, 0]
    final = champion_space(raw, effect)
    return final, raw, effect, {
        "operation": "row-keyed existing exact current-CAT5 seed42 OOF",
        "raw_probability_path": str(P2024.relative_to(ROOT)),
        "raw_probability_file_sha256": sha256_file(P2024),
        "effect_path": str(E2024.relative_to(ROOT)),
        "effect_file_sha256": sha256_file(E2024),
    }


def load_joined_baseline(year: int, canonical_ids: np.ndarray, endpoint: str) -> np.ndarray:
    path = LANE / f"baseline_y{year}_seed42.npz"
    with np.load(path) as packed:
        ids = packed["row_id"].astype(str)
        final = packed["final"].astype(np.float64)
        raw = packed["raw_probability"].astype(np.float64)
        effect = packed["v18_effect"].astype(np.float64)
    if len(ids) != len(final) or len(final) != len(raw) or len(raw) != len(effect):
        raise AssertionError(f"invalid immutable baseline lengths for {year}")
    if len(np.unique(ids)) != len(ids):
        raise AssertionError(f"invalid immutable baseline ids for {year}")
    if endpoint == "raw_pre_affine":
        values = np.clip(raw + GAMMA * effect, 0.0, 1.0)
    elif endpoint == "affine_current_blend":
        values = final
    else:
        raise ValueError(endpoint)
    indexed = pd.Series(values, index=pd.Index(ids, dtype="object"))
    joined = indexed.reindex(canonical_ids.astype(str))
    if joined.isna().any() or len(joined) != len(canonical_ids):
        raise AssertionError(f"row_id join failed for {year}: missing={int(joined.isna().sum())}")
    if row_id_hash(ids) != row_id_hash(canonical_ids):
        raise AssertionError(f"row-id fingerprint mismatch for {year}")
    return joined.to_numpy(np.float64)


def run_single_screen(
    e125: ModuleType, ids: dict[int, np.ndarray], endpoint: str
) -> dict[str, Any]:
    require_time("strict two-transition screen")
    log(f"running zero-fit raw/cutoff/combined screen: {endpoint}")
    counts = e125.season_row_counts()
    frame = e125.load_through_year(2024, counts)
    built = {year: e125.build_state(frame, year) for year in (2022, 2023, 2024)}
    states = {year: built[year][0] for year in (2022, 2023, 2024)}
    base = {
        year: load_joined_baseline(year, ids[year], endpoint)
        for year in (2022, 2023, 2024)
    }
    for year in (2022, 2023, 2024):
        if len(states[year]) != len(base[year]):
            raise AssertionError(f"state/baseline length mismatch for {year}")
    results: dict[str, Any] = {}
    selected: list[str] = []
    for name, columns in e125.BLOCKS.items():
        model22 = e125.fit_ridge(states[2022], base[2022], columns, 2022)
        effect23, diag23 = e125.apply_ridge(model22, states[2023])
        discovery = e125.evaluate(base[2023], effect23, states[2023])
        discovery["effect_diagnostics"] = diag23
        discovery_pass = bool(
            discovery["raw_gain"] >= 8.0
            and discovery["shape_gain"] >= 5.0
            and discovery["F_contribution"] >= 0.0
            and discovery["R_contribution"] >= 0.0
        )
        discovery["gate_pass"] = discovery_pass
        model23 = e125.fit_ridge(states[2023], base[2023], columns, 2023)
        effect24, diag24 = e125.apply_ridge(model23, states[2024])
        rolling = e125.evaluate(base[2024], effect24, states[2024])
        rolling["effect_diagnostics"] = diag24
        frozen_effect, frozen_diag = e125.apply_ridge(model22, states[2024])
        frozen = e125.evaluate(base[2024], frozen_effect, states[2024])
        frozen["effect_diagnostics"] = frozen_diag
        results[name] = {
            "discovery_2022_to_2023": discovery,
            "rolling_2023_to_2024": rolling,
            "frozen_2022_to_2024": frozen,
            "discovery_gate_pass": discovery_pass,
        }
        if discovery_pass:
            selected.append(name)
        log(
            f"[{endpoint}/{name}] 22->23 raw/shape={discovery['raw_gain']:+.3f}/{discovery['shape_gain']:+.3f} "
            f"F/R={discovery['F_contribution']:+.3f}/{discovery['R_contribution']:+.3f} "
            f"gate={'PASS' if discovery_pass else 'FAIL'}"
        )
        log(
            f"         23->24 rolling raw/shape={rolling['raw_gain']:+.3f}/{rolling['shape_gain']:+.3f} "
            f"F/R={rolling['F_contribution']:+.3f}/{rolling['R_contribution']:+.3f}; "
            f"frozen raw/shape={frozen['raw_gain']:+.3f}/{frozen['shape_gain']:+.3f}"
        )
    final_pass = False
    selected_name: str | None = None
    if selected:
        selected_name = max(
            selected,
            key=lambda name: (
                min(
                    results[name]["discovery_2022_to_2023"]["raw_gain"],
                    results[name]["discovery_2022_to_2023"]["shape_gain"],
                ),
                results[name]["discovery_2022_to_2023"]["raw_gain"],
            ),
        )
        d = results[selected_name]["discovery_2022_to_2023"]
        r = results[selected_name]["rolling_2023_to_2024"]
        f = results[selected_name]["frozen_2022_to_2024"]
        final_pass = bool(
            f["raw_gain"] > 0.0
            and f["shape_gain"] > 0.0
            and r["raw_gain"] > 0.0
            and r["shape_gain"] > 0.0
            and r["F_contribution"] >= 0.0
            and r["R_contribution"] >= 0.0
            and np.mean([d["raw_gain"], r["raw_gain"]]) >= 12.0
            and np.mean([d["shape_gain"], r["shape_gain"]]) >= 10.0
        )
    return {
        "results": results,
        "selected_block": selected_name,
        "final_gate_pass": final_pass,
        "endpoint": endpoint,
        "state_diagnostics": {str(year): built[year][1] for year in (2022, 2023, 2024)},
    }


def run_screen(e125: ModuleType, ids: dict[int, np.ndarray]) -> dict[str, Any]:
    raw = run_single_screen(e125, ids, "raw_pre_affine")
    require_time("affine current-blend screen")
    affine = run_single_screen(e125, ids, "affine_current_blend")
    return {"raw_pre_affine": raw, "affine_current_blend": affine}


def write_report(report: dict[str, Any]) -> None:
    (LANE / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (LANE / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


def main() -> None:
    if LANE.exists():
        raise FileExistsError(f"immutable lane already exists: {LANE}")
    LANE.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report: dict[str, Any] = {
        "experiment": 129,
        "purpose": "exp125 repaired seed42 exact current-season raw-state screen",
        "analysis_only": True,
        "test_opened": False,
        "submission_built": False,
        "timebox_minutes": ARGS.max_minutes,
        "source_hashes": source_hashes(),
        "endpoint": "clip(.49+1.06*(clip(p+.30*v18,0,1)-.49)-.0066,0,1)",
        "baseline_seed": SEED,
        "threads": ARGS.threads,
    }
    try:
        log("=== immutable repaired exp125 seed42 raw-state screen ===")
        log("official train.csv only; no test, ZIP, submission, or overwrite of lab/104/lab/122")
        ids = canonical_rows()
        exp122 = quiet_exp122()
        final, raw, effect, provenance = train_2022(exp122, ids[2022])
        report.setdefault("baselines", {})["2022"] = save_baseline(2022, ids[2022], final, raw, effect, provenance)
        del final, raw, effect
        require_time("2023 reproduction")
        final, raw, effect, provenance = reproduce_2023(exp122, ids[2023])
        report["baselines"]["2023"] = save_baseline(2023, ids[2023], final, raw, effect, provenance)
        del final, raw, effect
        final, raw, effect, provenance = existing_2024(ids[2024])
        report["baselines"]["2024"] = save_baseline(2024, ids[2024], final, raw, effect, provenance)
        del final, raw, effect
        e125 = load_module("exp125_for_129", ROOT / "exp" / "125_exact_cs_raw_gate.py", ["exp/125_exact_cs_raw_gate.py"])
        report["screen"] = run_screen(e125, ids)
        affine_pass = bool(report["screen"]["affine_current_blend"]["final_gate_pass"])
        report["status"] = "PASS" if affine_pass else "FAIL"
        log(f"FINAL {'PASS' if affine_pass else 'FAIL'} — no model arm or submission is authorized here")
    except TimeoutError as exc:
        report["status"] = "TIMEBOX_STOP"
        report["error"] = str(exc)
        log(f"TIMEBOX STOP: {exc}")
    except Exception as exc:
        report["status"] = "QA_OR_RUNTIME_STOP"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        log(f"STOP: {type(exc).__name__}: {exc}")
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["seconds_remaining"] = max(left_seconds(), 0.0)
        write_report(report)


if __name__ == "__main__":
    main()
