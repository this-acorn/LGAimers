"""Row-local current-anchor + official-train Tensor-EB inference runtime."""

from __future__ import annotations

import gc
import json
import runpy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
CHAMPION_WEIGHT = 0.6434428305247574
EXP021_WEIGHT = 0.35655716947524263
MODEL_DIR = Path("./model")
DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
TEST_PATH = DATA_DIR / "test.csv"


def _run_component(filename: str, namespace: str) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
    module = runpy.run_path(str(MODEL_DIR / filename), run_name=namespace)
    main = module.get("main")
    if not callable(main):
        raise RuntimeError(f"component has no callable main(): {filename}")
    main()
    if not OUTPUT_PATH.is_file():
        raise RuntimeError(f"component did not create {OUTPUT_PATH}: {filename}")
    prediction = pd.read_csv(OUTPUT_PATH, encoding="utf-8-sig")
    del main, module
    gc.collect()
    return prediction


def _validate_component(frame: pd.DataFrame, name: str) -> None:
    if list(frame.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"{name}: invalid columns {frame.columns.tolist()}")
    if frame[ID_COL].duplicated().any():
        raise ValueError(f"{name}: duplicate row_id")
    values = frame[TARGET_COL].to_numpy(np.float64)
    if not np.isfinite(values).all() or not ((values >= 0.0) & (values <= 1.0)).all():
        raise ValueError(f"{name}: invalid probability")


def _prepare_keys(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        ID_COL, "game_type", "pitcher_id", "pitcher_hand", "batter_hand",
        "balls_before", "strikes_before", "base_state",
    ]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"missing tensor columns: {missing}")
    output = frame.loc[:, required].copy()
    output["pitcher_id"] = pd.to_numeric(
        output["pitcher_id"], errors="coerce"
    ).fillna(-1).astype("int64")
    for column in ("pitcher_hand", "batter_hand", "base_state", "game_type"):
        output[column] = output[column].astype("string").fillna("__MISSING__").astype(str)
    balls = pd.to_numeric(output["balls_before"], errors="coerce").fillna(-1).astype("int16")
    strikes = pd.to_numeric(output["strikes_before"], errors="coerce").fillna(-1).astype("int16")
    output["_count_state"] = (balls * 3 + strikes).astype("int16")
    return output


def _lookup(rows: pd.DataFrame, table: pd.DataFrame, keys: list[str]) -> tuple[np.ndarray, float]:
    left = rows.loc[:, keys].copy()
    left["_lookup_order"] = np.arange(len(left), dtype=np.int64)
    joined = left.merge(
        table.loc[:, keys + ["correction"]], on=keys, how="left", sort=False,
        validate="many_to_one",
    ).sort_values("_lookup_order", kind="stable")
    if not np.array_equal(joined["_lookup_order"].to_numpy(), np.arange(len(rows))):
        raise AssertionError("tensor lookup changed row order")
    matched = joined["correction"].notna().to_numpy()
    return joined["correction"].fillna(0.0).to_numpy(np.float64), float(matched.mean())


def _tensor_effect(test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    payload = joblib.load(MODEL_DIR / "tensor_tables.joblib")
    metadata = json.loads((MODEL_DIR / "tensor_metadata.json").read_text(encoding="utf-8"))
    rows = _prepare_keys(test)
    effect = np.zeros(len(rows), np.float64)
    coverage: dict[str, float] = {}
    for specification in payload["specifications"]:
        name = specification["name"]
        correction, matched = _lookup(rows, payload["tables"][name], specification["keys"])
        effect += float(specification["weight"]) * correction
        coverage[name] = matched
    is_r = rows["game_type"].to_numpy() == "R"
    effect[~is_r] = 0.0
    if not np.isfinite(effect).all() or np.any(effect[~is_r] != 0.0):
        raise AssertionError("invalid tensor effect or F fallback")
    return pd.DataFrame({ID_COL: rows[ID_COL].to_numpy(), "effect": effect,
                         "is_r": is_r}), {**coverage, "scale": float(metadata["scale"])}


def main() -> None:
    champion = _run_component("champion_inference.py", "tensor_champion_component")
    _validate_component(champion, "champion")
    champion_values = champion[TARGET_COL].to_numpy(np.float64, copy=True)
    champion_ids = champion[ID_COL].to_numpy(copy=True)
    del champion
    gc.collect()

    exp021 = _run_component("exp021_inference.py", "tensor_exp021_component")
    _validate_component(exp021, "exp021")
    if not np.array_equal(champion_ids, exp021[ID_COL].to_numpy()):
        raise ValueError("component row_id order mismatch")
    anchor = CHAMPION_WEIGHT * champion_values + EXP021_WEIGHT * exp021[
        TARGET_COL
    ].to_numpy(np.float64)

    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
    sample.columns = [column.replace("\ufeff", "").replace("ï»¿", "").strip()
                      for column in sample.columns]
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig", low_memory=False)
    test.columns = [column.replace("\ufeff", "").replace("ï»¿", "").strip()
                    for column in test.columns]
    if len(sample) != len(anchor) or not np.array_equal(sample[ID_COL], champion_ids):
        raise ValueError("anchor/sample alignment failure")
    if test[ID_COL].duplicated().any() or sample[ID_COL].duplicated().any():
        raise ValueError("duplicate row_id")
    effect_frame, diagnostic = _tensor_effect(test)
    aligned = sample[[ID_COL]].merge(
        effect_frame, on=ID_COL, how="left", sort=False, validate="one_to_one"
    )
    if aligned["effect"].isna().any():
        raise ValueError("test/sample row_id set mismatch")
    scale = float(diagnostic.pop("scale"))
    effect = aligned["effect"].to_numpy(np.float64)
    is_r = aligned["is_r"].to_numpy(bool)
    prediction = np.clip(anchor + scale * effect, 0.0, 1.0)
    if not np.array_equal(prediction[~is_r], anchor[~is_r]):
        raise AssertionError("F predictions changed")
    if not np.isfinite(prediction).all():
        raise ValueError("non-finite final prediction")
    pd.DataFrame({ID_COL: sample[ID_COL], TARGET_COL: prediction}).to_csv(
        OUTPUT_PATH, index=False, encoding="utf-8"
    )
    print(
        f"Saved: {OUTPUT_PATH} rows={len(prediction)} scale={scale:.2f} "
        f"R={int(is_r.sum())} mean_effect_R={effect[is_r].mean():+.8f} "
        f"coverage={diagnostic}", flush=True,
    )


if __name__ == "__main__":
    main()
