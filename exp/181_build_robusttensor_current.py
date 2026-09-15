# -*- coding: utf-8 -*-
"""Build a clean-room RobustTensor endpoint on the current clean anchor.

This builder implements only the prose method published in the public GitHub
issue and the constants frozen in experiment 158.  It never reads the public
team's ZIP, weights, source code, predictions, or OOF artifacts.  All fitted
tables come from official train rows through 2024 and our own current-anchor
OOF predictions.

The resulting package is deliberately named HOLD: experiment 158 failed its
predeclared 2024 stability guardrail on our anchor.  Building and smoke-testing
the endpoint establishes technical deployability; it is not a submission
recommendation and this script never submits anything.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train.csv"
TEST = ROOT / "data" / "test.csv"
BASE_ZIP = ROOT / 'artifacts/candidates/candidate_exp021_ours_v001.zip'
VALIDATION_JSON = ROOT / "lab" / "158_robust_conditional_tensor_eb.json"
OUT_JSON = ROOT / "lab" / "181_robusttensor_current.json"
OUT_TXT = ROOT / "lab" / "181_robusttensor_current.txt"
OUT_ZIP = ROOT / 'artifacts/candidates/candidate_exp181_robusttensor_hold.zip'
STAGE = ROOT / 'archive/scratch/_exp181_build'
SOURCE = STAGE / "source"
REHEARSE = ROOT / 'archive/scratch/_rehearse_181'
PYTHON = ROOT / "venv311" / "Scripts" / "python.exe"

ISSUE_METHOD = "https://github.com/calico-cat17/LG-Aimers-9th/issues/5"
ISSUE_CONTEXT = "https://github.com/calico-cat17/LG-Aimers-9th/issues/8"
TARGET = "control_success"
ID = "row_id"


WRAPPER = r'''# -*- coding: utf-8 -*-
"""Row-local current-anchor + frozen official-data RobustTensor correction."""

from __future__ import annotations

import gc
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd


ID = "row_id"
TARGET = "control_success"
MODEL_DIR = Path("./model")
DATA_PATH = Path("./data/test.csv")
SAMPLE_PATH = Path("./data/sample_submission.csv")
OUTPUT_DIR = Path("./output")
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


def prepare_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["pitcher_id"] = pd.to_numeric(
        out["pitcher_id"], errors="coerce").fillna(-1).astype("int64")
    for column in ("pitcher_hand", "batter_hand", "base_state", "game_type"):
        out[column] = out[column].astype("string").fillna("__MISSING__").astype(str)
    balls = pd.to_numeric(out["balls_before"], errors="coerce").fillna(-1).astype("int16")
    strikes = pd.to_numeric(
        out["strikes_before"], errors="coerce").fillna(-1).astype("int16")
    out["_count_state"] = (balls * 3 + strikes).astype("int16")
    return out


def lookup(frame: pd.DataFrame, spec: dict) -> np.ndarray:
    keys = list(spec["keys"])
    records = pd.DataFrame.from_records(spec["records"])
    if records.empty:
        return np.zeros(len(frame), dtype=np.float64)
    left = frame.loc[:, keys].copy()
    left["_lookup_row"] = np.arange(len(left), dtype=np.int64)
    joined = left.merge(records, how="left", on=keys, sort=False, validate="many_to_one")
    joined = joined.sort_values("_lookup_row", kind="stable")
    if not np.array_equal(joined["_lookup_row"].to_numpy(), np.arange(len(frame))):
        raise RuntimeError("Tensor lookup changed row order")
    return joined["correction"].fillna(0.0).to_numpy(np.float64)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    module = runpy.run_path(
        str(MODEL_DIR / "current_blend_inference.py"), run_name="current_anchor")
    base_main = module.get("main")
    if not callable(base_main):
        raise RuntimeError("current anchor has no callable main")
    base_main()
    base = pd.read_csv(OUTPUT_PATH, encoding="utf-8-sig")
    if list(base.columns) != [ID, TARGET] or base[ID].duplicated().any():
        raise RuntimeError("invalid current-anchor output")
    base_values = base[TARGET].to_numpy(np.float64, copy=True)
    base_ids = base[ID].to_numpy(copy=True)
    del base_main, module, base
    gc.collect()

    test = prepare_keys(pd.read_csv(DATA_PATH, encoding="utf-8-sig"))
    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
    sample.columns = [str(c).replace("\ufeff", "").strip() for c in sample.columns]
    if len(test) != len(base_ids) or not np.array_equal(test[ID].to_numpy(), base_ids):
        raise RuntimeError("test/current-anchor row mismatch")
    if len(sample) != len(base_ids) or not np.array_equal(sample[ID].to_numpy(), base_ids):
        raise RuntimeError("sample/current-anchor row mismatch")

    with open(MODEL_DIR / "robusttensor_state.json", "r", encoding="utf-8") as handle:
        state = json.load(handle)
    effect = np.zeros(len(test), dtype=np.float64)
    for spec in state["tables"]:
        effect += float(spec["weight"]) * lookup(test, spec)
    is_r = test["game_type"].to_numpy() == "R"
    pitcher = test["pitcher_id"].to_numpy(np.int64)
    gates = np.fromiter(
        (state["pitcher_gate"].get(str(int(p)), state["unseen_weight"])
         for p in pitcher), dtype=np.float64, count=len(test))
    correction = np.where(is_r, gates * effect, 0.0)
    prediction = np.clip(base_values + correction, 0.0, 1.0)
    if not np.array_equal(prediction[~is_r], base_values[~is_r]):
        raise RuntimeError("F rows changed")
    if not np.isfinite(prediction).all() or np.any((prediction < 0.0) | (prediction > 1.0)):
        raise RuntimeError("invalid final probabilities")

    pd.DataFrame({ID: base_ids, TARGET: prediction}).to_csv(
        OUTPUT_PATH, index=False, encoding="utf-8")
    print(
        f"Saved: {OUTPUT_PATH} | rows={len(prediction)} | R={int(is_r.sum())} | "
        f"F_changed=0 | correction_mean_R={correction[is_r].mean():+.9f} | "
        f"correction_std_R={correction[is_r].std():.9f} | "
        f"mean={prediction.mean():.9f} min={prediction.min():.9f} "
        f"max={prediction.max():.9f}", flush=True)


if __name__ == "__main__":
    main()
'''


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def fit_table(
    source: pd.DataFrame,
    source_residual: np.ndarray,
    name: str,
    keys: tuple[str, ...],
    weight: float,
    alpha: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    work = source.loc[:, list(keys)].copy()
    work["_residual"] = source_residual
    table = (
        work.groupby(list(keys), sort=False, observed=True, dropna=False)["_residual"]
        .agg(["sum", "size"])
        .reset_index()
    )
    table["correction"] = table["sum"] / (table["size"] + alpha)
    records = table.loc[:, [*keys, "correction"]].to_dict(orient="records")
    state = {
        "name": name,
        "keys": list(keys),
        "weight": weight,
        "alpha": alpha,
        "records": native(records),
    }
    diagnostic = {
        "name": name,
        "groups": int(len(table)),
        "weight": weight,
        "alpha": alpha,
        "correction_mean": float(table["correction"].mean()),
        "correction_std": float(table["correction"].std()),
        "correction_max_abs": float(table["correction"].abs().max()),
    }
    return state, diagnostic


def fit_deployment_state(e158, e155) -> tuple[dict[str, Any], dict[str, Any]]:
    columns = [
        "season", "game_type", "pitcher_id", "pitcher_hand", "batter_hand",
        "balls_before", "strikes_before", "base_state", TARGET,
    ]
    frame = e158.prepare_keys(
        pd.read_csv(TRAIN, usecols=columns, encoding="utf-8-sig"))
    rows = {
        year: frame.loc[frame["season"].eq(year)].reset_index(drop=True)
        for year in (2023, 2024)
    }
    target = {year: rows[year][TARGET].to_numpy(np.float64) for year in rows}
    baseline = {
        year: e158.load_current_baseline(e155, rows[year], year)
        for year in rows
    }

    # This 2024 effect is strictly forward: its tables were fitted on 2023.
    # Its observed 2024 row gains define the frozen reliability map for 2025.
    effect24, effect_diag = e158.tensor_effect(
        rows[2023], target[2023], baseline[2023], rows[2024])
    reliability_map, reliability_diag = e158.learn_pitcher_reliability(
        rows[2024], target[2024], baseline[2024], effect24)

    is_r = rows[2024]["game_type"].to_numpy() == "R"
    source_r = rows[2024].loc[is_r].reset_index(drop=True)
    residual_r = target[2024][is_r] - baseline[2024][is_r]
    tables = []
    table_diagnostics = []
    for name, keys, weight, alpha in e158.TENSOR_SPECS:
        state, diagnostic = fit_table(
            source_r, residual_r, name, keys, weight, alpha)
        tables.append(state)
        table_diagnostics.append(diagnostic)

    state = {
        "schema_version": 1,
        "status": "HOLD_NO_SUBMISSION",
        "fit_source": "official train season 2024 R rows and our current-anchor OOF",
        "tables": tables,
        "pitcher_gate": {str(int(k)): float(v) for k, v in reliability_map.items()},
        "reliability_alpha": float(e158.RELIABILITY_ALPHA),
        "reliable_weight": float(e158.RELIABLE_WEIGHT),
        "unstable_weight": float(e158.UNSTABLE_WEIGHT),
        "unseen_weight": float(e158.UNSEEN_WEIGHT),
        "primary_scale": float(e158.PRIMARY_SCALE),
        "F_policy": "exact current-anchor fallback",
    }
    diagnostics = {
        "train_2024_rows": int(len(rows[2024])),
        "train_2024_R_rows": int(is_r.sum()),
        "residual_mean_R": float(residual_r.mean()),
        "residual_std_R": float(residual_r.std()),
        "forward_effect_2023_to_2024": effect_diag,
        "reliability_fit_on_2024_for_2025": reliability_diag,
        "tables_fit_on_2024_for_2025": table_diagnostics,
    }
    return state, diagnostics


def build_package(state: dict[str, Any]) -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    SOURCE.mkdir(parents=True)
    with zipfile.ZipFile(BASE_ZIP) as archive:
        archive.extractall(SOURCE)
    base_script = SOURCE / "script.py"
    base_script.replace(SOURCE / "model" / "current_blend_inference.py")
    base_script.write_text(WRAPPER, encoding="utf-8")
    (SOURCE / "model" / "robusttensor_state.json").write_text(
        json.dumps(native(state), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8")
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(SOURCE.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(SOURCE).as_posix())


def fake_server_rehearsal(rows: int = 245789, seed: int = 181) -> dict[str, Any]:
    if REHEARSE.exists():
        shutil.rmtree(REHEARSE)
    work = REHEARSE / "full"
    work.mkdir(parents=True)
    with zipfile.ZipFile(OUT_ZIP) as archive:
        archive.extractall(work)
        names = archive.namelist()
    (work / "data").mkdir()
    sys.path.insert(0, str(ROOT / "exp"))
    fake_builder = __import__("145_rehearse_reimpl").build_fake_test
    real = pd.read_csv(TEST, encoding="utf-8-sig")
    fake = fake_builder(real, rows, seed)
    fake.to_csv(work / "data" / "test.csv", index=False, encoding="utf-8")
    pd.DataFrame({ID: fake[ID], TARGET: 0.0}).to_csv(
        work / "data" / "sample_submission.csv", index=False, encoding="utf-8")
    started = time.time()
    result = subprocess.run(
        [str(PYTHON), "script.py"], cwd=work, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    elapsed = time.time() - started
    if result.returncode != 0:
        raise RuntimeError(
            "fake-server failure\n" + (result.stdout or "")[-3000:] +
            "\n" + (result.stderr or "")[-5000:])
    output = pd.read_csv(work / "output" / "submission.csv")
    values = output[TARGET].to_numpy(np.float64)
    tops = sorted({name.split("/")[0] for name in names})
    expected_tops = ["model", "requirements.txt", "script.py"]
    if tops != expected_tops:
        raise RuntimeError(f"bad package roots: {tops}")
    if list(output.columns) != [ID, TARGET] or len(output) != rows:
        raise RuntimeError("fake-server output schema/row count mismatch")
    if not np.array_equal(output[ID].to_numpy(), fake[ID].to_numpy()):
        raise RuntimeError("fake-server output row order mismatch")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise RuntimeError("fake-server probabilities invalid")
    return {
        "rows": rows,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "stdout_tail": (result.stdout or "").strip().splitlines()[-1],
        "zip_roots": tops,
        "output_mean": float(values.mean()),
        "output_std": float(values.std()),
        "output_min": float(values.min()),
        "output_max": float(values.max()),
        "schema_ok": True,
        "row_order_ok": True,
        "finite_range_ok": True,
    }


def row_independence_qa() -> dict[str, Any]:
    result = subprocess.run(
        [str(PYTHON), str(ROOT / "exp" / "147_row_independence_ours.py"),
         "--zip", OUT_ZIP.name, "--n", "300", "--seed", "181"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if result.returncode != 0:
        raise RuntimeError(
            "row-independence failure\n" + (result.stdout or "")[-4000:] +
            "\n" + (result.stderr or "")[-4000:])
    stdout = result.stdout or ""
    # Mojibake in the legacy QA labels does not affect this numeric assertion.
    if stdout.count("0.000e+00") < 7:
        raise RuntimeError("row-independence QA did not report exact zero")
    return {"rows": 300, "returncode": 0, "max_absolute_difference": 0.0,
            "stdout_tail": stdout[-3500:]}


def main() -> None:
    if not BASE_ZIP.is_file() or not VALIDATION_JSON.is_file() or not PYTHON.is_file():
        raise SystemExit("required clean base, exp158 validation, or venv311 is missing")
    started = time.time()
    e158 = load_module(ROOT / "exp" / "158_robust_conditional_tensor_eb.py", "e158_181")
    e155 = load_module(ROOT / "exp" / "155_cause_runner_currentblend_gate.py", "e155_181")
    validation = json.loads(VALIDATION_JSON.read_text(encoding="utf-8"))

    state, fit_diagnostics = fit_deployment_state(e158, e155)
    build_package(state)
    rehearsal = fake_server_rehearsal()
    independence = row_independence_qa()

    confirmation = validation["confirmation_2024_fixed_scale"]
    periods = validation["periods_2024"]
    bootstrap = validation["pitcher_cluster_bootstrap_2024"]
    payload = {
        "experiment": 181,
        "status": "TECHNICALLY_DEPLOYABLE_BUT_VALIDATION_NO_GO_DO_NOT_SUBMIT",
        "submitted": False,
        "cleanroom": {
            "third_party_code_read_or_copied": False,
            "third_party_weights_predictions_or_oof_used": False,
            "fit_data": "official train through season 2024 plus our own current-anchor OOF",
            "base_zip": BASE_ZIP.name,
            "base_zip_sha256": sha256(BASE_ZIP),
            "public_method_issue": ISSUE_METHOD,
            "public_context_issue": ISSUE_CONTEXT,
        },
        "frozen_method": {
            "tensor_specs": [
                {"name": n, "keys": list(k), "weight": w, "alpha": a}
                for n, k, w, a in e158.TENSOR_SPECS],
            "reliability_alpha": e158.RELIABILITY_ALPHA,
            "reliable_weight": e158.RELIABLE_WEIGHT,
            "unstable_weight": e158.UNSTABLE_WEIGHT,
            "unseen_weight": e158.UNSEEN_WEIGHT,
            "primary_scale": e158.PRIMARY_SCALE,
            "clip": [0.0, 1.0],
            "R_only": True,
            "F_exact_fallback": True,
        },
        "our_2024_confirmation_evidence": {
            "source": str(VALIDATION_JSON),
            "status": validation["status"],
            "raw_gain": confirmation["raw_gain"],
            "equal_mean_shape_gain": confirmation["equal_mean_shape_gain"],
            "early_gain": periods["early_month_le_6"]["gain"],
            "late_gain": periods["late_month_gt_6"]["gain"],
            "all_hands_positive": validation["all_hands_positive"],
            "pitcher_cluster_ci": [bootstrap["p025"], bootstrap["p975"]],
            "pitcher_cluster_prob_positive": bootstrap["prob_positive"],
            "decision": "FAIL_NO_SUBMISSION",
        },
        "external_transfer_context_not_our_validation": {
            "issue_5_claimed_anchor": "JM R Residual scale 1.00 (not our anchor)",
            "issue_5_reported_local_gain": 15.4889,
            "issue_5_reported_official_score": 1148.7730692448,
            "issue_8_repeats_robusttensor_as_input_score": 1148.77306,
            "issue_8_method_detail": (
                "Issue #8 identifies the RobustTensor ZIP and score but does not specify its "
                "Tensor/EB hyperparameters; those public settings are in issue #5."
            ),
            "issue_8_hybrid_followup_official_score": 1109.9797870635,
            "issue_8_hybrid_base_official_score": 1150.3783882339,
            "interpretation": (
                "Evidence that the method transferred on JM's different anchor; it does not "
                "override our failed cross-year confirmation and is not an expected score for this ZIP."
            ),
        },
        "deployment_refit_for_2025": fit_diagnostics,
        "artifact": {
            "zip": str(OUT_ZIP),
            "size_bytes": OUT_ZIP.stat().st_size,
            "sha256": sha256(OUT_ZIP),
            "label": "HOLD_NO_SUBMISSION",
        },
        "fake_server_qa": rehearsal,
        "row_independence_qa": independence,
        "elapsed_seconds": time.time() - started,
    }
    OUT_JSON.write_text(
        json.dumps(native(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "EXP181 RobustTensor on current clean anchor",
        "===========================================",
        f"status: {payload['status']}",
        "submitted: false",
        "",
        "Our locked 2024 confirmation (actual decision evidence)",
        f"  raw gain                 {confirmation['raw_gain']:+.6f}",
        f"  equal-mean shape gain    {confirmation['equal_mean_shape_gain']:+.6f}",
        f"  early / late             {periods['early_month_le_6']['gain']:+.6f} / "
        f"{periods['late_month_gt_6']['gain']:+.6f}",
        f"  all hands positive       {validation['all_hands_positive']}",
        f"  pitcher 95% CI           [{bootstrap['p025']:+.6f}, {bootstrap['p975']:+.6f}]",
        "  decision                 FAIL_NO_SUBMISSION",
        "",
        "External context (not evidence for our anchor)",
        "  Public issue #5 reports +15.4889 locally and official 1148.7730692448",
        "  on JM R Residual scale 1.00. Issue #8 repeats that endpoint as an input.",
        "  Issue #8 does not give Tensor settings; it reports a later Hybrid official",
        "  failure (1109.979787 vs its 1150.378388 base), reinforcing transfer risk.",
        "  No JM code, ZIP, weights, predictions, or OOF artifacts were read or used.",
        "",
        "Technical artifact (HOLD only)",
        f"  zip      {OUT_ZIP.name}",
        f"  bytes    {OUT_ZIP.stat().st_size}",
        f"  SHA256   {payload['artifact']['sha256']}",
        f"  fake QA  rows={rehearsal['rows']} rc=0 range_ok=true F_changed=0",
        "  row independence max abs diff = 0",
        "",
        "Bottom line",
        "  A clean 2025 implementation is technically possible and server-safe, but the",
        "  only valid transfer check on our current anchor failed. This ZIP must not consume",
        "  a leaderboard slot unless an explicit decision overrides the frozen guardrail.",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
