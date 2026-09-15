# -*- coding: utf-8 -*-
"""EXP-128: one-shot 2024 confirmation of the frozen EXP-127 p0 arms.

The two deployable rules were frozen before any 2024 metric was read:

* primary:   0.50 * exact-current-blend + 0.50 * hierarchical p0
* secondary: exact-current-blend on R, hierarchical p0 on F

The secondary may qualify only if the primary fails its fixed gate.  Direct p0
is diagnostic only.  This script reads official train through 2024 and saved
OOF vectors; it never reads test.csv, an EXP-021 inference source, or creates a
model, ZIP, or submission.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "lab"
DATA = ROOT / "data" / "train.csv"
EXP127 = ROOT / "exp" / "127_dynamic_hier_residual.py"
OUT_JSON = LAB / "128_p0_confirmation.json"
OUT_TXT = LAB / "128_p0_confirmation.txt"
P0_PATH = LAB / "128_y2024_p0.npy"

OWN_PATH = LAB / "89_cat5_probs_seed42.npy"
V18_PATH = LAB / "103_v18_effect_2024.npy"
EXP020_ROOT = (
    LAB
    / "115_mkis_exp021_work"
    / "artifacts"
    / "EXP-020"
    / "low_rank_pitcher_context_eb"
)
EXP020_PATH = EXP020_ROOT / "predictions_lowrank_s300_r6_2024.npy"
EXP020_TARGET_PATH = EXP020_ROOT / "targets_2024.npy"

EXP020_WEIGHT = 0.35655716947524263
OWN_WEIGHT = 1.0 - EXP020_WEIGHT
PRIMARY_WEIGHT_P0 = 0.50
# Exact seed-42 proxy used by EXP-126/127.  The 946.789763 value belongs to
# the two-seed own endpoint and must not be mixed into this coherent fold.
EXPECTED_ACUR_SCORE = 936.5839989479041

PRIMARY_GATE = {
    "raw_gain_min": 15.0,
    "equal_mean_gain_min": 10.0,
    "F_contribution_min": 0.0,
    "R_contribution_min": 0.0,
    "bootstrap_p025_min": 0.0,
}
SECONDARY_GATE = {
    "raw_gain_min": 25.0,
    "equal_mean_gain_min": 20.0,
    "F_contribution_min": 0.0,
    "bootstrap_p025_min": 0.0,
}


def load_exp127():
    original = sys.argv[:]
    try:
        sys.argv = [str(EXP127), "--smoke"]
        spec = importlib.util.spec_from_file_location("exp128_cleanroom_exp127", EXP127)
        if spec is None or spec.loader is None:
            raise ImportError(EXP127)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = original


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probability(path: Path, rows: int, label: str) -> np.ndarray:
    value = np.load(path, allow_pickle=False).astype(np.float64)
    if value.ndim == 2:
        if value.shape[1] != 5:
            raise ValueError(f"{label}: unexpected multiclass shape {value.shape}")
        value = value[:, 0]
    if value.shape != (rows,) or not np.isfinite(value).all():
        raise ValueError(f"{label}: shape/finite failure {value.shape}")
    if not ((value >= 0.0) & (value <= 1.0)).all():
        raise ValueError(f"{label}: probability outside [0,1]")
    return value


def vector(path: Path, rows: int, label: str) -> np.ndarray:
    value = np.load(path, allow_pickle=False).astype(np.float64)
    if value.shape != (rows,) or not np.isfinite(value).all():
        raise ValueError(f"{label}: shape/finite failure {value.shape}")
    return value


def contribution(base, candidate, target, mask) -> float:
    rate = float(target.mean())
    denominator = len(target) * rate * (1.0 - rate)
    improvement = (base - target) ** 2 - (candidate - target) ** 2
    return float(100000.0 * improvement[mask].sum() / denominator)


def evaluate(name, candidate, base, target, game, pitcher, m, draws=2000):
    base_score = m.raw_score(base, target)
    candidate_score = m.raw_score(candidate, target)
    f_mask = game == "F"
    return {
        "name": name,
        "score": candidate_score,
        "raw_gain": candidate_score - base_score,
        "equal_mean_gain_unclipped": m.equal_mean_gain(
            base, candidate, target, clip=False
        ),
        "equal_mean_gain_clipped": m.equal_mean_gain(
            base, candidate, target, clip=True
        ),
        "F_contribution": contribution(base, candidate, target, f_mask),
        "R_contribution": contribution(base, candidate, target, ~f_mask),
        "mean_shift": float(candidate.mean() - base.mean()),
        "pitcher_bootstrap": m.pitcher_bootstrap(
            base, candidate, target, pitcher, seed=128_024, draws=draws
        ),
    }


def main() -> None:
    started = time.time()
    m = load_exp127()
    required = [DATA, OWN_PATH, V18_PATH, EXP020_PATH, EXP020_TARGET_PATH]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    frame = pd.read_csv(DATA, encoding="utf-8-sig", low_memory=False)
    frame.columns = frame.columns.str.replace("\ufeff", "", regex=False).str.strip()
    history = frame.loc[frame["season"].le(2023)].copy()
    valid = frame.loc[frame["season"].eq(2024)].reset_index(drop=True)
    if valid["row_id"].duplicated().any():
        raise AssertionError("duplicate 2024 row_id")

    history = m.recover_pitch_labels(history)
    volatility, missing = m.row_volatility(history)
    finite = volatility[(~missing) & np.isfinite(volatility)]
    if len(finite) < 1000:
        raise AssertionError("too few train-only volatility rows")
    volatility_scale = max(float(np.quantile(finite, 0.75)), 0.01)
    hierarchical, diagnostics = m.attach_hierarchical(
        valid, history, volatility_scale, "confirmation-2024"
    )
    p0 = hierarchical["h_base"].to_numpy(np.float64)
    if not np.isfinite(p0).all() or not ((p0 > 0.0) & (p0 < 1.0)).all():
        raise AssertionError("invalid p0")
    np.save(P0_PATH, p0.astype(np.float32))

    rows = len(valid)
    target = valid["control_success"].to_numpy(np.float64)
    saved_target = vector(EXP020_TARGET_PATH, rows, "EXP020 saved target")
    if not np.array_equal(target, saved_target):
        raise AssertionError("official target/order differs from saved OOF target")
    own = probability(OWN_PATH, rows, "CAT5 seed42")
    v18 = vector(V18_PATH, rows, "V18 effect")
    exp020 = probability(EXP020_PATH, rows, "EXP020 rank6")
    own_endpoint = np.clip(
        0.49 + 1.06 * (np.clip(own + 0.30 * v18, 0.0, 1.0) - 0.49) - 0.0066,
        0.0,
        1.0,
    )
    acur = OWN_WEIGHT * own_endpoint + EXP020_WEIGHT * exp020
    acur_score = m.raw_score(acur, target)
    if abs(acur_score - EXPECTED_ACUR_SCORE) > 1e-8:
        raise AssertionError(f"Acur score {acur_score} != {EXPECTED_ACUR_SCORE}")

    game = valid["game_type"].astype(str).to_numpy()
    pitcher = valid["pitcher_id"].to_numpy()
    primary_prediction = np.clip(
        (1.0 - PRIMARY_WEIGHT_P0) * acur + PRIMARY_WEIGHT_P0 * p0, 0.0, 1.0
    )
    secondary_prediction = np.where(game == "F", p0, acur)
    variants = {
        "primary_global_p0_w050": evaluate(
            "primary_global_p0_w050",
            primary_prediction,
            acur,
            target,
            game,
            pitcher,
            m,
        ),
        "secondary_F_p0_R_acur": evaluate(
            "secondary_F_p0_R_acur",
            secondary_prediction,
            acur,
            target,
            game,
            pitcher,
            m,
        ),
        "diagnostic_direct_p0": evaluate(
            "diagnostic_direct_p0", p0, acur, target, game, pitcher, m
        ),
    }

    primary = variants["primary_global_p0_w050"]
    primary_checks = {
        "raw_gain": primary["raw_gain"] >= PRIMARY_GATE["raw_gain_min"],
        "equal_mean_gain": primary["equal_mean_gain_clipped"]
        >= PRIMARY_GATE["equal_mean_gain_min"],
        "F_nonnegative": primary["F_contribution"]
        >= PRIMARY_GATE["F_contribution_min"],
        "R_nonnegative": primary["R_contribution"]
        >= PRIMARY_GATE["R_contribution_min"],
        "bootstrap_lower_positive": primary["pitcher_bootstrap"]["p025"]
        > PRIMARY_GATE["bootstrap_p025_min"],
    }
    primary_pass = bool(all(primary_checks.values()))

    secondary = variants["secondary_F_p0_R_acur"]
    secondary_checks = {
        "primary_failed": not primary_pass,
        "raw_gain": secondary["raw_gain"] >= SECONDARY_GATE["raw_gain_min"],
        "equal_mean_gain": secondary["equal_mean_gain_clipped"]
        >= SECONDARY_GATE["equal_mean_gain_min"],
        "F_positive": secondary["F_contribution"]
        > SECONDARY_GATE["F_contribution_min"],
        "R_exact_zero": abs(secondary["R_contribution"]) <= 1e-12,
        "bootstrap_lower_positive": secondary["pitcher_bootstrap"]["p025"]
        > SECONDARY_GATE["bootstrap_p025_min"],
    }
    secondary_pass = bool(all(secondary_checks.values()))
    decision = (
        "PASS_PRIMARY_GLOBAL_P0_W050"
        if primary_pass
        else "PASS_SECONDARY_F_ONLY"
        if secondary_pass
        else "FAIL_NO_DEPLOYMENT"
    )

    result = {
        "experiment": 128,
        "protocol": "frozen 2023 rule -> one-shot untouched 2024 confirmation",
        "precommitted_before_2024": {
            "primary": "0.50*Acur + 0.50*p0",
            "secondary": "Acur on R; p0 on F",
            "direct_p0": "diagnostic only",
            "no_weight_tuning_after_2024": True,
        },
        "baseline": {
            "score": acur_score,
            "expected_score": EXPECTED_ACUR_SCORE,
            "weights": {"own": OWN_WEIGHT, "EXP020_rank6": EXP020_WEIGHT},
        },
        "alignment": {
            "rows": rows,
            "official_vs_saved_target_exact": True,
            "row_id_fingerprint": m.ordered_value_fingerprint(valid["row_id"]),
        },
        "volatility_scale": volatility_scale,
        "hierarchical_diagnostics": diagnostics,
        "variants": variants,
        "gates": {
            "primary": PRIMARY_GATE,
            "primary_checks": primary_checks,
            "primary_pass": primary_pass,
            "secondary": SECONDARY_GATE,
            "secondary_checks": secondary_checks,
            "secondary_pass": secondary_pass,
        },
        "decision": decision,
        "sources_sha256": {str(path): sha256(path) for path in required},
        "elapsed_seconds": time.time() - started,
        "prohibitions": {
            "test_read": False,
            "EXP021_inference_source": False,
            "model_fit": False,
            "zip": False,
            "submission": False,
        },
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "=== EXP-128 frozen p0 one-shot 2024 confirmation ===",
        "no test / no model fit / no EXP021 source / no ZIP / no submission",
        f"Acur exact score={acur_score:.6f} rows={rows:,} volatility_scale={volatility_scale:.6f}",
    ]
    for name, value in variants.items():
        boot = value["pitcher_bootstrap"]
        lines.append(
            f"{name}: score={value['score']:.6f} raw={value['raw_gain']:+.6f} "
            f"shape={value['equal_mean_gain_clipped']:+.6f} "
            f"F/R={value['F_contribution']:+.6f}/{value['R_contribution']:+.6f} "
            f"boot95=[{boot['p025']:+.6f},{boot['p975']:+.6f}]"
        )
    lines.extend(
        [
            f"primary_checks={primary_checks}",
            f"secondary_checks={secondary_checks}",
            f"FINAL={decision}",
            f"elapsed={result['elapsed_seconds']:.2f}s",
        ]
    )
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
