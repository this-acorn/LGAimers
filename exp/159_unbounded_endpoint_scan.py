# -*- coding: utf-8 -*-
"""Diagnostic scan of saved 2024 probability endpoints with signed weights.

This is deliberately analysis-only.  It reads official 2024 labels and saved
local OOF vectors, never test.csv, and does not build a submission.  The goal
is to detect directions that were missed by convex-only blending, especially
R-only anti-blends (negative endpoint weights).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "lab" / "159_unbounded_endpoint_scan.json"
OUT_TXT = ROOT / "lab" / "159_unbounded_endpoint_scan.txt"
SKIP_PARTS = {
    "target", "effect", "mask", "qS", "p0", "control_final",
    "features", "smoke", "fake", "pitchtype_prior",
}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("e155_for_scan", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(probability: np.ndarray, target: np.ndarray) -> float:
    rate = float(target.mean())
    return float(100000.0 * (1.0 - np.mean((probability - target) ** 2) / (rate * (1.0 - rate))))


def main() -> None:
    e155 = load_module(ROOT / "exp" / "155_cause_runner_currentblend_gate.py")
    frame = pd.read_csv(
        ROOT / "data" / "train.csv", encoding="utf-8-sig",
        usecols=["season", "game_type", "control_success"], low_memory=False,
    )
    rows = frame.loc[frame["season"].eq(2024)].reset_index(drop=True)
    target = rows["control_success"].to_numpy(np.float64)
    n = len(rows)
    cat = e155.load_probability(e155.CAT5[2024], n)
    v18 = e155.load_vector(e155.V18[2024], n)
    external = e155.load_probability(e155.EXP021[2024], n)
    champion = e155.champion_probability(cat, v18)
    baseline = e155.CHAMPION_WEIGHT * champion + e155.EXP021_WEIGHT * external
    base_score = score(baseline, target)
    is_r = rows["game_type"].astype(str).eq("R").to_numpy()
    segments = {"ALL": np.ones(n, dtype=bool), "R": is_r, "F": ~is_r}

    seen: set[str] = set()
    records: list[dict[str, object]] = []
    inspected = 0
    accepted = 0
    for path in sorted((ROOT / "lab").rglob("*.npy")):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        lower = rel.lower()
        if "116_exp021_build_env" in lower or any(part.lower() in lower for part in SKIP_PARTS):
            continue
        try:
            value = np.load(path, allow_pickle=False)
        except Exception:
            continue
        inspected += 1
        if value.shape == (n, 5):
            value = value[:, 0]
        if value.shape != (n,):
            continue
        value = np.asarray(value, dtype=np.float64)
        if not np.isfinite(value).all() or value.min() < -1e-6 or value.max() > 1.0 + 1e-6:
            continue
        if value.std() < 1e-4:
            continue
        digest = hashlib.sha256(value.astype(np.float32, copy=False).tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        accepted += 1
        endpoint_score = score(value, target)
        # Reject obvious label/proxy leakage and non-probability diagnostics.
        if endpoint_score > 2000.0:
            continue
        for segment, mask in segments.items():
            direction = np.where(mask, value - baseline, 0.0)
            curvature = float(direction @ direction)
            if curvature <= 1e-14:
                continue
            weight = float(((target - baseline) @ direction) / curvature)
            weight_capped = float(np.clip(weight, -3.0, 3.0))
            prediction = np.clip(baseline + weight_capped * direction, 0.0, 1.0)
            gain = score(prediction, target) - base_score
            convex_weight = float(np.clip(weight, 0.0, 1.0))
            convex = np.clip(baseline + convex_weight * direction, 0.0, 1.0)
            records.append({
                "path": rel,
                "sha256_f32": digest,
                "segment": segment,
                "endpoint_score": endpoint_score,
                "weight_unbounded": weight,
                "weight_used": weight_capped,
                "gain_signed": float(gain),
                "weight_convex": convex_weight,
                "gain_convex": float(score(convex, target) - base_score),
                "clipped_rate": float(np.mean((prediction <= 0.0) | (prediction >= 1.0))),
                "direction_std": float(direction.std()),
            })

    ranked = sorted(records, key=lambda row: float(row["gain_signed"]), reverse=True)
    negative = [row for row in ranked if float(row["weight_used"]) < 0.0]
    report = {
        "analysis_only": True,
        "test_opened": False,
        "baseline_score_2024": base_score,
        "n_rows": n,
        "files_inspected": inspected,
        "unique_probability_vectors": accepted,
        "top_signed": ranked[:40],
        "top_negative": negative[:40],
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "=== exp159 signed endpoint scan (2024 same-fold diagnostic) ===",
        "NO TEST / NO ZIP / NO SUBMISSION",
        f"baseline={base_score:.6f} rows={n:,} inspected={inspected} unique={accepted}",
        "",
        "Top signed directions:",
    ]
    for row in ranked[:25]:
        lines.append(
            f"{row['gain_signed']:+9.4f} w={row['weight_used']:+8.4f} "
            f"seg={row['segment']:>3s} endpoint={row['endpoint_score']:+9.3f} "
            f"clip={100.0 * row['clipped_rate']:.3f}% {row['path']}"
        )
    lines.extend(["", "Top genuine anti-blends:"])
    for row in negative[:25]:
        lines.append(
            f"{row['gain_signed']:+9.4f} w={row['weight_used']:+8.4f} "
            f"seg={row['segment']:>3s} endpoint={row['endpoint_score']:+9.3f} "
            f"clip={100.0 * row['clipped_rate']:.3f}% {row['path']}"
        )
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
