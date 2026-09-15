# -*- coding: utf-8 -*-
"""Diagnostic-only upper bound for segment calibration on the frozen champion.

This script uses only the labeled 2024 validation frame and saved 2024
predictions.  It does not read test.csv, build a bundle, or authorize an LB
probe.  The purpose is to decide whether a nine-submission symmetric segment
calibration scheme could plausibly explain a +100 gain before discussing its
separate rules question.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FRAME = ROOT / "lab" / "90_analysis_2024.parquet"
CAT5 = ROOT / "lab" / "89_cat5.npy"
V18 = ROOT / "lab" / "103_v18_effect_2024.npy"
OUT_JSON = ROOT / "lab" / "114_segment_calibration_capacity.json"
OUT_TEXT = ROOT / "lab" / "114_segment_calibration_capacity.txt"

CENTER = 0.49
SCALE = 1.06
SHIFT = 0.0066
GAMMA = 0.30


def score(probability: np.ndarray, target: np.ndarray) -> float:
    rate = float(target.mean())
    mse = float(np.mean((probability - target) ** 2))
    return 100000.0 * (1.0 - mse / (rate * (1.0 - rate)))


def audit_partition(
    name: str,
    labels: np.ndarray,
    probability: np.ndarray,
    target: np.ndarray,
    global_oracle_score: float,
) -> dict:
    labels = np.asarray(labels).astype(str)
    corrected = probability.copy()
    rows = []
    for value in pd.unique(labels):
        mask = labels == value
        delta = float(np.mean(target[mask] - probability[mask]))
        corrected[mask] = np.clip(probability[mask] + delta, 0.0, 1.0)
        rows.append(
            {
                "group": str(value),
                "rows": int(mask.sum()),
                "oracle_delta": delta,
            }
        )
    base_score = score(probability, target)
    group_score = score(corrected, target)
    return {
        "name": name,
        "groups": len(rows),
        "base_score": base_score,
        "group_oracle_score": group_score,
        "group_oracle_gain": group_score - base_score,
        "differential_gain_over_global_oracle": group_score - global_oracle_score,
        "group_rows": rows,
    }


def main() -> None:
    frame = pd.read_parquet(FRAME)
    target = frame["y"].to_numpy(np.float64)
    cat5 = np.load(CAT5).astype(np.float64)
    effect = np.load(V18).astype(np.float64)
    if not (len(frame) == len(cat5) == len(effect)):
        raise RuntimeError("2024 frame/prediction length mismatch")

    pre_affine = np.clip(cat5 + GAMMA * effect, 0.0, 1.0)
    probability = np.clip(
        CENTER + SCALE * (pre_affine - CENTER) - SHIFT,
        0.0,
        1.0,
    )
    base_score = score(probability, target)
    global_delta = float(np.mean(target - probability))
    global_probability = np.clip(probability + global_delta, 0.0, 1.0)
    global_score = score(global_probability, target)

    p4_fixed = pd.cut(
        probability,
        [-np.inf, 0.44, 0.49, 0.54, np.inf],
        labels=["p<.44", ".44-.49", ".49-.54", "p>=.54"],
    ).astype(str)
    game_p50 = frame["game_type"].astype(str) + "|" + np.where(
        probability >= 0.50, "p>=.50", "p<.50"
    )
    count12 = (
        frame["balls_before"].astype(str)
        + "-"
        + frame["strikes_before"].astype(str)
    )
    count3 = np.select(
        [
            frame["balls_before"].to_numpy() > frame["strikes_before"].to_numpy(),
            frame["balls_before"].to_numpy() < frame["strikes_before"].to_numpy(),
        ],
        ["hitter_ahead", "pitcher_ahead"],
        default="even",
    )
    rookie = (
        frame["game_type"].astype(str)
        + "|rookie="
        + (frame["p_first_season"] == 2024).astype(str)
    )

    partitions = [
        ("game_type", frame["game_type"].to_numpy()),
        ("probability_4_fixed", np.asarray(p4_fixed)),
        ("game_type_x_p50", np.asarray(game_p50)),
        ("count_12", np.asarray(count12)),
        ("count_3", count3),
        ("game_type_x_rookie", np.asarray(rookie)),
    ]
    audits = [
        audit_partition(name, labels, probability, target, global_score)
        for name, labels in partitions
    ]

    payload = {
        "experiment": 114,
        "scope": "2024 labeled diagnostic only; no test rows and no deployment",
        "champion_formula": {
            "gamma": GAMMA,
            "center": CENTER,
            "scale": SCALE,
            "shift": SHIFT,
        },
        "rows": len(target),
        "target_rate": float(target.mean()),
        "prediction_mean": float(probability.mean()),
        "base_score": base_score,
        "global_oracle_delta": global_delta,
        "global_oracle_gain": global_score - base_score,
        "partitions": audits,
        "decision": (
            "REJECT nine-slot segment tomography as a +100 path: the best tested "
            "four-group differential upper bound is below +8 and the twelve-count "
            "upper bound is below +11; rules approval would also be required."
        ),
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "exp/114 — segment calibration capacity (diagnostic only)",
        f"rows={len(target):,} target_rate={target.mean():.9f} pred_mean={probability.mean():.9f}",
        f"base_score={base_score:.6f}",
        f"global_oracle delta={global_delta:+.9f} gain={global_score-base_score:+.6f}",
        "",
    ]
    for row in audits:
        lines.append(
            f"{row['name']:<24} groups={row['groups']:>2} "
            f"total_oracle={row['group_oracle_gain']:+.6f} "
            f"differential_over_global={row['differential_gain_over_global_oracle']:+.6f}"
        )
    lines.extend(
        [
            "",
            "DECISION: REJECT as a +100 path. game_type×p50 (4 groups) adds only",
            "about +7.6 beyond a global oracle; all 12 counts add about +10.6.",
            "This is an upper-bound diagnostic using 2024 labels, not a deployable model.",
            "Segment-wise LB inversion also requires separate written rules approval.",
        ]
    )
    OUT_TEXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

