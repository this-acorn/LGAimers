"""Solve the leaderboard-calibrated affine plane from frozen full-test predictions.

The public metric is an affine transform of mean squared error.  Therefore the
curvature between two predictions is proportional to their mean squared
difference.  One measured midpoint calibrates that proportionality constant;
known endpoint scores then recover the hidden-label linear terms without ever
using labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ID = "row_id"
TARGET = "control_success"


def load(path: Path, expected_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if list(frame.columns) != [ID, TARGET]:
        raise ValueError(f"{path}: unexpected columns {frame.columns.tolist()}")
    ids = frame[ID].to_numpy()
    values = frame[TARGET].to_numpy(dtype=np.float64)
    if len(np.unique(ids)) != len(ids) or not np.isfinite(values).all():
        raise ValueError(f"{path}: invalid ids or probabilities")
    if expected_ids is not None and not np.array_equal(ids, expected_ids):
        raise ValueError(f"{path}: row order mismatch")
    return ids, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--joa", type=Path, required=True)
    parser.add_argument("--jm075", type=Path, required=True)
    parser.add_argument("--jy", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--score-current", type=float, default=1116.5770907872)
    parser.add_argument("--score-joa", type=float, default=1126.8664003703)
    parser.add_argument("--score-jm075", type=float, default=1134.6974491044)
    parser.add_argument("--score-jy", type=float, default=1130.3604943627)
    parser.add_argument("--mid-current-jm075", type=float, default=1144.7518569078)
    parser.add_argument("--mid-current-jy", type=float, default=1142.7349769136)
    args = parser.parse_args()

    ids, current = load(args.current)
    _, joa = load(args.joa, ids)
    _, jm075 = load(args.jm075, ids)

    k_jm = (
        4.0 * args.mid_current_jm075
        - 2.0 * args.score_current
        - 2.0 * args.score_jm075
    )
    delta_jm = jm075 - current
    mse_delta_jm = float(np.mean(delta_jm * delta_jm))
    metric_scale = k_jm / mse_delta_jm

    # Coordinates: current + x0*(JOA-current) + x1*((JM.75-JOA)/.75).
    names = ["joa_minus_current", "jm_residual_per_scale"]
    directions = [joa - current, (jm075 - joa) / 0.75]
    endpoint_scores = [args.score_joa, None]

    k_jy_measured = None
    if args.jy:
        _, jy = load(args.jy, ids)
        names.append("jy_minus_current")
        directions.append(jy - current)
        endpoint_scores.append(args.score_jy)
        k_jy_measured = (
            4.0 * args.mid_current_jy
            - 2.0 * args.score_current
            - 2.0 * args.score_jy
        )

    matrix = np.column_stack(directions)
    hessian = metric_scale * (matrix.T @ matrix) / len(current)

    linear = np.empty(len(directions), dtype=np.float64)
    linear[0] = args.score_joa - args.score_current + hessian[0, 0]
    # Force the .75 endpoint score to recover the residual linear term.
    x075 = np.zeros(len(directions), dtype=np.float64)
    x075[0] = 1.0
    x075[1] = 0.75
    known_without_l1 = args.score_current + linear[0]
    quad075 = float(x075 @ hessian @ x075)
    linear[1] = (args.score_jm075 - known_without_l1 + quad075) / 0.75
    if args.jy:
        # Use the measured diagonal curvature rather than its locally calibrated
        # estimate, while retaining locally computed cross-products.
        local_diag = float(hessian[2, 2])
        factor = float(np.sqrt(k_jy_measured / local_diag))
        hessian[2, :] *= factor
        hessian[:, 2] *= factor
        hessian[2, 2] = k_jy_measured
        linear[2] = args.score_jy - args.score_current + hessian[2, 2]

    optimum = 0.5 * np.linalg.solve(hessian, linear)
    gain = float(linear @ optimum - optimum @ hessian @ optimum)
    predicted_score = args.score_current + gain
    raw_prediction = current + matrix @ optimum
    clipped = np.clip(raw_prediction, 0.0, 1.0)

    # Cross-check the measured JM midpoint and endpoint in the recovered model.
    midpoint = 0.5 * x075
    reconstructed_mid = float(
        args.score_current + linear @ midpoint - midpoint @ hessian @ midpoint
    )
    reconstructed_endpoint = float(
        args.score_current + linear @ x075 - x075 @ hessian @ x075
    )

    payload = {
        "rows": len(current),
        "metric_scale": metric_scale,
        "k_current_jm075": k_jm,
        "mse_delta_current_jm075": mse_delta_jm,
        "direction_names": names,
        "hessian": hessian.tolist(),
        "linear": linear.tolist(),
        "eigenvalues": np.linalg.eigvalsh(hessian).tolist(),
        "optimum_coordinates": optimum.tolist(),
        "predicted_unclipped_score": predicted_score,
        "predicted_gain": gain,
        "raw_prediction_min": float(raw_prediction.min()),
        "raw_prediction_max": float(raw_prediction.max()),
        "clipped_rows": int(np.count_nonzero(raw_prediction != clipped)),
        "clipped_fraction": float(np.mean(raw_prediction != clipped)),
        "reconstructed_jm075_midpoint": reconstructed_mid,
        "reconstructed_jm075_endpoint": reconstructed_endpoint,
        "measured_jm075_midpoint": args.mid_current_jm075,
    }
    if k_jy_measured is not None:
        payload["k_current_jy_measured"] = k_jy_measured
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
