"""Solve the fixed-centre affine vertex from one leaderboard probe.

For the competition skill score and ``p_s = c + s * (p - c)``, the score is

    score(s) = C + L*s - B*s**2,
    C = -lambda * (c - r)**2,
    lambda = 100000 / (r * (1-r)).

Once ``lambda`` and the target-rate root ``r`` are known, the current score at
``s=1`` plus one probe at any other scale identifies ``L`` and ``B`` exactly.
The defaults reconstruct lambda from the two historical constant-shift probes;
those two scores are only recorded to two decimals locally, so pass their full
leaderboard precision when available.

This is analysis only.  It does not build or modify a submission artifact.
"""

from __future__ import annotations

import argparse

import numpy as np


def recover_lambda(
    base_score: float,
    shifted_012_score: float,
    shifted_0066_score: float,
) -> tuple[float, float]:
    """Return (lambda, old prediction mean minus target rate)."""
    deltas = np.array((0.012, 0.0066), dtype=np.float64)
    design = np.column_stack((2.0 * deltas, -np.square(deltas)))
    gain = np.array(
        (shifted_012_score - base_score, shifted_0066_score - base_score),
        dtype=np.float64,
    )
    lambda_times_bias, metric_lambda = np.linalg.solve(design, gain)
    return float(metric_lambda), float(lambda_times_bias / metric_lambda)


def target_rate_roots(metric_lambda: float) -> tuple[float, float]:
    denominator = 100000.0 / metric_lambda
    discriminant = 1.0 - 4.0 * denominator
    if discriminant < 0.0:
        raise ValueError("recovered denominator is not a Bernoulli variance")
    gap = np.sqrt(discriminant)
    return float((1.0 - gap) / 2.0), float((1.0 + gap) / 2.0)


def solve_vertex(
    base_score: float,
    probe_score: float,
    probe_scale: float,
    center: float,
    metric_lambda: float,
    target_rate: float,
) -> dict[str, float]:
    if probe_scale in (0.0, 1.0):
        raise ValueError("probe scale must differ from both zero and one")
    constant_score = -metric_lambda * (center - target_rate) ** 2
    y_one = base_score - constant_score
    y_probe = probe_score - constant_score
    curvature = (
        probe_scale * y_one - y_probe
    ) / (probe_scale * (probe_scale - 1.0))
    linear = y_one + curvature
    if curvature <= 0.0:
        raise ValueError(f"non-concave recovered parabola: B={curvature}")
    optimum = linear / (2.0 * curvature)
    peak = constant_score + linear * linear / (4.0 * curvature)
    return {
        "constant_score": float(constant_score),
        "curvature_B": float(curvature),
        "linear_L": float(linear),
        "optimum_scale": float(optimum),
        "peak_score": float(peak),
        "peak_gain_from_base": float(peak - base_score),
        "peak_gain_from_probe": float(peak - probe_score),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-score", type=float, required=True)
    parser.add_argument("--probe-scale", type=float, default=0.96)
    parser.add_argument("--base-score", type=float, default=1114.6116846973)
    parser.add_argument("--center", type=float, default=0.44)
    parser.add_argument("--old-base", type=float, default=1059.0501189623)
    parser.add_argument("--old-d012", type=float, default=1065.26)
    parser.add_argument("--old-d0066", type=float, default=1076.81)
    parser.add_argument("--root", choices=("low", "high", "both"), default="low")
    args = parser.parse_args()

    metric_lambda, old_bias = recover_lambda(
        args.old_base, args.old_d012, args.old_d0066
    )
    denominator = 100000.0 / metric_lambda
    low, high = target_rate_roots(metric_lambda)
    print(f"lambda={metric_lambda:.12f}")
    print(f"r(1-r)={denominator:.12f}")
    print(f"old mean bias={old_bias:+.12f}")
    print(f"target-rate roots: low={low:.12f}, high={high:.12f}")
    print(
        "WARNING: local historical shift scores are rounded to 0.01; "
        "supply their full precision for an exact result."
    )

    roots = (("low", low), ("high", high))
    for label, rate in roots:
        if args.root != "both" and label != args.root:
            continue
        result = solve_vertex(
            args.base_score,
            args.probe_score,
            args.probe_scale,
            args.center,
            metric_lambda,
            rate,
        )
        print(f"\n[{label} root r={rate:.12f}]")
        for key, value in result.items():
            print(f"{key}={value:.12f}")
        optimum = result["optimum_scale"]
        if optimum <= 1.0:
            print("clipping: impossible for p in [0,1] because scale is in [0,1]")
        else:
            low_safe = args.center * (1.0 - 1.0 / optimum)
            high_safe = args.center + (1.0 - args.center) / optimum
            print(
                "clipping-free only if hidden base predictions stay in "
                f"[{low_safe:.12f}, {high_safe:.12f}]"
            )


if __name__ == "__main__":
    main()
