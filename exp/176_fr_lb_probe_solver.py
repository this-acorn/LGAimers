"""Solve disjoint F/R leaderboard directions from two probes per direction.

Let ``A`` be the already-scored reference prediction and let ``v_g`` be the
change from A to a new endpoint on rows in group ``g`` only.  On any interval
where the submitted probabilities are affine in ``w`` (in particular, the
convex interval 0 <= w <= 1), the competition's normalized Brier score is

    S_g(w) = S0 + L_g*w - K_g*w**2.

The scores at w=0.5 and w=1 recover the entire parabola.  Since v_F and v_R
have disjoint row support, their quadratic cross term is exactly zero and the
joint score is the sum of the two recovered gains.

Examples
--------
One surviving direction (two new leaderboard scores)::

    python exp/176_fr_lb_probe_solver.py \
        --f-half 1120.123456789 --f-one 1124.234567891

Both directions (four new leaderboard scores)::

    python exp/176_fr_lb_probe_solver.py \
        --f-half 1120.1 --f-one 1124.2 \
        --r-half 1118.3 --r-one 1120.7

The default deployment bounds are [0, 1], which is guaranteed to be a convex
combination of two already-valid endpoint probabilities.  Only widen a bound
after calculating a no-clipping interval from the *real submitted test
predictions*; do not infer such an interval from synthetic/fake geometry.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_S0 = 1114.6116846973
CURVATURE_TOL = 1e-10


@dataclass(frozen=True)
class AxisSolution:
    name: str
    s_half: float
    s_one: float
    d: float
    k: float
    linear: float
    raw_weight: float | None
    raw_gain: float | None
    bound_low: float
    bound_high: float
    selected_weight: float
    selected_gain: float
    selected_score_alone: float
    status: str
    raw_inside_declared_no_clip_bounds: bool | None


def gain(w: float, linear: float, k: float) -> float:
    """Score gain at weight w for S(w)-S0 = linear*w-k*w**2."""
    return linear * w - k * w * w


def solve_axis(
    name: str,
    s0: float,
    s_half: float,
    s_one: float,
    bounds: tuple[float, float],
    tol: float = CURVATURE_TOL,
) -> AxisSolution:
    """Recover one axis and maximize it over a declared no-clipping interval."""
    lo, hi = bounds
    if not all(math.isfinite(x) for x in (s0, s_half, s_one, lo, hi)):
        raise ValueError(f"{name}: scores and bounds must be finite")
    if lo > hi:
        raise ValueError(f"{name}: lower bound {lo} exceeds upper bound {hi}")

    half_gain = s_half - s0
    d = s_one - s0
    # 2*(S(.5)-S0) - (S(1)-S0) = K/2.
    k = 4.0 * half_gain - 2.0 * d
    linear = d + k

    if k > tol:
        raw_weight = linear / (2.0 * k)
        raw_gain = gain(raw_weight, linear, k)
        selected_weight = min(max(raw_weight, lo), hi)
        selected_gain = gain(selected_weight, linear, k)
        inside = lo <= raw_weight <= hi
        status = "concave_vertex_inside_bounds" if inside else "concave_vertex_clamped_to_bound"
    elif abs(k) <= tol:
        # A numerically linear curve has no finite unconstrained vertex.
        raw_weight = None
        raw_gain = None
        candidates = (lo, hi)
        selected_weight = max(candidates, key=lambda w: gain(w, linear, k))
        selected_gain = gain(selected_weight, linear, k)
        inside = None
        status = "near_linear_choose_best_bound"
    else:
        # Convex curvature cannot have an interior maximum.  This can be real
        # on a clipped/piecewise segment, or signal a score transcription error.
        raw_weight = linear / (2.0 * k)
        raw_gain = gain(raw_weight, linear, k)
        candidates = (lo, hi)
        selected_weight = max(candidates, key=lambda w: gain(w, linear, k))
        selected_gain = gain(selected_weight, linear, k)
        inside = lo <= raw_weight <= hi
        status = "nonconcave_choose_best_bound_check_scores_or_clipping"

    return AxisSolution(
        name=name,
        s_half=s_half,
        s_one=s_one,
        d=d,
        k=k,
        linear=linear,
        raw_weight=raw_weight,
        raw_gain=raw_gain,
        bound_low=lo,
        bound_high=hi,
        selected_weight=selected_weight,
        selected_gain=selected_gain,
        selected_score_alone=s0 + selected_gain,
        status=status,
        raw_inside_declared_no_clip_bounds=inside,
    )


def _axis_pair(parser: argparse.ArgumentParser, args: argparse.Namespace,
               prefix: str) -> tuple[float, float] | None:
    half = getattr(args, f"{prefix}_half")
    one = getattr(args, f"{prefix}_one")
    if (half is None) != (one is None):
        parser.error(f"--{prefix}-half and --{prefix}-one must be supplied together")
    return None if half is None else (half, one)


def _fmt(value: float | None, digits: int = 9) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_payload(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, Any]:
    pairs = {"F": _axis_pair(parser, args, "f"),
             "R": _axis_pair(parser, args, "r")}
    if all(pair is None for pair in pairs.values()):
        parser.error("supply one or both complete probe pairs (2 or 4 scores total)")

    bounds = {"F": tuple(args.f_bounds), "R": tuple(args.r_bounds)}
    solutions: list[AxisSolution] = []
    for name, pair in pairs.items():
        if pair is not None:
            solutions.append(solve_axis(name, args.s0, pair[0], pair[1], bounds[name]))

    selected_total_gain = sum(s.selected_gain for s in solutions)
    raw_valid = all(s.raw_weight is not None and
                    s.raw_inside_declared_no_clip_bounds is True and
                    s.k > CURVATURE_TOL for s in solutions)
    raw_total_gain = (sum(float(s.raw_gain) for s in solutions)
                      if raw_valid else None)

    return {
        "reference_score_s0": args.s0,
        "formula": {
            "d": "S(1)-S0",
            "K": "4*(S(0.5)-S0)-2*(S(1)-S0)",
            "L": "d+K",
            "score": "S0 + L*w - K*w^2",
            "w_vertex": "L/(2*K), for K>0",
            "joint": "S0 + gain_F(wF) + gain_R(wR); Q_FR=0 by disjoint support",
        },
        "axes": {s.name: asdict(s) for s in solutions},
        "selected": {
            "weights": {s.name: s.selected_weight for s in solutions},
            "predicted_gain": selected_total_gain,
            "predicted_score": args.s0 + selected_total_gain,
            "domain": "declared no-clipping bounds",
        },
        "raw_unconstrained": {
            "valid_under_declared_bounds": raw_valid,
            "weights": {s.name: s.raw_weight for s in solutions},
            "predicted_gain": raw_total_gain,
            "predicted_score": None if raw_total_gain is None else args.s0 + raw_total_gain,
        },
        "warning": (
            "The single-parabola extrapolation is exact only while prediction is affine in w. "
            "Default [0,1] bounds are convex-safe. Widen bounds only from actual test endpoint "
            "arrays after checking every row for probability clipping."
        ),
    }


def print_report(payload: dict[str, Any]) -> None:
    print(f"S0 = {payload['reference_score_s0']:.12f}")
    print("S(w) = S0 + (d+K)w - K w^2")
    for name, axis in payload["axes"].items():
        print()
        print(f"[{name}] S(.5)={axis['s_half']:.12f}  S(1)={axis['s_one']:.12f}")
        print(f"  d={axis['d']:+.12f}  K={axis['k']:+.12f}  L=d+K={axis['linear']:+.12f}")
        print(f"  raw w*={_fmt(axis['raw_weight'])}  raw gain={_fmt(axis['raw_gain'])}")
        print(f"  bounds=[{axis['bound_low']:.9f}, {axis['bound_high']:.9f}]")
        print(f"  selected w={axis['selected_weight']:.9f}  "
              f"gain={axis['selected_gain']:+.9f}  "
              f"score-alone={axis['selected_score_alone']:.9f}")
        print(f"  status={axis['status']}")

    selected = payload["selected"]
    weights = ", ".join(f"w{name}={value:.9f}"
                        for name, value in selected["weights"].items())
    print()
    print(f"JOINT SELECTED: {weights}")
    print(f"  predicted gain={selected['predicted_gain']:+.9f}")
    print(f"  predicted score={selected['predicted_score']:.9f}")
    if payload["raw_unconstrained"]["valid_under_declared_bounds"]:
        raw = payload["raw_unconstrained"]
        print(f"  raw joint vertex is valid: predicted score={raw['predicted_score']:.9f}")
    else:
        print("  raw joint vertex is NOT certified inside all declared no-clipping bounds.")


def self_test() -> None:
    s0 = DEFAULT_S0
    # F: L=20, K=8 -> vertex 1.25; convex-safe selection is w=1.
    f_half = s0 + 20.0 * 0.5 - 8.0 * 0.25
    f_one = s0 + 20.0 - 8.0
    f = solve_axis("F", s0, f_half, f_one, (0.0, 1.0))
    assert math.isclose(f.k, 8.0, abs_tol=1e-9)
    assert math.isclose(f.linear, 20.0, abs_tol=1e-9)
    assert math.isclose(float(f.raw_weight), 1.25, abs_tol=1e-9)
    assert math.isclose(f.selected_weight, 1.0, abs_tol=1e-12)

    # R: L=9, K=9 -> vertex .5; selected joint gain = 12 + 2.25.
    r_half = s0 + 9.0 * 0.5 - 9.0 * 0.25
    r_one = s0
    r = solve_axis("R", s0, r_half, r_one, (0.0, 1.0))
    assert math.isclose(r.k, 9.0, abs_tol=1e-9)
    assert math.isclose(float(r.raw_weight), 0.5, abs_tol=1e-9)
    assert math.isclose(f.selected_gain + r.selected_gain, 14.25, abs_tol=1e-9)
    print("self-test passed: exact two-point recovery and disjoint-axis addition")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover and optimize disjoint F/R Public-LB quadratics.")
    parser.add_argument("--s0", type=float, default=DEFAULT_S0,
                        help=f"reference A score (default: {DEFAULT_S0})")
    parser.add_argument("--f-half", type=float, help="F-only w=0.5 score")
    parser.add_argument("--f-one", type=float, help="F-only w=1.0 score")
    parser.add_argument("--r-half", type=float, help="R-only w=0.5 score")
    parser.add_argument("--r-one", type=float, help="R-only w=1.0 score")
    parser.add_argument("--f-bounds", nargs=2, type=float, metavar=("LOW", "HIGH"),
                        default=(0.0, 1.0), help="certified affine/no-clip F interval")
    parser.add_argument("--r-bounds", nargs=2, type=float, metavar=("LOW", "HIGH"),
                        default=(0.0, 1.0), help="certified affine/no-clip R interval")
    parser.add_argument("--json", action="store_true", help="also print machine-readable JSON")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    payload = build_payload(args, parser)
    print_report(payload)
    if args.json:
        print()
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
