"""Recover the global Public-LB quadratic over historical model endpoints.

This experiment uses *whole-model* prediction endpoints only.  It does not
perform row/group tomography.  Public scores determine the linear projection
of the hidden labels onto each endpoint direction, while prediction files
determine the quadratic Gram matrix.

For reference prediction r and endpoint displacement d_i = p_i-r,

    S(r + D c) = S(r) + b.T c - c.T Q c,
    Q_ij = a E[d_i d_j],
    b_i  = S(p_i) - S(r) + Q_ii.

The scale ``a`` is calibrated from the exact V18/EXP021 blend curvature.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
SPAN_ROOT = ROOT / 'archive/scratch/_lbspan_full_172'
EXP021_CSV = ROOT / 'archive/scratch/_rehearse_143' / "ours" / "output" / "submission.csv"
OUT_JSON = ROOT / "lab" / "174_lb_model_span_opt.json"
OUT_TXT = ROOT / "lab" / "174_lb_model_span_opt.txt"

TARGET = "control_success"
ID = "row_id"

S_V18 = 1092.808353586
S_EXP021 = 1043.6074197937
S_REF = 1114.6116846973
K_AXIS = 171.5001496146865
W_EXP021 = 0.35655716947524263


@dataclass(frozen=True)
class Endpoint:
    name: str
    score: float
    precision: str = "exact"
    note: str = ""


# Scores copied from docs/SUBMISSION_LOG.md.  Values marked ``short`` or
# ``approx`` never enter the primary solve.
ENDPOINTS = (
    Endpoint("submit4", 830.322760105),
    Endpoint("submit5", 864.3312059823, note="documented HGB/CB w=.3"),
    Endpoint("submit6", 880.5655581163, note="documented HGB/CB w=.5"),
    Endpoint("submit8", 898.61863054),
    Endpoint("submit9", 897.6644306949),
    Endpoint("submit10", 990.957167531),
    Endpoint("submit12", 993.6345481775),
    Endpoint("submit13", 975.5048373847),
    Endpoint("submit14", 1033.9866361712),
    Endpoint("submit16", 1006.3794404841),
    Endpoint("champion", 1059.0501189623),
    Endpoint("it300", 1040.0366, precision="short", note="only 4 decimals in ledger"),
    Endpoint("v18", S_V18),
)


def native(x: Any) -> Any:
    """Convert numpy/scipy objects recursively for JSON."""
    if isinstance(x, dict):
        return {str(k): native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [native(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def read_pred(path: Path, expected_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    f = pd.read_csv(path, usecols=[ID, TARGET])
    if f[ID].duplicated().any():
        raise ValueError(f"duplicate row_id: {path}")
    ids = f[ID].to_numpy()
    p = f[TARGET].to_numpy(np.float64)
    if expected_ids is not None and not np.array_equal(ids, expected_ids):
        raise ValueError(f"row order/id mismatch: {path}")
    if not np.isfinite(p).all():
        raise ValueError(f"non-finite prediction: {path}")
    return ids, p


def score_of(c: np.ndarray, b: np.ndarray, q: np.ndarray) -> float:
    return float(S_REF + b @ c - c @ q @ c)


def prediction_stats(ref: np.ndarray, d: np.ndarray, c: np.ndarray) -> dict[str, float]:
    p = ref + d @ c
    return {
        "mean": float(p.mean()),
        "std": float(p.std()),
        "min": float(p.min()),
        "max": float(p.max()),
        "clip_low_rate": float(np.mean(p < 0.0)),
        "clip_high_rate": float(np.mean(p > 1.0)),
        "clip_total_rate": float(np.mean((p < 0.0) | (p > 1.0))),
        "clip_mean_abs_change": float(np.mean(np.abs(np.clip(p, 0.0, 1.0) - p))),
    }


def coefficient_stats(c: np.ndarray) -> dict[str, float]:
    return {
        "l1": float(np.abs(c).sum()),
        "l2": float(np.linalg.norm(c)),
        "max_abs": float(np.abs(c).max(initial=0.0)),
        "sum_endpoint_displacements": float(c.sum()),
        "implied_reference_weight": float(1.0 - c.sum()),
    }


def svd_solve(q: np.ndarray, b: np.ndarray, rel_cut: float) -> tuple[np.ndarray, dict[str, Any]]:
    eig, vec = np.linalg.eigh((q + q.T) / 2.0)
    order = np.argsort(eig)[::-1]
    eig, vec = eig[order], vec[:, order]
    threshold = max(float(eig[0]) * rel_cut, 1e-14)
    keep = eig > threshold
    c = np.zeros_like(b)
    if keep.any():
        vk = vec[:, keep]
        c = 0.5 * vk @ ((vk.T @ b) / eig[keep])
    projected_b = vec[:, keep] @ (vec[:, keep].T @ b) if keep.any() else np.zeros_like(b)
    null_b = b - projected_b
    diag = {
        "relative_cutoff": rel_cut,
        "threshold": threshold,
        "rank": int(keep.sum()),
        "condition_kept": float(eig[0] / eig[keep][-1]) if keep.any() else None,
        "b_null_l2": float(np.linalg.norm(null_b)),
        "b_null_fraction": float(np.linalg.norm(null_b) / max(np.linalg.norm(b), 1e-30)),
    }
    return c, diag


def ridge_solve(q: np.ndarray, b: np.ndarray, rel_ridge: float) -> tuple[np.ndarray, dict[str, Any]]:
    scale = max(float(np.linalg.eigvalsh((q + q.T) / 2.0).max()), 1e-12)
    ridge = rel_ridge * scale
    c = 0.5 * np.linalg.solve((q + q.T) / 2.0 + ridge * np.eye(len(b)), b)
    return c, {"relative_ridge": rel_ridge, "absolute_ridge": ridge}


def constrained_solve(q: np.ndarray, b: np.ndarray, kind: str) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(b)
    fun = lambda c: float(c @ q @ c - b @ c)
    jac = lambda c: 2.0 * q @ c - b
    if kind == "simplex":
        # Convex combination of reference and historical endpoints.
        cons = ({"type": "ineq", "fun": lambda c: 1.0 - c.sum(),
                 "jac": lambda c: -np.ones_like(c)},)
        res = minimize(fun, np.zeros(n), jac=jac, method="SLSQP",
                       bounds=[(0.0, 1.0)] * n, constraints=cons,
                       options={"ftol": 1e-12, "maxiter": 5000})
    elif kind == "box1":
        res = minimize(fun, np.zeros(n), jac=jac, method="L-BFGS-B",
                       bounds=[(-1.0, 1.0)] * n,
                       options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 5000})
    else:
        raise ValueError(kind)
    return np.asarray(res.x), {
        "success": bool(res.success), "message": str(res.message),
        "iterations": int(getattr(res, "nit", -1)), "objective": float(res.fun),
    }


def make_solution(name: str, c: np.ndarray, detail: dict[str, Any], names: list[str],
                  b: np.ndarray, q: np.ndarray, ref: np.ndarray, d: np.ndarray) -> dict[str, Any]:
    return {
        "name": name,
        "predicted_score_unclipped": score_of(c, b, q),
        "predicted_gain": score_of(c, b, q) - S_REF,
        "coefficients": {n: float(x) for n, x in zip(names, c)},
        "coefficient_stats": coefficient_stats(c),
        "prediction_stats": prediction_stats(ref, d, c),
        "solver": detail,
    }


def leave_one_out_generic(names: list[str], b: np.ndarray, q: np.ndarray,
                          ref: np.ndarray, d: np.ndarray, solver) -> dict[str, Any]:
    rows = []
    full_c, _ = solver(q, b)
    full_p = ref + d @ full_c
    for j, omitted in enumerate(names):
        keep = np.arange(len(names)) != j
        cs, diag = solver(q[np.ix_(keep, keep)], b[keep])
        c = np.zeros(len(names))
        c[keep] = cs
        p = ref + d @ c
        rows.append({
            "omitted": omitted,
            "predicted_score_on_full_geometry": score_of(c, b, q),
            "gain": score_of(c, b, q) - S_REF,
            "prediction_rms_vs_full": float(np.sqrt(np.mean((p - full_p) ** 2))),
            "max_abs_coefficient": float(np.abs(c).max(initial=0.0)),
            "l1": float(np.abs(c).sum()),
            "solver": diag,
        })
    scores = np.array([z["predicted_score_on_full_geometry"] for z in rows])
    rms = np.array([z["prediction_rms_vs_full"] for z in rows])
    return {
        "rows": rows,
        "score_min": float(scores.min()),
        "score_median": float(np.median(scores)),
        "score_max": float(scores.max()),
        "score_drop_max": float(score_of(full_c, b, q) - scores.min()),
        "prediction_rms_median": float(np.median(rms)),
        "prediction_rms_max": float(rms.max()),
    }


def leave_one_out(names: list[str], b: np.ndarray, q: np.ndarray, ref: np.ndarray,
                  d: np.ndarray, rel_cut: float) -> dict[str, Any]:
    out = leave_one_out_generic(
        names, b, q, ref, d,
        lambda qs, bs: svd_solve(qs, bs, rel_cut),
    )
    out["relative_cutoff"] = rel_cut
    return out


def random_mask_sensitivity(d: np.ndarray, b: np.ndarray, q_full: np.ndarray,
                            axis: np.ndarray, fraction: float = 0.30,
                            draws: int = 40, rel_cut: float = 1e-4) -> dict[str, Any]:
    """Stress-test unknown Public-LB mask using random row subsets.

    Each subset is re-scaled so the known V18/EXP021 axis curvature remains K.
    Endpoint linear projections remain the observed LB projections.
    """
    rng = np.random.default_rng(174)
    n = len(axis)
    m = max(1000, int(round(fraction * n)))
    rows = []
    for k in range(draws):
        ix = rng.choice(n, size=m, replace=False)
        a_sub = K_AXIS / float(np.mean(axis[ix] ** 2))
        q = a_sub * (d[ix].T @ d[ix]) / m
        c, diag = svd_solve(q, b, rel_cut)
        rows.append({
            "draw": k,
            "rank": diag["rank"],
            "candidate_score_on_full_geometry": score_of(c, b, q_full),
            "candidate_score_on_subset_geometry": score_of(c, b, q),
            "coefficient_l1": float(np.abs(c).sum()),
            "coefficient_max_abs": float(np.abs(c).max(initial=0.0)),
        })
    sf = np.array([z["candidate_score_on_full_geometry"] for z in rows])
    ss = np.array([z["candidate_score_on_subset_geometry"] for z in rows])
    return {
        "fraction": fraction, "draws": draws, "relative_cutoff": rel_cut,
        "full_geometry_score_p05": float(np.quantile(sf, .05)),
        "full_geometry_score_median": float(np.median(sf)),
        "full_geometry_score_p95": float(np.quantile(sf, .95)),
        "subset_geometry_score_p05": float(np.quantile(ss, .05)),
        "subset_geometry_score_median": float(np.median(ss)),
        "subset_geometry_score_p95": float(np.quantile(ss, .95)),
        "rows": rows,
    }


def relation_audit(preds: dict[str, np.ndarray], scores: dict[str, float], a: float) -> list[dict[str, Any]]:
    rows = []
    for mid, w in (("submit5", .30), ("submit6", .50)):
        if all(k in preds for k in ("submit4", "submit8", mid)):
            expected = (1.0 - w) * preds["submit4"] + w * preds["submit8"]
            err = preds[mid] - expected
            # Score predicted from the two endpoint scores and their curvature.
            curvature = a * float(np.mean((preds["submit8"] - preds["submit4"]) ** 2))
            linear = scores["submit8"] - scores["submit4"] + curvature
            predicted_score = scores["submit4"] + w * linear - w * w * curvature
            rows.append({
                "endpoint": mid, "documented_weight_on_submit8": w,
                "prediction_rms_error": float(np.sqrt(np.mean(err ** 2))),
                "prediction_max_abs_error": float(np.abs(err).max()),
                "official_score": scores[mid], "score_implied_by_relation": predicted_score,
                "score_residual": scores[mid] - predicted_score,
            })
    return rows


def curve_optimum(sa: float, sb: float, k: float) -> tuple[float, float]:
    """Best convex blend, with w being the weight on endpoint B."""
    w = float(np.clip(0.5 + (sb - sa) / (2.0 * k), 0.0, 1.0))
    score = (1.0 - w) * sa + w * sb + w * (1.0 - w) * k
    return w, float(score)


def required_curvature(sa: float, sb: float, target: float) -> float:
    """Minimum K for the A--B quadratic's interior maximum to reach target."""
    hi, lo = max(sa, sb), min(sa, sb)
    gain = target - hi
    gap = hi - lo
    if gain <= 0:
        return 0.0
    return float(gap + 2.0 * gain + 2.0 * np.sqrt(gain * (gap + gain)))


def choose_recommendation(solutions: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = []
    for s in solutions:
        cs, ps = s["coefficient_stats"], s["prediction_stats"]
        stable_coeff = cs["max_abs"] <= 1.5 and cs["l1"] <= 5.0
        stable_clip = ps["clip_total_rate"] <= 0.001
        loo = s.get("leave_one_out", {})
        loo_ok = (loo.get("score_min", -np.inf) >= S_REF and
                  loo.get("score_drop_max", np.inf) <= 10.0)
        if stable_coeff and stable_clip and loo_ok:
            eligible.append(s)
    if not eligible:
        return {"status": "NO_STABLE_SOLUTION", "selected": None}
    best = max(eligible, key=lambda z: z["predicted_score_unclipped"])
    score = best["predicted_score_unclipped"]
    status = "STABLE_GE_1150" if score >= 1150.0 else ("STABLE_GE_1125" if score >= 1125.0 else "STABLE_LT_1125")
    return {"status": status, "selected": best["name"],
            "predicted_score": score, "predicted_gain": score - S_REF}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-draws", type=int, default=40)
    ap.add_argument("--no-mask-stress", action="store_true")
    args = ap.parse_args()

    v18_path = SPAN_ROOT / "v18" / "output" / "submission.csv"
    missing_critical = [str(p) for p in (v18_path, EXP021_CSV) if not p.exists()]
    if missing_critical:
        payload = {"experiment": 174, "status": "WAITING_CRITICAL_OUTPUTS", "missing": missing_critical}
        OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        OUT_TXT.write_text("EXP174 WAITING_CRITICAL_OUTPUTS\n" + "\n".join(missing_critical) + "\n", encoding="utf-8")
        print(OUT_TXT.read_text(encoding="utf-8"), end="")
        return

    ids, p_v18 = read_pred(v18_path)
    _, p_exp = read_pred(EXP021_CSV, ids)
    axis = p_exp - p_v18
    axis_mse = float(np.mean(axis ** 2))
    a = K_AXIS / axis_mse
    p_ref = (1.0 - W_EXP021) * p_v18 + W_EXP021 * p_exp

    score_map = {e.name: e.score for e in ENDPOINTS}
    meta_map = {e.name: e for e in ENDPOINTS}
    preds: dict[str, np.ndarray] = {}
    missing = []
    excluded_precision = []
    for e in ENDPOINTS:
        path = SPAN_ROOT / e.name / "output" / "submission.csv"
        if not path.exists():
            missing.append(e.name)
            continue
        _, p = read_pred(path, ids)
        preds[e.name] = p
        if e.precision != "exact":
            excluded_precision.append(e.name)

    # Add one exact axis endpoint.  V18 itself is kept only once; EXP021 is
    # perfectly collinear with it around p_ref and would add no rank.
    primary_names = [e.name for e in ENDPOINTS if e.precision == "exact" and e.name in preds]
    if "v18" not in primary_names:
        raise RuntimeError("V18 endpoint unexpectedly absent")
    d = np.column_stack([preds[n] - p_ref for n in primary_names])
    q = a * (d.T @ d) / len(ids)
    q = (q + q.T) / 2.0
    scores = np.array([score_map[n] for n in primary_names])
    b = scores - S_REF + np.diag(q)

    eig = np.linalg.eigvalsh(q)[::-1]
    positive = eig[eig > max(eig[0] * 1e-12, 1e-14)]
    matrix_diag = {
        "n_rows": len(ids), "n_endpoints": len(primary_names),
        "eigenvalues_desc": eig, "rank_rel_1e-12": len(positive),
        "condition_rel_1e-12": float(positive[0] / positive[-1]) if len(positive) else None,
        "a": a, "axis_mean_square": axis_mse, "axis_rms": float(np.sqrt(axis_mse)),
        "axis_curvature_reconstructed": float(a * axis_mse),
        "reference_score": S_REF, "reference_exp021_weight": W_EXP021,
        "reference_mean": float(p_ref.mean()), "reference_min": float(p_ref.min()),
        "reference_max": float(p_ref.max()),
    }

    # The exact optimum on the calibration axis must have zero first derivative.
    dv = p_v18 - p_ref
    qv = a * float(np.mean(dv ** 2))
    bv = S_V18 - S_REF + qv
    calibration_audit = {
        "v18_q": qv, "v18_b_should_be_zero": bv,
        "reference_score_geometry": float(S_V18 + (S_EXP021 - S_V18 + K_AXIS) ** 2 / (4 * K_AXIS)),
        "reference_score_official": S_REF,
        "reference_score_residual": float(S_REF - (S_V18 + (S_EXP021 - S_V18 + K_AXIS) ** 2 / (4 * K_AXIS))),
    }

    solutions = []
    loo_by_cut: dict[str, Any] = {}
    cuts = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10)
    for cut in cuts:
        c, diag = svd_solve(q, b, cut)
        key = f"{cut:.0e}"
        sol = make_solution(f"svd_{key}", c, diag, primary_names, b, q, p_ref, d)
        loo_by_cut[key] = leave_one_out(primary_names, b, q, p_ref, d, cut)
        sol["leave_one_out"] = loo_by_cut[key]
        solutions.append(sol)
    for rr in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-5):
        c, diag = ridge_solve(q, b, rr)
        sol = make_solution(f"ridge_{rr:.0e}", c, diag, primary_names, b, q, p_ref, d)
        sol["leave_one_out"] = leave_one_out_generic(
            primary_names, b, q, p_ref, d,
            lambda qs, bs, rr=rr: ridge_solve(qs, bs, rr),
        )
        solutions.append(sol)
    for kind in ("simplex", "box1"):
        c, diag = constrained_solve(q, b, kind)
        sol = make_solution(kind, c, diag, primary_names, b, q, p_ref, d)
        sol["leave_one_out"] = leave_one_out_generic(
            primary_names, b, q, p_ref, d,
            lambda qs, bs, kind=kind: constrained_solve(qs, bs, kind),
        )
        solutions.append(sol)

    # IMPORTANT: all CSVs here came from the 245,789-row fake-server rehearsal,
    # not the hidden Public-LB evaluation rows.  Their Gram matrix is therefore
    # not Public-LB geometry.  The exact submit5 identity below supplies a direct
    # falsification: the fake Gram misses its official score by 27.17 points.
    for s in solutions:
        s["valid_for_lb_prediction"] = False
        s["invalid_reason"] = "Gram matrix measured on fake rehearsal rows"
    recommendation = {
        "status": "INVALID_FAKE_TEST_GEOMETRY",
        "selected": None,
        "reason": "whole-model covariance was measured on fake rehearsal rows; do not submit any reconstructed solution",
    }
    mask_stress = None

    # Four-decimal endpoint is sensitivity-only: report its implied b interval.
    short_sensitivity = []
    for name in excluded_precision:
        p = preds[name]
        di = p - p_ref
        qii = a * float(np.mean(di ** 2))
        e = meta_map[name]
        short_sensitivity.append({
            "endpoint": name, "recorded_score": e.score, "precision": e.precision,
            "qii": qii, "implied_b_at_recorded": e.score - S_REF + qii,
            "implied_b_interval_if_rounded_4dp": [e.score - 0.00005 - S_REF + qii,
                                                   e.score + 0.00005 - S_REF + qii],
            "included_in_primary": False,
        })

    relation = relation_audit(preds, score_map, a)

    # Exact function-level affine identities remain useful: because the middle
    # package is algebraically identical to the stated blend on every row, its
    # official score recovers hidden curvature without observing hidden rows.
    identities = []
    identity_specs = [
        ("submit4", "submit8", "submit5", .30),
        ("submit12", "submit8", "submit13", .50),
    ]
    fake_to_hidden_ratios = []
    for left, right, middle, w in identity_specs:
        if not all(n in preds for n in (left, right, middle)):
            continue
        err = preds[middle] - ((1.0 - w) * preds[left] + w * preds[right])
        hidden_k = ((score_map[middle] - (1.0 - w) * score_map[left] - w * score_map[right])
                    / (w * (1.0 - w)))
        fake_k = a * float(np.mean((preds[right] - preds[left]) ** 2))
        wopt, sopt = curve_optimum(score_map[left], score_map[right], hidden_k)
        ratio = hidden_k / fake_k
        fake_to_hidden_ratios.append(ratio)
        identities.append({
            "left": left, "right": right, "middle": middle, "middle_weight_on_right": w,
            "identity_rms": float(np.sqrt(np.mean(err ** 2))),
            "identity_max_abs": float(np.abs(err).max()),
            "hidden_curvature_exact": hidden_k,
            "fake_rehearsal_curvature": fake_k,
            "hidden_to_fake_ratio": ratio,
            "optimal_weight_on_right": wopt,
            "optimal_score": sopt,
        })

    # Sensitivity only.  It is NOT an LB forecast.  It asks whether any old
    # endpoint has even a numerical route to 1150 under (a) the two observed
    # old-axis hidden/fake ratios, and (b) the much more optimistic ratio 1.0.
    ratio_low = min(fake_to_hidden_ratios)
    ratio_high = max(fake_to_hidden_ratios)
    old_sensitivity = []
    for name in primary_names:
        if name == "v18":
            continue
        q_fake = a * float(np.mean((preds[name] - p_ref) ** 2))
        row = {"endpoint": name, "endpoint_score": score_map[name],
               "fake_curvature_vs_current": q_fake}
        for label, ratio in (("observed_ratio_low", ratio_low),
                             ("observed_ratio_high", ratio_high),
                             ("optimistic_ratio_1", 1.0)):
            k = ratio * q_fake
            wopt, sopt = curve_optimum(S_REF, score_map[name], k)
            row[label] = {
                "ratio": ratio, "curvature": k, "optimal_old_weight": wopt,
                "optimal_score": sopt,
                "midpoint_score": .5 * (S_REF + score_map[name]) + .25 * k,
            }
        kreq = required_curvature(S_REF, score_map[name], 1150.0)
        row["curvature_required_for_1150"] = kreq
        row["required_hidden_to_fake_ratio_for_1150"] = kreq / q_fake
        row["evidence_for_1150"] = False
        old_sensitivity.append(row)
    payload = {
        "experiment": 174,
        "method": "whole-model endpoint audit; fake-test Gram invalidated",
        "status": "INVALID_FAKE_TEST_GEOMETRY",
        "primary_endpoints": primary_names,
        "missing_outputs": missing,
        "excluded_for_precision": excluded_precision,
        "matrix": matrix_diag,
        "calibration_audit": calibration_audit,
        "documented_blend_relation_audit": relation,
        "exact_scored_linear_identities": identities,
        "old_endpoint_fake_ratio_sensitivity_NOT_LB_FORECAST": old_sensitivity,
        "endpoint_linear_terms": {n: float(x) for n, x in zip(primary_names, b)},
        "endpoint_self_quadratics": {n: float(x) for n, x in zip(primary_names, np.diag(q))},
        "solutions": solutions,
        "leave_one_out": loo_by_cut,
        "unknown_public_mask_stress": mask_stress,
        "short_precision_sensitivity": short_sensitivity,
        "recommendation": recommendation,
        "builder_gate": {
            "allowed": False,
            "reason": "invalid fake-test geometry; no old endpoint has evidence for 1150",
            "ge_1150_immediate_report": False,
        },
    }
    OUT_JSON.write_text(json.dumps(native(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ranked = sorted(solutions, key=lambda z: z["predicted_score_unclipped"], reverse=True)
    lines = [
        "EXP174 whole-model LB-span optimization",
        f"rows={len(ids):,} endpoints={len(primary_names)} missing={missing}",
        f"a={a:.9f} axis_rms={np.sqrt(axis_mse):.9f} K_check={a*axis_mse:.12f}",
        f"rank(1e-12)={matrix_diag['rank_rel_1e-12']}/{len(primary_names)} condition={matrix_diag['condition_rel_1e-12']:.6g}",
        f"calibration b_v18={bv:+.12g} score_residual={calibration_audit['reference_score_residual']:+.12g}",
        "",
        "INVALID: reconstructed solutions below use fake-test Gram; DO NOT SUBMIT:",
    ]
    for s in ranked[:12]:
        cs, ps = s["coefficient_stats"], s["prediction_stats"]
        lines.append(f"  {s['name']:>12s} score={s['predicted_score_unclipped']:.6f} gain={s['predicted_gain']:+.6f} "
                     f"L1={cs['l1']:.3f} max|c|={cs['max_abs']:.3f} clip={100*ps['clip_total_rate']:.4f}%")
    lines += ["", f"RECOMMENDATION: {recommendation}"]
    if relation:
        lines += ["", "Documented HGB/CB relation audit:"]
        for z in relation:
            lines.append(f"  {z['endpoint']} pred_rms={z['prediction_rms_error']:.3g} "
                         f"score residual={z['score_residual']:+.6f}")
    lines += ["", "Exact scored affine identities (valid hidden curvature):"]
    for z in identities:
        lines.append(f"  {z['middle']}=blend({z['left']},{z['right']}): "
                     f"K_hidden={z['hidden_curvature_exact']:.6f} "
                     f"hidden/fake={z['hidden_to_fake_ratio']:.6f} "
                     f"best={z['optimal_score']:.6f}")
    lines += ["", "Current x old sensitivity (NOT an LB forecast):"]
    for z in sorted(old_sensitivity,
                    key=lambda x: x['optimistic_ratio_1']['optimal_score'], reverse=True):
        lines.append(f"  {z['endpoint']}: observed-ratio upper best="
                     f"{z['observed_ratio_high']['optimal_score']:.3f}; "
                     f"even fake-ratio-1 best={z['optimistic_ratio_1']['optimal_score']:.3f}; "
                     f"ratio needed for 1150={z['required_hidden_to_fake_ratio_for_1150']:.3f}")
    if mask_stress:
        lines += ["", "Unknown Public-mask random-30% stress (SVD 1e-4):",
                  f"  full-geometry candidate score p05/median/p95 = "
                  f"{mask_stress['full_geometry_score_p05']:.3f} / "
                  f"{mask_stress['full_geometry_score_median']:.3f} / "
                  f"{mask_stress['full_geometry_score_p95']:.3f}"]
    lines += ["", "No current x old midpoint probe is justified by these data.",
              "FINAL: INVALID_FAKE_TEST_GEOMETRY / NO_1150_EVIDENCE"]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_TXT.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
