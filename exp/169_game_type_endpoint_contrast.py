"""Forward-only F/R-specific mixing of the two deployed endpoints.

The public-LB optimum fixes the global champion/EXP021 weight.  This screen asks
one narrow question outside that one-dimensional line: does the *difference*
between the endpoints need a different coefficient for F and R rows?

Coefficients are fitted once on 2023 OOF residuals and transferred unchanged to
2024.  The 2024 labels are confirmation-only.  No test data or submission is
read or written.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train.csv"
LAB = ROOT / "lab"
OUT_JSON = LAB / "169_game_type_endpoint_contrast.json"
OUT_TXT = LAB / "169_game_type_endpoint_contrast.txt"

EXP_ROOT = (
    LAB / "115_mkis_exp021_work" / "artifacts" / "EXP-020"
    / "low_rank_pitcher_context_eb"
)
CAT5_BASE = {
    2023: LAB / "122_y2023_seed42_base_final.npy",
    2024: LAB / "89_cat5_probs_seed42.npy",
}
V18_EFFECT = {
    2023: LAB / "104_v18_effect_y2023.npy",
    2024: LAB / "103_v18_effect_2024.npy",
}
EXP021 = {
    year: EXP_ROOT / f"predictions_lowrank_s300_r6_{year}.npy"
    for year in (2023, 2024)
}
EXP_TARGET = {
    year: EXP_ROOT / f"targets_{year}.npy" for year in (2023, 2024)
}

W2 = 0.35655716947524263
W1 = 1.0 - W2
V18_GAMMA = 0.30
AFFINE_CENTER = 0.49
AFFINE_SCALE = 1.06
AFFINE_SHIFT = -0.0066
SCORE_SCALE = 100000.0


def probability(path: Path, rows: int) -> np.ndarray:
    value = np.load(path, allow_pickle=False).astype(np.float64)
    if value.ndim == 2:
        value = value[:, 0]
    if value.shape != (rows,) or not np.isfinite(value).all():
        raise ValueError(f"invalid array {path}: {value.shape}")
    return value


def score(p: np.ndarray, y: np.ndarray) -> float:
    rate = float(y.mean())
    return float(SCORE_SCALE * (1.0 - np.mean((p - y) ** 2) / (rate * (1.0 - rate))))


def champion(base: np.ndarray, effect: np.ndarray) -> np.ndarray:
    corrected = np.clip(base + V18_GAMMA * effect, 0.0, 1.0)
    return np.clip(
        AFFINE_CENTER + AFFINE_SCALE * (corrected - AFFINE_CENTER) + AFFINE_SHIFT,
        0.0,
        1.0,
    )


def load_year(train: pd.DataFrame, year: int) -> dict[str, np.ndarray]:
    rows = train.loc[train["season"].eq(year)].reset_index(drop=True)
    y = rows["control_success"].to_numpy(np.float64)
    target = probability(EXP_TARGET[year], len(rows))
    if not np.array_equal(y, target):
        raise ValueError(f"target alignment failure in {year}")
    p1 = champion(
        probability(CAT5_BASE[year], len(rows)),
        probability(V18_EFFECT[year], len(rows)),
    )
    p2 = probability(EXP021[year], len(rows))
    current = W1 * p1 + W2 * p2
    return {
        "y": y,
        "p1": p1,
        "p2": p2,
        "current": current,
        "direction": p2 - p1,
        "game": rows["game_type"].astype(str).to_numpy(),
        "pitcher": rows["pitcher_id"].astype(str).to_numpy(),
    }


def fit_group_beta(data: dict[str, np.ndarray], centered: bool) -> dict[str, float]:
    result: dict[str, float] = {}
    for group in ("F", "R"):
        mask = data["game"] == group
        d = data["direction"][mask].copy()
        if centered:
            d -= d.mean()
        residual = data["y"][mask] - data["current"][mask]
        denominator = float(d @ d)
        result[group] = float((d @ residual) / denominator) if denominator else 0.0
    return result


def correction(
    data: dict[str, np.ndarray], beta: dict[str, float], centered: bool, shrink: float
) -> np.ndarray:
    out = np.zeros_like(data["current"])
    for group in ("F", "R"):
        mask = data["game"] == group
        d = data["direction"][mask].copy()
        if centered:
            d -= d.mean()
        out[mask] = shrink * beta[group] * d
    return out


def diagnostics(data: dict[str, np.ndarray], corr: np.ndarray) -> dict[str, float]:
    base = score(data["current"], data["y"])
    candidate = np.clip(data["current"] + corr, 0.0, 1.0)
    result = {
        "base": base,
        "candidate": score(candidate, data["y"]),
        "gain": score(candidate, data["y"]) - base,
        "mean_shift": float(candidate.mean() - data["current"].mean()),
        "effect_std": float(corr.std()),
    }
    for group in ("F", "R"):
        mask = data["game"] == group
        result[f"{group}_contribution"] = float(
            SCORE_SCALE
            * (
                np.sum((data["current"][mask] - data["y"][mask]) ** 2)
                - np.sum((candidate[mask] - data["y"][mask]) ** 2)
            )
            / (len(data["y"]) * data["y"].mean() * (1.0 - data["y"].mean()))
        )
    half = len(candidate) // 2
    for label, indices in (
        ("early", np.arange(half)),
        ("late", np.arange(half, len(candidate))),
    ):
        y = data["y"][indices]
        result[f"{label}_gain"] = score(candidate[indices], y) - score(
            data["current"][indices], y
        )
    return result


def pitcher_bootstrap(
    data: dict[str, np.ndarray], corr: np.ndarray, draws: int = 3000
) -> dict[str, float]:
    candidate = np.clip(data["current"] + corr, 0.0, 1.0)
    y = data["y"]
    denom = float(y.mean() * (1.0 - y.mean()))
    delta = (data["current"] - y) ** 2 - (candidate - y) ** 2
    frame = pd.DataFrame({"pitcher": data["pitcher"], "delta": delta})
    grouped = frame.groupby("pitcher", sort=False)["delta"].agg(["sum", "size"])
    sums = grouped["sum"].to_numpy(np.float64)
    sizes = grouped["size"].to_numpy(np.int64)
    rng = np.random.default_rng(20260901)
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        take = rng.integers(0, len(sums), size=len(sums))
        values[draw] = SCORE_SCALE * sums[take].sum() / (sizes[take].sum() * denom)
    return {
        "p025": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "p975": float(np.quantile(values, 0.975)),
        "prob_positive": float(np.mean(values > 0.0)),
    }


def main() -> None:
    train = pd.read_csv(
        TRAIN,
        usecols=["season", "control_success", "game_type", "pitcher_id"],
        low_memory=False,
    )
    source = load_year(train, 2023)
    confirm = load_year(train, 2024)
    report: dict[str, object] = {
        "protocol": "fit coefficients on 2023; transfer unchanged to 2024",
        "rows": {"2023": len(source["y"]), "2024": len(confirm["y"])},
        "variants": {},
    }
    lines = ["EXP169 game-type endpoint contrast", report["protocol"], ""]
    for centered in (False, True):
        name = "centered" if centered else "raw"
        beta = fit_group_beta(source, centered)
        for shrink in (0.25, 0.50, 0.75, 1.00):
            key = f"{name}_s{shrink:.2f}"
            source_corr = correction(source, beta, centered, shrink)
            confirm_corr = correction(confirm, beta, centered, shrink)
            source_diag = diagnostics(source, source_corr)
            confirm_diag = diagnostics(confirm, confirm_corr)
            item = {
                "centered": centered,
                "shrink": shrink,
                "locked_beta": beta,
                "discovery_2023": source_diag,
                "confirmation_2024": confirm_diag,
                "pitcher_bootstrap_2024": pitcher_bootstrap(confirm, confirm_corr),
            }
            report["variants"][key] = item
            boot = item["pitcher_bootstrap_2024"]
            lines.append(
                f"{key:16s} betaF={beta['F']:+.4f} betaR={beta['R']:+.4f} "
                f"2023={source_diag['gain']:+.3f} 2024={confirm_diag['gain']:+.3f} "
                f"F/R={confirm_diag['F_contribution']:+.3f}/{confirm_diag['R_contribution']:+.3f} "
                f"early/late={confirm_diag['early_gain']:+.3f}/{confirm_diag['late_gain']:+.3f} "
                f"CI=[{boot['p025']:+.3f},{boot['p975']:+.3f}]"
            )
    eligible = []
    for key, item in report["variants"].items():
        c = item["confirmation_2024"]
        b = item["pitcher_bootstrap_2024"]
        if (
            c["gain"] >= 8.0
            and c["F_contribution"] >= 0.0
            and c["R_contribution"] >= 0.0
            and c["early_gain"] >= 0.0
            and c["late_gain"] >= 0.0
            and b["p025"] >= 0.0
        ):
            eligible.append(key)
    report["eligible"] = eligible
    report["status"] = "PASS" if eligible else "FAIL_NO_DEPLOY"
    lines += ["", f"eligible={eligible}", f"status={report['status']}"]
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
