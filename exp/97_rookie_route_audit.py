# -*- coding: utf-8 -*-
"""
[97] 2024 first-seen pitcher routing audit (training-free).

The champion stays unchanged outside the 2024 first-seen-pitcher rows.  On
those rows, sweep champion/partner blends and report the gain on the *full*
253,507-row score scale.  Confidence intervals resample pitchers, not rows.

Inputs are aligned artifacts produced by exp/89 and exp/90.  Optional future
partners can be added to CANDIDATE_FILES without changing the scoring logic.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS = Path("lab/90_analysis_2024.parquet")
CHAMPION = Path("lab/89_cat5.npy")
BOOTSTRAPS = 5000
BOOTSTRAP_SEED = 20260829
WEIGHTS = (0.25, 0.50, 0.75, 1.00)

CANDIDATE_COLUMNS = {
    "MC04": "p_mc04",
    "BIN04": "p_bin04",
}
CANDIDATE_FILES = {
    "M4": Path("lab/93_M4.npy"),
    "OH": Path("lab/93_OH.npy"),
    "CTR1": Path("lab/93_CTR1.npy"),
    # Written by exp/98 when the specialist run is complete.
    "rookie_expert": Path("lab/98_rookie_expert.npy"),
    # Reserved for a future cat5 model that removes pitcher_id only.
    "cat5_no_pitcher": Path("lab/98_cat5_no_pitcher.npy"),
}


def raw_score(pred, target):
    pred = np.asarray(pred, dtype="float64")
    target = np.asarray(target, dtype="float64")
    rate = float(target.mean())
    return 100000.0 * (1.0 - np.mean((pred - target) ** 2) / (rate * (1.0 - rate)))


def load_vector(path, n_rows):
    pred = np.load(path).astype("float64")
    if pred.ndim == 2:
        pred = pred[:, 0]
    if pred.shape != (n_rows,):
        raise ValueError(f"{path}: expected {(n_rows,)}, got {pred.shape}")
    if not np.isfinite(pred).all():
        raise ValueError(f"{path}: predictions contain NaN/inf")
    return pred


def optimal_weight(d_gain, diversity_k):
    if diversity_k <= 0:
        return float(d_gain > 0)
    return float(np.clip(0.5 + d_gain / (2.0 * diversity_k), 0.0, 1.0))


def global_gain(base, partner, target, mask, weight, denominator):
    blended = base[mask] + weight * (partner[mask] - base[mask])
    row_gain = ((base[mask] - target[mask]) ** 2
                - (blended - target[mask]) ** 2)
    return 100000.0 * float(row_gain.sum()) / (len(target) * denominator)


def pitcher_bootstrap(base, partner, target, mask, pitcher_id, weight, denominator):
    b = base[mask]
    p = partner[mask]
    y = target[mask]
    pid = pitcher_id[mask]
    blended = b + weight * (p - b)
    row_gain = (b - y) ** 2 - (blended - y) ** 2

    _, inverse = np.unique(pid, return_inverse=True)
    n_pitchers = int(inverse.max()) + 1
    counts = np.bincount(inverse).astype("float64")
    gains = np.bincount(inverse, weights=row_gain, minlength=n_pitchers)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, n_pitchers, size=(BOOTSTRAPS, n_pitchers))
    sampled_mean = gains[draws].sum(axis=1) / counts[draws].sum(axis=1)
    rookie_share = float(mask.mean())
    values = 100000.0 * rookie_share * sampled_mean / denominator
    lo, med, hi = np.quantile(values, [0.025, 0.5, 0.975])
    return float(lo), float(med), float(hi), float(np.mean(values > 0))


def main():
    cols = ["y", "pitcher_id", "p_first_season", "asof_pitcher_n",
            *CANDIDATE_COLUMNS.values()]
    frame = pd.read_parquet(ANALYSIS, columns=cols)
    n_rows = len(frame)
    target = frame["y"].to_numpy("float64")
    pitcher_id = frame["pitcher_id"].to_numpy()
    rookie = frame["p_first_season"].to_numpy() == 2024
    champion = load_vector(CHAMPION, n_rows)

    # lab/24b is an independent alignment check made by the old no-ID harness.
    old = np.load("lab/24b_preds.npz")
    if not np.array_equal(target.astype("int8"), old["2024_y"]):
        raise ValueError("2024 target order differs between exp/24b and exp/90")
    if not np.array_equal(rookie, old["2024_meta_p_new"]):
        raise ValueError("2024 rookie mask differs between exp/24b and exp/90")

    rate = float(target.mean())
    denominator = rate * (1.0 - rate)
    champion_score = raw_score(champion, target)
    rookie_rate = float(target[rookie].mean())
    print("=== 2024 first-seen-pitcher routing audit ===")
    print(f"rows={n_rows:,} rookie_rows={rookie.sum():,} ({rookie.mean():.2%}) "
          f"rookie_pitchers={np.unique(pitcher_id[rookie]).size} "
          f"r_all={rate:.6f} r_rookie={rookie_rate:.6f}")
    print(f"champion=cat5 score={champion_score:.2f}")
    print("All gains below are global-equivalent points on the full 2024 score.")

    candidates = {name: frame[col].to_numpy("float64")
                  for name, col in CANDIDATE_COLUMNS.items()}
    for name, path in CANDIDATE_FILES.items():
        if path.exists():
            candidates[name] = load_vector(path, n_rows)
    # Old HGB prediction is diagnostic only; it is not the requested CatBoost partner.
    candidates["HGB65_no_pitcher"] = old["2024_no_pitcher"].astype("float64")

    scale = 100000.0 / (n_rows * denominator)
    for name, partner in candidates.items():
        if partner.shape != champion.shape:
            raise ValueError(f"{name}: alignment length {partner.shape} != {champion.shape}")
        d_gain = scale * float(np.sum(
            rookie * ((champion - target) ** 2 - (partner - target) ** 2)))
        diversity_k = scale * float(np.sum(rookie * (partner - champion) ** 2))
        weight = optimal_weight(d_gain, diversity_k)
        optimum = global_gain(champion, partner, target, rookie, weight, denominator)
        swept = [global_gain(champion, partner, target, rookie, w, denominator)
                 for w in WEIGHTS]
        print("")
        print(f"[{name}] d={d_gain:+.2f} K={diversity_k:.2f} "
              f"w*={weight:.3f} optimum={optimum:+.2f}")
        print(f"  rookie means: y={target[rookie].mean():.5f} "
              f"champion={champion[rookie].mean():.5f} "
              f"partner={partner[rookie].mean():.5f}")
        print("  sweep " + " ".join(
            f"w={w:.2f}:{g:+.2f}" for w, g in zip(WEIGHTS, swept)))
        for label, bootstrap_weight in [
                *[(f"w={fixed:.2f}", fixed) for fixed in WEIGHTS],
                ("w*=diagnostic", weight)]:
            lo, med, hi, prob = pitcher_bootstrap(
                champion, partner, target, rookie, pitcher_id,
                bootstrap_weight, denominator)
            print(f"  pitcher-bootstrap {label}: 95% [{lo:+.2f}, {hi:+.2f}] "
                  f"median={med:+.2f} P(gain>0)={prob:.3f}")
        print("  note: w* is selected on these labels; fixed-weight intervals are the "
              "less selection-biased evidence.")

        # For first-seen pitchers, official career n is also current-season n.
        # Bucket-specific optima are exploratory (label-selected), but expose a
        # specialist that only helps during the earliest cold-start window.
        n_prior = frame.loc[rookie, "asof_pitcher_n"].fillna(0).to_numpy("float64")
        bucket_edges = (0.0, 25.0, 100.0, 300.0, np.inf)
        labels = ("0-24", "25-99", "100-299", "300+")
        pieces = []
        for left, right, label in zip(bucket_edges[:-1], bucket_edges[1:], labels):
            local = (n_prior >= left) & (n_prior < right)
            full_mask = np.zeros(n_rows, dtype=bool)
            full_mask[np.flatnonzero(rookie)[local]] = True
            bucket_d = scale * float(np.sum(
                full_mask * ((champion - target) ** 2 - (partner - target) ** 2)))
            bucket_k = scale * float(np.sum(
                full_mask * (partner - champion) ** 2))
            bucket_w = optimal_weight(bucket_d, bucket_k)
            bucket_gain = global_gain(
                champion, partner, target, full_mask, bucket_w, denominator)
            pieces.append(f"{label}:n={local.sum():,},d={bucket_d:+.2f},K={bucket_k:.2f},"
                          f"w*={bucket_w:.2f},g={bucket_gain:+.2f}")
        print("  asof_pitcher_n bucket optima (exploratory): " + " | ".join(pieces))


if __name__ == "__main__":
    main()
