# -*- coding: utf-8 -*-
"""Fast official-only std-z MLP endpoint, isolated from exp168.

Discovery is strictly <=2022 -> 2023.  A single small dual-head network is
trained with contiguous 65,536-row slices: one head predicts probability
directly from all historical rows, while the second learns a logit correction
around the frozen current OOF anchor on the latest training season (2022).
Only if a 2023 endpoint passes the predeclared gain/stability gate may the
unchanged recipe be refit <=2023 and checked on 2024.

The official feature construction and audit boundary are imported from exp168;
exp168 itself is never modified.  No test row or third-party artifact is read.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "lab"
TRAIN = ROOT / "data" / "train.csv"
STEM = "179_fast_stdmlp_endpoint"
LOCK = LAB / f"{STEM}_lock.json"
REPORT = LAB / f"{STEM}.json"
TEXT = LAB / f"{STEM}.txt"
THREADS = 6
SEED = 179
EPOCHS = 1
BATCH = 65536
PREDICT_BATCH = 65536
LR = 3.0e-3
WEIGHT_DECAY = 3.0e-4
GAIN_GATE = 20.0
BOOTSTRAP_DRAWS = 3000
SCORE_SCALE = 100000.0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E168 = load_module(ROOT / "exp" / "168_elephant_stdmlp_clean_oof.py", "e168_for_179")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(p: np.ndarray, y: np.ndarray) -> float:
    rate = float(y.mean())
    return float(SCORE_SCALE * (1.0 - np.mean(np.square(p - y)) / (rate * (1.0 - rate))))


def geometry(current: np.ndarray, endpoint: np.ndarray, y: np.ndarray) -> dict[str, float]:
    direction = endpoint - current
    ss = float(direction @ direction)
    raw_w = float(((y - current) @ direction) / ss) if ss > 0 else 0.0
    w = float(np.clip(raw_w, 0.0, 1.0))
    candidate = current + w * direction
    d = float(score(endpoint, y) - score(current, y))
    rate = float(y.mean())
    k = float(SCORE_SCALE * np.mean(np.square(direction)) / (rate * (1.0 - rate)))
    return {
        "d_endpoint_score_minus_current": d,
        "K_direction_curvature": k,
        "optimal_weight_unconstrained": raw_w,
        "locked_convex_weight": w,
        "current_score": score(current, y),
        "endpoint_score": score(endpoint, y),
        "optimal_blend_score": score(candidate, y),
        "optimal_blend_gain": float(score(candidate, y) - score(current, y)),
        "correlation": float(np.corrcoef(current, endpoint)[0, 1]),
    }


def latest_season_anchor(train_year: int, expected_rows: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frame = pd.read_csv(
        TRAIN,
        encoding="utf-8-sig",
        usecols=["season", "control_success"],
        low_memory=False,
    )
    frame.columns = [c.replace("\ufeff", "").strip() for c in frame.columns]
    seasons = pd.to_numeric(frame["season"], errors="coerce").to_numpy(np.int64)
    keep = seasons <= train_year
    seasons = seasons[keep]
    frame = frame.loc[keep].reset_index(drop=True)
    if len(frame) != expected_rows:
        raise AssertionError(f"training order mismatch: {len(frame)} != {expected_rows}")
    latest = seasons == train_year
    rows = frame.loc[latest, ["control_success"]].reset_index(drop=True)
    _, _, current = E168.load_current(rows, train_year)
    target = rows["control_success"].to_numpy(np.float64)
    anchor = np.full(expected_rows, np.nan, dtype=np.float32)
    anchor[latest] = current.astype(np.float32)
    return seasons, anchor, {
        "anchor_year": train_year,
        "anchor_rows": int(latest.sum()),
        "anchor_target_rate": float(target.mean()),
        "anchor_current_score": score(current, target),
    }


def make_network(torch, nn, d_in: int, target_rate: float):
    class FastDualMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(d_in, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Dropout(0.12),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Dropout(0.08),
            )
            self.direct = nn.Linear(64, 1)
            self.residual = nn.Linear(64, 1)
            nn.init.zeros_(self.residual.weight)
            nn.init.zeros_(self.residual.bias)
            nn.init.zeros_(self.direct.weight)
            bias = math.log(target_rate / (1.0 - target_rate))
            nn.init.constant_(self.direct.bias, bias)

        def forward(self, x):
            h = self.trunk(x)
            return self.direct(h).squeeze(-1), self.residual(h).squeeze(-1)

    return FastDualMLP()


def predict(network, matrix: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    network.eval()
    x = torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float32))
    direct_parts: list[np.ndarray] = []
    delta_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), PREDICT_BATCH):
            direct_logit, delta = network(x[start : start + PREDICT_BATCH])
            direct_parts.append(torch.sigmoid(direct_logit).numpy())
            delta_parts.append(delta.numpy())
    direct = np.concatenate(direct_parts).astype(np.float64)
    delta = np.concatenate(delta_parts).astype(np.float64)
    clipped = np.clip(current, 1e-6, 1.0 - 1e-6)
    base_logit = np.log(clipped / (1.0 - clipped))
    skip = 1.0 / (1.0 + np.exp(-np.clip(base_logit + delta, -30.0, 30.0)))
    return direct, skip


def train_fold(fold: dict[str, Any], train_year: int, current_valid: np.ndarray):
    import torch
    import torch.nn as nn

    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)

    x = torch.from_numpy(np.ascontiguousarray(fold["z_train"], dtype=np.float32))
    y = torch.from_numpy(np.ascontiguousarray(fold["y_train"], dtype=np.float32))
    _, anchor_np, anchor_meta = latest_season_anchor(train_year, len(x))
    anchor = torch.from_numpy(anchor_np)
    network = make_network(torch, nn, x.shape[1], float(fold["y_train"].mean()))
    optimizer = torch.optim.AdamW(network.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    batches = math.ceil(len(x) / BATCH)
    history: list[dict[str, Any]] = []
    all_started = time.time()

    for epoch in range(EPOCHS):
        network.train()
        epoch_started = time.time()
        direct_sum = residual_sum = 0.0
        direct_n = residual_n = 0
        starts = list(range(0, len(x), BATCH))
        if epoch % 2:
            starts.reverse()
        for batch_index, start in enumerate(starts):
            xb = x[start : start + BATCH]
            yb = y[start : start + BATCH]
            ab = anchor[start : start + BATCH]
            direct_logit, delta = network(xb)
            direct_probability = torch.sigmoid(direct_logit)
            direct_loss = torch.square(direct_probability - yb).mean()
            finite = torch.isfinite(ab)
            if bool(finite.any()):
                safe_anchor = torch.clamp(ab[finite], 1e-6, 1.0 - 1e-6)
                anchor_logit = torch.logit(safe_anchor)
                residual_probability = torch.sigmoid(anchor_logit + delta[finite])
                residual_loss = torch.square(residual_probability - yb[finite]).mean()
                loss = direct_loss + residual_loss
                count = int(finite.sum())
                residual_sum += float(residual_loss.detach()) * count
                residual_n += count
            else:
                loss = direct_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count_direct = int(len(xb))
            direct_sum += float(direct_loss.detach()) * count_direct
            direct_n += count_direct
            if batch_index == 1:
                two_batch = time.time() - epoch_started
                eta = two_batch * batches / 2.0
                print(
                    f"[speed] 2_batches={two_batch:.2f}s batches={batches} "
                    f"epoch_eta={eta:.1f}s total_eta={eta * EPOCHS / 60.0:.2f}m",
                    flush=True,
                )
        elapsed = time.time() - epoch_started
        direct, skip = predict(network, fold["z_valid"], current_valid)
        epoch_geometry = {
            "direct": geometry(current_valid, direct, fold["y_valid"]),
            "current_logit_residual_skip": geometry(current_valid, skip, fold["y_valid"]),
        }
        row = {
            "epoch": epoch + 1,
            "seconds": elapsed,
            "direct_train_brier": direct_sum / direct_n,
            "residual_train_brier_latest_season": residual_sum / max(residual_n, 1),
            "residual_rows_seen": residual_n,
            "geometry": epoch_geometry,
        }
        history.append(row)
        print(f"[epoch {epoch + 1}/{EPOCHS}] seconds={elapsed:.2f}", flush=True)
        for name, geo in epoch_geometry.items():
            print(
                f"[geometry] {name} d={geo['d_endpoint_score_minus_current']:+.4f} "
                f"K={geo['K_direction_curvature']:.4f} "
                f"w={geo['locked_convex_weight']:.6f} "
                f"gain={geo['optimal_blend_gain']:+.4f}",
                flush=True,
            )

    probe = min(32, len(fold["z_valid"]))
    batch_direct, batch_skip = predict(network, fold["z_valid"][:probe], current_valid[:probe])
    single_direct, single_skip = [], []
    for i in range(probe):
        d, s = predict(network, fold["z_valid"][i : i + 1], current_valid[i : i + 1])
        single_direct.append(d[0])
        single_skip.append(s[0])
    independence = max(
        float(np.max(np.abs(batch_direct - np.asarray(single_direct)), initial=0.0)),
        float(np.max(np.abs(batch_skip - np.asarray(single_skip)), initial=0.0)),
    )
    del network, x, y, anchor
    gc.collect()
    return direct, skip, history, anchor_meta, float(time.time() - all_started), independence


def subset_gain(current, candidate, target, mask) -> float:
    return float(score(candidate[mask], target[mask]) - score(current[mask], target[mask]))


def pitcher_bootstrap(current, candidate, target, pitcher) -> dict[str, float | int]:
    work = pd.DataFrame({
        "pitcher": pitcher,
        "n": 1,
        "y": target,
        "base_se": np.square(current - target),
        "cand_se": np.square(candidate - target),
    })
    grouped = work.groupby("pitcher", sort=False).agg(
        n=("n", "sum"), y=("y", "sum"), base_se=("base_se", "sum"), cand_se=("cand_se", "sum")
    ).to_numpy(np.float64)
    rng = np.random.default_rng(SEED)
    gains = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for start in range(0, BOOTSTRAP_DRAWS, 100):
        width = min(100, BOOTSTRAP_DRAWS - start)
        sampled = grouped[rng.integers(0, len(grouped), size=(width, len(grouped)))].sum(axis=1)
        rate = sampled[:, 1] / sampled[:, 0]
        gains[start : start + width] = SCORE_SCALE * (sampled[:, 2] - sampled[:, 3]) / (
            sampled[:, 0] * rate * (1.0 - rate)
        )
    return {
        "draws": BOOTSTRAP_DRAWS,
        "clusters": int(len(grouped)),
        "p025": float(np.quantile(gains, 0.025)),
        "median": float(np.median(gains)),
        "p975": float(np.quantile(gains, 0.975)),
        "prob_positive": float(np.mean(gains > 0.0)),
    }


def endpoint_diagnostics(name, current, endpoint, target, rows) -> dict[str, Any]:
    geo = geometry(current, endpoint, target)
    candidate = current + geo["locked_convex_weight"] * (endpoint - current)
    month = pd.to_numeric(rows["game_month"], errors="coerce").to_numpy(np.int64)
    pitcher = pd.to_numeric(rows["pitcher_id"], errors="coerce").fillna(-1).to_numpy(np.int64)
    early = subset_gain(current, candidate, target, month <= 6)
    late = subset_gain(current, candidate, target, month > 6)
    bootstrap = pitcher_bootstrap(current, candidate, target, pitcher)
    passed = bool(
        geo["optimal_blend_gain"] >= GAIN_GATE
        and early > 0.0
        and late > 0.0
        and bootstrap["p025"] > 0.0
    )
    return {
        "name": name,
        "geometry": geo,
        "early_gain": early,
        "late_gain": late,
        "pitcher_cluster_bootstrap": bootstrap,
        "gate": {
            "gain_at_least_20": geo["optimal_blend_gain"] >= GAIN_GATE,
            "early_positive": early > 0.0,
            "late_positive": late > 0.0,
            "pitcher_p025_positive": bootstrap["p025"] > 0.0,
            "passed": passed,
        },
        "candidate": candidate,
    }


def discovery() -> None:
    if LOCK.exists():
        raise FileExistsError(LOCK)
    LAB.mkdir(parents=True, exist_ok=True)
    started = time.time()
    audit = E168.source_audit()
    print("[discovery] official <=2022 -> 2023", flush=True)
    fold = E168.build_fold(2023)
    print(f"[features] seconds={fold['metadata']['feature_build_seconds']:.2f} shape={fold['z_train'].shape}", flush=True)
    _, _, current = E168.load_current(fold["rows"], 2023)
    direct, skip, history, anchor_meta, fit_seconds, independence = train_fold(fold, 2022, current)
    if independence > 1e-5:
        raise AssertionError(f"row independence failed: {independence}")
    diagnostics = [
        endpoint_diagnostics("direct_probability", current, direct, fold["y_valid"], fold["rows"]),
        endpoint_diagnostics("current_logit_residual_skip", current, skip, fold["y_valid"], fold["rows"]),
    ]
    eligible = [item for item in diagnostics if item["gate"]["passed"]]
    selected = max(eligible, key=lambda item: item["geometry"]["optimal_blend_gain"]) if eligible else None
    artifacts = {}
    for name, value in (("direct", direct), ("skip", skip), ("current", current)):
        path = LAB / f"179_{name}_2023.npy"
        np.save(path, np.asarray(value, np.float32), allow_pickle=False)
        artifacts[name] = {"path": str(path), "sha256": sha256(path), "rows": len(value)}
    payload = {
        "experiment": 179,
        "phase": "DISCOVERY_LOCKED_ON_2023_BEFORE_2024",
        "status": "ELIGIBLE_FOR_2024_CONFIRMATION" if selected else "FAIL_STOP_BEFORE_2024",
        "recipe": {
            "epochs": EPOCHS, "batch": BATCH, "threads": THREADS, "seed": SEED,
            "architecture": "160->128(LayerNorm/SiLU/dropout)->64(SiLU/dropout)->dual heads",
            "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
            "training_slices": "contiguous; no DataLoader; no sampling",
        },
        "gate": "gain>=20 AND early>0 AND late>0 AND pitcher bootstrap p025>0",
        "audit": audit,
        "fold": fold["metadata"],
        "anchor": anchor_meta,
        "fit_seconds": fit_seconds,
        "wall_seconds": time.time() - started,
        "epoch_history": history,
        "row_independence_max_abs": independence,
        "endpoints": [{k: v for k, v in item.items() if k != "candidate"} for item in diagnostics],
        "selected": None if selected is None else {
            "name": selected["name"],
            "locked_weight": selected["geometry"]["locked_convex_weight"],
        },
        "artifacts": artifacts,
        "confirmation_2024_opened": False,
        "test_read": False,
        "zip_created": False,
    }
    LOCK.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "=== exp179 fast std-z dual-head MLP discovery ===",
        f"status={payload['status']} fit_seconds={fit_seconds:.2f} row_independence={independence:.3e}",
    ]
    for item in diagnostics:
        g = item["geometry"]
        b = item["pitcher_cluster_bootstrap"]
        lines.append(
            f"{item['name']}: d={g['d_endpoint_score_minus_current']:+.4f} K={g['K_direction_curvature']:.4f} "
            f"w={g['locked_convex_weight']:.6f} gain={g['optimal_blend_gain']:+.4f} "
            f"early={item['early_gain']:+.4f} late={item['late_gain']:+.4f} p025={b['p025']:+.4f} "
            f"pass={item['gate']['passed']}"
        )
    lines.append("2024 unopened; test unopened; no ZIP")
    TEXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def confirm() -> None:
    if not LOCK.is_file():
        raise FileNotFoundError("run discovery first")
    if REPORT.exists():
        raise FileExistsError(REPORT)
    locked = json.loads(LOCK.read_text(encoding="utf-8"))
    if locked["status"] != "ELIGIBLE_FOR_2024_CONFIRMATION":
        raise RuntimeError("2023 discovery gate failed; 2024 is forbidden")
    selected = locked["selected"]
    if selected["name"] != "direct_probability":
        raise RuntimeError(f"unsupported locked endpoint: {selected['name']}")
    locked_weight = float(selected["locked_weight"])
    started = time.time()
    print(
        f"[confirm] locked endpoint=direct_probability weight={locked_weight:.12g}; "
        "official <=2023 -> 2024",
        flush=True,
    )
    fold = E168.build_fold(2024)
    print(
        f"[features] seconds={fold['metadata']['feature_build_seconds']:.2f} "
        f"shape={fold['z_train'].shape}",
        flush=True,
    )
    _, _, current = E168.load_current(fold["rows"], 2024)
    direct, skip, history, anchor_meta, fit_seconds, independence = train_fold(
        fold, 2023, current
    )
    if independence > 1e-5:
        raise AssertionError(f"row independence failed: {independence}")
    target = fold["y_valid"]
    candidate = current + locked_weight * (direct - current)
    month = pd.to_numeric(fold["rows"]["game_month"], errors="coerce").to_numpy(np.int64)
    pitcher = (
        pd.to_numeric(fold["rows"]["pitcher_id"], errors="coerce")
        .fillna(-1)
        .to_numpy(np.int64)
    )
    bootstrap = pitcher_bootstrap(current, candidate, target, pitcher)
    fixed = {
        "locked_endpoint": "direct_probability",
        "locked_weight_from_2023": locked_weight,
        "current_score": score(current, target),
        "endpoint_score": score(direct, target),
        "candidate_score": score(candidate, target),
        "gain": float(score(candidate, target) - score(current, target)),
        "early_gain": subset_gain(current, candidate, target, month <= 6),
        "late_gain": subset_gain(current, candidate, target, month > 6),
        "pitcher_cluster_bootstrap": bootstrap,
    }
    artifacts = {}
    for name, value in (
        ("direct", direct), ("skip", skip), ("current", current),
        ("locked_candidate", candidate),
    ):
        path = LAB / f"179_{name}_2024.npy"
        np.save(path, np.asarray(value, np.float32), allow_pickle=False)
        artifacts[name] = {"path": str(path), "sha256": sha256(path), "rows": len(value)}
    report = {
        "experiment": 179,
        "phase": "UNTOUCHED_2024_CONFIRMATION_OF_2023_LOCK",
        "status": "CONFIRMED" if fixed["gain"] > 0.0 else "FAILED_CONFIRMATION",
        "lock_path": str(LOCK),
        "lock_sha256": sha256(LOCK),
        "recipe_unchanged": True,
        "post_2024_tuning": False,
        "fold": fold["metadata"],
        "anchor": anchor_meta,
        "fit_seconds": fit_seconds,
        "wall_seconds": time.time() - started,
        "epoch_history": history,
        "row_independence_max_abs": independence,
        "fixed_2024": fixed,
        "oracle_geometry_diagnostic_only": geometry(current, direct, target),
        "artifacts": artifacts,
        "test_read": False,
        "zip_created": False,
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "=== exp179 untouched 2024 confirmation ===",
        f"locked w={locked_weight:.9f} gain={fixed['gain']:+.4f}",
        f"current={fixed['current_score']:.4f} endpoint={fixed['endpoint_score']:.4f} "
        f"candidate={fixed['candidate_score']:.4f}",
        f"early={fixed['early_gain']:+.4f} late={fixed['late_gain']:+.4f} "
        f"pitcher_p025={bootstrap['p025']:+.4f}",
        f"row_independence={independence:.3e}",
        "post-2024 tuning=false; test unopened; no ZIP",
    ]
    with TEXT.open("a", encoding="utf-8") as handle:
        handle.write("\n" + "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("discovery", "confirm"), default="discovery", nargs="?"
    )
    args = parser.parse_args()
    if args.phase == "discovery":
        discovery()
    else:
        confirm()


if __name__ == "__main__":
    main()
