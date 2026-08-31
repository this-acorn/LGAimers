# -*- coding: utf-8 -*-
"""CUDA/AMP handoff for the official-only fast std-z MLP endpoint.

Modes
-----
validate
    Fit official seasons <=2023, predict 2024, save three seed endpoints and
    states, then report geometry against the frozen current 2024 OOF anchor.
    Every seed uses its final epoch; validation labels never select an epoch.
final_train
    Rebuild the identical <=2023-fitted preprocessing, append official 2024
    rows, fit all <=2024 labels, and save per-seed states for later packaging.

No test.csv is read.  The source feature builder is the audited official-only
exp168 implementation; model/training code here is original and row-local.
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

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp"
LAB = ROOT / "lab"
CURRENT_2024 = LAB / "179_current_2024.npy"
EXPECTED_CURRENT_SHA256 = "e78b168244675adea4bd6a3d56f021a9376cd4b78c428d536a9b53b02fcebe9b"
EXPECTED_ROW_SHA256 = "6e5fb9e3c20ab4363de4c3be8d58246a87245646ff156eca38c7eeb9e7d70787"
EXPECTED_TARGET_SHA256 = "e0483e6a48d9f699ca9c4abf7b4b81aed4d679c359c7e9c3752b3f9e6e27c9d3"
SCORE_SCALE = 100000.0
AUDITED_SOURCES = {
    ROOT / "reference" / "LA9elephantmiracle" / "LICENSE":
        "76c97c2298b40d458f02fb30dd6a045864c3d9f6b73a3ffbea0487753192d493",
    ROOT / "reference" / "LA9elephantmiracle" / "cowork" / "cw" / "v17" / "src" / "common.py":
        "91d84161d6e4a085ffa8219f1ea9e453cfd4c5409d15a2baee0dbf694671313a",
    ROOT / "reference" / "LA9elephantmiracle" / "performance_tracking" / "models" / "sj_stdmlp" / "prep_mlp.py":
        "aadfd85573fb76ba13cc4f0c95b24886a2ea698a81399922e65ac6b520d2e8c8",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E168 = load_module(EXP / "168_elephant_stdmlp_clean_oof.py", "e168_for_182")
E179 = load_module(EXP / "179_fast_stdmlp_endpoint.py", "e179_for_182")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_audited_sources() -> None:
    for path, expected in AUDITED_SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing audited source: {path}")
        if file_sha256(path).lower() != expected:
            raise AssertionError(f"audited source SHA256 mismatch: {path}")


def resolve_device(torch, requested: str):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA PyTorch/device is unavailable")
    return torch.device(requested)


def build_network(torch, nn, d_in: int, target_rate: float):
    class FastMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(d_in, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Dropout(0.12),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Dropout(0.08),
                nn.Linear(64, 1),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.constant_(
                self.network[-1].bias,
                math.log(target_rate / (1.0 - target_rate)),
            )

        def forward(self, x):
            return self.network(x).squeeze(-1)

    return FastMLP()


def predict(torch, network, matrix: np.ndarray, device, batch: int, use_amp: bool) -> np.ndarray:
    network.eval()
    source = torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float32))
    out: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(source), batch):
            xb = source[start : start + batch].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                probability = torch.sigmoid(network(xb))
            out.append(probability.float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def fit_seed(
    matrix: np.ndarray,
    target: np.ndarray,
    validation: np.ndarray | None,
    current_validation: np.ndarray | None,
    seed: int,
    args: argparse.Namespace,
    output_dir: Path,
    state_name: str,
) -> tuple[np.ndarray | None, list[dict[str, Any]], Path, float]:
    import torch
    import torch.nn as nn

    device = resolve_device(torch, args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    else:
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    x = torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float32))
    y = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32))
    network = build_network(torch, nn, x.shape[1], float(target.mean())).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    batches = math.ceil(len(x) / args.batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=batches * args.epochs,
        pct_start=0.15,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, Any]] = []
    started_all = time.time()
    final_prediction = None

    for epoch in range(args.epochs):
        network.train()
        started = time.time()
        loss_sum = 0.0
        seen = 0
        starts = list(range(0, len(x), args.batch))
        if epoch % 2:
            starts.reverse()
        for batch_index, start in enumerate(starts):
            xb = x[start : start + args.batch].to(device, non_blocking=True)
            yb = y[start : start + args.batch].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                probability = torch.sigmoid(network(xb))
                loss = torch.square(probability - yb).mean()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count = int(len(xb))
            loss_sum += float(loss.detach()) * count
            seen += count
            if epoch == 0 and batch_index == 1:
                elapsed2 = time.time() - started
                print(
                    f"[seed {seed} speed] device={device} amp={use_amp} "
                    f"2_batches={elapsed2:.2f}s epoch_eta={elapsed2 * batches / 2:.1f}s",
                    flush=True,
                )
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "seconds": time.time() - started,
            "train_brier": loss_sum / seen,
            "learning_rate_end": optimizer.param_groups[0]["lr"],
        }
        if validation is not None:
            final_prediction = predict(
                torch, network, validation, device, args.predict_batch, use_amp
            )
            if current_validation is not None:
                record["validation_geometry_diagnostic"] = E179.geometry(
                    current_validation, final_prediction, VALIDATION_TARGET
                )
                geo = record["validation_geometry_diagnostic"]
                print(
                    f"[seed {seed} epoch {epoch + 1}/{args.epochs}] "
                    f"seconds={record['seconds']:.1f} d={geo['d_endpoint_score_minus_current']:+.3f} "
                    f"K={geo['K_direction_curvature']:.3f} "
                    f"blend_gain={geo['optimal_blend_gain']:+.3f}",
                    flush=True,
                )
        else:
            print(
                f"[seed {seed} epoch {epoch + 1}/{args.epochs}] "
                f"seconds={record['seconds']:.1f} train_brier={record['train_brier']:.6f}",
                flush=True,
            )
        history.append(record)

    state_path = output_dir / f"{state_name}_seed{seed}.pt"
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in network.state_dict().items()},
            "seed": seed,
            "input_features": int(matrix.shape[1]),
            "architecture": "160->128(LayerNorm/SiLU/dropout)->64(SiLU/dropout)->1",
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "preprocessing": "robust-z+missing mask fitted on official <=2023",
            "test_read": False,
        },
        state_path,
    )
    elapsed_all = time.time() - started_all
    del network, optimizer, scheduler, scaler, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return final_prediction, history, state_path, elapsed_all


VALIDATION_TARGET: np.ndarray


def load_current_and_check(fold: dict[str, Any]) -> np.ndarray:
    if not CURRENT_2024.is_file():
        raise FileNotFoundError(CURRENT_2024)
    if file_sha256(CURRENT_2024).lower() != EXPECTED_CURRENT_SHA256:
        raise AssertionError("frozen 2024 current OOF SHA256 mismatch")
    if fold["metadata"]["row_id_order_sha256"] != EXPECTED_ROW_SHA256:
        raise AssertionError("2024 row_id/order mismatch")
    if fold["metadata"]["target_float64_sha256"] != EXPECTED_TARGET_SHA256:
        raise AssertionError("2024 target/order mismatch")
    current = np.load(CURRENT_2024, allow_pickle=False).astype(np.float64)
    if current.shape != (len(fold["y_valid"]),) or not np.isfinite(current).all():
        raise AssertionError("invalid frozen current OOF")
    return current


def validation_stability(current, endpoint, target, rows, geo) -> dict[str, Any]:
    candidate = current + geo["locked_convex_weight"] * (endpoint - current)
    month = pd.to_numeric(rows["game_month"], errors="coerce").to_numpy(np.int64)
    pitcher = (
        pd.to_numeric(rows["pitcher_id"], errors="coerce").fillna(-1).to_numpy(np.int64)
    )
    return {
        "early_gain": E179.subset_gain(current, candidate, target, month <= 6),
        "late_gain": E179.subset_gain(current, candidate, target, month > 6),
        "pitcher_cluster_bootstrap": E179.pitcher_bootstrap(
            current, candidate, target, pitcher
        ),
    }


def validate(args: argparse.Namespace) -> None:
    global VALIDATION_TARGET
    verify_audited_sources()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "validation_report.json").exists():
        raise FileExistsError(out / "validation_report.json")
    started = time.time()
    print("[exp182] build official <=2023 -> 2024", flush=True)
    fold = E168.build_fold(2024)
    VALIDATION_TARGET = fold["y_valid"]
    current = load_current_and_check(fold)
    predictions: list[np.ndarray] = []
    per_seed = []
    for seed in args.seeds:
        prediction, history, state_path, elapsed = fit_seed(
            fold["z_train"], fold["y_train"], fold["z_valid"], current,
            seed, args, out, "validation",
        )
        assert prediction is not None
        endpoint_path = out / f"endpoint_2024_seed{seed}.npy"
        np.save(endpoint_path, prediction.astype(np.float32), allow_pickle=False)
        predictions.append(prediction)
        per_seed.append({
            "seed": seed,
            "elapsed_seconds": elapsed,
            "history": history,
            "state": {"path": str(state_path), "sha256": file_sha256(state_path)},
            "endpoint": {"path": str(endpoint_path), "sha256": file_sha256(endpoint_path)},
            "final_geometry": E179.geometry(current, prediction, VALIDATION_TARGET),
        })
    ensemble = np.mean(predictions, axis=0)
    ensemble_path = out / "endpoint_2024_seed_ensemble.npy"
    np.save(ensemble_path, ensemble.astype(np.float32), allow_pickle=False)
    geo = E179.geometry(current, ensemble, VALIDATION_TARGET)
    stability = validation_stability(
        current, ensemble, VALIDATION_TARGET, fold["rows"], geo
    )
    passed = bool(
        geo["optimal_blend_gain"] >= 20.0
        and stability["early_gain"] > 0.0
        and stability["late_gain"] > 0.0
        and stability["pitcher_cluster_bootstrap"]["p025"] > 0.0
    )
    report = {
        "experiment": 182,
        "mode": "2024_VALIDATION_FINAL_EPOCH_ONLY",
        "status": "PASS_CANDIDATE_FOR_FINAL_TRAIN" if passed else "FAIL_NO_DEPLOY",
        "device_request": args.device,
        "amp_request": args.amp,
        "recipe": {
            "epochs": args.epochs, "batch": args.batch, "seeds": args.seeds,
            "lr": args.lr, "weight_decay": args.weight_decay,
        },
        "fold": fold["metadata"],
        "current_anchor": {
            "path": str(CURRENT_2024), "sha256": file_sha256(CURRENT_2024),
            "score": E179.score(current, VALIDATION_TARGET),
        },
        "per_seed": per_seed,
        "ensemble": {
            "path": str(ensemble_path), "sha256": file_sha256(ensemble_path),
            "geometry": geo,
            "recommended_convex_weight": geo["locked_convex_weight"],
            "stability": stability,
        },
        "gate": "gain>=20 AND early>0 AND late>0 AND pitcher p025>0",
        "wall_seconds": time.time() - started,
        "test_read": False,
        "zip_created": False,
    }
    (out / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[FINAL] status={report['status']} d={geo['d_endpoint_score_minus_current']:+.4f} "
        f"K={geo['K_direction_curvature']:.4f} w={geo['locked_convex_weight']:.6f} "
        f"gain={geo['optimal_blend_gain']:+.4f} early={stability['early_gain']:+.4f} "
        f"late={stability['late_gain']:+.4f} "
        f"p025={stability['pitcher_cluster_bootstrap']['p025']:+.4f}",
        flush=True,
    )


def final_train(args: argparse.Namespace) -> None:
    verify_audited_sources()
    out = Path(args.output_dir).resolve()
    report_path = out / "validation_report.json"
    if not report_path.is_file():
        raise FileNotFoundError("validation_report.json is required")
    validation_report = json.loads(report_path.read_text(encoding="utf-8"))
    if validation_report["status"] != "PASS_CANDIDATE_FOR_FINAL_TRAIN":
        raise RuntimeError("validation gate failed; final training is forbidden")
    locked = validation_report["recipe"]
    if (
        args.epochs != int(locked["epochs"])
        or args.batch != int(locked["batch"])
        or args.seeds != [int(v) for v in locked["seeds"]]
        or args.lr != float(locked["lr"])
        or args.weight_decay != float(locked["weight_decay"])
    ):
        raise RuntimeError("final_train arguments must exactly match validation recipe")
    final_report_path = out / "final_train_report.json"
    if final_report_path.exists():
        raise FileExistsError(final_report_path)
    print("[exp182] final train all official <=2024; test remains unopened", flush=True)
    fold = E168.build_fold(2024)
    matrix = np.ascontiguousarray(
        np.concatenate((fold["z_train"], fold["z_valid"]), axis=0), dtype=np.float32
    )
    target = np.ascontiguousarray(
        np.concatenate((fold["y_train"], fold["y_valid"],), axis=0), dtype=np.float32
    )
    states = []
    for seed in args.seeds:
        _, history, state_path, elapsed = fit_seed(
            matrix, target, None, None, seed, args, out, "final_all_through_2024"
        )
        states.append({
            "seed": seed, "elapsed_seconds": elapsed, "history": history,
            "state": {"path": str(state_path), "sha256": file_sha256(state_path)},
        })
    report = {
        "experiment": 182,
        "mode": "FINAL_TRAIN_OFFICIAL_THROUGH_2024",
        "status": "STATES_READY_PACKAGING_STILL_REQUIRED",
        "validation_report_sha256": file_sha256(report_path),
        "rows": int(len(matrix)),
        "features": int(matrix.shape[1]),
        "recipe": locked,
        "states": states,
        "preprocessing_lock": (
            "robust-z fitted <=2023 exactly as validation, then used for 2024; "
            "this is intentional to keep the validated recipe unchanged"
        ),
        "test_read": False,
        "zip_created": False,
    }
    final_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[FINAL TRAIN] {final_report_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "final_train"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--predict-batch", type=int, default=32768)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=3.0e-4)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output-dir", default=str(LAB / "182_gpu_stdmlp"))
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch <= 0 or args.predict_batch <= 0:
        parser.error("epochs and batch sizes must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "validate":
        validate(args)
    else:
        final_train(args)


if __name__ == "__main__":
    main()
