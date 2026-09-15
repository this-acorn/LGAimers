# -*- coding: utf-8 -*-
"""Build a flat, row-local blend of current affine-opt and Calico JM s=.75.

The public Calico archive stores ``scale=0.075``.  The authorized leaderboard
endpoint is reproduced by changing only that manifest value to ``0.75``.  All
component files are copied into one package at build time, so evaluation does
not spend time extracting a nested 280 MB ZIP or duplicating ``test.csv``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / 'artifacts/candidates/candidate_affine_opt.zip'
JM_SOURCE = (
    ROOT
    / "reference"
    / "calico_JY_lfs_download"
    / "final"
    / "sub_JM_R_res_scale075_RE.zip"
)
DEFAULT_OUTPUT = ROOT / 'artifacts/candidates/candidate_current_jm0750_w050.zip'

CURRENT_SHA256 = "E772CA86DAF99209A50AB9E29F030EE680C0770520D98FF5ABE9C3FD4832EC65"
JM_SOURCE_SHA256 = "46AA2A15130ED9BA8D302F203FA42C8E3F1730ACE5D47B3D4EF80429AE2B053F"
OLD_SCALE = 0.075
JM_SCALE = 0.75
BUFFER_BYTES = 4 * 1024 * 1024


WRAPPER = r'''# -*- coding: utf-8 -*-
"""Sequential, row-local convex blend of two frozen probability endpoints."""

from __future__ import annotations

import gc
import runpy
from pathlib import Path

import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
CURRENT_WEIGHT = __CURRENT_WEIGHT__
JM_WEIGHT = __JM_WEIGHT__
MODEL_DIR = Path("./model")
OUTPUT_DIR = Path("./output")
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"
SAMPLE_PATH = Path("./data/sample_submission.csv")


def _run_component(filename: str, namespace: str) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
    module = runpy.run_path(str(MODEL_DIR / filename), run_name=namespace)
    component_main = module.get("main")
    if not callable(component_main):
        raise RuntimeError(f"component has no callable main(): {filename}")
    component_main()
    if not OUTPUT_PATH.is_file():
        raise RuntimeError(f"component did not create {OUTPUT_PATH}: {filename}")
    prediction = pd.read_csv(OUTPUT_PATH, encoding="utf-8-sig")
    del component_main, module
    gc.collect()
    return prediction


def _validated_series(frame: pd.DataFrame, name: str) -> pd.Series:
    frame.columns = [str(column).replace("\ufeff", "").strip() for column in frame.columns]
    if list(frame.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"{name}: invalid columns {frame.columns.tolist()}")
    if frame[ID_COL].duplicated().any():
        raise ValueError(f"{name}: duplicate row_id")
    ids = frame[ID_COL].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"{name}: duplicate normalized row_id")
    values = frame[TARGET_COL].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: non-finite probability")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name}: probability outside [0,1]")
    return pd.Series(values, index=ids.to_numpy(), name=name)


def main() -> None:
    current = _validated_series(
        _run_component("current_inference.py", "current_affine_component"), "current")
    current_values = current.copy()
    del current
    gc.collect()

    jm = _validated_series(
        _run_component("jm0750_inference.py", "calico_jm0750_component"), "jm0750")

    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
    sample.columns = [str(column).replace("\ufeff", "").strip() for column in sample.columns]
    if list(sample.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample: invalid columns {sample.columns.tolist()}")
    if sample[ID_COL].duplicated().any():
        raise ValueError("sample: duplicate row_id")
    ids = sample[ID_COL].astype(str)
    if ids.duplicated().any():
        raise ValueError("sample: duplicate normalized row_id")
    if len(current_values) != len(sample) or len(jm) != len(sample):
        raise ValueError("component/sample row-count mismatch")

    current_aligned = current_values.reindex(ids.to_numpy()).to_numpy(np.float64)
    jm_aligned = jm.reindex(ids.to_numpy()).to_numpy(np.float64)
    if not np.isfinite(current_aligned).all() or not np.isfinite(jm_aligned).all():
        raise ValueError("row_id alignment produced missing component predictions")

    blended = np.clip(CURRENT_WEIGHT * current_aligned + JM_WEIGHT * jm_aligned, 0.0, 1.0)
    if not np.isfinite(blended).all() or np.any((blended < 0.0) | (blended > 1.0)):
        raise ValueError("invalid blended probabilities")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COL: sample[ID_COL], TARGET_COL: blended}).to_csv(
        OUTPUT_PATH, index=False, encoding="utf-8")
    print(
        f"Saved: {OUTPUT_PATH} | rows={len(sample):,} | "
        f"current={CURRENT_WEIGHT:.9f} jm0750={JM_WEIGHT:.9f} | "
        f"mean={blended.mean():.9f} min={blended.min():.9f} max={blended.max():.9f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def safe_member(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise AssertionError(f"unsafe ZIP member: {name!r}")


def zip_info(name: str, source: zipfile.ZipInfo | None = None) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, source.date_time if source else (2026, 9, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (source.external_attr if source else 0o100644 << 16)
    return info


def stream_member(
    source_zip: zipfile.ZipFile,
    source_info: zipfile.ZipInfo,
    target_zip: zipfile.ZipFile,
    target_name: str,
) -> None:
    safe_member(source_info.filename)
    safe_member(target_name)
    with source_zip.open(source_info, "r") as source, target_zip.open(
        zip_info(target_name, source_info), "w", force_zip64=True
    ) as target:
        shutil.copyfileobj(source, target, length=BUFFER_BYTES)


def write_bytes(target: zipfile.ZipFile, name: str, value: bytes) -> None:
    safe_member(name)
    target.writestr(zip_info(name), value, compresslevel=6)


def patched_manifest(source: bytes) -> tuple[bytes, dict]:
    text = source.decode("utf-8")
    sentinel = '"scale": 0.075'
    replacement = '"scale": 0.75'
    if text.count(sentinel) != 1 or replacement in text:
        raise AssertionError("JM manifest scale sentinel drifted")
    patched_text = text.replace(sentinel, replacement)
    before = json.loads(text)
    after = json.loads(patched_text)
    if float(before.get("scale", -1.0)) != OLD_SCALE:
        raise AssertionError("JM source manifest scale is not 0.075")
    if float(after.get("scale", -1.0)) != JM_SCALE:
        raise AssertionError("JM patched manifest scale is not 0.75")
    expected = dict(before)
    expected["scale"] = JM_SCALE
    if after != expected:
        raise AssertionError("JM manifest changed beyond the scale field")
    return patched_text.encode("utf-8"), {
        "path": "model/r_residual_manifest.json",
        "old_scale": OLD_SCALE,
        "new_scale": JM_SCALE,
        "source_sha256": sha256_bytes(source),
        "patched_sha256": sha256_bytes(patched_text.encode("utf-8")),
        "only_semantic_change": True,
    }


def mapped_members(
    current: zipfile.ZipFile, jm: zipfile.ZipFile
) -> tuple[list[tuple[zipfile.ZipFile, zipfile.ZipInfo, str]], bytes, dict]:
    required_current = {"script.py", "requirements.txt", "model/blend_metadata.json"}
    required_jm = {
        "script.py",
        "requirements.txt",
        "joa_base_script.py",
        "model/r_residual_manifest.json",
        "model/sota_common_pipeline.py",
        "src/__init__.py",
    }
    current_names = set(current.namelist())
    jm_names = set(jm.namelist())
    if not required_current.issubset(current_names):
        raise AssertionError(f"current ZIP structure drift: {sorted(required_current-current_names)}")
    if not required_jm.issubset(jm_names):
        raise AssertionError(f"JM ZIP structure drift: {sorted(required_jm-jm_names)}")

    rows: list[tuple[zipfile.ZipFile, zipfile.ZipInfo, str]] = []
    for info in current.infolist():
        if info.is_dir() or info.filename == "requirements.txt":
            continue
        target = "model/current_inference.py" if info.filename == "script.py" else info.filename
        rows.append((current, info, target))
    for info in jm.infolist():
        if info.is_dir() or info.filename in {"requirements.txt", "model/r_residual_manifest.json"}:
            continue
        target = "model/jm0750_inference.py" if info.filename == "script.py" else info.filename
        rows.append((jm, info, target))

    targets = [target for _, _, target in rows]
    reserved = {
        "script.py",
        "requirements.txt",
        "model/current_jm0750_blend_metadata.json",
        "model/r_residual_manifest.json",
    }
    duplicates = sorted({name for name in targets if targets.count(name) > 1})
    collisions = sorted(set(targets) & reserved)
    if duplicates or collisions:
        raise AssertionError(f"component path collision: duplicates={duplicates}, reserved={collisions}")

    requirements = current.read("requirements.txt")
    jm_requirements = jm.read("requirements.txt").decode("utf-8", errors="strict")
    if "catboost==1.2.10" not in requirements.decode("utf-8"):
        raise AssertionError("current requirements lost catboost 1.2.10")
    if "catboost==1.2.10" not in jm_requirements:
        raise AssertionError("JM requirements drifted")
    manifest, patch = patched_manifest(jm.read("model/r_residual_manifest.json"))
    return rows, requirements, {"bytes": manifest, "evidence": patch}


def build(output: Path, jm_weight: float) -> dict:
    if not 0.0 <= jm_weight <= 1.0:
        raise ValueError("--jm-weight must be in [0,1]")
    current_weight = 1.0 - jm_weight
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if sha256(CURRENT) != CURRENT_SHA256:
        raise AssertionError("locked candidate_affine_opt.zip SHA256 mismatch")
    if sha256(JM_SOURCE) != JM_SOURCE_SHA256:
        raise AssertionError("locked Calico JM source SHA256 mismatch")

    wrapper = (
        WRAPPER.replace("__CURRENT_WEIGHT__", repr(current_weight))
        .replace("__JM_WEIGHT__", repr(jm_weight))
    ).encode("utf-8")
    partial = output.with_name(output.name + ".building")
    if partial.exists():
        raise FileExistsError(f"refusing to overwrite stale partial {partial}")
    output.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict = {
        "name": "current_affine_opt__calico_jm0750_blend",
        "weights": {"current_affine_opt": current_weight, "calico_jm0750": jm_weight},
        "row_local": True,
        "component_execution": "sequential",
        "packaging": "flat_build_time_merge_no_runtime_extraction",
        "current": {"name": CURRENT.name, "sha256": CURRENT_SHA256},
        "calico_jm_source": {
            "name": JM_SOURCE.name,
            "sha256": JM_SOURCE_SHA256,
            "source_scale": OLD_SCALE,
            "deployed_scale": JM_SCALE,
            "authorized_by_user": True,
        },
    }
    try:
        with zipfile.ZipFile(CURRENT) as current, zipfile.ZipFile(JM_SOURCE) as jm:
            members, requirements, manifest = mapped_members(current, jm)
            metadata["manifest_patch"] = manifest["evidence"]
            with zipfile.ZipFile(
                partial,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
                allowZip64=True,
            ) as target:
                write_bytes(target, "script.py", wrapper)
                write_bytes(target, "requirements.txt", requirements)
                for source_zip, source_info, target_name in members:
                    stream_member(source_zip, source_info, target, target_name)
                write_bytes(target, "model/r_residual_manifest.json", manifest["bytes"])
                write_bytes(
                    target,
                    "model/current_jm0750_blend_metadata.json",
                    (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
                )

        with zipfile.ZipFile(partial) as archive:
            bad = archive.testzip()
            names = archive.namelist()
            if bad is not None:
                raise AssertionError(f"output ZIP CRC failure: {bad}")
            if names[:2] != ["script.py", "requirements.txt"]:
                raise AssertionError("output ZIP root entry order drift")
            if len(names) != len(set(names)):
                raise AssertionError("output ZIP has duplicate entries")
            forbidden = [
                name
                for name in names
                if name.startswith(("data/", "output/")) or "__pycache__" in name
            ]
            if forbidden:
                raise AssertionError(f"output ZIP contains runtime files: {forbidden}")
            deployed = json.loads(
                archive.read("model/r_residual_manifest.json").decode("utf-8"))
            if float(deployed["scale"]) != JM_SCALE:
                raise AssertionError("output JM scale verification failed")
        partial.replace(output)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise

    result = {
        "artifact": str(output),
        "sha256": sha256(output),
        "bytes": output.stat().st_size,
        "entries": len(names),
        "crc": "passed",
        "weights": metadata["weights"],
        "jm_scale": JM_SCALE,
        "manifest_patch": metadata["manifest_patch"],
    }
    return result


def smoke_test(output: Path) -> dict:
    """Run the exact artifact on the checked-in five-row official schema sample."""
    import os
    import subprocess
    import sys
    import time

    import numpy as np
    import pandas as pd

    python = ROOT / "venv311" / "Scripts" / "python.exe"
    executable = str(python if python.is_file() else Path(sys.executable))
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="exp188_smoke_", dir=ROOT) as raw:
        work = Path(raw)
        with zipfile.ZipFile(output) as archive:
            archive.extractall(work)
        data_dir = work / "data"
        data_dir.mkdir()
        shutil.copy2(ROOT / "data" / "test.csv", data_dir / "test.csv")
        shutil.copy2(
            ROOT / "data" / "sample_submission.csv", data_dir / "sample_submission.csv")
        completed = subprocess.run(
            [executable, "script.py"],
            cwd=work,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "smoke test failed\n"
                + (completed.stdout or "")[-4000:]
                + "\n"
                + (completed.stderr or "")[-8000:]
            )
        submission = pd.read_csv(work / "output" / "submission.csv")
        sample = pd.read_csv(data_dir / "sample_submission.csv")
        values = submission["control_success"].to_numpy(np.float64)
        if list(submission.columns) != ["row_id", "control_success"]:
            raise AssertionError("smoke output schema failure")
        if not np.array_equal(submission["row_id"].to_numpy(), sample["row_id"].to_numpy()):
            raise AssertionError("smoke output row-order failure")
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise AssertionError("smoke output probability failure")
        return {
            "rows": len(submission),
            "seconds": time.perf_counter() - started,
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "stdout_tail": (completed.stdout or "").strip().splitlines()[-1],
        }


def server_scale_test(output: Path, rows: int, seed: int) -> dict:
    """Run the exact package on a deterministic evaluation-size synthetic frame."""
    import importlib.util
    import os
    import subprocess
    import sys
    import time

    import numpy as np
    import pandas as pd

    if rows <= 0:
        raise ValueError("--rows must be positive")
    helper_path = ROOT / "exp" / "145_rehearse_reimpl.py"
    spec = importlib.util.spec_from_file_location("exp145_for_exp188", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fake-test helper {helper_path}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    real = pd.read_csv(ROOT / "data" / "test.csv", encoding="utf-8-sig")
    fake = helper.build_fake_test(real, rows, seed)

    python = ROOT / "venv311" / "Scripts" / "python.exe"
    executable = str(python if python.is_file() else Path(sys.executable))
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="exp188_scale_", dir=ROOT) as raw:
        work = Path(raw)
        with zipfile.ZipFile(output) as archive:
            archive.extractall(work)
        data_dir = work / "data"
        data_dir.mkdir()
        fake.to_csv(data_dir / "test.csv", index=False, encoding="utf-8")
        pd.DataFrame({"row_id": fake["row_id"], "control_success": 0.0}).to_csv(
            data_dir / "sample_submission.csv", index=False, encoding="utf-8")
        inference_started = time.perf_counter()
        completed = subprocess.run(
            [executable, "script.py"],
            cwd=work,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        inference_seconds = time.perf_counter() - inference_started
        if completed.returncode != 0:
            raise RuntimeError(
                "server-scale test failed\n"
                + (completed.stdout or "")[-5000:]
                + "\n"
                + (completed.stderr or "")[-10000:]
            )
        submission = pd.read_csv(work / "output" / "submission.csv")
        values = submission["control_success"].to_numpy(np.float64)
        if list(submission.columns) != ["row_id", "control_success"] or len(submission) != rows:
            raise AssertionError("server-scale output schema/row-count failure")
        if not np.array_equal(submission["row_id"].to_numpy(), fake["row_id"].to_numpy()):
            raise AssertionError("server-scale output row-order failure")
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise AssertionError("server-scale output probability failure")
        return {
            "rows": rows,
            "seed": seed,
            "inference_seconds": inference_seconds,
            "total_seconds": time.perf_counter() - started,
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "schema_ok": True,
            "row_order_ok": True,
            "finite_range_ok": True,
            "stdout_tail": (completed.stdout or "").strip().splitlines()[-1],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jm-weight",
        type=float,
        default=0.5,
        help="convex weight of the patched Calico JM s=.75 endpoint (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output ZIP (default: {DEFAULT_OUTPUT.name})",
    )
    parser.add_argument("--smoke", action="store_true", help="run the five-row package smoke test")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="smoke-test an already-built --output without rebuilding it",
    )
    parser.add_argument(
        "--server-only",
        action="store_true",
        help="run evaluation-size QA on an already-built --output without rebuilding it",
    )
    parser.add_argument("--rows", type=int, default=245789, help="rows for --server-only")
    parser.add_argument("--seed", type=int, default=188, help="seed for --server-only")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    if args.smoke_only or args.server_only:
        if args.smoke_only and args.server_only:
            raise ValueError("choose at most one of --smoke-only and --server-only")
        if not output.is_file():
            raise FileNotFoundError(output)
        result = {
            "artifact": str(output),
            "sha256": sha256(output),
            "bytes": output.stat().st_size,
        }
        if args.smoke_only:
            result["smoke_test"] = smoke_test(output)
        else:
            result["server_scale_test"] = server_scale_test(output, args.rows, args.seed)
    else:
        result = build(output, args.jm_weight)
        if args.smoke:
            result["smoke_test"] = smoke_test(output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
