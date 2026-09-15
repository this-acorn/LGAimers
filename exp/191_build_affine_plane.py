"""Build one-run current/JOA/JM-residual affine-plane submission.

The frozen JM endpoint internally computes the JOA anchor before applying its
R-only residual.  This builder makes an isolated patch that saves that already
computed anchor, so the packaged wrapper obtains current, JOA, and JM(.75)
predictions with only two model executions (current once, JM once).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / 'artifacts/candidates/candidate_current_jm0750_w050.zip'
PYTHON = ROOT / "venv311" / "Scripts" / "python.exe"


WRAPPER = r'''"""Leaderboard-calibrated affine plane; row-local inference only."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
ID = "row_id"
TARGET = "control_success"
X_JOA = __X_JOA__
X_RESIDUAL = __X_RESIDUAL__
CURRENT_BUNDLE = ROOT / "model" / "current_bundle"
JM_BUNDLE = ROOT / "model" / "endpoint_bundle"


def attach_data(bundle: Path) -> None:
    target = bundle / "data"
    if target.exists() or target.is_symlink():
        return
    try:
        os.symlink(DATA, target, target_is_directory=True)
        return
    except OSError:
        target.mkdir(parents=True, exist_ok=True)
    for name in ("test.csv", "sample_submission.csv"):
        source = DATA / name
        destination = target / name
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def run(bundle: Path, name: str) -> None:
    attach_data(bundle)
    out = bundle / "output"
    out.mkdir(parents=True, exist_ok=True)
    for filename in ("submission.csv", "anchor_submission.csv"):
        path = out / filename
        if path.exists():
            path.unlink()
    started = time.time()
    result = subprocess.run(
        [sys.executable, "script.py"], cwd=bundle, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(f"[{name}] rc={result.returncode} elapsed={time.time()-started:.2f}s", flush=True)
    if result.stdout:
        print(result.stdout[-4000:], flush=True)
    if result.returncode:
        if result.stderr:
            print(result.stderr[-8000:], file=sys.stderr, flush=True)
        raise RuntimeError(f"{name} failed")


def aligned(path: Path, ids: pd.Series, name: str) -> np.ndarray:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if list(frame.columns) != [ID, TARGET] or frame[ID].duplicated().any():
        raise ValueError(f"{name}: invalid output")
    values = frame.set_index(ID)[TARGET].reindex(ids.to_numpy()).to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: missing/nonfinite output")
    return values


def main() -> None:
    sample = pd.read_csv(DATA / "sample_submission.csv", encoding="utf-8-sig")
    sample.columns = [str(c).replace("\ufeff", "").strip() for c in sample.columns]
    if list(sample.columns[:2]) != [ID, TARGET] or sample[ID].duplicated().any():
        raise ValueError("invalid sample submission")
    run(CURRENT_BUNDLE, "current")
    current = aligned(CURRENT_BUNDLE / "output" / "submission.csv", sample[ID], "current")
    run(JM_BUNDLE, "jm075")
    jm075 = aligned(JM_BUNDLE / "output" / "submission.csv", sample[ID], "jm075")
    joa = aligned(JM_BUNDLE / "output" / "anchor_submission.csv", sample[ID], "joa")
    raw = current + X_JOA * (joa - current) + X_RESIDUAL * ((jm075 - joa) / 0.75)
    prediction = np.clip(raw, 0.0, 1.0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: sample[ID], TARGET: prediction}).to_csv(
        OUTPUT / "submission.csv", index=False, encoding="utf-8")
    print(
        f"[plane] x_joa={X_JOA:.12g} x_residual={X_RESIDUAL:.12g} "
        f"rows={len(prediction)} raw_range=[{raw.min():.9f},{raw.max():.9f}] "
        f"clipped={int(np.count_nonzero(raw != prediction))} mean={prediction.mean():.9f}",
        flush=True)


if __name__ == "__main__":
    main()
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def safe_extract(path: Path, destination: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe ZIP member: {info.filename}")
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failure: {bad}")
        archive.extractall(destination)
    for stale in destination.rglob("output"):
        if stale.is_dir():
            shutil.rmtree(stale)
    for stale in destination.rglob("data"):
        if stale.is_dir() and not stale.is_symlink():
            shutil.rmtree(stale)


def patch_endpoint(endpoint_script: Path) -> None:
    text = endpoint_script.read_text(encoding="utf-8")
    needle = "    base_submission = run_joa_anchor()\n"
    if text.count(needle) != 1:
        raise ValueError("cannot uniquely patch JM anchor capture")
    replacement = needle + (
        "    # Packaging-only capture of the already-computed frozen JOA anchor.\n"
        "    base_submission.to_csv(\"./output/anchor_submission.csv\", "
        "index=False, encoding=\"utf-8\")\n"
    )
    endpoint_script.write_text(text.replace(needle, replacement), encoding="utf-8")


def make_zip(source: Path, output: Path) -> None:
    with zipfile.ZipFile(
        output, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
    ) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(source).as_posix())


def smoke(zip_path: Path, data: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="lg191_smoke_") as raw:
        work = Path(raw)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(work)
        shutil.copytree(data, work / "data")
        result = subprocess.run(
            [str(PYTHON), "script.py"], cwd=work, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        print((result.stdout or "")[-6000:])
        if result.returncode:
            print((result.stderr or "")[-10000:])
            raise SystemExit("smoke failed")
        output = pd.read_csv(work / "output" / "submission.csv", encoding="utf-8-sig")
        sample = pd.read_csv(data / "sample_submission.csv", encoding="utf-8-sig")
        if not output[ID].equals(sample[ID]):
            raise SystemExit("smoke row order mismatch")
        print(
            f"smoke PASS rows={len(output)} mean={output[TARGET].mean():.9f} "
            f"range=[{output[TARGET].min():.9f},{output[TARGET].max():.9f}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--x-joa", type=float, required=True)
    parser.add_argument("--x-residual", type=float, required=True)
    parser.add_argument("--name", default='artifacts/candidates/last.zip')
    parser.add_argument("--smoke-data", type=Path, default=ROOT / "data")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    output = ROOT / args.name
    if output.suffix.lower() != ".zip":
        output = output.with_suffix(".zip")
    source = ROOT / "lab" / "191_final_src"
    if source.exists():
        shutil.rmtree(source)
    if output.exists():
        output.unlink()
    source.mkdir(parents=True)
    safe_extract(args.template.resolve(), source)
    patch_endpoint(source / "model" / "endpoint_bundle" / "script.py")
    (source / "script.py").write_text(
        WRAPPER.replace("__X_JOA__", repr(float(args.x_joa))).replace(
            "__X_RESIDUAL__", repr(float(args.x_residual))),
        encoding="utf-8")
    make_zip(source, output)
    print(f"built={output} bytes={output.stat().st_size} sha256={sha256(output)}")
    if not args.skip_smoke:
        smoke(output, args.smoke_data.resolve())


if __name__ == "__main__":
    main()
