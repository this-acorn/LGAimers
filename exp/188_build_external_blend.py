"""Build a row-local blend of the frozen affine champion and an external endpoint.

The two original submission bundles are preserved under separate directories.
At inference time the root wrapper runs them sequentially, aligns predictions by
``row_id`` against ``sample_submission.csv``, and writes their weighted average.

Example (JY endpoint, 50:50 probe)::

    python exp/188_build_external_blend.py \
      --endpoint-zip reference/calico_JY_lfs_download/final/sub_JY_team_residual_scale015.zip \
      --weight 0.5 --name probe_affine_jy_w050

For the JM bundle, ``--endpoint-scale`` updates only the frozen top-level scale
in ``model/r_residual_manifest.json`` before packaging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURRENT = ROOT / 'artifacts/candidates/candidate_affine_opt.zip'
PYTHON = ROOT / "venv311" / "Scripts" / "python.exe"


WRAPPER = r'''"""Sequential row-local blend of two frozen submission endpoints."""
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
ID_COL = "row_id"
TARGET_COL = "control_success"
ENDPOINT_WEIGHT = __WEIGHT__
BUNDLES = (("current", ROOT / "model" / "current_bundle"),
           ("endpoint", ROOT / "model" / "endpoint_bundle"))


def _attach_data(bundle: Path) -> None:
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


def _run(name: str, bundle: Path) -> pd.DataFrame:
    _attach_data(bundle)
    out_dir = bundle / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "submission.csv"
    if out_path.exists():
        out_path.unlink()
    started = time.time()
    result = subprocess.run(
        [sys.executable, "script.py"], cwd=bundle, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    elapsed = time.time() - started
    print(f"[{name}] rc={result.returncode} elapsed={elapsed:.2f}s", flush=True)
    if result.stdout:
        print(result.stdout[-4000:], flush=True)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr[-8000:], file=sys.stderr, flush=True)
        raise RuntimeError(f"{name} endpoint failed")
    if not out_path.is_file():
        raise RuntimeError(f"{name} endpoint did not create {out_path}")
    return pd.read_csv(out_path, encoding="utf-8-sig")


def _align(frame: pd.DataFrame, ids: pd.Series, name: str) -> np.ndarray:
    if list(frame.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"{name}: invalid columns {frame.columns.tolist()}")
    if frame[ID_COL].duplicated().any():
        raise ValueError(f"{name}: duplicate row_id")
    indexed = frame.set_index(ID_COL)[TARGET_COL]
    missing = pd.Index(ids).difference(indexed.index)
    extra = indexed.index.difference(pd.Index(ids))
    if len(missing) or len(extra):
        raise ValueError(f"{name}: row_id mismatch missing={len(missing)} extra={len(extra)}")
    values = indexed.reindex(ids.to_numpy()).to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not ((values >= 0.0).all() and (values <= 1.0).all()):
        raise ValueError(f"{name}: invalid probability")
    return values


def main() -> None:
    sample = pd.read_csv(DATA / "sample_submission.csv", encoding="utf-8-sig")
    sample.columns = [str(c).replace("\ufeff", "").strip() for c in sample.columns]
    if ID_COL not in sample or sample[ID_COL].duplicated().any():
        raise ValueError("invalid sample_submission row_id")
    current_frame = _run(*BUNDLES[0])
    endpoint_frame = _run(*BUNDLES[1])
    current = _align(current_frame, sample[ID_COL], "current")
    endpoint = _align(endpoint_frame, sample[ID_COL], "endpoint")
    prediction = np.clip((1.0 - ENDPOINT_WEIGHT) * current + ENDPOINT_WEIGHT * endpoint, 0.0, 1.0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({ID_COL: sample[ID_COL], TARGET_COL: prediction})
    out.to_csv(OUTPUT / "submission.csv", index=False, encoding="utf-8")
    print(f"[blend] weight_endpoint={ENDPOINT_WEIGHT:.12g} rows={len(out)} "
          f"current_mean={current.mean():.9f} endpoint_mean={endpoint.mean():.9f} "
          f"blend_mean={prediction.mean():.9f}", flush=True)


if __name__ == "__main__":
    main()
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def safe_extract(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe ZIP member: {info.filename}")
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC failure: {bad}")
        archive.extractall(destination)
    required = (destination / "script.py", destination / "model", destination / "requirements.txt")
    if not all(path.exists() for path in required):
        raise ValueError(f"invalid submission structure: {zip_path}")
    shutil.rmtree(destination / "output", ignore_errors=True)
    shutil.rmtree(destination / "data", ignore_errors=True)


def patch_endpoint_scale(endpoint: Path, scale: float) -> str:
    manifest = endpoint / "model" / "r_residual_manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if "scale" not in payload:
            raise ValueError(f"scale absent from {manifest}")
        old = float(payload["scale"])
        payload["scale"] = float(scale)
        manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return f"{manifest.relative_to(endpoint).as_posix()}: {old} -> {scale}"
    script = endpoint / "script.py"
    text = script.read_text(encoding="utf-8")
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if line.startswith("RESIDUAL_SCALE = ")]
    if len(hits) != 1:
        raise ValueError("cannot identify one endpoint residual scale")
    i = hits[0]
    old = lines[i]
    lines[i] = f"RESIDUAL_SCALE = {float(scale)!r}"
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f"script.py: {old} -> {lines[i]}"


def merge_requirements(current: Path, endpoint: Path) -> str:
    values: dict[str, str] = {}
    order: list[str] = []
    for source in (current / "requirements.txt", endpoint / "requirements.txt"):
        for raw in source.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key = line.split("==", 1)[0].split("[", 1)[0].lower()
            if key not in values:
                order.append(key)
            values[key] = line
    # The endpoint requirements are the environment in which its official score
    # was produced; endpoint pins therefore deliberately override current pins.
    return "\n".join(values[key] for key in order) + "\n"


def make_zip(source: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(source).as_posix())


def smoke(zip_path: Path, data_dir: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="lg188_smoke_") as raw:
        work = Path(raw)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(work)
        shutil.copytree(data_dir, work / "data")
        started = time.time()
        result = subprocess.run(
            [str(PYTHON), "script.py"], cwd=work, capture_output=True, text=True,
            encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        elapsed = time.time() - started
        print(f"smoke rc={result.returncode} elapsed={elapsed:.2f}s")
        print((result.stdout or "")[-5000:].encode("ascii", "backslashreplace").decode("ascii"))
        if result.returncode != 0:
            print((result.stderr or "")[-10000:].encode("ascii", "backslashreplace").decode("ascii"),
                  file=sys.stderr)
            raise SystemExit("smoke failed")
        output = pd.read_csv(work / "output" / "submission.csv", encoding="utf-8-sig")
        sample = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
        if list(output.columns) != ["row_id", "control_success"]:
            raise SystemExit("smoke output schema mismatch")
        if not output["row_id"].equals(sample["row_id"]):
            raise SystemExit("smoke output row_id mismatch")
        print(f"smoke rows={len(output)} mean={output.control_success.mean():.9f} "
              f"min={output.control_success.min():.9f} max={output.control_success.max():.9f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-zip", type=Path, required=True)
    parser.add_argument("--current-zip", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--endpoint-scale", type=float)
    parser.add_argument("--smoke-data", type=Path, default=ROOT / "data")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.weight <= 1.0:
        raise SystemExit("--weight must be in [0,1]")
    current_zip = args.current_zip.resolve()
    endpoint_zip = args.endpoint_zip.resolve()
    for path in (current_zip, endpoint_zip):
        if not path.is_file():
            raise SystemExit(f"missing ZIP: {path}")

    source = ROOT / f"{args.name}_src"
    output = ROOT / f"{args.name}.zip"
    if source.exists() or output.exists():
        if not args.overwrite:
            raise SystemExit(f"refusing to overwrite {source} or {output}; pass --overwrite")
        shutil.rmtree(source, ignore_errors=True)
        if output.exists():
            output.unlink()
    current = source / "model" / "current_bundle"
    endpoint = source / "model" / "endpoint_bundle"
    current.mkdir(parents=True)
    endpoint.mkdir(parents=True)
    safe_extract(current_zip, current)
    safe_extract(endpoint_zip, endpoint)
    patch_note = None
    if args.endpoint_scale is not None:
        patch_note = patch_endpoint_scale(endpoint, args.endpoint_scale)
    (source / "script.py").write_text(
        WRAPPER.replace("__WEIGHT__", repr(float(args.weight))), encoding="utf-8")
    (source / "requirements.txt").write_text(
        merge_requirements(current, endpoint), encoding="utf-8")
    make_zip(source, output)
    print(f"built={output}")
    print(f"bytes={output.stat().st_size} entries={sum(1 for p in source.rglob('*') if p.is_file())}")
    print(f"sha256={sha256(output)}")
    print(f"current={current_zip} sha256={sha256(current_zip)}")
    print(f"endpoint={endpoint_zip} sha256={sha256(endpoint_zip)}")
    print(f"weight_endpoint={args.weight:.12g}")
    if patch_note:
        print(f"endpoint_patch={patch_note}")
    if not args.skip_smoke:
        smoke(output, args.smoke_data.resolve())


if __name__ == "__main__":
    main()
