"""Build a row-local 50:50 probe between the current champion and a Calico endpoint.

The Calico archive is kept as a stored nested ZIP so the large model payload is
not needlessly recompressed.  At inference time it is extracted into a fixed,
package-local directory, run independently, aligned by row_id, and blended with
the current endpoint.  No statistic over evaluation rows is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / 'artifacts/candidates/candidate_affine_opt.zip'
CALICO_DIR = ROOT / "reference" / "calico_JY_lfs_download" / "final"
ENDPOINTS = {
    "jy": CALICO_DIR / "sub_JY_team_residual_scale015.zip",
    "jm": CALICO_DIR / "sub_JM_R_res_scale075_RE.zip",
}
PYTHON = ROOT / "venv311" / "Scripts" / "python.exe"

REQUIREMENTS = """numpy==1.26.4
pandas==2.3.3
scikit-learn==1.8.0
joblib==1.5.3
catboost==1.2.10
lightgbm==4.3.0
xgboost==2.0.3
"""

WRAPPER = r'''"""Sequential row-local blend of the frozen current and Calico endpoints."""
from __future__ import annotations

import gc
import json
import os
import runpy
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"
ROOT = Path(".").resolve()
OUTPUT = ROOT / "output" / "submission.csv"
SAMPLE = ROOT / "data" / "sample_submission.csv"
ENDPOINT_ARCHIVE = ROOT / "model" / "calico_endpoint.zip"
RUNTIME = ROOT / "_calico_runtime"
ENDPOINT_KIND = __ENDPOINT_KIND__
ENDPOINT_WEIGHT = __ENDPOINT_WEIGHT__
JM_SCALE = __JM_SCALE__


def _invoke(script: Path, cwd: Path, namespace: str) -> pd.DataFrame:
    old = Path.cwd()
    try:
        os.chdir(cwd)
        module = runpy.run_path(str(script), run_name=namespace)
        main = module.get("main")
        if not callable(main):
            raise RuntimeError(f"component has no callable main(): {script}")
        main()
        output = cwd / "output" / "submission.csv"
        if not output.is_file():
            raise RuntimeError(f"component did not create {output}")
        frame = pd.read_csv(output, encoding="utf-8-sig")
        del main, module
        gc.collect()
        return frame
    finally:
        os.chdir(old)


def _validate(frame: pd.DataFrame, name: str) -> pd.Series:
    frame.columns = [str(c).replace("\ufeff", "").strip() for c in frame.columns]
    if list(frame.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"{name}: bad columns {frame.columns.tolist()}")
    if frame[ID_COL].duplicated().any():
        raise ValueError(f"{name}: duplicate row_id")
    values = frame[TARGET_COL].to_numpy(np.float64)
    if not np.isfinite(values).all() or not ((values >= 0).all() and (values <= 1).all()):
        raise ValueError(f"{name}: invalid probabilities")
    return pd.Series(values, index=frame[ID_COL].astype(str), name=name)


def main() -> None:
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    RUNTIME.mkdir(parents=True)
    try:
        current = _validate(_invoke(ROOT / "current_script.py", ROOT, "current_component"), "current")

        with zipfile.ZipFile(ENDPOINT_ARCHIVE) as archive:
            archive.extractall(RUNTIME)
        data = RUNTIME / "data"
        data.mkdir(exist_ok=True)
        shutil.copy2(ROOT / "data" / "test.csv", data / "test.csv")
        shutil.copy2(SAMPLE, data / "sample_submission.csv")

        if ENDPOINT_KIND == "jm":
            manifest_path = RUNTIME / "model" / "r_residual_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["scale"] = float(JM_SCALE)
            manifest["version"] = f"calico-authorized-runtime-scale-{JM_SCALE:g}"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        endpoint = _validate(_invoke(RUNTIME / "script.py", RUNTIME, "calico_component"), ENDPOINT_KIND)
        sample = pd.read_csv(SAMPLE, encoding="utf-8-sig")
        sample.columns = [str(c).replace("\ufeff", "").strip() for c in sample.columns]
        ids = sample[ID_COL].astype(str)
        p_current = current.reindex(ids).to_numpy(np.float64)
        p_endpoint = endpoint.reindex(ids).to_numpy(np.float64)
        if not np.isfinite(p_current).all() or not np.isfinite(p_endpoint).all():
            raise ValueError("row_id alignment produced missing predictions")
        probability = (1.0 - ENDPOINT_WEIGHT) * p_current + ENDPOINT_WEIGHT * p_endpoint
        if not np.isfinite(probability).all() or not ((probability >= 0).all() and (probability <= 1).all()):
            raise ValueError("invalid convex blend")
        (ROOT / "output").mkdir(exist_ok=True)
        pd.DataFrame({ID_COL: sample[ID_COL], TARGET_COL: probability}).to_csv(
            OUTPUT, index=False, encoding="utf-8")
        print(
            f"Saved {OUTPUT}: rows={len(sample):,} endpoint={ENDPOINT_KIND} "
            f"weight={ENDPOINT_WEIGHT:.9f} mean={probability.mean():.9f}", flush=True)
    finally:
        shutil.rmtree(RUNTIME, ignore_errors=True)


if __name__ == "__main__":
    main()
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def wrapper_text(endpoint: str, weight: float, jm_scale: float) -> str:
    return (
        WRAPPER.replace("__ENDPOINT_KIND__", repr(endpoint))
        .replace("__ENDPOINT_WEIGHT__", repr(float(weight)))
        .replace("__JM_SCALE__", repr(float(jm_scale)))
    )


def build(endpoint: str, weight: float, jm_scale: float, output: Path) -> dict:
    endpoint_zip = ENDPOINTS[endpoint]
    if not CURRENT.is_file() or not endpoint_zip.is_file():
        raise FileNotFoundError((CURRENT, endpoint_zip))
    metadata = {
        "kind": "authorized_calico_pair_probe",
        "current_zip": CURRENT.name,
        "current_sha256": sha256(CURRENT),
        "endpoint": endpoint,
        "endpoint_zip": endpoint_zip.name,
        "endpoint_sha256": sha256(endpoint_zip),
        "endpoint_weight": weight,
        "jm_scale": jm_scale if endpoint == "jm" else None,
        "row_local": True,
    }
    with zipfile.ZipFile(CURRENT) as source, zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as target:
        for item in source.infolist():
            if item.filename in {"script.py", "requirements.txt"}:
                continue
            target.writestr(item, source.read(item.filename))
        target.writestr("current_script.py", source.read("script.py"))
        target.writestr("script.py", wrapper_text(endpoint, weight, jm_scale))
        target.writestr("requirements.txt", REQUIREMENTS)
        target.writestr("model/calico_blend_metadata.json", json.dumps(metadata, indent=2))
        nested = zipfile.ZipInfo("model/calico_endpoint.zip")
        nested.compress_type = zipfile.ZIP_STORED
        nested.external_attr = 0o100644 << 16
        with endpoint_zip.open("rb") as handle:
            target.writestr(nested, handle.read())
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"CRC failure: {bad}")
    metadata["output"] = output.name
    metadata["output_bytes"] = output.stat().st_size
    metadata["output_sha256"] = sha256(output)
    return metadata


def smoke(output: Path) -> dict:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="calico_pair_") as raw:
        work = Path(raw)
        with zipfile.ZipFile(output) as archive:
            archive.extractall(work)
        (work / "data").mkdir()
        shutil.copy2(ROOT / "data" / "test.csv", work / "data" / "test.csv")
        shutil.copy2(ROOT / "data" / "sample_submission.csv", work / "data" / "sample_submission.csv")
        result = subprocess.run(
            [str(PYTHON), "script.py"], cwd=work, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=900,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode:
            raise RuntimeError((result.stdout + "\n" + result.stderr)[-8000:])
        import pandas as pd
        import numpy as np
        out = pd.read_csv(work / "output" / "submission.csv")
        values = out["control_success"].to_numpy(np.float64)
        if len(out) != 5 or not np.isfinite(values).all() or not ((values >= 0).all() and (values <= 1).all()):
            raise RuntimeError("invalid smoke output")
        return {
            "seconds": time.perf_counter() - started,
            "rows": len(out),
            "mean": float(values.mean()),
            "stdout_tail": result.stdout[-2000:],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", choices=ENDPOINTS, required=True)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--jm-scale", type=float, default=0.75)
    parser.add_argument("--output")
    parser.add_argument("--no-smoke", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.weight <= 1.0:
        raise SystemExit("probe weight must be convex")
    suffix = "jy015" if args.endpoint == "jy" else f"jm{args.jm_scale:.4f}".replace(".", "")
    output = ROOT / (args.output or f"candidate_current_{suffix}_w{args.weight:.4f}.zip".replace(".", ""))
    info = build(args.endpoint, args.weight, args.jm_scale, output)
    if not args.no_smoke:
        info["smoke"] = smoke(output)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
