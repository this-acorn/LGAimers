"""Build the final current/JM/JY affine submission with cross-ZIP deduplication.

The deployed prediction is::

    current
    + X0 * (joa - current)
    + XR * ((jm075 - joa) / 0.75)
    + XJY * (jy - current)

JM is executed once and its internal JOA anchor submission is captured before
the residual endpoint overwrites ``output/submission.csv``.  Files shared by
the frozen JM and JY ZIPs (same uncompressed size and CRC32) are stored once;
the root wrapper recreates both package layouts using hard links, falling back
to ordinary copies when hard links are unavailable.
"""

from __future__ import annotations

import argparse
import collections
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

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CURRENT_ZIP = ROOT / 'artifacts/candidates/candidate_affine_opt.zip'
JM_ZIP = ROOT / "reference/calico_JY_lfs_download/final/sub_JM_R_res_scale075_RE.zip"
JY_ZIP = ROOT / "reference/calico_JY_lfs_download/final/sub_JY_team_residual_scale015.zip"
OUTPUT_ZIP = ROOT / 'artifacts/candidates/last.zip'
PYTHON = ROOT / "venv311/Scripts/python.exe"

LOCKED_SHA256 = {
    "current": "E772CA86DAF99209A50AB9E29F030EE680C0770520D98FF5ABE9C3FD4832EC65",
    "jm": "46AA2A15130ED9BA8D302F203FA42C8E3F1730ACE5D47B3D4EF80429AE2B053F",
    "jy": "990B6441D03BC31CD1A9FE93AB0716CC12EFACD15DA2F024B1E059760DC6799D",
}

X0 = -0.4478235679645368
XR = 1.7571769012742058
XJY = 1.2342702603060989
JM_SCALE = 0.75
BUFFER_SIZE = 4 * 1024 * 1024


WRAPPER = r'''"""Final frozen current/JM/JY affine endpoint."""
from __future__ import annotations

import json
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
X0 = __X0__
XR = __XR__
XJY = __XJY__
JM_SCALE = __JM_SCALE__
BUNDLES = {
    "current": ROOT / "model/current_bundle",
    "jm": ROOT / "model/jm_bundle",
    "jy": ROOT / "model/jy_bundle",
}


def _materialize_shared() -> None:
    manifest = json.loads((ROOT / "model/dedup_manifest.json").read_text(encoding="utf-8"))
    linked = copied = existing = 0
    materialized_bytes = 0
    for item in manifest["shared_files"]:
        source = ROOT / item["source"]
        if not source.is_file() or source.stat().st_size != int(item["size"]):
            raise RuntimeError(f"invalid shared payload: {source}")
        for relative in item["targets"]:
            target = ROOT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not target.is_file() or target.stat().st_size != int(item["size"]):
                    raise RuntimeError(f"invalid existing materialized file: {target}")
                existing += 1
                continue
            try:
                os.link(source, target)
                linked += 1
            except OSError:
                shutil.copy2(source, target)
                copied += 1
            materialized_bytes += int(item["size"])
    print(
        f"[dedup] linked={linked} copied={copied} existing={existing} "
        f"logical_bytes={materialized_bytes}", flush=True)


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


def _run(name: str) -> pd.DataFrame:
    bundle = BUNDLES[name]
    _attach_data(bundle)
    out_dir = bundle / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "submission.csv"
    if out_path.exists():
        out_path.unlink()
    anchor_path = out_dir / "joa_anchor.csv"
    if anchor_path.exists():
        anchor_path.unlink()
    started = time.time()
    result = subprocess.run(
        [sys.executable, "script.py"], cwd=bundle, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    elapsed = time.time() - started
    print(f"[{name}] rc={result.returncode} elapsed={elapsed:.2f}s", flush=True)
    if result.stdout:
        print(result.stdout[-5000:], flush=True)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr[-10000:], file=sys.stderr, flush=True)
        raise RuntimeError(f"{name} endpoint failed")
    if not out_path.is_file():
        raise RuntimeError(f"{name} endpoint did not create {out_path}")
    return pd.read_csv(out_path, encoding="utf-8-sig")


def _provide_jy_anchor(source: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"missing captured JOA anchor: {source}")
    destination = BUNDLES["jy"] / "joa_anchor.csv"
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
        method = "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        method = "copy"
    print(f"[anchor] JM -> JY via {method}: {destination}", flush=True)


def _align(frame: pd.DataFrame, ids: pd.Series, name: str) -> np.ndarray:
    frame.columns = [str(column).replace("\ufeff", "").strip() for column in frame.columns]
    if list(frame.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"{name}: invalid columns {frame.columns.tolist()}")
    normalized = frame[ID_COL].astype(str)
    if normalized.duplicated().any():
        raise ValueError(f"{name}: duplicate row_id")
    indexed = pd.Series(frame[TARGET_COL].to_numpy(dtype=np.float64), index=normalized)
    wanted = ids.astype(str)
    values = indexed.reindex(wanted.to_numpy()).to_numpy(dtype=np.float64)
    if len(indexed) != len(wanted) or not np.isfinite(values).all():
        raise ValueError(f"{name}: row_id mismatch or non-finite prediction")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name}: component probability outside [0,1]")
    return values


def main() -> None:
    _materialize_shared()
    sample = pd.read_csv(DATA / "sample_submission.csv", encoding="utf-8-sig")
    sample.columns = [str(column).replace("\ufeff", "").strip() for column in sample.columns]
    if list(sample.columns) != [ID_COL, TARGET_COL] or sample[ID_COL].astype(str).duplicated().any():
        raise ValueError("invalid sample_submission")

    current_frame = _run("current")
    jm_frame = _run("jm")
    anchor_path = BUNDLES["jm"] / "output/joa_anchor.csv"
    if not anchor_path.is_file():
        raise RuntimeError("instrumented JM endpoint did not preserve JOA anchor")
    joa_frame = pd.read_csv(anchor_path, encoding="utf-8-sig")
    _provide_jy_anchor(anchor_path)
    jy_frame = _run("jy")

    ids = sample[ID_COL]
    current = _align(current_frame, ids, "current")
    joa = _align(joa_frame, ids, "joa")
    jm = _align(jm_frame, ids, "jm075")
    jy = _align(jy_frame, ids, "jy")
    raw_prediction = (
        current
        + X0 * (joa - current)
        + XR * ((jm - joa) / JM_SCALE)
        + XJY * (jy - current)
    )
    if not np.isfinite(raw_prediction).all():
        raise ValueError("final affine prediction contains non-finite values")
    clipped_low = int((raw_prediction < 0.0).sum())
    clipped_high = int((raw_prediction > 1.0).sum())
    prediction = np.clip(raw_prediction, 0.0, 1.0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COL: sample[ID_COL], TARGET_COL: prediction}).to_csv(
        OUTPUT / "submission.csv", index=False, encoding="utf-8")
    print(
        f"[last] rows={len(prediction)} x0={X0:.16g} xr={XR:.16g} xjy={XJY:.16g} "
        f"raw_min={raw_prediction.min():.9f} raw_max={raw_prediction.max():.9f} "
        f"clipped_low={clipped_low} clipped_high={clipped_high} "
        f"mean={prediction.mean():.9f} min={prediction.min():.9f} max={prediction.max():.9f}", flush=True)


if __name__ == "__main__":
    main()
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def safe_name(name: str) -> None:
    value = PurePosixPath(name)
    if value.is_absolute() or ".." in value.parts or "\\" in name:
        raise ValueError(f"unsafe ZIP member: {name!r}")


def make_info(name: str, source: zipfile.ZipInfo | None = None) -> zipfile.ZipInfo:
    safe_name(name)
    info = zipfile.ZipInfo(name, source.date_time if source else (2026, 9, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = source.external_attr if source else (0o100644 << 16)
    return info


def write_bytes(target: zipfile.ZipFile, name: str, value: bytes) -> None:
    target.writestr(make_info(name), value)


def stream_member(
    source_zip: zipfile.ZipFile,
    source_info: zipfile.ZipInfo,
    target_zip: zipfile.ZipFile,
    target_name: str,
) -> None:
    safe_name(source_info.filename)
    with source_zip.open(source_info, "r") as source, target_zip.open(
        make_info(target_name, source_info), "w", force_zip64=True
    ) as target:
        shutil.copyfileobj(source, target, length=BUFFER_SIZE)


def patched_jm_script(source: bytes) -> bytes:
    text = source.decode("utf-8")
    needle = '    return pd.read_csv(output, encoding="utf-8-sig")'
    replacement = (
        '    base = pd.read_csv(output, encoding="utf-8-sig")\n'
        '    base.to_csv("./output/joa_anchor.csv", index=False, encoding="utf-8")\n'
        '    return base'
    )
    if text.count(needle) != 1 or "joa_anchor.csv" in text:
        raise AssertionError("JM anchor-capture sentinel drifted")
    return text.replace(needle, replacement).encode("utf-8")


def patched_jm_manifest(source: bytes) -> bytes:
    payload = json.loads(source.decode("utf-8"))
    if float(payload.get("scale", -1.0)) != 0.075:
        raise AssertionError(f"unexpected JM source scale: {payload.get('scale')}")
    payload["scale"] = JM_SCALE
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def patched_jy_script(source: bytes) -> bytes:
    """Replace only JY's duplicate JOA inference with strict captured-anchor loading."""
    text = source.decode("utf-8")
    needle = "    p_anchor = predict_teammate(test)"
    replacement = '''    anchor_frame = pd.read_csv("./joa_anchor.csv", encoding="utf-8-sig")
    anchor_frame.columns = [str(c).replace("\\ufeff", "").strip() for c in anchor_frame.columns]
    if list(anchor_frame.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"captured JOA anchor schema mismatch: {anchor_frame.columns.tolist()}")
    anchor_ids = anchor_frame[ID_COL].astype(str)
    if anchor_ids.duplicated().any():
        raise ValueError("captured JOA anchor has duplicate row_id")
    wanted_ids = test[ID_COL].astype(str)
    if len(anchor_frame) != len(test) or set(anchor_ids) != set(wanted_ids):
        raise ValueError("captured JOA anchor row_id mismatch")
    anchor_by_id = pd.Series(anchor_frame[TARGET_COL].to_numpy(float), index=anchor_ids)
    p_anchor = anchor_by_id.reindex(wanted_ids.to_numpy()).to_numpy(dtype=np.float64)
    if not np.isfinite(p_anchor).all() or np.any((p_anchor < 0.0) | (p_anchor > 1.0)):
        raise ValueError("captured JOA anchor contains invalid probability")'''
    if text.count(needle) != 1 or "captured JOA anchor schema mismatch" in text:
        raise AssertionError("JY anchor-reuse sentinel drifted")
    return text.replace(needle, replacement).encode("utf-8")


def non_directory(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    result = []
    for info in archive.infolist():
        safe_name(info.filename)
        if not info.is_dir():
            result.append(info)
    return result


def merged_requirements() -> bytes:
    # Pins were jointly smoke-tested in the workspace venv.  JY's pandas,
    # LightGBM, and XGBoost versions take precedence over the current bundle.
    return (
        "numpy==1.26.4\n"
        "pandas==2.3.3\n"
        "scikit-learn==1.8.0\n"
        "joblib==1.5.3\n"
        "catboost==1.2.10\n"
        "lightgbm==4.3.0\n"
        "xgboost==2.0.3\n"
    ).encode("utf-8")


def build(output: Path) -> dict:
    for name, path in (("current", CURRENT_ZIP), ("jm", JM_ZIP), ("jy", JY_ZIP)):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != LOCKED_SHA256[name]:
            raise AssertionError(f"{name} frozen ZIP SHA mismatch: {actual}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    partial = output.with_name(output.name + ".building")
    if partial.exists():
        raise FileExistsError(f"refusing stale partial {partial}")

    wrapper = (
        WRAPPER.replace("__X0__", repr(X0))
        .replace("__XR__", repr(XR))
        .replace("__XJY__", repr(XJY))
        .replace("__JM_SCALE__", repr(JM_SCALE))
    ).encode("utf-8")

    metadata = {
        "name": "last_current_jm075_jy_affine",
        "formula": "current+x0*(joa-current)+xr*((jm075-joa)/0.75)+xjy*(jy-current)",
        "coordinates": {"x0": X0, "xr": XR, "xjy": XJY},
        "jm_source_scale": 0.075,
        "jm_deployed_scale": JM_SCALE,
        "sources": {
            "current": {"name": CURRENT_ZIP.name, "sha256": LOCKED_SHA256["current"]},
            "jm": {"name": JM_ZIP.name, "sha256": LOCKED_SHA256["jm"]},
            "jy": {"name": JY_ZIP.name, "sha256": LOCKED_SHA256["jy"]},
        },
        "jm_instrumentation": "capture exact internal JOA output before residual overwrite",
        "jy_instrumentation": "strictly reuse JM-captured JOA; skip duplicate teammate inference",
        "dedup_key": "uncompressed_size+CRC32; source member verified by ZIP CRC",
    }

    try:
        with zipfile.ZipFile(CURRENT_ZIP) as current, zipfile.ZipFile(JM_ZIP) as jm, zipfile.ZipFile(JY_ZIP) as jy:
            archives = {"current": current, "jm": jm, "jy": jy}
            infos = {name: non_directory(archive) for name, archive in archives.items()}
            # The complete source bytes are already locked by SHA-256 above;
            # re-inflating all three large archives here would duplicate that
            # integrity check and materially delay the final build.  Member
            # streams still validate while being read, and the completed
            # output is fully CRC-tested below.

            by_key: dict[str, dict[tuple[int, int], list[zipfile.ZipInfo]]] = {}
            for name in ("jm", "jy"):
                grouped: dict[tuple[int, int], list[zipfile.ZipInfo]] = collections.defaultdict(list)
                for info in infos[name]:
                    if info.filename != "requirements.txt":
                        grouped[(info.file_size, info.CRC)].append(info)
                by_key[name] = grouped
            common_keys = sorted(set(by_key["jm"]) & set(by_key["jy"]), reverse=True)
            shared_members = []
            shared_targets: set[tuple[str, str]] = set()
            for index, key in enumerate(common_keys):
                size, crc = key
                source_name = f"model/shared/s{index:04d}.bin"
                targets = []
                for package in ("jm", "jy"):
                    for info in by_key[package][key]:
                        target_name = f"model/{package}_bundle/{info.filename}"
                        targets.append(target_name)
                        shared_targets.add((package, info.filename))
                shared_members.append(
                    {
                        "source": source_name,
                        "size": size,
                        "crc32": f"{crc:08X}",
                        "targets": targets,
                        "source_member": by_key["jm"][key][0].filename,
                    }
                )

            metadata["dedup"] = {
                "shared_groups": len(shared_members),
                "shared_target_files": sum(len(item["targets"]) for item in shared_members),
                "saved_uncompressed_bytes": sum(
                    item["size"] * (len(item["targets"]) - 1) for item in shared_members
                ),
            }
            manifest = {"version": 1, "shared_files": shared_members}

            with zipfile.ZipFile(
                partial,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
                allowZip64=True,
            ) as target:
                write_bytes(target, "script.py", wrapper)
                write_bytes(target, "requirements.txt", merged_requirements())

                for info in infos["current"]:
                    if info.filename == "requirements.txt":
                        continue
                    stream_member(current, info, target, f"model/current_bundle/{info.filename}")

                jm_script = patched_jm_script(jm.read("script.py"))
                jm_manifest = patched_jm_manifest(jm.read("model/r_residual_manifest.json"))
                for info in infos["jm"]:
                    if info.filename == "requirements.txt" or ("jm", info.filename) in shared_targets:
                        continue
                    target_name = f"model/jm_bundle/{info.filename}"
                    if info.filename == "script.py":
                        write_bytes(target, target_name, jm_script)
                    elif info.filename == "model/r_residual_manifest.json":
                        write_bytes(target, target_name, jm_manifest)
                    else:
                        stream_member(jm, info, target, target_name)

                for info in infos["jy"]:
                    if info.filename == "requirements.txt" or ("jy", info.filename) in shared_targets:
                        continue
                    target_name = f"model/jy_bundle/{info.filename}"
                    if info.filename == "script.py":
                        write_bytes(target, target_name, patched_jy_script(jy.read("script.py")))
                    else:
                        stream_member(jy, info, target, target_name)

                for item in shared_members:
                    info = by_key["jm"][(item["size"], int(item["crc32"], 16))][0]
                    stream_member(jm, info, target, item["source"])

                write_bytes(
                    target,
                    "model/dedup_manifest.json",
                    (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
                )
                write_bytes(
                    target,
                    "model/last_metadata.json",
                    (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
                )

        with zipfile.ZipFile(partial) as result:
            names = result.namelist()
            if names[:2] != ["script.py", "requirements.txt"]:
                raise AssertionError("root entry order drift")
            if len(names) != len(set(names)):
                raise AssertionError("duplicate output ZIP entries")
            if any(name.startswith(("data/", "output/")) or "__pycache__" in name for name in names):
                raise AssertionError("runtime files leaked into output ZIP")
            bad = result.testzip()
            if bad is not None:
                raise AssertionError(f"output CRC failure: {bad}")
            metadata["output_entries"] = len(names)
            metadata["output_uncompressed_bytes"] = sum(info.file_size for info in result.infolist())
        partial.replace(output)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise

    metadata["output_bytes"] = output.stat().st_size
    metadata["output_sha256"] = sha256(output)
    return metadata


def smoke(zip_path: Path, full_data: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="lg191_last_") as raw:
        work = Path(raw)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(work)
        data = work / "data"
        data.mkdir()
        test = pd.read_csv(full_data / "test.csv", encoding="utf-8-sig", nrows=5)
        sample = pd.read_csv(full_data / "sample_submission.csv", encoding="utf-8-sig", nrows=5)
        if not test["row_id"].astype(str).equals(sample["row_id"].astype(str)):
            raise AssertionError("smoke source row_id mismatch")
        test.to_csv(data / "test.csv", index=False, encoding="utf-8")
        sample.to_csv(data / "sample_submission.csv", index=False, encoding="utf-8")
        started = time.time()
        result = subprocess.run(
            [str(PYTHON), "script.py"], cwd=work, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        elapsed = time.time() - started
        print((result.stdout or "")[-12000:].encode("ascii", "backslashreplace").decode("ascii"))
        if result.returncode != 0:
            print((result.stderr or "")[-16000:].encode("ascii", "backslashreplace").decode("ascii"), file=sys.stderr)
            raise RuntimeError(f"smoke failed rc={result.returncode}")
        output = pd.read_csv(work / "output/submission.csv", encoding="utf-8-sig")
        if list(output.columns) != ["row_id", "control_success"]:
            raise AssertionError("smoke schema mismatch")
        if not output["row_id"].astype(str).equals(sample["row_id"].astype(str)):
            raise AssertionError("smoke row order mismatch")

        def component(name: str, filename: str = "submission.csv") -> np.ndarray:
            frame = pd.read_csv(work / f"model/{name}_bundle/output/{filename}", encoding="utf-8-sig")
            indexed = pd.Series(
                frame["control_success"].to_numpy(float), index=frame["row_id"].astype(str))
            return indexed.reindex(sample["row_id"].astype(str)).to_numpy(float)

        current = component("current")
        jm = component("jm")
        joa = component("jm", "joa_anchor.csv")
        jy = component("jy")
        expected = np.clip(
            current + X0 * (joa - current) + XR * ((jm - joa) / JM_SCALE) + XJY * (jy - current),
            0.0,
            1.0,
        )
        actual = output["control_success"].to_numpy(float)
        max_error = float(np.max(np.abs(expected - actual)))
        if max_error > 5e-15:
            raise AssertionError(f"smoke affine mismatch: {max_error}")
        return {
            "returncode": result.returncode,
            "elapsed_seconds": elapsed,
            "rows": len(output),
            "row_order_exact": True,
            "mean": float(actual.mean()),
            "min": float(actual.min()),
            "max": float(actual.max()),
            "affine_max_abs_error": max_error,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ZIP)
    parser.add_argument("--smoke-data", type=Path, default=ROOT / 'archive/scratch/_rehearse_181/full/data')
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    started = time.time()
    report = build(args.output.resolve())
    report["build_seconds"] = time.time() - started
    print(json.dumps(report, indent=2))
    if not args.skip_smoke:
        report["smoke"] = smoke(args.output.resolve(), args.smoke_data.resolve())
        print(json.dumps(report["smoke"], indent=2))


if __name__ == "__main__":
    main()
