"""Rebase two already-built, row-local Tensor candidates on the affine champion.

This is a packaging-only operation.  It neither fits a model nor reads test data.
The source Tensor tables are our own official-train fits from EXP180/181.  The
affine anchor is our own submitted candidate.  Outputs remain HOLD artifacts
until their external-transfer risk is explicitly accepted.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AFFINE = ROOT / 'artifacts/candidates/candidate_affine_opt.zip'
SOURCES = {
    "s020": ROOT / 'artifacts/candidates/candidate_tensor_s020.zip',
    "robust": ROOT / 'artifacts/candidates/candidate_exp181_robusttensor_hold.zip',
}
OUTPUTS = {
    "s020": ROOT / 'artifacts/candidates/candidate_exp185_affine_tensor_s020_hold.zip',
    "robust": ROOT / 'artifacts/candidates/candidate_exp185_affine_robusttensor_hold.zip',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_zip(source: Path, output: Path, replacements: dict[str, bytes]) -> None:
    with zipfile.ZipFile(source, "r") as incoming, zipfile.ZipFile(
        output, "w", zipfile.ZIP_DEFLATED
    ) as outgoing:
        names = incoming.namelist()
        for name in names:
            payload = replacements.get(name, incoming.read(name))
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 1, 7, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            outgoing.writestr(info, payload)
    with zipfile.ZipFile(output, "r") as check:
        if check.testzip() is not None:
            raise RuntimeError(f"corrupt output: {output}")


def build_s020() -> dict[str, object]:
    source = SOURCES["s020"]
    with zipfile.ZipFile(source, "r") as archive:
        script = archive.read("script.py").decode("utf-8")
    substitutions = {
        "CHAMPION_WEIGHT = 0.6434428305247574": "CHAMPION_WEIGHT = 0.655318",
        "EXP021_WEIGHT = 0.35655716947524263": "EXP021_WEIGHT = 0.38781",
        "anchor = CHAMPION_WEIGHT * champion_values + EXP021_WEIGHT * exp021[\n        TARGET_COL\n    ].to_numpy(np.float64)": (
            "anchor = CHAMPION_WEIGHT * champion_values + EXP021_WEIGHT * exp021[\n"
            "        TARGET_COL\n    ].to_numpy(np.float64)\n"
            "    anchor = np.clip(0.44 + (anchor - 0.44) - 0.020206959, 0.0, 1.0)"
        ),
    }
    for old, new in substitutions.items():
        if script.count(old) != 1:
            raise RuntimeError(f"s020 patch token count != 1: {old[:40]}")
        script = script.replace(old, new)
    output = OUTPUTS["s020"]
    write_zip(source, output, {"script.py": script.encode("utf-8")})
    return {"source": source.name, "output": output.name, "sha256": sha256(output),
            "bytes": output.stat().st_size}


def build_robust() -> dict[str, object]:
    source = SOURCES["robust"]
    with zipfile.ZipFile(AFFINE, "r") as archive:
        affine_script = archive.read("script.py")
    output = OUTPUTS["robust"]
    write_zip(source, output, {"model/current_blend_inference.py": affine_script})
    return {"source": source.name, "output": output.name, "sha256": sha256(output),
            "bytes": output.stat().st_size}


def main() -> None:
    for path in [AFFINE, *SOURCES.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    report = {
        "experiment": 185,
        "status": "HOLD_EXTERNAL_TRANSFER_LOTTERY_NOT_SUBMITTED",
        "affine_anchor": {"file": AFFINE.name, "sha256": sha256(AFFINE)},
        "artifacts": [build_s020(), build_robust()],
        "note": "Packaging-only rebase; EXP180/181 validation failures still apply.",
    }
    destination = ROOT / "lab" / "185_affine_tensor_rebase.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
