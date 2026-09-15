# -*- coding: utf-8 -*-
"""Create the analytic optimum EXP021 blend from the fully-QA'd 45% parent."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'artifacts/candidates/candidate_champ55.zip'
SOURCE = ROOT / 'candidates/candidate_exp021_w0356557_v002_src'
OUTPUT = ROOT / 'artifacts/candidates/candidate_exp021_w0356557_v002.zip'
REPORT = ROOT / "lab" / "133_exp021_w0356557_build.json"

PARENT_SHA256 = "32d7f275afd34134ecd34891e67b3128b20d817732364507b3f4fbe2f1630faf"
EXP021_WEIGHT = 0.35655716947524263
CHAMPION_WEIGHT = 1.0 - EXP021_WEIGHT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if sha256(PARENT) != PARENT_SHA256:
        raise AssertionError("parent blend SHA256 mismatch")
    if SOURCE.exists() or OUTPUT.exists():
        raise FileExistsError("refusing to overwrite an existing optimum artifact")
    if abs(CHAMPION_WEIGHT + EXP021_WEIGHT - 1.0) > 1e-15:
        raise AssertionError("weights do not sum to one")

    SOURCE.mkdir()
    with zipfile.ZipFile(PARENT) as archive:
        if archive.testzip() is not None:
            raise AssertionError("parent blend CRC failure")
        names = archive.namelist()
        archive.extractall(SOURCE)

    script_path = SOURCE / "script.py"
    script = script_path.read_text(encoding="utf-8")
    if script.count("CHAMPION_WEIGHT = 0.55") != 1 or script.count("EXP021_WEIGHT = 0.45") != 1:
        raise AssertionError("parent wrapper weight sentinels drifted")
    script = script.replace(
        "CHAMPION_WEIGHT = 0.55", f"CHAMPION_WEIGHT = {CHAMPION_WEIGHT!r}"
    ).replace(
        "EXP021_WEIGHT = 0.45", f"EXP021_WEIGHT = {EXP021_WEIGHT!r}"
    )
    script_path.write_text(script, encoding="utf-8", newline="\n")

    metadata_path = SOURCE / "model" / "blend_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "name": "exp021_w0356557_v002",
            "weights": {
                "champion": CHAMPION_WEIGHT,
                "exp021_strict": EXP021_WEIGHT,
            },
            "parent_blend": {
                "name": PARENT.name,
                "sha256": PARENT_SHA256,
                "public_score": 1113.1142204091,
            },
            "geometry": {
                "champion_score": 1092.808353586,
                "exp021_endpoint_score": 1043.6074197937,
                "probe_weight": 0.45,
                "probe_score": 1113.1142204091,
                "measured_K": 171.5001496146865,
                "analytic_weight": EXP021_WEIGHT,
                "projected_score": 1114.611684697336,
            },
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )

    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(SOURCE / "script.py", "script.py")
        archive.write(SOURCE / "requirements.txt", "requirements.txt")
        for path in sorted((SOURCE / "model").iterdir()):
            if path.is_file():
                archive.write(path, f"model/{path.name}")
    with zipfile.ZipFile(OUTPUT) as archive:
        bad = archive.testzip()
        output_names = archive.namelist()
        if bad is not None:
            raise AssertionError(f"optimum ZIP CRC failure: {bad}")
        if output_names != names:
            raise AssertionError("optimum ZIP entry order/schema drift")

    report = {
        "artifact": str(OUTPUT),
        "sha256": sha256(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "parent_sha256": PARENT_SHA256,
        "weights": {"champion": CHAMPION_WEIGHT, "exp021_strict": EXP021_WEIGHT},
        "measured_K": 171.5001496146865,
        "projected_score": 1114.611684697336,
        "entries": output_names,
        "crc": "passed",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
