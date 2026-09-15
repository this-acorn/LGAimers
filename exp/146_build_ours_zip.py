"""EXP-021 컴포넌트를 우리 자체 구현으로 교체한 제출 zip 을 빌드하고 종단 리허설한다.

절차
  1. 챔피언 소스를 복사하고 model/exp021_inference.py 만 우리 구현으로 교체
  2. __pycache__ 제거 후 zip (최상위 폴더 없이 model/ · script.py · requirements.txt)
  3. 245,789행 가짜 평가셋으로 script.py 를 그대로 실행 — 평가 서버와 같은 진입점
  4. 기존 챔피언 zip 의 최종 혼합 결과와 최대 절대차 0 인지 확인
  5. 선택적으로 사후 아핀 상수(--scale/--center/--shift)를 최종 혼합에 주입

사용법:
  python exp/146_build_ours_zip.py --name candidate_exp021_ours_v001
  python exp/146_build_ours_zip.py --name probe_a --scale 1.08 --skip-diff
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'candidates/candidate_exp021_w0356557_v002_src'
OURS = ROOT / "exp" / "143_exp021_component.py"
STAGE = ROOT / "_rehearse_146"
PYTHON = ROOT / "venv311" / "Scripts" / "python.exe"

AFFINE_BLOCK = '''
# --- 최종 혼합 확률에 대한 사후 선형 재조정 (행 단위 상수 연산, 행 간 독립) ---
BLEND_SCALE = {scale!r}
BLEND_CENTER = {center!r}
BLEND_SHIFT = {shift!r}
'''

AFFINE_APPLY = (
    "    blended = np.clip(\n"
    "        BLEND_CENTER + BLEND_SCALE * (blended - BLEND_CENTER) - BLEND_SHIFT, 0.0, 1.0)\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_source(dest: Path, scale: float, center: float, shift: float,
                 w_champ: float | None = None, w_exp021: float | None = None,
                 v18_gamma: float | None = None,
                 corr_scale: float | None = None) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(SRC, dest, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(OURS, dest / "model" / "exp021_inference.py")

    if corr_scale is not None:
        # EXP-021 의 해석적 베이스 대비 보정 성분 배율 — (p1,p2) 평면 밖 방향
        comp = dest / "model" / "exp021_inference.py"
        text = comp.read_text(encoding="utf-8")
        if text.count("CORRECTION_SCALE = 1.0") != 1:
            raise SystemExit("exp021_inference.py 의 CORRECTION_SCALE 앵커를 찾지 못했습니다")
        comp.write_text(text.replace("CORRECTION_SCALE = 1.0",
                                     f"CORRECTION_SCALE = {corr_scale!r}", 1), encoding="utf-8")

    if v18_gamma is not None:
        # 혼합이 V18 을 beta1 배로 희석하므로 챔피언 내부 감마를 되돌린다 (평면 밖 방향)
        champ = dest / "model" / "champion_inference.py"
        text = champ.read_text(encoding="utf-8")
        if text.count("V18_GAMMA = 0.30") != 1:
            raise SystemExit("champion_inference.py 의 V18_GAMMA 앵커를 찾지 못했습니다")
        champ.write_text(text.replace("V18_GAMMA = 0.30", f"V18_GAMMA = {v18_gamma!r}", 1),
                         encoding="utf-8")

    if w_champ is not None:
        # 두 컴포넌트에 독립 가중을 준다 (합이 1 이 아니어도 된다 = 3파라미터 아핀)
        script = (dest / "script.py").read_text(encoding="utf-8")
        for name, value in (("CHAMPION_WEIGHT", w_champ), ("EXP021_WEIGHT", w_exp021)):
            old = next(line for line in script.splitlines() if line.startswith(name + " = "))
            script = script.replace(old, f"{name} = {value!r}", 1)
        (dest / "script.py").write_text(script, encoding="utf-8")

    if (scale, shift) != (1.0, 0.0):
        script = (dest / "script.py").read_text(encoding="utf-8")
        anchor = 'SAMPLE_PATH = Path("./data/sample_submission.csv")\n'
        if anchor not in script:
            raise SystemExit("script.py 앵커를 찾지 못했습니다 (상수 블록)")
        script = script.replace(
            anchor,
            anchor + AFFINE_BLOCK.format(scale=scale, center=center, shift=shift), 1)
        target = ("    blended = CHAMPION_WEIGHT * champion_values "
                  "+ EXP021_WEIGHT * exp021_values\n")
        if target not in script:
            raise SystemExit("script.py 앵커를 찾지 못했습니다 (혼합식)")
        script = script.replace(target, target + AFFINE_APPLY, 1)
        (dest / "script.py").write_text(script, encoding="utf-8")


def make_zip(source: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(source).as_posix())


def rehearse(zip_path: Path, tag: str, rows: int, seed: int) -> np.ndarray:
    work = STAGE / tag
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if any(name.startswith(("/", "..")) for name in names):
            raise SystemExit(f"{tag}: zip 경로가 안전하지 않습니다")
        tops = {name.split("/")[0] for name in names}
        if tops != {"model", "script.py", "requirements.txt"}:
            raise SystemExit(f"{tag}: zip 최상위 구조 불일치 {sorted(tops)}")
        archive.extractall(work)
    shutil.copytree(STAGE / "data", work / "data")

    started = time.time()
    result = subprocess.run(
        [str(PYTHON), "script.py"], cwd=work, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    elapsed = time.time() - started
    if result.returncode != 0:
        print((result.stdout or "")[-3000:])
        print((result.stderr or "")[-4000:], file=sys.stderr)
        raise SystemExit(f"{tag}: script.py 실행 실패")
    print(f"  [{tag}] {elapsed:6.1f}s")
    for line in (result.stdout or "").strip().splitlines():
        print(f"      {line}")

    out = pd.read_csv(work / "output" / "submission.csv")
    if list(out.columns) != ["row_id", "control_success"]:
        raise SystemExit(f"{tag}: 제출 컬럼 불일치")
    if len(out) != rows:
        raise SystemExit(f"{tag}: 제출 행 수 불일치")
    return out["control_success"].to_numpy(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--center", type=float, default=0.44)
    parser.add_argument("--shift", type=float, default=0.0)
    parser.add_argument("--rows", type=int, default=245789)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--skip-diff", action="store_true")
    parser.add_argument("--w-champ", type=float, help="챔피언 컴포넌트 가중 (합 1 불필요)")
    parser.add_argument("--w-exp021", type=float, help="EXP-021 컴포넌트 가중")
    parser.add_argument("--v18-gamma", type=float, help="챔피언 내부 V18 감마 (기본 0.30)")
    parser.add_argument("--corr-scale", type=float, help="EXP-021 보정성분 배율 (기본 1.0)")
    args = parser.parse_args()
    if (args.w_champ is None) != (args.w_exp021 is None):
        raise SystemExit("--w-champ 와 --w-exp021 은 함께 지정해야 합니다")

    source = ROOT / f"{args.name}_src"
    zip_path = ROOT / f"{args.name}.zip"
    build_source(source, args.scale, args.center, args.shift, args.w_champ, args.w_exp021,
                 args.v18_gamma, args.corr_scale)
    make_zip(source, zip_path)
    print(f"빌드: {zip_path.name}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
    print(f"SHA256: {sha256(zip_path)}")
    if (args.scale, args.shift) != (1.0, 0.0):
        print(f"사후 아핀: center={args.center} scale={args.scale} shift={args.shift}")
    print()

    # 가짜 평가셋은 exp/145 와 같은 규칙으로 만든다
    sys.path.insert(0, str(ROOT / "exp"))
    build_fake_test = __import__("145_rehearse_reimpl").build_fake_test
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "data").mkdir(parents=True)
    real = pd.read_csv(ROOT / "data" / "test.csv", encoding="utf-8-sig")
    fake = build_fake_test(real, args.rows, args.seed)
    fake.to_csv(STAGE / "data" / "test.csv", index=False, encoding="utf-8")
    pd.DataFrame({"row_id": fake["row_id"], "control_success": 0.0}).to_csv(
        STAGE / "data" / "sample_submission.csv", index=False, encoding="utf-8")
    print(f"가짜 평가셋 {len(fake):,}행 — script.py 종단 리허설\n")

    ours = rehearse(zip_path, "ours", args.rows, args.seed)
    if args.skip_diff:
        print(f"\n평균 {ours.mean():.12f}  min {ours.min():.9f}  max {ours.max():.9f}")
        return

    reference = rehearse(ROOT / 'artifacts/candidates/candidate_exp021_w0356557.zip', "reference",
                         args.rows, args.seed)
    diff = np.abs(ours - reference)
    print(f"\n최대 절대차 : {diff.max():.3e}")
    print(f"불일치 행수 : {int((diff > 0).sum()):,} / {len(diff):,}")
    print(f"평균        : 기존 {reference.mean():.12f} / 우리 {ours.mean():.12f}")
    print("판정        : " + ("완전 일치 — 점수 1114.6116846973 유지"
                              if diff.max() == 0.0 else "불일치"))


if __name__ == "__main__":
    main()
