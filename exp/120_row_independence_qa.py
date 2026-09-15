# -*- coding: utf-8 -*-
"""Validate row-independent inference for a selected submission archive.

Compare batch, singleton, reverse-order, shuffled, split, and duplicated-row
execution with an absolute tolerance of 1e-12. Extract and execute the chosen
ZIP in a temporary directory without rebuilding it.
"""

_PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--zip", default='artifacts/candidates/candidate_v18g030.zip')
ap.add_argument("--n", type=int, default=4000, help="검사에 쓸 행 수 (train 앞부분을 test 형식으로)")
args = ap.parse_args()

ROOT = Path(str(_PROJECT_ROOT))
ZIP = ROOT / args.zip
WORK = ROOT / "_tmp_qa111"
VENV_PY = (str(_PROJECT_ROOT / "venv311/Scripts/python.exe"))
TOL = 1e-12
OUT_TXT = ROOT / "lab/120_row_independence_qa.txt"
OUT_JSON = ROOT / "lab/120_row_independence_qa.json"

lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


ok_all = True


def check(cond, ok, ng):
    global ok_all
    log(f"    [{'OK' if cond else '!!'}] {ok if cond else ng}")
    if not cond:
        ok_all = False
    return cond


log("=" * 78)
log(f"[120] 행 독립성 QA — {args.zip}")
log("=" * 78)
check(ZIP.exists(), f"ZIP 존재 {ZIP.name}", "ZIP 없음")
sha = hashlib.sha256(ZIP.read_bytes()).hexdigest().upper()
log(f"    SHA256 {sha}")

if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir()
with zipfile.ZipFile(ZIP) as z:
    names = sorted(z.namelist())
    log(f"    구성 {names}")

# --- 검사용 test 프레임 (train 앞부분을 test 형식으로; canonical test.csv 는 열지 않는다) ---
test_cols = [c.replace("\ufeff", "").strip()
             for c in pd.read_csv(ROOT / "data/test.csv", encoding="utf-8-sig", nrows=0).columns]
src = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", nrows=args.n)
src.columns = [c.replace("\ufeff", "").strip() for c in src.columns]
base = src[[c for c in test_cols if c != "row_id"]].copy()
base.insert(0, "row_id", [f"TEST_{i:06d}" for i in range(len(base))])
log(f"    검사 행 {len(base):,}  (train 앞부분을 test 형식으로 구성; canonical test.csv 미사용)")


def run_case(name, frame):
    """frame 을 test.csv 로 두고 script.py 실행 → {row_id: prediction} 반환"""
    d = WORK / name
    if d.exists():
        shutil.rmtree(d)
    (d / "data").mkdir(parents=True)
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(d)
    frame.to_csv(d / "data/test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"row_id": frame["row_id"], "control_success": 0.5}).to_csv(
        d / "data/sample_submission.csv", index=False, encoding="utf-8-sig")
    t0 = time.time()
    r = subprocess.run([VENV_PY, "-u", "script.py"], cwd=d, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    if r.returncode != 0:
        log(f"    [!!] {name} exit {r.returncode}")
        for ln in (r.stderr or "").splitlines()[-12:]:
            log(f"       | {ln}")
        return None, dt
    sub = pd.read_csv(d / "output/submission.csv")
    return dict(zip(sub["row_id"].astype(str), sub["control_success"].astype(float))), dt


results, timings = {}, {}
rng = np.random.default_rng(42)

cases = {
    "batch": base,
    "reverse": base.iloc[::-1].reset_index(drop=True),
    "permutation": base.iloc[rng.permutation(len(base))].reset_index(drop=True),
    "duplicate": pd.concat([base, base], ignore_index=True).assign(
        row_id=lambda d: [f"{r}_{i//len(base)}" for i, r in enumerate(d["row_id"])]),
}
log("")
log("케이스별 실행")
for name, frame in cases.items():
    res, dt = run_case(name, frame)
    results[name], timings[name] = res, dt
    log(f"    {name:12s} rows={len(frame):6,}  {dt:5.1f}s  {'OK' if res else 'FAIL'}")

# split: 절반씩 두 번 실행해 합침
half = len(base) // 2
res_a, ta = run_case("split_a", base.iloc[:half].reset_index(drop=True))
res_b, tb = run_case("split_b", base.iloc[half:].reset_index(drop=True))
if res_a and res_b:
    results["split"] = {**res_a, **res_b}
    timings["split"] = ta + tb
    log(f"    {'split':12s} rows={len(base):6,}  {ta+tb:5.1f}s  OK (2회 분할)")

# singleton: 처음 5행을 각각 1행짜리로
single = {}
for i in range(5):
    res, _ = run_case(f"single{i}", base.iloc[[i]].reset_index(drop=True))
    if res:
        single.update(res)
results["singleton5"] = single
log(f"    {'singleton5':12s} rows=     5  (1행씩 5회)  {'OK' if len(single) == 5 else 'FAIL'}")

log("")
log(f"batch 대비 최대 절대차 (tolerance {TOL:g})")
ref = results.get("batch")
detail = {}
if ref:
    for name in ["reverse", "permutation", "duplicate", "split", "singleton5"]:
        res = results.get(name)
        if not res:
            check(False, "", f"{name} 실행 실패")
            continue
        if name == "duplicate":
            pairs = [(abs(res[f"{k}_0"] - v), abs(res[f"{k}_1"] - v)) for k, v in ref.items()
                     if f"{k}_0" in res and f"{k}_1" in res]
            diff = max(max(a, b) for a, b in pairs) if pairs else float("nan")
            n_cmp = len(pairs) * 2
        else:
            common = [k for k in res if k in ref]
            diff = max(abs(res[k] - ref[k]) for k in common) if common else float("nan")
            n_cmp = len(common)
        detail[name] = {"max_abs_diff": float(diff), "n_compared": int(n_cmp)}
        check(diff <= TOL, f"{name:11s} n={n_cmp:6,}  최대차 {diff:.3e}",
              f"{name:11s} 최대차 {diff:.3e} > tol — 행 독립 위반")

pred = np.array(list(ref.values())) if ref else np.array([])
if pred.size:
    log("")
    log(f"    예측 통계: mean {pred.mean():.6f}  min {pred.min():.6f}  max {pred.max():.6f}  NaN {int(np.isnan(pred).sum())}")

shutil.rmtree(WORK, ignore_errors=True)
log("")
log("=" * 78)
log(("전 케이스 통과 — 각 행의 예측은 동반 행 구성·순서와 무관 (규정 4 충족)"
     if ok_all else "!! 위반 항목 있음"))
log("=" * 78)
OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
OUT_JSON.write_text(json.dumps({"zip": args.zip, "sha256": sha, "n_rows": int(len(base)),
                                "tolerance": TOL, "cases": detail, "timings": timings,
                                "all_pass": bool(ok_all)}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
sys.exit(0 if ok_all else 1)
