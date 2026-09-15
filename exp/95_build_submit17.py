# -*- coding: utf-8 -*-
"""
[95] submit17.zip 빌드+검증 — submit17_src/{script.py(v9), requirements.txt, model/model.pkl}

하민 exp/74 의 검증 절차 그대로 (구조/무결성 → 가짜 서버 245,789행 → 진짜 2025 5행) + pyarrow 부재 확인.
실행 (아무 python): PYTHONIOENCODING=utf-8 python -u exp/95_build_submit17.py [--zip submit17.zip]
★ 사용자 검토·승인 후에만 실행. LB 제출은 사용자가 직접.
"""

_PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--zip", default="submit17.zip")
ap.add_argument("--allow-smoke", action="store_true", help="exp/94 --smoke 번들로 빌드 경로만 검증할 때")
args = ap.parse_args()

ROOT = str(_PROJECT_ROOT)
SRC = f"{ROOT}/submissions/submit17_src"
OUT_ZIP = f"{ROOT}/{args.zip}"
VENV_PY = (str(_PROJECT_ROOT / "venv311/Scripts/python.exe"))
SIM = f"{ROOT}/_tmp_sim_submit17"
SIM5 = f"{ROOT}/_tmp_sim_submit17_5"
N_EVAL = 245789


def log(*a):
    print(*a, flush=True)


ok_all = True


def check(cond, ok, ng):
    global ok_all
    log(f"    [{'OK' if cond else '!!'}] {ok if cond else ng}")
    if not cond:
        ok_all = False
    return cond


log("=" * 80)
log("0. 환경 — venv311 (서버 동일 버전, pyarrow 없음)")
log("=" * 80)
r0 = subprocess.run([VENV_PY, "-c", "import sys,numpy,pandas;print(sys.version.split()[0],numpy.__version__,pandas.__version__)"],
                    capture_output=True, text=True)
log(f"    venv311: {r0.stdout.strip()}")
check(r0.stdout.startswith("3.11") and "1.26.4" in r0.stdout and "2.0.3" in r0.stdout, "버전 서버 동일", "버전 불일치")
r0b = subprocess.run([VENV_PY, "-c", "import pyarrow"], capture_output=True, text=True)
check(r0b.returncode != 0, "pyarrow 없음 (서버 동일)", "pyarrow 존재 — 서버와 다름, 로드 테스트 무의미")

log("\n" + "=" * 80)
log(f"A. {args.zip} 생성 및 구조 검사")
log("=" * 80)
req = open(f"{SRC}/requirements.txt", encoding="utf-8").read()
check("catboost==1.2.10" in req, "catboost 명시", "catboost 없음")
check(os.path.exists(f"{SRC}/model/model.pkl"), "model.pkl 존재", "model.pkl 없음 — exp/94 먼저")
check(not os.path.exists(f"{SRC}/model/model_partial.pkl"), "partial 없음(학습 완주)", "model_partial.pkl 남아있음 — 학습 미완주?")
items = [(f"{SRC}/script.py", "script.py"), (f"{SRC}/requirements.txt", "requirements.txt"),
         (f"{SRC}/model/model.pkl", "model/model.pkl")]
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for real, arc in items:
        z.write(real, arc)
with zipfile.ZipFile(OUT_ZIP) as z:
    check(sorted(z.namelist()) == ["model/model.pkl", "requirements.txt", "script.py"] and z.testzip() is None,
          f"구조/무결성 정상 ({os.path.getsize(OUT_ZIP)/1024**2:.1f} MB)", "zip 이상")

log("\n" + "=" * 80)
log(f"B. 가짜 평가 서버 ({N_EVAL:,}행, train 앞부분을 test 형식으로)")
log("=" * 80)
if os.path.exists(SIM):
    shutil.rmtree(SIM)
os.makedirs(f"{SIM}/data")
with zipfile.ZipFile(OUT_ZIP) as z:
    z.extractall(SIM)
test_cols = [c.replace("\ufeff", "").strip() for c in pd.read_csv(f"{ROOT}/data/test.csv", encoding="utf-8-sig", nrows=1).columns]
src = pd.read_csv(f"{ROOT}/data/train.csv", encoding="utf-8-sig", nrows=N_EVAL)
src.columns = [c.replace("\ufeff", "").strip() for c in src.columns]
fake = src[[c for c in test_cols if c != "row_id"]].copy()
fake.insert(0, "row_id", [f"TEST_{i:06d}" for i in range(len(fake))])
fake.to_csv(f"{SIM}/data/test.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"row_id": fake["row_id"], "control_success": 0.5}).to_csv(f"{SIM}/data/sample_submission.csv", index=False, encoding="utf-8-sig")
t0 = time.time()
r = subprocess.run([VENV_PY, "-u", "script.py"], cwd=SIM, capture_output=True, text=True, encoding="utf-8", errors="replace")
elapsed = time.time() - t0
for line in (r.stdout or "").splitlines():
    log(f"    | {line}")
if r.returncode != 0:
    for line in (r.stderr or "").splitlines()[-25:]:
        log(f"    | {line}")
check(r.returncode == 0, f"정상 종료 ({elapsed:.0f}s)", f"exit {r.returncode}")
mf = re.search(r"features=(\d+)", r.stdout or "")
mc = re.search(r"combo_cats=\[(.*?)\]", r.stdout or "")
n_combo = len([x for x in (mc.group(1).split(",") if mc and mc.group(1).strip() else [])])
if mf:
    check(int(mf.group(1)) == 79 + n_combo, f"피처 {mf.group(1)}개 = 79 + 조합 {n_combo}", f"피처 수 이상: {mf.group(1)} (조합 {n_combo})")
check("multiclass P(success) averaged" in (r.stdout or ""), "멀티클래스 추론 경로 확인", "멀티클래스 로그 없음")
mv = re.search(r"version=(\S+)", r.stdout or "")
ver = mv.group(1) if mv else "?"
check(("smoke" not in ver) or args.allow_smoke, f"번들 version={ver}", f"★ SMOKE 번들({ver})로 빌드됨 — 제출 금지")
sub = pd.read_csv(f"{SIM}/output/submission.csv")
p = sub["control_success"].to_numpy(float)
check(list(sub.columns) == ["row_id", "control_success"] and len(sub) == N_EVAL and sub["row_id"].tolist() == fake["row_id"].tolist(),
      "형식/순서 정상", "형식 이상")
check(not np.isnan(p).any() and p.min() >= 0 and p.max() <= 1 and ((p == 0) | (p == 1)).mean() == 0,
      f"확률 정상 [{p.min():.4f}, {p.max():.4f}] 평균 {p.mean():.6f}", "확률 이상")
check(elapsed < 600, f"시간 여유 {600/max(elapsed,0.1):.0f}배", "10분 초과")

# 행 독립성 프로브: 앞 1,000행만 따로 넣어도 예측이 비트 단위로 같아야 한다
log("\n    행 독립성 프로브 (앞 1,000행 단독 실행 vs 전체 실행)")
SIMK = f"{SIM}_k"
if os.path.exists(SIMK):
    shutil.rmtree(SIMK)
os.makedirs(f"{SIMK}/data")
with zipfile.ZipFile(OUT_ZIP) as z:
    z.extractall(SIMK)
fake.head(1000).to_csv(f"{SIMK}/data/test.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"row_id": fake["row_id"].head(1000), "control_success": 0.5}).to_csv(f"{SIMK}/data/sample_submission.csv", index=False, encoding="utf-8-sig")
rk = subprocess.run([VENV_PY, "-u", "script.py"], cwd=SIMK, capture_output=True, text=True, encoding="utf-8", errors="replace")
check(rk.returncode == 0, "1,000행 단독 정상 종료", f"exit {rk.returncode}")
if rk.returncode == 0:
    pk = pd.read_csv(f"{SIMK}/output/submission.csv")["control_success"].to_numpy(float)
    check(np.array_equal(pk, p[:1000]), f"행 독립 확인 (최대차 {np.abs(pk - p[:1000]).max():.2e})", "행 독립 위반 — 예측이 다른 행에 의존")

log("\n" + "=" * 80)
log("C. 대회 배포 원본 test.csv (5행, 진짜 2025)")
log("=" * 80)
if os.path.exists(SIM5):
    shutil.rmtree(SIM5)
os.makedirs(f"{SIM5}/data")
with zipfile.ZipFile(OUT_ZIP) as z:
    z.extractall(SIM5)
for f in ["test.csv", "sample_submission.csv"]:
    shutil.copy(f"{ROOT}/data/{f}", f"{SIM5}/data/{f}")
r5 = subprocess.run([VENV_PY, "-u", "script.py"], cwd=SIM5, capture_output=True, text=True, encoding="utf-8", errors="replace")
check(r5.returncode == 0, "정상 종료", f"exit {r5.returncode}")
if r5.returncode == 0:
    s5 = pd.read_csv(f"{SIM5}/output/submission.csv")
    p5 = s5["control_success"].to_numpy(float)
    log(f"    예측값: {p5.round(6).tolist()}")
    check(not np.isnan(p5).any(), "2025 5행 NaN 없음", "2025 5행 NaN")
else:
    for line in (r5.stderr or "").splitlines()[-25:]:
        log(f"    | {line}")

for dpath in (SIM, SIMK, SIM5):
    shutil.rmtree(dpath, ignore_errors=True)
log("\n" + "=" * 80)
log(("모든 검사 통과 — " if ok_all else "!! 검사 실패 항목 있음 — ") + f"{OUT_ZIP}")
log("=" * 80)
sys.exit(0 if ok_all else 1)
