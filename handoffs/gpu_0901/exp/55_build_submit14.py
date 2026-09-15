# -*- coding: utf-8 -*-
"""
[55] submit14.zip 빌드+검증 — 타겟분해 멀티클래스 CS79 CatBoost 8시드

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/55_build_submit14.py
"""

import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np
import pandas as pd

ROOT = "c:/Users/gwonn/Desktop/open"
SRC = f"{ROOT}/submit14_src"
OUT_ZIP = f"{ROOT}/submit14.zip"
SP = ("C:/Users/gwonn/AppData/Local/Temp/claude/c--Users-gwonn-Desktop-open/"
      "ac9f75b2-20b2-4bd0-a3a5-9f91191c2d24/scratchpad")
VENV_SP = ("C:/Users/gwonn/AppData/Local/Temp/claude/c--Users-gwonn-Desktop-open/"
           "aecbab82-c8d9-4dad-8fa6-306809344e0a/scratchpad")
VENV_PY = f"{VENV_SP}/venv311/Scripts/python.exe"
SIM = f"{SP}/submit14_sim"
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
log("A. submit14.zip 생성 및 구조 검사")
log("=" * 80)
req = open(f"{SRC}/requirements.txt", encoding="utf-8").read()
check("catboost==1.2.10" in req, "catboost 명시", "catboost 없음")
items = [(f"{SRC}/script.py", "script.py"),
         (f"{SRC}/requirements.txt", "requirements.txt"),
         (f"{SRC}/model/model.pkl", "model/model.pkl")]
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for real, arc in items:
        z.write(real, arc)
with zipfile.ZipFile(OUT_ZIP) as z:
    check(sorted(z.namelist()) == ["model/model.pkl", "requirements.txt", "script.py"]
          and z.testzip() is None,
          f"구조/무결성 정상 ({os.path.getsize(OUT_ZIP)/1024**2:.1f} MB)", "zip 이상")

log("\n" + "=" * 80)
log(f"B. 가짜 평가 서버 ({N_EVAL:,}행)")
log("=" * 80)
if os.path.exists(SIM):
    shutil.rmtree(SIM)
os.makedirs(f"{SIM}/data")
with zipfile.ZipFile(OUT_ZIP) as z:
    z.extractall(SIM)
test_cols = [c.replace("\ufeff", "").strip() for c in
             pd.read_csv(f"{ROOT}/data/test.csv", encoding="utf-8-sig", nrows=1).columns]
src = pd.read_csv(f"{ROOT}/data/train.csv", encoding="utf-8-sig", nrows=N_EVAL)
src.columns = [c.replace("\ufeff", "").strip() for c in src.columns]
fake = src[[c for c in test_cols if c != "row_id"]].copy()
fake.insert(0, "row_id", [f"TEST_{i:06d}" for i in range(len(fake))])
fake.to_csv(f"{SIM}/data/test.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"row_id": fake["row_id"], "control_success": 0.5}).to_csv(
    f"{SIM}/data/sample_submission.csv", index=False, encoding="utf-8-sig")

t0 = time.time()
r = subprocess.run([VENV_PY, "-u", "script.py"], cwd=SIM,
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
elapsed = time.time() - t0
for line in (r.stdout or "").splitlines():
    log(f"    | {line}")
if r.returncode != 0:
    for line in (r.stderr or "").splitlines()[-25:]:
        log(f"    | {line}")
check(r.returncode == 0, f"정상 종료 ({elapsed:.0f}s)", f"exit {r.returncode}")
mf = re.search(r"features=(\d+)", r.stdout or "")
check(mf and int(mf.group(1)) == 79, "피처 79개", f"피처 이상: {mf.group(1) if mf else '?'}")
check("cb model 8/8 done" in (r.stdout or ""), "CB 8시드 완주", "CB 수 이상")
check("multiclass P(success)" in (r.stdout or ""), "멀티클래스 경로 확인", "멀티클래스 로그 없음")
# 가짜 데이터(=train 앞부분)에서는 CS가 대부분 0/NaN이어야 정상 (자기 자신이 상수표에 포함)
mc = re.search(r"f_cs_p_rate coverage=([\d.]+)%", r.stdout or "")
if mc:
    log(f"    · (참고) 가짜 test에서 CS 커버리지 {mc.group(1)}% — train 재사용 데이터라 낮은 게 정상")

sub = pd.read_csv(f"{SIM}/output/submission.csv")
p = sub["control_success"].to_numpy(float)
check(list(sub.columns) == ["row_id", "control_success"] and len(sub) == N_EVAL
      and sub["row_id"].tolist() == fake["row_id"].tolist(), "형식/순서 정상", "형식 이상")
check(not np.isnan(p).any() and p.min() >= 0 and p.max() <= 1
      and ((p == 0) | (p == 1)).mean() == 0,
      f"확률 정상 [{p.min():.4f}, {p.max():.4f}] 평균 {p.mean():.6f}", "확률 이상")
check(elapsed < 600, f"시간 여유 {600/max(elapsed,0.1):.0f}배", "10분 초과")

log("\n" + "=" * 80)
log("C. 대회 배포 원본 test.csv (5행, 진짜 2025)")
log("=" * 80)
SIM5 = f"{SP}/submit14_sim5"
if os.path.exists(SIM5):
    shutil.rmtree(SIM5)
os.makedirs(f"{SIM5}/data")
with zipfile.ZipFile(OUT_ZIP) as z:
    z.extractall(SIM5)
for f in ["test.csv", "sample_submission.csv"]:
    shutil.copy(f"{ROOT}/data/{f}", f"{SIM5}/data/{f}")
r5 = subprocess.run([VENV_PY, "-u", "script.py"], cwd=SIM5,
                    capture_output=True, text=True, encoding="utf-8", errors="replace")
check(r5.returncode == 0, "정상 종료", f"exit {r5.returncode}")
if r5.returncode == 0:
    s5 = pd.read_csv(f"{SIM5}/output/submission.csv")
    log(f"    예측값: {s5['control_success'].round(6).tolist()}")
    mc5 = re.search(r"f_cs_p_rate coverage=([\d.]+)%", r5.stdout or "")
    check(mc5 and float(mc5.group(1)) >= 60,
          f"2025 5행 CS 커버리지 {mc5.group(1) if mc5 else '?'}% (진짜 데이터에서 작동)",
          "2025 CS 커버리지 이상")
else:
    for line in (r5.stderr or "").splitlines()[-15:]:
        log(f"    | {line}")

log("\n" + "=" * 80)
log("최종: " + ("✅ 제출 가능" if ok_all else "❌ 문제"))
log(f"  {OUT_ZIP}")
log("  (14번째 제출 — 타겟분해 멀티클래스(5클래스) P(성공). 로컬 exp/53 +36.1,")
log("   구종 라벨은 train 내 복원(공식 허용), 추론 입력은 P(구종|사전정보)뿐)")
log("=" * 80)
sys.exit(0 if ok_all else 1)
