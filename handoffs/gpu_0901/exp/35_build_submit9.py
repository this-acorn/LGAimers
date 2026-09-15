# -*- coding: utf-8 -*-
"""
[35] submit9.zip — CatBoost 16시드, w=1.0 (exp/33 증량분 반영)

실행: PYTHONIOENCODING=utf-8 <venv311>/Scripts/python.exe -u exp/35_build_submit9.py
"""

import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
import joblib
import numpy as np
import pandas as pd

ROOT = "c:/Users/gwonn/Desktop/open"
SP = ("C:/Users/gwonn/AppData/Local/Temp/claude/c--Users-gwonn-Desktop-open/"
      "ac9f75b2-20b2-4bd0-a3a5-9f91191c2d24/scratchpad")
VENV_SP = ("C:/Users/gwonn/AppData/Local/Temp/claude/c--Users-gwonn-Desktop-open/"
           "aecbab82-c8d9-4dad-8fa6-306809344e0a/scratchpad")
VENV_PY = f"{VENV_SP}/venv311/Scripts/python.exe"
N_EVAL = 245789
W = 1.0


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
log("submit9.zip (CB 16시드, w=1.0)")
log("=" * 80)
b = joblib.load(f"{ROOT}/submit/model/model.pkl")
check(len(b["cb_models"]) == 16, f"CB 모델 {len(b['cb_models'])}개 (=16)",
      f"CB {len(b['cb_models'])}개 != 16 — exp/33 미완?")
stage = f"{SP}/submit9_src"
if os.path.exists(stage):
    shutil.rmtree(stage)
os.makedirs(f"{stage}/model")
shutil.copy(f"{ROOT}/submit/script.py", f"{stage}/script.py")
shutil.copy(f"{ROOT}/submit/requirements.txt", f"{stage}/requirements.txt")
b["w_cb"] = W
joblib.dump(b, f"{stage}/model/model.pkl", compress=3)

out_zip = f"{ROOT}/submit9.zip"
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for real, arc in [(f"{stage}/script.py", "script.py"),
                      (f"{stage}/requirements.txt", "requirements.txt"),
                      (f"{stage}/model/model.pkl", "model/model.pkl")]:
        z.write(real, arc)
with zipfile.ZipFile(out_zip) as z:
    check(sorted(z.namelist()) == ["model/model.pkl", "requirements.txt", "script.py"]
          and z.testzip() is None,
          f"zip 정상 ({os.path.getsize(out_zip)/1024**2:.1f} MB)", "zip 이상")

sim = f"{SP}/submit9_sim"
if os.path.exists(sim):
    shutil.rmtree(sim)
os.makedirs(f"{sim}/data")
with zipfile.ZipFile(out_zip) as z:
    z.extractall(sim)
src = pd.read_csv(f"{ROOT}/data/train.csv", encoding="utf-8-sig", nrows=N_EVAL)
src.columns = [c.replace("\ufeff", "").strip() for c in src.columns]
test_cols = [c.replace("\ufeff", "").strip() for c in
             pd.read_csv(f"{ROOT}/data/test.csv", encoding="utf-8-sig", nrows=1).columns]
fake = src[[c for c in test_cols if c != "row_id"]].copy()
fake.insert(0, "row_id", [f"TEST_{i:06d}" for i in range(len(fake))])
fake.to_csv(f"{sim}/data/test.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"row_id": fake["row_id"], "control_success": 0.5}).to_csv(
    f"{sim}/data/sample_submission.csv", index=False, encoding="utf-8-sig")
t0 = time.time()
r = subprocess.run([VENV_PY, "-u", "script.py"], cwd=sim,
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
check(r.returncode == 0, f"가짜 서버 정상 종료 ({time.time()-t0:.0f}s)",
      f"exit {r.returncode}")
if r.returncode != 0:
    for line in (r.stderr or "").splitlines()[-10:]:
        log(f"    | {line}")
else:
    check("cb model 16/16 done" in (r.stdout or ""), "CB 16개 완주", "CB 수 이상")
    mw = re.search(r"mixed: w_cb=([\d.]+)", r.stdout or "")
    check(mw and abs(float(mw.group(1)) - W) < 1e-9, f"w={W} 확인", "w 이상")
    sub = pd.read_csv(f"{sim}/output/submission.csv")
    p = sub["control_success"].to_numpy(float)
    check(len(sub) == N_EVAL and not np.isnan(p).any()
          and 0 <= p.min() and p.max() <= 1, f"출력 정상 (평균 {p.mean():.6f})", "출력 이상")

log("\n최종: " + ("✅ submit9.zip 제출 가능 (CB16, w=1.0)" if ok_all else "❌ 문제"))
sys.exit(0 if ok_all else 1)
