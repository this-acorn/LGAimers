# -*- coding: utf-8 -*-
"""
[50] submit13.zip — CS79(993.63) ⊗ CB65(898.62) 혼합, w65=0.5 (곡선 측정용)

번들 = submit12 번들 전체 + cb65_models(submit8 검증본) + feats65 + w65 스칼라.
50:50 측정 → 곡선 계산 → w65 최적값 재빌드(재학습 없음) → 수확 제출.
로컬 곡선: D=4.78e-4, 최적 w65≈0.15 (+5.4). LB 예상: 최적점 997~1004.

★ venv311:  <venv311>/Scripts/python.exe -u exp/50_build_submit13.py
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
SRC = f"{ROOT}/submit13_src"
OUT_ZIP = f"{ROOT}/submit13.zip"
SP = ("C:/Users/gwonn/AppData/Local/Temp/claude/c--Users-gwonn-Desktop-open/"
      "ac9f75b2-20b2-4bd0-a3a5-9f91191c2d24/scratchpad")
VENV_SP = ("C:/Users/gwonn/AppData/Local/Temp/claude/c--Users-gwonn-Desktop-open/"
           "aecbab82-c8d9-4dad-8fa6-306809344e0a/scratchpad")
VENV_PY = f"{VENV_SP}/venv311/Scripts/python.exe"
SIM = f"{SP}/submit13_sim"
N_EVAL = 245789
W65 = 0.5


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
log("A. 번들 병합 (CS79 + CB65)")
log("=" * 80)
b12 = joblib.load(f"{ROOT}/submit12_src/model/model.pkl")
b8 = joblib.load(f"{ROOT}/submit8_extract/model/model.pkl")
check(len(b12["cb_models"]) == 8 and b12.get("version") == "cs79",
      "CS79 번들 정상 (LB 993.63 검증본)", "CS79 번들 이상")
check(len(b8["cb_models"]) == 8, "CB65 8시드 (LB 898.62 검증본)", "CB65 이상")
merged = dict(b12)
merged["cb65_models"] = b8["cb_models"]
merged["feats65"] = b8["cb_feats_num"] + b8["cb_feats_cat"]
merged["w65"] = W65
merged["version"] = "mix_v6"
joblib.dump(merged, f"{SRC}/model/model.pkl", compress=3)
log(f"  저장: {os.path.getsize(f'{SRC}/model/model.pkl')/1024**2:.1f} MB, w65={W65}")

log("\n" + "=" * 80)
log(f"B. zip + 가짜 서버 ({N_EVAL:,}행)")
log("=" * 80)
items = [(f"{SRC}/script.py", "script.py"),
         (f"{SRC}/requirements.txt", "requirements.txt"),
         (f"{SRC}/model/model.pkl", "model/model.pkl")]
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for real, arc in items:
        z.write(real, arc)
with zipfile.ZipFile(OUT_ZIP) as z:
    check(sorted(z.namelist()) == ["model/model.pkl", "requirements.txt", "script.py"]
          and z.testzip() is None,
          f"zip 정상 ({os.path.getsize(OUT_ZIP)/1024**2:.1f} MB)", "zip 이상")

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
el = time.time() - t0
for line in [l for l in (r.stdout or "").splitlines()
             if any(k in l for k in ("cs79=", "mixed", "pred mean", "Saved"))]:
    log(f"    | {line}")
if r.returncode != 0:
    for line in (r.stderr or "").splitlines()[-15:]:
        log(f"    | {line}")
check(r.returncode == 0, f"정상 종료 ({el:.0f}s)", f"exit {r.returncode}")
check("cs79 model 8/8 done" in (r.stdout or "")
      and "cb65 model 8/8 done" in (r.stdout or ""), "두 계열 완주", "모델 수 이상")
mw = re.search(r"mixed: w65=([\d.]+)", r.stdout or "")
check(mw and abs(float(mw.group(1)) - W65) < 1e-9, f"w65={W65}", "w 이상")
sub = pd.read_csv(f"{SIM}/output/submission.csv")
p = sub["control_success"].to_numpy(float)
check(len(sub) == N_EVAL and not np.isnan(p).any() and 0 <= p.min() and p.max() <= 1,
      f"출력 정상 (평균 {p.mean():.6f})", "출력 이상")
check(el < 600, f"시간 여유 {600/max(el,0.1):.0f}배", "10분 초과")

log("\n" + "=" * 80)
log("C. 진짜 test.csv 5행")
log("=" * 80)
SIM5 = f"{SP}/submit13_sim5"
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

log("\n" + "=" * 80)
log("최종: " + ("✅ 제출 가능 (50:50 곡선 측정용)" if ok_all else "❌ 문제"))
log(f"  {OUT_ZIP}")
log("  판독: M=50:50 점수 → K=4(M−946.13) → w65* = 0.5 − 95.01/(2K)")
log("  손익분기: M ≤ 969.9 면 혼합 이득 없음 → 수확 제출 생략")
log("=" * 80)
sys.exit(0 if ok_all else 1)
