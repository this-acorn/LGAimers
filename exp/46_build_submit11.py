# -*- coding: utf-8 -*-
"""
[46] artifacts/submissions/submit11.zip — CB65(898.62) ⊗ CS76(990.96) 혼합 번들, w65=0.5 (곡선 측정용)

번들 병합: submit8의 cb_models(8, LB 검증본) + submit10의 전체(cs76 8시드 + CS 상수표).
w65는 스칼라 — 50:50 실측 후 곡선 계산해서 최적 w로 재빌드 (재학습 불필요).

★ venv311 실행:  <venv311>/Scripts/python.exe -u exp/46_build_submit11.py
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

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
SRC = f"{ROOT}/submissions/submit11_src"
OUT_ZIP = f"{ROOT}/artifacts/submissions/submit11.zip"
SP = ("C:/Users/gwonn/AppData/Local/Temp/claude/c--Users-gwonn-Desktop-open/"
      "ac9f75b2-20b2-4bd0-a3a5-9f91191c2d24/scratchpad")
VENV_SP = ("C:/Users/gwonn/AppData/Local/Temp/claude/c--Users-gwonn-Desktop-open/"
           "aecbab82-c8d9-4dad-8fa6-306809344e0a/scratchpad")
VENV_PY = f"{VENV_SP}/venv311/Scripts/python.exe"
SIM = f"{SP}/submit11_sim"
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
log("A. 번들 병합")
log("=" * 80)
b8 = joblib.load(f"{ROOT}/submissions/submit8_extract/model/model.pkl")
b10 = joblib.load(f"{ROOT}/submissions/submit10_src/model/model.pkl")
cb65 = b8["cb_models"]
check(len(cb65) == 8, f"CB65 모델 8개 (submit8 = LB 898.62 검증본)", f"CB65 {len(cb65)}개")
check(len(b10["cb_models"]) == 8 and b10.get("version") == "cs76",
      "CS76 모델 8개 (submit10 = LB 990.96 검증본)", "CS76 번들 이상")
# CB65 입력 피처 순서: submit8 script.py의 fcb 경로 = cb_feats_num + CAT(str)
feats65 = b8["cb_feats_num"] + b8["cb_feats_cat"]
merged = {"cs76_models": b10["cb_models"], "cb65_models": cb65,
          "feats76": b10["feats"], "feats65": feats65,
          "prior": b10["prior"], "cs_const_p": b10["cs_const_p"],
          "cs_const_b": b10["cs_const_b"], "w65": W65, "version": "mix_v3"}
joblib.dump(merged, f"{SRC}/model/model.pkl", compress=3)
log(f"  저장: {SRC}/model/model.pkl "
    f"({os.path.getsize(f'{SRC}/model/model.pkl')/1024**2:.1f} MB, w65={W65})")

log("\n" + "=" * 80)
log("B. zip + 가짜 서버 검증")
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
elapsed = time.time() - t0
tail_lines = [l for l in (r.stdout or "").splitlines()
              if any(k in l for k in ("cs76=", "mixed", "pred mean", "Saved", "coverage"))]
for line in tail_lines:
    log(f"    | {line}")
if r.returncode != 0:
    for line in (r.stderr or "").splitlines()[-20:]:
        log(f"    | {line}")
check(r.returncode == 0, f"정상 종료 ({elapsed:.0f}s)", f"exit {r.returncode}")
check("cs76 8/8 done" in (r.stdout or "") and "cb65 8/8 done" in (r.stdout or ""),
      "두 계열 8시드 완주", "모델 수 이상")
mw = re.search(r"mixed: w65=([\d.]+)", r.stdout or "")
check(mw and abs(float(mw.group(1)) - W65) < 1e-9, f"w65={W65}", "w 이상")
sub = pd.read_csv(f"{SIM}/output/submission.csv")
p = sub["control_success"].to_numpy(float)
check(len(sub) == N_EVAL and not np.isnan(p).any() and 0 <= p.min() and p.max() <= 1,
      f"출력 정상 (평균 {p.mean():.6f})", "출력 이상")
check(elapsed < 600, f"시간 여유 {600/max(elapsed,0.1):.0f}배", "10분 초과")

log("\n" + "=" * 80)
log("C. 진짜 test.csv 5행")
log("=" * 80)
SIM5 = f"{SP}/submit11_sim5"
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
log("  절차: ① 이것 제출 → ② 3점(991.0/898.6/50:50)으로 곡선 정확 결정")
log("        → ③ w65를 최적값으로 바꿔 재빌드(재학습 없음) → 제출")
log("=" * 80)
sys.exit(0 if ok_all else 1)
