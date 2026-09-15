# -*- coding: utf-8 -*-
"""
[31] artifacts/submissions/submit6.zip — w_cb=0.5 변형 (재학습 없음, 번들 스칼라만 변경)

배경: submit5(w=0.3) 실전 2025 = 864.33 (+34.0). 로컬 연도별 최적 w는
  2021/22≈0.2~0.3, 2024≈0.5, 2023≈0.9 로 갈렸고, 2025의 +34.0은 중간~CB친화 구간.
  → w=0.5 한 점을 실측하면 2025 혼합곡선의 위치가 잡힌다 (곡선은 w의 2차식).
  w 선택은 제출 간 모델 선택(정상적 제출 활용)이지, test 데이터로 계산한
  사후 보정값이 아니다 — 규칙 위반 아님. 어떤 w든 각 행 예측은 행 독립.

절차: submissions/submit/의 script.py·requirements.txt 그대로 + model.pkl 사본에 w_cb=0.5
  → 스테이징 → artifacts/submissions/submit6.zip → 가짜 서버 245,789행 검증 (venv311)

실행: PYTHONIOENCODING=utf-8 <venv311>/Scripts/python.exe -u exp/31_build_submit6_w05.py
  (pkl 재저장이 있으므로 venv311 필수)
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
SP = ("C:/Users/gwonn/AppData/Local/Temp/claude/c--Users-gwonn-Desktop-open/"
      "ac9f75b2-20b2-4bd0-a3a5-9f91191c2d24/scratchpad")
VENV_SP = ("C:/Users/gwonn/AppData/Local/Temp/claude/c--Users-gwonn-Desktop-open/"
           "aecbab82-c8d9-4dad-8fa6-306809344e0a/scratchpad")
VENV_PY = f"{VENV_SP}/venv311/Scripts/python.exe"
STAGE = f"{SP}/submit6_src"
OUT_ZIP = f"{ROOT}/artifacts/submissions/submit6.zip"
SIM = f"{SP}/submit6_sim"
N_EVAL = 245789
NEW_W = 0.5


def log(*a):
    print(*a, flush=True)


ok_all = True


def check(cond, msg_ok, msg_ng):
    global ok_all
    log(f"    [{'OK' if cond else '!!'}] {msg_ok if cond else msg_ng}")
    if not cond:
        ok_all = False
    return cond


# ---- 1. 스테이징: w만 0.5로 바꾼 번들 ----
log("=" * 80)
log(f"A. 스테이징 (w_cb 0.3 → {NEW_W})")
log("=" * 80)
if os.path.exists(STAGE):
    shutil.rmtree(STAGE)
os.makedirs(f"{STAGE}/model")
shutil.copy(f"{ROOT}/submissions/submit/script.py", f"{STAGE}/script.py")
shutil.copy(f"{ROOT}/submissions/submit/requirements.txt", f"{STAGE}/requirements.txt")
b = joblib.load(f"{ROOT}/submissions/submit/model/model.pkl")
assert abs(b["w_cb"] - 0.3) < 1e-9, f"원본 w_cb={b['w_cb']} != 0.3 — 상태 확인 필요"
assert len(b["models"]) == 8 and len(b["cb_models"]) == 8
b["w_cb"] = NEW_W
joblib.dump(b, f"{STAGE}/model/model.pkl", compress=3)
log(f"  원본 확인(HGB8+CB8, w=0.3) → 사본 w_cb={NEW_W} 저장")
log(f"  ※ submissions/submit/model/model.pkl 원본(w=0.3, 864.33 확정)은 건드리지 않음")

# ---- 2. zip ----
items = [(f"{STAGE}/script.py", "script.py"),
         (f"{STAGE}/requirements.txt", "requirements.txt"),
         (f"{STAGE}/model/model.pkl", "model/model.pkl")]
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for real, arc in items:
        z.write(real, arc)
with zipfile.ZipFile(OUT_ZIP) as z:
    names = z.namelist()
    corrupt = z.testzip()
log(f"  {OUT_ZIP}  ({os.path.getsize(OUT_ZIP)/1024**2:.1f} MB)")
check(sorted(names) == ["model/model.pkl", "requirements.txt", "script.py"],
      "구조 정확", f"구조 이상 {names}")
check(corrupt is None, "압축 무결성", f"손상 {corrupt}")

# ---- 3. 가짜 서버 ----
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
tail = [l for l in (r.stdout or "").splitlines() if "mixed" in l or "pred mean" in l
        or "features" in l or "Saved" in l]
for line in tail:
    log(f"    | {line}")
check(r.returncode == 0, f"정상 종료 ({elapsed:.0f}s)", f"exit {r.returncode}")
if r.returncode != 0:
    for line in (r.stderr or "").splitlines()[-15:]:
        log(f"    | {line}")
mw = re.search(r"mixed: w_cb=([\d.]+)", r.stdout or "")
check(mw and abs(float(mw.group(1)) - NEW_W) < 1e-9,
      f"w_cb={NEW_W} 적용 확인", f"w 불일치: {mw.group(1) if mw else '로그 없음'}")
sub = pd.read_csv(f"{SIM}/output/submission.csv")
p = sub["control_success"].to_numpy(float)
check(len(sub) == N_EVAL and not np.isnan(p).any()
      and p.min() >= 0 and p.max() <= 1, "출력 형식/범위 정상", "출력 이상")

log("\n" + "=" * 80)
log("최종: " + ("✅ 제출 가능" if ok_all else "❌ 문제"))
log(f"  {OUT_ZIP}  (w_cb={NEW_W} 변형 · 제출 여부는 사용자 결정)")
log(f"  현재 확정: submit5(w=0.3)=864.33. w=0.5 실측 시 2025 혼합곡선 위치 파악 가능")
log("=" * 80)
sys.exit(0 if ok_all else 1)
