# -*- coding: utf-8 -*-
"""
[32] submit7(w=0.9) / submit8(w=1.0) 빌드 — 실측 2025 혼합곡선의 상단 공략

실측 곡선 (LB 3점으로 정확히 결정, 근사 아님):
  G(w) = 132.68w - 64.38w²  →  w*=1.03 (구간 밖) → [0,1] 최적 w=1.0
  예측: w=0.9 → 897.58 / w=1.0 → 898.62   (현재 880.57)

실행: PYTHONIOENCODING=utf-8 <venv311>/Scripts/python.exe -u exp/32_build_w_variants.py
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
VARIANTS = [("submit7.zip", 0.9, 897.58), ("submit8.zip", 1.0, 898.62)]


def log(*a):
    print(*a, flush=True)


ok_all = True


def check(cond, msg_ok, msg_ng):
    global ok_all
    log(f"    [{'OK' if cond else '!!'}] {msg_ok if cond else msg_ng}")
    if not cond:
        ok_all = False
    return cond


# 가짜 test 데이터는 한 번만 준비
src = pd.read_csv(f"{ROOT}/data/train.csv", encoding="utf-8-sig", nrows=N_EVAL)
src.columns = [c.replace("\ufeff", "").strip() for c in src.columns]
test_cols = [c.replace("\ufeff", "").strip() for c in
             pd.read_csv(f"{ROOT}/data/test.csv", encoding="utf-8-sig", nrows=1).columns]
fake = src[[c for c in test_cols if c != "row_id"]].copy()
fake.insert(0, "row_id", [f"TEST_{i:06d}" for i in range(len(fake))])

for zip_name, w, pred in VARIANTS:
    log("\n" + "=" * 80)
    log(f"{zip_name}  (w_cb={w}, 곡선 예측 LB {pred})")
    log("=" * 80)
    stage = f"{SP}/{zip_name.replace('.zip','')}_src"
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(f"{stage}/model")
    shutil.copy(f"{ROOT}/submit/script.py", f"{stage}/script.py")
    shutil.copy(f"{ROOT}/submit/requirements.txt", f"{stage}/requirements.txt")
    b = joblib.load(f"{ROOT}/submit/model/model.pkl")
    assert len(b["models"]) == 8 and len(b["cb_models"]) == 8
    b["w_cb"] = w
    joblib.dump(b, f"{stage}/model/model.pkl", compress=3)

    out_zip = f"{ROOT}/{zip_name}"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for real, arc in [(f"{stage}/script.py", "script.py"),
                          (f"{stage}/requirements.txt", "requirements.txt"),
                          (f"{stage}/model/model.pkl", "model/model.pkl")]:
            z.write(real, arc)
    with zipfile.ZipFile(out_zip) as z:
        check(sorted(z.namelist()) == ["model/model.pkl", "requirements.txt",
                                       "script.py"] and z.testzip() is None,
              f"zip 구조/무결성 ({os.path.getsize(out_zip)/1024**2:.1f} MB)", "zip 이상")

    sim = f"{SP}/{zip_name.replace('.zip','')}_sim"
    if os.path.exists(sim):
        shutil.rmtree(sim)
    os.makedirs(f"{sim}/data")
    with zipfile.ZipFile(out_zip) as z:
        z.extractall(sim)
    fake.to_csv(f"{sim}/data/test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"row_id": fake["row_id"], "control_success": 0.5}).to_csv(
        f"{sim}/data/sample_submission.csv", index=False, encoding="utf-8-sig")
    t0 = time.time()
    r = subprocess.run([VENV_PY, "-u", "script.py"], cwd=sim,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    el = time.time() - t0
    check(r.returncode == 0, f"가짜 서버 245,789행 정상 종료 ({el:.0f}s)",
          f"exit {r.returncode}")
    if r.returncode != 0:
        for line in (r.stderr or "").splitlines()[-10:]:
            log(f"    | {line}")
        continue
    mw = re.search(r"mixed: w_cb=([\d.]+)", r.stdout or "")
    check(mw and abs(float(mw.group(1)) - w) < 1e-9, f"w_cb={w} 적용 확인",
          f"w 불일치: {mw.group(1) if mw else '없음'}")
    sub = pd.read_csv(f"{sim}/output/submission.csv")
    p = sub["control_success"].to_numpy(float)
    check(len(sub) == N_EVAL and not np.isnan(p).any()
          and p.min() >= 0 and p.max() <= 1,
          f"출력 정상 (평균 {p.mean():.6f})", "출력 이상")

log("\n" + "=" * 80)
log("최종: " + ("✅ 두 변형 모두 제출 가능" if ok_all else "❌ 문제 있음"))
log("  submit7.zip = w0.9 (헤지)  /  submit8.zip = w1.0 (곡선 최적)")
log("  ※ submit/model/model.pkl 원본(w=0.3)은 그대로")
log("=" * 80)
sys.exit(0 if ok_all else 1)
