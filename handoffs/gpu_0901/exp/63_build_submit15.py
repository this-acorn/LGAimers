# -*- coding: utf-8 -*-
"""
[63] submit15.zip — 3-way 앙상블 (CB65 0.02 / CS76 0.46 / CS79 0.52), 재학습 없음

가중치 산출 근거 (전부 이미 확보된 실측):
  단독 LB:  CB65 898.62 / CS76 990.96 / CS79 993.63
  실측 다양성: CS79⊗CB65 50:50 = 975.50 → K2025(CB65,CS79)=117.5
              로컬/2025 비율 0.614를 나머지 쌍에 적용
  S(blend) = Σwᵢ Sᵢ + Σᵢ<ⱼ wᵢwⱼ Kᵢⱼ  (혼합 브라이어의 항등식)
  → 최적 (0.02, 0.46, 0.52), 예상 LB 1003.05 (+9.41)
앙상블 가중치를 LB 점수 참고로 선택하는 것은 공식 Q&A(08-12) 명시 허용.

번들 병합: submit8(CB65 8시드) + submit10(CS76 8시드+CS상수표)
          + submit12(CS79 8시드+구종테이블)
실행: <venv311>/Scripts/python.exe -u exp/63_build_submit15.py
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
SRC = f"{ROOT}/submit15_src"
OUT_ZIP = f"{ROOT}/submit15.zip"
SP = ("C:/Users/gwonn/AppData/Local/Temp/claude/c--Users-gwonn-Desktop-open/"
      "ac9f75b2-20b2-4bd0-a3a5-9f91191c2d24/scratchpad")
VENV_PY = ("C:/Users/gwonn/AppData/Local/Temp/claude/c--Users-gwonn-Desktop-open/"
           "aecbab82-c8d9-4dad-8fa6-306809344e0a/scratchpad/venv311/Scripts/python.exe")
SIM = f"{SP}/submit15_sim"
N_EVAL = 245789
W = {"CB65": 0.02, "CS76": 0.46, "CS79": 0.52}


def log(*a):
    print(*a, flush=True)


ok_all = True


def check(c, ok, ng):
    global ok_all
    log(f"    [{'OK' if c else '!!'}] {ok if c else ng}")
    if not c:
        ok_all = False
    return c


log("=" * 80)
log("A. 세 번들 병합")
log("=" * 80)
os.makedirs(f"{SRC}/model", exist_ok=True)
shutil.copy(f"{ROOT}/submit12_src/requirements.txt", f"{SRC}/requirements.txt")

if not os.path.exists(f"{ROOT}/submit8_extract/model/model.pkl"):
    with zipfile.ZipFile(f"{ROOT}/submit8.zip") as z:
        z.extract("model/model.pkl", f"{ROOT}/submit8_extract")
b8 = joblib.load(f"{ROOT}/submit8_extract/model/model.pkl")
b10 = joblib.load(f"{ROOT}/submit10_src/model/model.pkl")
b12 = joblib.load(f"{ROOT}/submit12_src/model/model.pkl")
check(len(b8["cb_models"]) == 8, "CB65 8시드 (LB 898.62 검증본)", "CB65 이상")
check(len(b10["cb_models"]) == 8 and len(b10["feats"]) == 76,
      "CS76 8시드/76피처 (LB 990.96)", "CS76 이상")
check(len(b12["cb_models"]) == 8 and len(b12["feats"]) == 79,
      "CS79 8시드/79피처 (LB 993.63)", "CS79 이상")
# CS 상수표는 b10/b12가 동일해야 함 (둘 다 ≤2024 전체로 생성)
same = (len(b10["cs_const_p"]) == len(b12["cs_const_p"]))
check(same, "CS 상수표 정합", "CS 상수표 불일치")

merged = {
    "cb65_models": b8["cb_models"], "cs76_models": b10["cb_models"],
    "cs79_models": b12["cb_models"],
    "feats65": b8["cb_feats_num"] + b8["cb_feats_cat"],
    "feats76": b10["feats"], "feats79": b12["feats"],
    "cs_const_p": b12["cs_const_p"], "cs_const_b": b12["cs_const_b"],
    "mix_tbl": b12["mix_tbl"], "pfb_model": b12["pfb_model"],
    "prior": b12["prior"], "weights": W, "version": "blend3",
}
joblib.dump(merged, f"{SRC}/model/model.pkl", compress=3)
log(f"  저장 {os.path.getsize(f'{SRC}/model/model.pkl')/1024**2:.1f} MB  가중치 {W}")

log("\n" + "=" * 80)
log(f"B. zip + 가짜 서버 ({N_EVAL:,}행)")
log("=" * 80)
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for real, arc in [(f"{SRC}/script.py", "script.py"),
                      (f"{SRC}/requirements.txt", "requirements.txt"),
                      (f"{SRC}/model/model.pkl", "model/model.pkl")]:
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
tc = [c.replace("\ufeff", "").strip() for c in
      pd.read_csv(f"{ROOT}/data/test.csv", encoding="utf-8-sig", nrows=1).columns]
src = pd.read_csv(f"{ROOT}/data/train.csv", encoding="utf-8-sig", nrows=N_EVAL)
src.columns = [c.replace("\ufeff", "").strip() for c in src.columns]
fake = src[[c for c in tc if c != "row_id"]].copy()
fake.insert(0, "row_id", [f"TEST_{i:06d}" for i in range(len(fake))])
fake.to_csv(f"{SIM}/data/test.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"row_id": fake["row_id"], "control_success": 0.5}).to_csv(
    f"{SIM}/data/sample_submission.csv", index=False, encoding="utf-8-sig")
t0 = time.time()
r = subprocess.run([VENV_PY, "-u", "script.py"], cwd=SIM, capture_output=True,
                   text=True, encoding="utf-8", errors="replace")
el = time.time() - t0
for line in [l for l in (r.stdout or "").splitlines()
             if any(k in l for k in ("weights", "blended", "pred mean", "Saved",
                                     "coverage", "X65"))]:
    log(f"    | {line}")
if r.returncode != 0:
    for line in (r.stderr or "").splitlines()[-20:]:
        log(f"    | {line}")
check(r.returncode == 0, f"정상 종료 ({el:.0f}s)", f"exit {r.returncode}")
for tag in ["cb65 8/8 done", "cs76 8/8 done", "cs79 8/8 done"]:
    check(tag in (r.stdout or ""), f"{tag.split()[0]} 완주", f"{tag} 없음")
check("blended 3-way" in (r.stdout or ""), "3-way 혼합 실행", "혼합 로그 없음")
sub = pd.read_csv(f"{SIM}/output/submission.csv")
p = sub["control_success"].to_numpy(float)
check(len(sub) == N_EVAL and not np.isnan(p).any() and 0 <= p.min() and p.max() <= 1,
      f"출력 정상 (평균 {p.mean():.6f})", "출력 이상")
check(el < 600, f"시간 여유 {600/max(el,0.1):.0f}배", "10분 초과")

log("\n" + "=" * 80)
log("C. 진짜 test.csv 5행")
log("=" * 80)
SIM5 = f"{SP}/submit15_sim5"
if os.path.exists(SIM5):
    shutil.rmtree(SIM5)
os.makedirs(f"{SIM5}/data")
with zipfile.ZipFile(OUT_ZIP) as z:
    z.extractall(SIM5)
for f in ["test.csv", "sample_submission.csv"]:
    shutil.copy(f"{ROOT}/data/{f}", f"{SIM5}/data/{f}")
r5 = subprocess.run([VENV_PY, "-u", "script.py"], cwd=SIM5, capture_output=True,
                    text=True, encoding="utf-8", errors="replace")
check(r5.returncode == 0, "정상 종료", f"exit {r5.returncode}")
if r5.returncode == 0:
    s5 = pd.read_csv(f"{SIM5}/output/submission.csv")
    log(f"    예측값: {s5['control_success'].round(6).tolist()}")
else:
    for line in (r5.stderr or "").splitlines()[-15:]:
        log(f"    | {line}")

log("\n" + "=" * 80)
log("최종: " + ("✅ 제출 가능" if ok_all else "❌ 문제"))
log(f"  {OUT_ZIP}   가중치 {W}")
log("  예상 LB 1003.05 (현재 993.63 대비 +9.41) — 재학습 0, 순수 앙상블 수확")
log("=" * 80)
sys.exit(0 if ok_all else 1)
