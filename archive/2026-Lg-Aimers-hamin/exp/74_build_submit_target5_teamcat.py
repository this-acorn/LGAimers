# -*- coding: utf-8 -*-
"""
[74] submit_target5_teamcat.zip 빌드+검증 — target5 멀티클래스(dou 5클래스 분해) +
     team_cat cat_features 콤보, CatBoost 8시드

exp/65_build_submit_team_cat.py를 그대로 따르되 target5_teamcat용으로 경로/피처 수만
갱신한 버전. submit12_src/{script.py(v8, script_v8_target5_teamcat.py로 교체됨),
requirements.txt, model/model.pkl} 3종이 갖춰진 상태(= exp/73_train_target5_teamcat.py
실행 완료)에서 실행해야 함.

★ 경로 안내: 아래 ROOT/VENV_PY는 이 파이프라인을 원래 실행했던 로컬 환경(Windows,
  개인 경로) 그대로 남겨둔 것입니다. 다른 환경에서 재현할 때는 본인의 저장소 루트 및
  가상환경 python 경로로 이 두 값만 바꿔서 사용하면 됩니다.

실행: (venv 활성화 상태에서) python -u exp/74_build_submit_target5_teamcat.py

리더보드 결과: 1059.0501189623 (팀 최고 기록, 2026-08-27 확정)
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

ROOT = "C:/Users/hamin/Downloads/2026-Lg-Aimers-dou"
SRC = f"{ROOT}/submit12_src"
OUT_ZIP = f"{ROOT}/submit_target5_teamcat.zip"
VENV_PY = f"{ROOT}/venv311/Scripts/python.exe"
SIM = f"{ROOT}/_tmp_sim_target5_teamcat"
SIM5 = f"{ROOT}/_tmp_sim_target5_teamcat_5"
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
log("A. submit_target5_teamcat.zip 생성 및 구조 검사")
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
if mf:
    check(int(mf.group(1)) == 79, "피처 79개 (target5/team_cat 모두 새 컬럼 없음)",
          f"피처 이상: {mf.group(1)}")
else:
    log("    [참고] 'features=' 출력 패턴 없음 — 위 로그에서 피처 수 직접 확인")
check("multiclass P(success) averaged" in (r.stdout or ""),
      "멀티클래스 추론 경로 확인(script.py가 v8인지)", "멀티클래스 로그 없음 — script.py가 v8인지 확인 필요")

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
    p5 = s5["control_success"].to_numpy(float)
    log(f"    예측값: {p5.round(6).tolist()}")
    check(not np.isnan(p5).any(), "2025 5행 NaN 없음", "2025 5행에 NaN 있음 — 피처 결측 처리 확인")
else:
    for line in (r5.stderr or "").splitlines()[-15:]:
        log(f"    | {line}")

log("\n" + "=" * 80)
log("최종: " + ("✅ 제출 가능" if ok_all else "❌ 문제"))
log(f"  {OUT_ZIP}")
log("  (target5_teamcat = dou submit14 5클래스 타겟 분해(LB 1033.99) + team_cat")
log("   cat_features(LB 1018.1353에서 CS79 대비 +24.51) 콤보. 실전 리더보드 결과:")
log("   1059.0501189623 — 팀 최고 기록 확정, 2026-08-27)")
log("=" * 80)
sys.exit(0 if ok_all else 1)
