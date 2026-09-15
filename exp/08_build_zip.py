"""
[08] artifacts/submissions/submit.zip 생성 + 전체 검증

실행:  PYTHONIOENCODING=utf-8 py -3.12 -u exp/08_build_zip.py
       (zip 만들기와 구조 검사는 어느 파이썬으로 해도 무방.
        단, '가짜 서버 실행' 단계는 venv311로 돌린다 — 서버와 같은 환경이어야 의미가 있으므로)

검사 항목:
  A. zip 내부 구조     최상위가 정확히 model/ + script.py + requirements.txt 인가
                       경로 구분자가 '/' 인가 (윈도우 '\\'면 리눅스에서 깨짐)
  B. 가짜 서버 실행    빈 폴더에 압축 해제 → data/ 붙임 → script.py 실행
                       (245,789행으로 실제 평가 규모 재현)
  C. 출력 검증         컬럼명 / 행수 / row_id 일치 / 확률 0~1 / 결측 / 반올림 여부
  D. 추론 시간         10분 제한 대비 여유
  E. 원본 test.csv     배포된 5행으로도 정상 동작하는가
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

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
SRC = f"{ROOT}/submissions/submit"
OUT_ZIP = f"{ROOT}/artifacts/submissions/submit_gbdt.zip"
SP = ("C:/Users/gwonn/AppData/Local/Temp/claude/c--Users-gwonn-Desktop-open/"
      "aecbab82-c8d9-4dad-8fa6-306809344e0a/scratchpad")
VENV_PY = f"{SP}/venv311/Scripts/python.exe"
SIM = f"{SP}/submit_sim"
N_EVAL = 245789          # 실제 평가 데이터 행 수


def log(*a):
    print(*a, flush=True)


ok_all = True


def check(cond, msg_ok, msg_ng):
    global ok_all
    if cond:
        log(f"    [OK] {msg_ok}")
    else:
        log(f"    [!!] {msg_ng}")
        ok_all = False
    return cond


# =====================================================================
# A. zip 생성 + 구조 검사
# =====================================================================
log("=" * 84)
log("A. zip 생성 및 구조 검사")
log("=" * 84)

model_files = sorted(os.listdir(f"{SRC}/model"))
if not model_files:
    raise SystemExit("submissions/submit/model/ 이 비어있다. 07번 학습을 먼저 돌릴 것.")

items = [(f"{SRC}/script.py", "script.py"),
         (f"{SRC}/requirements.txt", "requirements.txt")]
items += [(f"{SRC}/model/{f}", f"model/{f}") for f in model_files]

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for real, arc in items:
        z.write(real, arc)          # arcname에 '/' 명시 → 리눅스 호환

with zipfile.ZipFile(OUT_ZIP) as z:
    names = z.namelist()
    corrupt = z.testzip()

log(f"  {OUT_ZIP}  ({os.path.getsize(OUT_ZIP)/1024**2:.1f} MB)")
for n in names:
    log(f"    - {n}")

check("script.py" in names, "script.py 최상위 존재", "script.py 없음")
check("requirements.txt" in names, "requirements.txt 최상위 존재", "requirements.txt 없음")
check(any(n.startswith("model/") for n in names), "model/ 존재", "model/ 없음")
check(not any("\\" in n for n in names), "경로 구분자 '/' 정상", "역슬래시 발견 → 리눅스에서 깨짐")
tops = {n.split("/")[0] for n in names}
check(tops <= {"model", "script.py", "requirements.txt"},
      f"최상위 항목 정상 {sorted(tops)}", f"불필요한 최상위 항목 {sorted(tops)}")
check(corrupt is None, "압축 무결성 정상", f"손상된 엔트리 {corrupt}")
check(os.path.getsize(OUT_ZIP) < 10 * 1024**3, "용량 10GB 이내", "용량 초과")


# =====================================================================
# B. 가짜 평가 서버 재현
# =====================================================================
log("\n" + "=" * 84)
log(f"B. 가짜 평가 서버 실행 ({N_EVAL:,}행)")
log("=" * 84)

if os.path.exists(SIM):
    shutil.rmtree(SIM)
os.makedirs(f"{SIM}/data")

with zipfile.ZipFile(OUT_ZIP) as z:
    z.extractall(SIM)
log("  zip 압축 해제 완료")

# train.csv 앞부분을 test.csv 형태로 가공 (정답은 따로 보관)
test_cols = [c.replace("\ufeff", "").strip() for c in
             pd.read_csv(f"{ROOT}/data/test.csv", encoding="utf-8-sig", nrows=1).columns]
src = pd.read_csv(f"{ROOT}/data/train.csv", encoding="utf-8-sig", nrows=N_EVAL)
src.columns = [c.replace("\ufeff", "").strip() for c in src.columns]
y_true = src["control_success"].to_numpy()

fake = src[[c for c in test_cols if c != "row_id"]].copy()
fake.insert(0, "row_id", [f"TEST_{i:06d}" for i in range(len(fake))])
fake.to_csv(f"{SIM}/data/test.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"row_id": fake["row_id"], "control_success": 0.5}).to_csv(
    f"{SIM}/data/sample_submission.csv", index=False, encoding="utf-8-sig")
log(f"  가짜 data/ 준비 완료 (test {len(fake):,}행)")

log(f"\n  실행: {VENV_PY} script.py   (서버와 같은 numpy/pandas 버전)")
t0 = time.time()
r = subprocess.run([VENV_PY, "-u", "script.py"], cwd=SIM,
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
elapsed = time.time() - t0
for line in (r.stdout or "").splitlines():
    log(f"    | {line}")
if r.returncode != 0:
    log("  --- stderr ---")
    for line in (r.stderr or "").splitlines()[-30:]:
        log(f"    | {line}")
check(r.returncode == 0, f"정상 종료 (exit 0)", f"비정상 종료 (exit {r.returncode})")

# hand delta 조인 검증 (감사 중간-2): 조용한 전-NaN 실패를 자동으로 잡는다
mf = re.search(r"features=(\d+)", r.stdout or "")
if check(mf is not None, "features 로그 존재", "features 로그 없음"):
    check(int(mf.group(1)) == 67, f"피처 {mf.group(1)}개 (기존65+hand delta2)",
          f"피처 {mf.group(1)}개 ≠ 67 — 구버전 model.pkl 의심")
for feat in ["f_hand_delta", "f_bhand_delta"]:
    mn = re.search(rf"{feat}: NaN ([\d.]+)%", r.stdout or "")
    if check(mn is not None, f"{feat} NaN율 로그 존재",
             f"{feat} NaN율 로그 없음 (조인 경로 미실행?)"):
        check(float(mn.group(1)) < 5.0, f"{feat} NaN {mn.group(1)}% (<5%)",
              f"{feat} NaN {mn.group(1)}% — 조인 실패 의심 (dtype 회귀로 전-NaN일 수 있음)")


# =====================================================================
# C. 출력 검증
# =====================================================================
log("\n" + "=" * 84)
log("C. output/submission.csv 검증")
log("=" * 84)

sub_path = f"{SIM}/output/submission.csv"
if check(os.path.exists(sub_path), "제출 파일 생성됨", "제출 파일 없음"):
    sub = pd.read_csv(sub_path)
    p = sub["control_success"].to_numpy(float) if "control_success" in sub.columns else None

    check(list(sub.columns) == ["row_id", "control_success"],
          f"컬럼 정확 {list(sub.columns)}", f"컬럼 불일치 {list(sub.columns)}")
    check(len(sub) == N_EVAL, f"행수 {len(sub):,} 일치", f"행수 {len(sub):,} ≠ {N_EVAL:,}")
    check(sub["row_id"].tolist() == fake["row_id"].tolist(),
          "row_id 순서 일치", "row_id 순서 불일치")
    if p is not None:
        check(not np.isnan(p).any(), "결측 없음", f"결측 {int(np.isnan(p).sum())}건")
        check(p.min() >= 0 and p.max() <= 1,
              f"확률 범위 정상 [{p.min():.6f}, {p.max():.6f}]", "0~1 벗어남")
        rounded = float(((p == 0) | (p == 1)).mean())
        check(rounded == 0, "0/1 반올림 없음 (연속 확률)", f"0/1값 {rounded*100:.2f}%")
        log(f"    · 예측 평균 {p.mean():.6f} / 고유값 {len(np.unique(p)):,}개")

        # 참고용 점수 (학습에 쓴 행이라 과대평가 — 정상동작 확인 용도로만)
        r_ = y_true.mean()
        brier = float(np.mean((p - y_true) ** 2))
        log(f"    · (참고) in-sample 점수 "
            f"{100000*(1-brier/(r_*(1-r_))):.1f}  ※ 학습데이터라 부풀려진 값")


# =====================================================================
# D. 추론 시간
# =====================================================================
log("\n" + "=" * 84)
log("D. 추론 시간")
log("=" * 84)
log(f"  소요 {elapsed:.1f}초  (제한 600초)")
check(elapsed < 600, f"제한 대비 {600/max(elapsed,0.1):.0f}배 여유", "10분 초과!")


# =====================================================================
# E. 배포된 진짜 test.csv (5행)
# =====================================================================
log("\n" + "=" * 84)
log("E. 대회 배포 원본 test.csv (5행)")
log("=" * 84)

SIM5 = f"{SP}/submit_sim5"
if os.path.exists(SIM5):
    shutil.rmtree(SIM5)
os.makedirs(f"{SIM5}/data")
with zipfile.ZipFile(OUT_ZIP) as z:
    z.extractall(SIM5)
for f in ["test.csv", "sample_submission.csv"]:
    shutil.copy(f"{ROOT}/data/{f}", f"{SIM5}/data/{f}")

r5 = subprocess.run([VENV_PY, "-u", "script.py"], cwd=SIM5,
                    capture_output=True, text=True, encoding="utf-8", errors="replace")
check(r5.returncode == 0, "정상 종료", f"비정상 종료 (exit {r5.returncode})")
if r5.returncode != 0:
    for line in (r5.stderr or "").splitlines()[-20:]:
        log(f"    | {line}")
else:
    s5 = pd.read_csv(f"{SIM5}/output/submission.csv")
    log(f"    예측값: {s5['control_success'].round(6).tolist()}")


# =====================================================================
log("\n" + "=" * 84)
log("최종 판정: " + ("✅ 제출 가능" if ok_all else "❌ 문제 있음 — 수정 필요"))
log(f"제출 파일: {OUT_ZIP}")
log("=" * 84)
sys.exit(0 if ok_all else 1)
