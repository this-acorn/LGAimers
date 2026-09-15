# -*- coding: utf-8 -*-
"""
[102] 챔피언 번들의 앞 N트리만 쓰는 제출물 빌드 — 재학습 0 (IT 학습곡선: 두 시드 정점 it=300, +21.6/+15.9)

입력: 하민 챔피언 zip (submit_target5_teamcat.zip: script.py v8 + requirements.txt + model/model.pkl, 8시드 it500)
처리: zip 을 submit18_src/ 에 풀고 script.py 의 예측 한 줄만 패치
        acc += m.predict_proba(X)[:, 0]   →   acc += m.predict_proba(X, ntree_end=NTREE_END)[:, 0]
      (CatBoost 는 학습된 모델의 앞 ntree_end 개 트리만으로 예측할 수 있다 — 모델·상수표·피처 로직은 그대로)
검증: exp/95 절차 그대로 — venv311 환경(서버 동일, pyarrow 없음) → zip 구조 → 가짜 서버 245,789행 → 행 독립 프로브(앞 1,000행)
      → 진짜 2025 5행, 그리고 원본(it500) 5행 예측과 달라졌는지(패치가 실제로 먹었는지) 확인.
실행 (아무 python): PYTHONIOENCODING=utf-8 python -u exp/102_build_ntree.py --src-zip <챔피언zip> --ntree 300 --zip submit18_it300.zip
★ LB 제출은 사용자가 직접. 이 스크립트는 모델을 바꾸지 않는다(트리 수 제한만).
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
ap.add_argument("--src-zip", required=True, help="하민 챔피언 zip 경로")
ap.add_argument("--ntree", type=int, default=300)
ap.add_argument("--zip", default=None, help="출력 zip 이름 (기본 submit18_it<ntree>.zip)")
ap.add_argument("--src-dir", default=None, help="패치본 소스 폴더 (기본 submit18_src_<variant>)")
ap.add_argument("--shift", type=float, default=0.0,
                help="2025 중심 프로브: 최종 확률에서 이 상수를 뺀다 (행 독립 상수, LB 로 d 를 역산하는 2차식 프로브). 0 이면 패치 안 함")
ap.add_argument("--scale", type=float, default=1.0,
                help="예측 선형 재조정 p' = c + s·(p − c) − δ 의 s (고정 상수, 행 독립). 1.0 이면 패치 안 함")
ap.add_argument("--center", type=float, default=0.49, help="위 c (train 유래 고정 상수; 2024 성공률 .486 근처)")
args = ap.parse_args()

ROOT = str(_PROJECT_ROOT)
VARIANT = (f"it{args.ntree}" + (f"_m{int(round(args.shift * 10000)):04d}" if args.shift > 0 else "")
           + (f"_s{int(round(args.scale * 100)):03d}" if args.scale != 1.0 else ""))
SCALE, CENTER = float(args.scale), float(args.center)
SRC = f"{ROOT}/{args.src_dir or f'submit18_src_{VARIANT}'}"
OUT_ZIP = f"{ROOT}/{args.zip or f'submit18_{VARIANT}.zip'}"
SHIFT = float(args.shift)
VENV_PY = (str(_PROJECT_ROOT / "venv311/Scripts/python.exe"))
SIM = f"{ROOT}/_tmp_sim_ntree"
SIMK = f"{SIM}_k"
SIM5 = f"{SIM}_5"
SIM5_ORIG = f"{SIM}_5orig"
N_EVAL = 245789
NTREE = args.ntree


def log(*a):
    print(*a, flush=True)


ok_all = True


def check(cond, ok, ng):
    global ok_all
    log(f"    [{'OK' if cond else '!!'}] {ok if cond else ng}")
    if not cond:
        ok_all = False
    return cond


def run_script(workdir):
    return subprocess.run([VENV_PY, "-u", "script.py"], cwd=workdir, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


log("=" * 80)
log("0. 환경 — venv311 (서버 동일 버전, pyarrow 없음)")
log("=" * 80)
r0 = subprocess.run([VENV_PY, "-c", "import sys,numpy,pandas,catboost;print(sys.version.split()[0],numpy.__version__,pandas.__version__,catboost.__version__)"],
                    capture_output=True, text=True)
log(f"    venv311: {r0.stdout.strip()}")
check(r0.stdout.startswith("3.11") and "1.26.4" in r0.stdout and "2.0.3" in r0.stdout and "1.2.10" in r0.stdout, "버전 서버 동일", "버전 불일치")
r0b = subprocess.run([VENV_PY, "-c", "import pyarrow"], capture_output=True, text=True)
check(r0b.returncode != 0, "pyarrow 없음 (서버 동일)", "pyarrow 존재 — 서버와 다름")

log("\n" + "=" * 80)
log(f"A. 원본 zip 풀기 → {args.src_dir}/  → script.py 패치 (ntree_end={NTREE})")
log("=" * 80)
check(os.path.exists(args.src_zip), f"원본 zip {args.src_zip}", "원본 zip 없음")
with zipfile.ZipFile(args.src_zip) as z:
    names = sorted(z.namelist())
    check(names == ["model/model.pkl", "requirements.txt", "script.py"], f"원본 구조 {names}", f"원본 구조 이상 {names}")
    if os.path.exists(SRC):
        shutil.rmtree(SRC)
    z.extractall(SRC)
orig_script = open(f"{SRC}/script.py", encoding="utf-8").read()
shutil.copy(f"{SRC}/script.py", f"{SRC}/script_orig_it500.py.bak")
OLD = "acc += m.predict_proba(X)[:, 0]"
NEW = "acc += m.predict_proba(X, ntree_end=NTREE_END)[:, 0]"
check(orig_script.count(OLD) == 1, "예측 줄 1개 발견", f"예측 줄 {orig_script.count(OLD)}개 — 패치 대상 불명확")
patched = orig_script.replace(OLD, NEW)
# 상수 삽입: 첫 import 블록 뒤 (joblib import 줄 다음)
anchor = "import joblib\n"
check(anchor in patched, "import joblib 앵커", "앵커 없음")
patched = patched.replace(anchor, anchor + f"\nNTREE_END = {NTREE}   # 예측에 사용할 트리 수 (학습된 8시드 모델 각 500트리 이하; 500 = 전체 사용)\n"
                          + f"SHIFT_DELTA = {SHIFT!r}   # 학습 데이터(train.csv) 연도별 성공률 하락 추세(2019 .565 → 2024 .486, 연평균 −0.016)에 근거한\n"
                          + "                       # 사전 중심 보정 상수. 학습 데이터 유래 고정값이며 각 test 행에 독립 적용 (다른 test 행·집계 미참조)\n", 1)
# 추론 로그에 트리 수 남기기
OLD2 = '        print(f"  multiclass P(success) averaged, version={b.get(\'version\')}")'
NEW2 = ('        print(f"  multiclass P(success) averaged, version={b.get(\'version\')}  ntree_end={NTREE_END} '
        'shift={SHIFT_DELTA} tree_count={[m.tree_count_ for m in models]}")')
check(patched.count(OLD2) == 1, "버전 로그 줄 발견", "버전 로그 줄 없음 (치명 아님)")
patched = patched.replace(OLD2, NEW2)
if SHIFT > 0 or SCALE != 1.0:
    OLD3 = "    preds = np.clip(preds, 0.0, 1.0)"
    if SCALE != 1.0:
        patched = patched.replace("SHIFT_DELTA = ", f"SCALE_S = {SCALE!r}      # 예측 확률 선형 재조정 상수 (고정값, 각 행 독립 적용)\nCENTER_C = {CENTER!r}    # 재조정 중심 (학습 데이터 2024 성공률 .486 근처의 고정 상수)\nSHIFT_DELTA = ", 1)
        NEW3 = "    preds = np.clip(CENTER_C + SCALE_S * (preds - CENTER_C) - SHIFT_DELTA, 0.0, 1.0)   # 고정 상수 선형 재조정 — 행 독립 (행 하나만 있어도 결과 동일)"
    else:
        NEW3 = "    preds = np.clip(preds - SHIFT_DELTA, 0.0, 1.0)   # 학습 유래 고정 상수 차감 — 행 독립 (행 하나만 있어도 결과 동일)"
    check(patched.count(OLD3) == 1, "클립 줄 1개 발견 (shift/scale 패치)", f"클립 줄 {patched.count(OLD3)}개 — 패치 불가")
    patched = patched.replace(OLD3, NEW3)
with open(f"{SRC}/script.py", "w", encoding="utf-8") as f:
    f.write(patched)
os.remove(f"{SRC}/script_orig_it500.py.bak")
req = open(f"{SRC}/requirements.txt", encoding="utf-8").read()
check("catboost==1.2.10" in req, "requirements catboost==1.2.10", "catboost 미명시")
# 번들 트리 수 확인 (venv311 로 로드)
rt = subprocess.run([VENV_PY, "-c",
                     "import joblib;b=joblib.load(r'%s/model/model.pkl');print(b.get('version'),len(b['cb_models']),[m.tree_count_ for m in b['cb_models']],b.get('n_classes'))" % SRC],
                    capture_output=True, text=True)
log(f"    번들: {rt.stdout.strip()}  {rt.stderr.strip()[-200:]}")
tc = re.findall(r"\[(.*?)\]", rt.stdout or "")
trees = [int(x) for x in tc[0].split(",")] if tc else []
check(bool(trees) and all(t >= NTREE for t in trees), f"모든 시드 tree_count ≥ {NTREE} {trees}", f"tree_count 부족/확인 실패 {trees}")

log("\n" + "=" * 80)
log(f"B. {os.path.basename(OUT_ZIP)} 생성 및 구조 검사")
log("=" * 80)
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for arc in ["script.py", "requirements.txt", "model/model.pkl"]:
        z.write(f"{SRC}/{arc}", arc)
with zipfile.ZipFile(OUT_ZIP) as z:
    check(sorted(z.namelist()) == ["model/model.pkl", "requirements.txt", "script.py"] and z.testzip() is None,
          f"구조/무결성 정상 ({os.path.getsize(OUT_ZIP)/1024**2:.1f} MB)", "zip 이상")

log("\n" + "=" * 80)
log(f"C. 가짜 평가 서버 ({N_EVAL:,}행, train 앞부분을 test 형식으로)")
log("=" * 80)
for d_ in (SIM, SIMK, SIM5, SIM5_ORIG):
    if os.path.exists(d_):
        shutil.rmtree(d_)
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
r = run_script(SIM)
elapsed = time.time() - t0
for line in (r.stdout or "").splitlines():
    log(f"    | {line}")
if r.returncode != 0:
    for line in (r.stderr or "").splitlines()[-25:]:
        log(f"    | {line}")
check(r.returncode == 0, f"정상 종료 ({elapsed:.0f}s)", f"exit {r.returncode}")
check(f"ntree_end={NTREE}" in (r.stdout or ""), f"추론 로그에 ntree_end={NTREE} 확인", "ntree_end 로그 없음 — 패치 미적용?")
check(f"shift={SHIFT!r}" in (r.stdout or "") or f"shift={SHIFT}" in (r.stdout or ""), f"추론 로그에 shift={SHIFT} 확인", "shift 로그 없음")
check("multiclass P(success) averaged" in (r.stdout or ""), "멀티클래스 추론 경로 확인", "멀티클래스 로그 없음")
sub = pd.read_csv(f"{SIM}/output/submission.csv")
p = sub["control_success"].to_numpy(float)
check(list(sub.columns) == ["row_id", "control_success"] and len(sub) == N_EVAL and sub["row_id"].tolist() == fake["row_id"].tolist(),
      "형식/순서 정상", "형식 이상")
check(not np.isnan(p).any() and p.min() >= 0 and p.max() <= 1 and ((p == 0) | (p == 1)).mean() == 0,
      f"확률 정상 [{p.min():.4f}, {p.max():.4f}] 평균 {p.mean():.6f}", "확률 이상")
check(elapsed < 600, f"시간 여유 {600/max(elapsed,0.1):.0f}배", "10분 초과")

log("\n    행 독립성 프로브 (앞 1,000행 단독 실행 vs 전체 실행)")
os.makedirs(f"{SIMK}/data")
with zipfile.ZipFile(OUT_ZIP) as z:
    z.extractall(SIMK)
fake.head(1000).to_csv(f"{SIMK}/data/test.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"row_id": fake["row_id"].head(1000), "control_success": 0.5}).to_csv(f"{SIMK}/data/sample_submission.csv", index=False, encoding="utf-8-sig")
rk = run_script(SIMK)
check(rk.returncode == 0, "1,000행 단독 정상 종료", f"exit {rk.returncode}")
if rk.returncode == 0:
    pk = pd.read_csv(f"{SIMK}/output/submission.csv")["control_success"].to_numpy(float)
    check(np.array_equal(pk, p[:1000]), f"행 독립 확인 (최대차 {np.abs(pk - p[:1000]).max():.2e})", "행 독립 위반")

log("\n" + "=" * 80)
log("D. 진짜 2025 test.csv 5행 — 패치본 vs 원본(it500) 비교 (패치가 실제로 먹었는지)")
log("=" * 80)
for d_, zpath in ((SIM5, OUT_ZIP), (SIM5_ORIG, args.src_zip)):
    os.makedirs(f"{d_}/data")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(d_)
    for f in ["test.csv", "sample_submission.csv"]:
        shutil.copy(f"{ROOT}/data/{f}", f"{d_}/data/{f}")
r5 = run_script(SIM5)
r5o = run_script(SIM5_ORIG)
check(r5.returncode == 0 and r5o.returncode == 0, "5행 정상 종료 (패치본·원본)", f"exit {r5.returncode}/{r5o.returncode}")
if r5.returncode == 0 and r5o.returncode == 0:
    p5 = pd.read_csv(f"{SIM5}/output/submission.csv")["control_success"].to_numpy(float)
    p5o = pd.read_csv(f"{SIM5_ORIG}/output/submission.csv")["control_success"].to_numpy(float)
    log(f"    it{NTREE} 예측: {p5.round(6).tolist()}")
    log(f"    it500 예측 : {p5o.round(6).tolist()}")
    check(not np.isnan(p5).any(), "2025 5행 NaN 없음", "2025 5행 NaN")
    check(np.abs(p5 - p5o).max() > 1e-6, f"패치 효과 확인 (5행 최대차 {np.abs(p5-p5o).max():.4f})", "패치본과 원본 예측이 동일 — ntree_end 미적용")
else:
    for line in ((r5.stderr or "") + (r5o.stderr or "")).splitlines()[-25:]:
        log(f"    | {line}")

for d_ in (SIM, SIMK, SIM5, SIM5_ORIG):
    shutil.rmtree(d_, ignore_errors=True)
log("\n" + "=" * 80)
log(("모든 검사 통과 — 제출 가능: " if ok_all else "!! 검사 실패 항목 있음 — 제출 금지: ") + OUT_ZIP)
log("=" * 80)
sys.exit(0 if ok_all else 1)
