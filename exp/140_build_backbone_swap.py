# -*- coding: utf-8 -*-
"""[140] 백본 교체 빌더 — 현 챔피언 zip 의 CatBoost 번들만 갈아끼우고 나머지는 전부 고정 (Claude 소유)

현 챔피언 `candidate_exp021_w0356557.zip` 구조:
    script.py                       두 endpoint 를 순차 실행 후 고정 가중 혼합
    model/champion_inference.py     CatBoost 8시드(model.pkl) → V18 잔차 γ=.30 → 아핀(0.49+1.06(p−.49)−.0066)
    model/exp021_inference.py       Temporal residual endpoint (LGBM/HGB/EB/SVD)
    model/model.pkl                 ← **이것만 교체한다**
    model/v18_tables.npz, *.json    나머지 전부 그대로 복사

이 스크립트가 하는 것 (그 외에는 아무것도 바꾸지 않는다):
  1) --model 로 받은 새 번들을 model/model.pkl 로 교체
  2) champion_inference.py 의 NTREE_END 를 --ntree 로 교체 (CatBoost 는 앞 N개 트리만 사용)
  3) 필요 시 script.py 의 혼합 가중을 --exp021-weight 로 교체 (기본: 챔피언과 동일하게 유지)
  4) zip 빌드 후 전체 검증: 환경 → 구조 → 번들 무결성 → 가짜서버 245,789행 → 행 독립 6종 → 진짜 2025 5행
     → 현 챔피언과의 5행 예측 차이(교체가 실제로 먹었는지)

실행:
  PYTHONIOENCODING=utf-8 python -u exp/140_build_backbone_swap.py \
      --model <새 model.pkl 경로> --ntree 1000 --zip candidate_bb1000_w0356557.zip
★ 제출은 사용자가 직접. 이 스크립트는 zip 만 만들고 검증한다.
"""

_PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="새 CatBoost 번들 (model.pkl)")
ap.add_argument("--ntree", type=int, required=True, help="예측에 쓸 트리 수 (번들의 tree_count 이하)")
ap.add_argument("--exp021-weight", type=float, default=None,
                help="EXP-021 가중. 미지정이면 현 챔피언 값 유지 (백본 효과만 격리하려면 미지정 권장)")
ap.add_argument("--src-zip", default='artifacts/candidates/candidate_exp021_w0356557.zip')
ap.add_argument("--zip", dest="out_zip", default=None)
ap.add_argument("--src-dir", default=None)
ap.add_argument("--n-qa", type=int, default=4000, help="행 독립 QA 에 쓸 행 수")
args = ap.parse_args()

ROOT = Path(str(_PROJECT_ROOT))
VENV_PY = ROOT / "venv311/Scripts/python.exe"        # 08-30 23:05 재구축 (구 스크래치패드 venv311 은 손상됨)
TAG = f"bb{args.ntree}" + ("" if args.exp021_weight is None else f"_w{int(round(args.exp021_weight*1e7)):07d}")
SRC = ROOT / (args.src_dir or f"candidate_{TAG}_src")
OUT_ZIP = ROOT / (args.out_zip or f"candidate_{TAG}.zip")
SIM = ROOT / 'archive/scratch/_tmp_sim_140'
N_EVAL = 245789
TOL = 1e-12

lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


ok_all = True


def check(cond, ok, ng):
    global ok_all
    log(f"    [{'OK' if cond else '!!'}] {ok if cond else ng}")
    if not cond:
        ok_all = False
    return cond


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()


def run(workdir):
    return subprocess.run([str(VENV_PY), "-u", "script.py"], cwd=workdir, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


log("=" * 82)
log(f"[140] 백본 교체 빌더 — ntree={args.ntree}  model={args.model}")
log("=" * 82)

log("0. 환경 (서버 동일: py3.11 / numpy1.26.4 / pandas2.0.3 / catboost1.2.10 / pyarrow 없음)")
r0 = subprocess.run([str(VENV_PY), "-c",
                     "import sys,numpy,pandas,sklearn,joblib,catboost;print(sys.version.split()[0],numpy.__version__,pandas.__version__,sklearn.__version__,joblib.__version__,catboost.__version__)"],
                    capture_output=True, text=True)
log(f"    {r0.stdout.strip()}  {r0.stderr.strip()[-120:]}")
check(r0.stdout.startswith("3.11") and "1.26.4" in r0.stdout and "2.0.3" in r0.stdout and "1.2.10" in r0.stdout,
      "버전 서버 동일", "버전 불일치")
check(subprocess.run([str(VENV_PY), "-c", "import pyarrow"], capture_output=True).returncode != 0,
      "pyarrow 없음", "pyarrow 존재 — 서버와 다름")
try:
    import lightgbm  # noqa
except ImportError:
    pass
r0c = subprocess.run([str(VENV_PY), "-c", "import lightgbm;print(lightgbm.__version__)"], capture_output=True, text=True)
check(r0c.returncode == 0, f"lightgbm {r0c.stdout.strip()} (EXP-021 endpoint 에 필요)",
      "lightgbm 없음 — EXP-021 endpoint 실행 불가. `venv311/Scripts/python.exe -m pip install lightgbm==4.6.0`")

log("")
log("A. 원본 zip 전개 + 백본 교체")
check(Path(args.src_zip).exists(), f"원본 {args.src_zip}", "원본 zip 없음")
check(Path(args.model).exists(), f"새 번들 {args.model} ({os.path.getsize(args.model)/1024**2:.1f} MB)", "새 번들 없음")
if SRC.exists():
    shutil.rmtree(SRC)
with zipfile.ZipFile(args.src_zip) as z:
    names = sorted(z.namelist())
    z.extractall(SRC)
log(f"    원본 구성 {len(names)}개, SHA256 {sha(args.src_zip)[:24]}…")
old_pkl_sha = sha(SRC / "model/model.pkl")
shutil.copy(args.model, SRC / "model/model.pkl")
new_pkl_sha = sha(SRC / "model/model.pkl")
check(old_pkl_sha != new_pkl_sha, f"model.pkl 교체됨 ({old_pkl_sha[:12]}… → {new_pkl_sha[:12]}…)", "번들이 동일 — 교체 무의미")

# 번들 무결성: 시드 수·트리 수·클래스·피처 + ★범주형 컬럼 배치(v8 뒤로몰기 vs v9 제자리)
probe = subprocess.run([str(VENV_PY), "-c", f"""
import joblib, json
b = joblib.load(r'{SRC}/model/model.pkl')
ms = b['cb_models']; f = list(b['feats']); idx = list(ms[0].get_cat_feature_indices())
print(json.dumps({{"version": b.get('version'), "n_models": len(ms),
                  "trees": [m.tree_count_ for m in ms], "classes": [int(c) for c in ms[0].classes_],
                  "n_feats": len(f), "cat_idx": idx, "cat_names": [f[i] for i in idx],
                  "tail5": f[-5:]}}, ensure_ascii=False))
"""], capture_output=True, text=True)
try:
    info = json.loads((probe.stdout or "").strip().splitlines()[-1])
except Exception:
    info = {}
    log(f"    !! 번들 조사 실패: {(probe.stderr or '')[-300:]}")
log(f"    번들: version={info.get('version')} 시드={info.get('n_models')} trees={info.get('trees')} "
    f"클래스={info.get('classes')} feats={info.get('n_feats')}")
log(f"          cat_idx={info.get('cat_idx')} → {info.get('cat_names')}")
trees = info.get("trees", [])
CAT5 = ["top_bottom", "game_type", "base_state", "pitcher_team_id", "batter_team_id"]
check(info.get("cat_names") == CAT5, f"범주형 5개 정상 {info.get('cat_names')}", f"범주형 이상 {info.get('cat_names')}")
check(info.get("n_feats") == 79, f"피처 79개", f"피처 {info.get('n_feats')}개 — 챔피언과 불일치")
# ★ 컬럼 순서 판정: 챔피언 v8 은 cat 을 맨 뒤로 몰고(74~78), exp/94 가 쓰는 v9 는 feats 제자리에 둔다.
#   모델의 cat_idx 가 어느 쪽인지 보고, 추론 스크립트의 build_matrix 를 그에 맞춰 교체한다.
n_feats = info.get("n_feats", 79)
cat_idx = info.get("cat_idx", [])
tail_layout = cat_idx == list(range(n_feats - 5, n_feats))
log(f"    → 학습 시 컬럼 배치: {'v8(범주형 맨 뒤)' if tail_layout else 'v9(feats 순서 유지)'}")
check(bool(trees) and all(t >= args.ntree for t in trees),
      f"모든 시드 tree_count ≥ {args.ntree} {trees}", f"tree_count 부족 {trees}")
check(len(trees) >= 2, f"시드 {len(trees)}개", "시드 수 이상")

# champion_inference.py 의 NTREE_END 교체
ci = SRC / "model/champion_inference.py"
src_ci = ci.read_text(encoding="utf-8")
m = re.search(r"^NTREE_END = (\d+)", src_ci, re.M)
check(m is not None, f"NTREE_END 발견 (현재 {m.group(1) if m else '?'})", "NTREE_END 없음")
src_ci = re.sub(r"^NTREE_END = \d+", f"NTREE_END = {args.ntree}", src_ci, count=1, flags=re.M)

# ★ build_matrix 를 학습 시 배치에 맞춘다. 안 맞추면 CatBoost 가 cat_feature 위치를 잘못 잡아 서버에서 죽는다
#   (리허설에서 실제로 재현: "Invalid type for cat_feature[feature_idx=18]=0.87").
V9_BUILD = '''def build_matrix(d, feats):
    """[exp/140 주입] feats 순서를 그대로 유지하는 배치 — exp/94(v9) 로 학습한 번들과 맞춘다.
    원본 v8 은 범주형을 맨 뒤로 몰았고, 그 배치로 학습된 번들에는 이 함수가 주입되지 않는다."""
    out = pd.DataFrame(index=d.index)
    for c in feats:
        if c in CAT:
            out[c] = d[c].astype(str)
        elif c in TEAM_CAT:
            out[c] = d[c].fillna(-1).astype("int64").astype(str)
        else:
            out[c] = d[c]
    return out
'''
if not tail_layout:
    m_bm = re.search(r"^def build_matrix\(d, feats\):.*?(?=\n\n\ndef |\n\n\n# |\Z)", src_ci, re.S | re.M)
    check(m_bm is not None, "build_matrix 발견", "build_matrix 없음 — 주입 불가")
    if m_bm:
        src_ci = src_ci[:m_bm.start()] + V9_BUILD.rstrip("\n") + src_ci[m_bm.end():]
        log("    → build_matrix 를 v9(순서 유지)로 교체 주입")
else:
    log("    → build_matrix 원본(v8) 유지")
ci.write_text(src_ci, encoding="utf-8")
check(re.search(rf"^NTREE_END = {args.ntree}\b", src_ci, re.M) is not None, f"NTREE_END = {args.ntree} 적용", "적용 실패")
check(("exp/140 주입" in src_ci) == (not tail_layout), "build_matrix 배치 일치", "build_matrix 주입 상태 불일치")
r_syn = subprocess.run([str(VENV_PY), "-c", f"import ast;ast.parse(open(r'{ci}',encoding='utf-8').read())"],
                       capture_output=True, text=True)
check(r_syn.returncode == 0, "champion_inference.py 문법 정상", f"문법 오류: {r_syn.stderr.strip()[-200:]}")

# 혼합 가중 (지정 시에만)
sp = SRC / "script.py"
src_sp = sp.read_text(encoding="utf-8")
w_old = float(re.search(r"EXP021_WEIGHT = ([\d.]+)", src_sp).group(1))
if args.exp021_weight is None:
    log(f"    혼합 가중 유지: EXP-021 {w_old:.10f} / 챔피언 {1-w_old:.10f}  (백본 효과만 격리)")
    w_new = w_old
else:
    w_new = float(args.exp021_weight)
    src_sp = re.sub(r"CHAMPION_WEIGHT = [\d.]+", f"CHAMPION_WEIGHT = {1-w_new!r}", src_sp, count=1)
    src_sp = re.sub(r"EXP021_WEIGHT = [\d.]+", f"EXP021_WEIGHT = {w_new!r}", src_sp, count=1)
    sp.write_text(src_sp, encoding="utf-8")
    log(f"    혼합 가중 변경: EXP-021 {w_old:.10f} → {w_new:.10f}")
check(abs((1 - w_new) + w_new - 1.0) < 1e-12, "가중 합 = 1", "가중 합 이상")

log("")
log(f"B. {OUT_ZIP.name} 빌드")
if OUT_ZIP.exists():
    OUT_ZIP.unlink()
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(SRC.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            z.write(p, p.relative_to(SRC).as_posix())
with zipfile.ZipFile(OUT_ZIP) as z:
    new_names = sorted(z.namelist())
    check(new_names == names and z.testzip() is None,
          f"구성/무결성 원본과 동일 ({len(new_names)}개, {OUT_ZIP.stat().st_size/1024**2:.1f} MB)",
          f"구성 불일치: 추가 {set(new_names)-set(names)} / 누락 {set(names)-set(new_names)}")
log(f"    SHA256 {sha(OUT_ZIP)}")

log("")
log(f"C. 가짜 평가 서버 ({N_EVAL:,}행)")
if SIM.exists():
    shutil.rmtree(SIM)
(SIM / "data").mkdir(parents=True)
with zipfile.ZipFile(OUT_ZIP) as z:
    z.extractall(SIM)
test_cols = [c.replace("\ufeff", "").strip()
             for c in pd.read_csv(ROOT / "data/test.csv", encoding="utf-8-sig", nrows=0).columns]
src_rows = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", nrows=N_EVAL)
src_rows.columns = [c.replace("\ufeff", "").strip() for c in src_rows.columns]
fake = src_rows[[c for c in test_cols if c != "row_id"]].copy()
# ★ 실제 test.csv 와 같은 시즌으로 맞춘다. EXP-021 endpoint 는 "추론 시즌이 학습 이력 뒤"를 강제하므로
#   train 앞부분(2019) 그대로 쓰면 ValueError 로 죽는다 (챔피언 단독 검증에는 없던 제약).
real_test = pd.read_csv(ROOT / "data/test.csv", encoding="utf-8-sig", nrows=5)
real_test.columns = [c.replace("﻿", "").strip() for c in real_test.columns]
TEST_SEASON = int(real_test["season"].iloc[0])
fake["season"] = TEST_SEASON
fake.insert(0, "row_id", [f"TEST_{i:06d}" for i in range(len(fake))])
log(f"    가짜 test 의 season 을 실제 평가 시즌 {TEST_SEASON} 으로 설정 (EXP-021 시간 가드 충족)")
fake.to_csv(SIM / "data/test.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"row_id": fake["row_id"], "control_success": 0.5}).to_csv(
    SIM / "data/sample_submission.csv", index=False, encoding="utf-8-sig")
t0 = time.time()
r = run(SIM)
el = time.time() - t0
for ln in (r.stdout or "").splitlines()[-16:]:
    log(f"    | {ln}")
if r.returncode != 0:
    for ln in (r.stderr or "").splitlines()[-20:]:
        log(f"    | {ln}")
check(r.returncode == 0, f"정상 종료 ({el:.0f}s)", f"exit {r.returncode}")
check(f"ntree_end={args.ntree}" in (r.stdout or ""), f"로그에 ntree_end={args.ntree}", "ntree_end 미적용?")
check(el < 600, f"추론 {el:.0f}s < 600s (여유 {600/max(el,.1):.1f}배)", "10분 초과")
sub = pd.read_csv(SIM / "output/submission.csv")
p_full = sub["control_success"].to_numpy(float)
check(list(sub.columns) == ["row_id", "control_success"] and len(sub) == N_EVAL
      and sub["row_id"].tolist() == fake["row_id"].tolist(), "형식/순서 정상", "형식 이상")
check(np.isfinite(p_full).all() and p_full.min() >= 0 and p_full.max() <= 1,
      f"확률 정상 [{p_full.min():.4f}, {p_full.max():.4f}] 평균 {p_full.mean():.6f}", "확률 이상")

log("")
log(f"D. 행 독립 QA 6종 (앞 {args.n_qa:,}행, tol {TOL:g})")
base = fake.head(args.n_qa).reset_index(drop=True)
rng = np.random.default_rng(42)
cases = {"batch": base,
         "reverse": base.iloc[::-1].reset_index(drop=True),
         "permutation": base.iloc[rng.permutation(len(base))].reset_index(drop=True),
         "duplicate": pd.concat([base, base], ignore_index=True).assign(
             row_id=lambda d: [f"{r}_{i//len(base)}" for i, r in enumerate(d["row_id"])])}
res = {}


def one(name, frame):
    d = SIM.parent / f"_tmp_sim_140_{name}"
    if d.exists():
        shutil.rmtree(d)
    (d / "data").mkdir(parents=True)
    with zipfile.ZipFile(OUT_ZIP) as z:
        z.extractall(d)
    frame.to_csv(d / "data/test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"row_id": frame["row_id"], "control_success": 0.5}).to_csv(
        d / "data/sample_submission.csv", index=False, encoding="utf-8-sig")
    rr = run(d)
    out = None
    if rr.returncode == 0:
        s = pd.read_csv(d / "output/submission.csv")
        out = dict(zip(s["row_id"].astype(str), s["control_success"].astype(float)))
    shutil.rmtree(d, ignore_errors=True)
    return out


for nm, fr in cases.items():
    res[nm] = one(nm, fr)
half = len(base) // 2
a_, b_ = one("split_a", base.iloc[:half].reset_index(drop=True)), one("split_b", base.iloc[half:].reset_index(drop=True))
res["split"] = {**(a_ or {}), **(b_ or {})} if a_ and b_ else None
single = {}
for i in range(3):
    rr = one(f"single{i}", base.iloc[[i]].reset_index(drop=True))
    if rr:
        single.update(rr)
res["singleton3"] = single
ref = res.get("batch")
detail = {}
if ref:
    for nm in ["reverse", "permutation", "duplicate", "split", "singleton3"]:
        rr = res.get(nm)
        if not rr:
            check(False, "", f"{nm} 실행 실패")
            continue
        if nm == "duplicate":
            pairs = [max(abs(rr[f"{k}_0"] - v), abs(rr[f"{k}_1"] - v)) for k, v in ref.items() if f"{k}_0" in rr]
            diff, n = (max(pairs) if pairs else float("nan")), len(pairs) * 2
        else:
            common = [k for k in rr if k in ref]
            diff, n = (max(abs(rr[k] - ref[k]) for k in common) if common else float("nan")), len(common)
        detail[nm] = {"max_abs_diff": float(diff), "n": int(n)}
        check(diff <= TOL, f"{nm:11s} n={n:6,} 최대차 {diff:.3e}", f"{nm} 최대차 {diff:.3e} — 행 독립 위반")

log("")
log("E. 진짜 2025 test.csv 5행 — 현 챔피언과 비교")
p5 = {}
for nm, zp in (("new", OUT_ZIP), ("champion", ROOT / args.src_zip)):
    d = SIM.parent / f"_tmp_sim_140_5_{nm}"
    if d.exists():
        shutil.rmtree(d)
    (d / "data").mkdir(parents=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(d)
    for f in ["test.csv", "sample_submission.csv"]:
        shutil.copy(ROOT / "data" / f, d / "data" / f)
    rr = run(d)
    if rr.returncode == 0:
        p5[nm] = pd.read_csv(d / "output/submission.csv")["control_success"].to_numpy(float)
    else:
        for ln in (rr.stderr or "").splitlines()[-12:]:
            log(f"    | {nm}: {ln}")
    shutil.rmtree(d, ignore_errors=True)
check("new" in p5 and "champion" in p5, "5행 정상 종료 (신규·챔피언)", "5행 실행 실패")
if "new" in p5 and "champion" in p5:
    log(f"    신규    {np.round(p5['new'], 6).tolist()}")
    log(f"    챔피언  {np.round(p5['champion'], 6).tolist()}")
    check(np.isfinite(p5["new"]).all(), "NaN 없음", "NaN")
    d5 = float(np.abs(p5["new"] - p5["champion"]).max())
    check(d5 > 1e-9, f"교체 효과 확인 (5행 최대차 {d5:.5f})", "챔피언과 동일 — 교체 미적용")

for d in ROOT.glob("_tmp_sim_140*"):
    shutil.rmtree(d, ignore_errors=True)
log("")
log("=" * 82)
log(("모든 검사 통과 — 제출 가능: " if ok_all else "!! 검사 실패 — 제출 금지: ") + str(OUT_ZIP))
log("=" * 82)
Path(ROOT / f"lab/140_build_{TAG}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
Path(ROOT / f"lab/140_build_{TAG}.json").write_text(json.dumps(
    {"zip": OUT_ZIP.name, "sha256": sha(OUT_ZIP) if OUT_ZIP.exists() else None, "ntree": args.ntree,
     "model": args.model, "model_sha256": new_pkl_sha, "exp021_weight": w_new,
     "tree_counts": trees, "runtime_sec": el, "row_independence": detail,
     "pred5_new": p5.get("new", np.array([])).tolist(), "pred5_champion": p5.get("champion", np.array([])).tolist(),
     "all_pass": bool(ok_all)}, ensure_ascii=False, indent=1), encoding="utf-8")
sys.exit(0 if ok_all else 1)
