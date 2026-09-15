# -*- coding: utf-8 -*-
"""
[82] 번들 재포장 2단계 — venv311(서버 동일 환경)에서 재조립 + zip + 가짜 서버 검증

★ 반드시 venv311 로 실행할 것 (python 3.11.8 / numpy 1.26.4 / pandas 2.0.3 / catboost 1.2.10)
  exp/81 이 시스템 python(pandas 3.0.1)에서 내보낸 조각들을 여기서 다시 묶는다.
  이렇게 하면 pickle 이 pandas 3.0 / PyArrow 를 참조하지 않는다.

검증 단계 (하나라도 실패하면 제출 금지):
  A. 재조립 후 진짜 test.csv 5행 예측이 exp/81 의 기준값과 **비트 단위로 일치**
  B. zip 구조 = {script.py, requirements.txt, model/model.pkl} 정확히 셋
  C. 가짜 서버 245,789행 완주 + 10분 제한 + 확률 범위 + 행수/순서
  D. 진짜 test.csv 5행 재실행
  E. pyarrow 없는 환경에서 로드되는지 (이 스크립트가 그 환경이다)

실행:
  <venv311>/Scripts/python.exe -u exp/82_rebuild_submit16.py
"""

_PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier

ROOT = str(_PROJECT_ROOT)
SRC = f"{ROOT}/submissions/submit16_src"
STAGE = f"{SRC}/_stage"
OUT_ZIP = f"{ROOT}/artifacts/submissions/submit16.zip"
SP = ("C:/Users/gwonn/AppData/Local/Temp/claude/c--Users-gwonn-Desktop-open/"
      "ac9f75b2-20b2-4bd0-a3a5-9f91191c2d24/scratchpad")
N_EVAL = 245789

ok_all = True


def log(*a):
    print(*a, flush=True)


def check(cond, ok, ng):
    global ok_all
    log(f"    [{'OK' if cond else '!!'}] {ok if cond else ng}")
    if not cond:
        ok_all = False
    return cond


log("=" * 84)
log("환경 확인 — 서버와 같아야 한다")
log("=" * 84)
log(f"  python {sys.version.split()[0]} / numpy {np.__version__} / "
    f"pandas {pd.__version__}")
check(np.__version__.startswith("1.26"), "numpy 1.26.x", f"numpy {np.__version__} — 서버와 다름")
check(pd.__version__.startswith("2.0"), "pandas 2.0.x", f"pandas {pd.__version__} — 서버와 다름")
try:
    import pyarrow  # noqa: F401
    log("    [!!] 이 환경에 pyarrow 가 있다 — 서버에는 없으므로 검증이 무의미해진다")
    ok_all = False
except ImportError:
    log("    [OK] pyarrow 없음 — 서버와 같은 조건에서 검증한다")

log("")
log("=" * 84)
log("A. 번들 재조립")
log("=" * 84)
meta = json.load(open(f"{STAGE}/meta.json", encoding="utf-8"))

models = []
for i in range(meta["n_models"]):
    m = CatBoostClassifier()
    m.load_model(f"{STAGE}/cb_{i}.cbm")
    models.append(m)
pfb = CatBoostClassifier()
pfb.load_model(f"{STAGE}/pfb.cbm")
check(len(models) == 8, f"CatBoost {len(models)}시드 + PFB 로드", "모델 수 이상")


def load_table(key):
    z = np.load(f"{STAGE}/{key}.npz", allow_pickle=False)
    spec = meta["tables"][key]
    df = pd.DataFrame({c: np.asarray(z[c]).astype(spec["dtypes"][c])
                       for c in spec["columns"]})
    assert list(df.shape) == spec["shape"], f"{key} shape 불일치"
    assert dict(df.dtypes.astype(str)) == spec["dtypes"], f"{key} dtype 불일치"
    return df


cp = load_table("cs_const_p")
cb_ = load_table("cs_const_b")
mix = load_table("mix_tbl")
check(True, f"표 복원 cs_const_p{cp.shape} cs_const_b{cb_.shape} mix_tbl{mix.shape}", "")

bundle = {
    "cb_models": models, "feats": meta["feats"], "prior": meta["prior"],
    "cs_const_p": cp, "cs_const_b": cb_, "mix_tbl": mix, "pfb_model": pfb,
    "seeds": meta["seeds"], "version": meta["version"],
    "season_decay": meta["season_decay"], "classes": meta["classes"],
}
joblib.dump(bundle, f"{SRC}/model/model.pkl", compress=3)
mb = os.path.getsize(f"{SRC}/model/model.pkl") / 1024 ** 2
check(mb < 200, f"저장 {mb:.1f} MB (pandas {pd.__version__} 로 pickle)", "용량 이상")

# 자기 자신을 다시 로드해서 순환 검증
b2 = joblib.load(f"{SRC}/model/model.pkl")
check(len(b2["cb_models"]) == 8 and len(b2["feats"]) == 79,
      "재로드 정상 (8시드 / 79피처)", "재로드 실패")

log("")
log("=" * 84)
log("B. ★ 기준 예측값과 비트 단위 일치 검증")
log("=" * 84)
import importlib.util
spec = importlib.util.spec_from_file_location("s16", f"{SRC}/script.py")
s16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s16)

t5 = pd.read_csv(f"{ROOT}/data/test.csv", encoding="utf-8-sig")
t5.columns = [c.replace("\ufeff", "").strip() for c in t5.columns]
ft = s16.attach_pt(
    s16.attach_cs(s16.add_features(t5, b2["prior"]), b2["cs_const_p"], b2["cs_const_b"]),
    b2["mix_tbl"], b2["pfb_model"])
X = s16.build_matrix(ft, b2["feats"])
per_seed = np.array([m.predict_proba(X)[:, 0] for m in b2["cb_models"]], dtype="float64")
got = per_seed.mean(axis=0)

ref = np.load(f"{STAGE}/ref_pred.npz")
exp_ens = ref["ens"]
exp_seed = ref["per_seed"]
log(f"  기준(3.12/pandas3.0): " + "  ".join(f"{v:.10f}" for v in exp_ens))
log(f"  재조립(3.11/pandas2.0): " + "  ".join(f"{v:.10f}" for v in got))
d_seed = float(np.abs(per_seed - exp_seed).max())
d_ens = float(np.abs(got - exp_ens).max())
log(f"  시드별 최대차 {d_seed:.3e}   앙상블 최대차 {d_ens:.3e}")
check(np.array_equal(per_seed, exp_seed),
      "시드별 예측 완전 동일 — 재포장이 모델을 바꾸지 않았다",
      f"예측이 달라졌다 (최대차 {d_seed:.3e}) — 재포장 실패")

log("")
log("=" * 84)
log(f"C. zip 빌드 + 가짜 서버 ({N_EVAL:,}행)")
log("=" * 84)
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for real, arc in [(f"{SRC}/script.py", "script.py"),
                      (f"{SRC}/requirements.txt", "requirements.txt"),
                      (f"{SRC}/model/model.pkl", "model/model.pkl")]:
        z.write(real, arc)
with zipfile.ZipFile(OUT_ZIP) as z:
    names = sorted(z.namelist())
    check(names == ["model/model.pkl", "requirements.txt", "script.py"]
          and z.testzip() is None,
          f"zip 정상 ({os.path.getsize(OUT_ZIP)/1024**2:.1f} MB) {names}", f"zip 이상 {names}")
req = open(f"{SRC}/requirements.txt", encoding="utf-8").read()
check("catboost" in req, f"requirements 확인: {req.split()}", "catboost 누락")
check("pyarrow" not in req, "pyarrow 의존 없음", "pyarrow 가 필요하면 서버에서 실패")

SIM = f"{SP}/submit16_sim"
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
r = subprocess.run([sys.executable, "-u", "script.py"], cwd=SIM,
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
el = time.time() - t0
for line in [l for l in (r.stdout or "").splitlines()
             if any(k in l for k in ("rows", "pred mean", "Saved", "coverage",
                                     "X=", "done", "multiclass"))]:
    log(f"    | {line}")
if r.returncode != 0:
    for line in (r.stderr or "").splitlines()[-20:]:
        log(f"    | {line}")
check(r.returncode == 0, f"정상 종료 ({el:.0f}s)", f"exit {r.returncode}")
check(el < 600, f"추론 시간 {el:.0f}s — 제한 600s의 {600/max(el,0.1):.0f}배 여유", "10분 초과")

sub = pd.read_csv(f"{SIM}/output/submission.csv")
p = sub["control_success"].to_numpy(float)
check(len(sub) == N_EVAL, f"행수 {len(sub):,}", f"행수 불일치 {len(sub):,}")
check(list(sub.columns) == ["row_id", "control_success"],
      f"컬럼 {list(sub.columns)}", f"컬럼 이상 {list(sub.columns)}")
check(not np.isnan(p).any() and p.min() >= 0 and p.max() <= 1,
      f"확률 정상 [{p.min():.4f}, {p.max():.4f}] 평균 {p.mean():.6f}", "확률 이상")
check((sub["row_id"].to_numpy() == fake["row_id"].to_numpy()).all(),
      "row_id 순서 = sample_submission 순서", "순서 불일치")

log("")
log("=" * 84)
log("D. 진짜 test.csv 5행")
log("=" * 84)
SIM5 = f"{SP}/submit16_sim5"
if os.path.exists(SIM5):
    shutil.rmtree(SIM5)
os.makedirs(f"{SIM5}/data")
with zipfile.ZipFile(OUT_ZIP) as z:
    z.extractall(SIM5)
for f in ["test.csv", "sample_submission.csv"]:
    shutil.copy(f"{ROOT}/data/{f}", f"{SIM5}/data/{f}")
r5 = subprocess.run([sys.executable, "-u", "script.py"], cwd=SIM5,
                    capture_output=True, text=True, encoding="utf-8", errors="replace")
check(r5.returncode == 0, "정상 종료", f"exit {r5.returncode}")
if r5.returncode == 0:
    s5 = pd.read_csv(f"{SIM5}/output/submission.csv")
    v5 = s5["control_success"].to_numpy(float)
    log(f"    예측값: {[round(v, 6) for v in v5]}")
    check(np.allclose(v5, exp_ens, atol=1e-12),
          "기준 예측값과 일치", f"기준과 다름 (최대차 {np.abs(v5-exp_ens).max():.3e})")
else:
    for line in (r5.stderr or "").splitlines()[-15:]:
        log(f"    | {line}")

log("")
log("=" * 84)
log("최종: " + ("✅ 제출 가능" if ok_all else "❌ 문제 — 제출 금지"))
log("=" * 84)
log(f"  {OUT_ZIP}")
log("  멀티클래스 5클래스 + lr0.04 + 시즌가중치 0.9^age, CatBoost 8시드")
log("  로컬 2024 폴드 proj8 867.6 (현 배포본 853.6 대비 +14.0)")
log("  전이율 1.12 가정 시 LB 약 1050 (현재 1033.99)")
sys.exit(0 if ok_all else 1)
