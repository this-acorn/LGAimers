# -*- coding: utf-8 -*-
"""exp/69 준비 블록 재사용 -> exp/83 (EBM 다양성 파트너 프로브) 생성기."""
import io

SRC = "exp/69_drift_axis.py"
DST = "exp/83_ebm_probe.py"
lines = io.open(SRC, encoding="utf-8").read().splitlines(True)

# 준비 블록: import 시작부터 "준비 완료" tick 까지 (exp/65와 동일함이 검증된 블록)
i0 = next(i for i, l in enumerate(lines) if l.startswith("import importlib.util"))
i1 = next(i for i, l in enumerate(lines) if l.startswith('tick(f"준비 완료'))
prep = "".join(lines[i0:i1 + 1])

HEAD = '''# -*- coding: utf-8 -*-
"""
[83] EBM 다양성 파트너 프로브 — 40만 행 서브샘플 짝지음 (챔피언 후보 아님)

왜 재는가 (exp/77·78 의 결론에서 직접 도출):
  · 우리는 사실상 단일 모델이다. 혼합 이득의 닫힌 해:
        gain = (d + K)^2 / (4K)   — 파트너는 이길 필요가 없다. K 가 크면 져도 된다.
  · OVA 는 "손실함수만 바꾸면 같은 함수로 수렴"해서 죽었다 (K_local 11.7).
    EBM(GA2M: 피처별 형태함수 + 쌍 상호작용 선형합)은 **함수 클래스 자체가 달라서**
    K 가 진짜로 클 수 있는 유일하게 남은 후보다.
  · 챔피언으로는 가망 없다 — 이 문제는 고차 상호작용이 크다(매치업 +107).
    질문은 오직 "d 가 K 를 살릴 만큼 얕은가"이다.

설계 (전체 학습은 수 시간이라 서브샘플 삼단 논법):
  1) 같은 40만 행에서 CatBoost(it500/d6/lr0.04, thread 8, seed 42)와 EBM 을 나란히 학습
     -> 함수클래스 격차 gap 을 짝지음으로 측정
  2) 같은 설정 CatBoost 의 전체 122만 행 점수는 이미 있다 (exp/65 seed42 = 830.9)
     -> 데이터 3배의 값 scale_gain 을 실측으로 안다
  3) 낙관 가정(EBM 도 데이터 3배에서 CatBoost 만큼 번다 — 가법모델은 보통 더 일찍
     포화하므로 이건 EBM 에 유리한 상한)으로 전체학습 EBM 점수를 추정
     -> 그 상한으로도 MC04 대비 혼합이득 < +15 면 영구 종결

판정:
  · 낙관 시나리오 혼합이득 < +15  ->  기각, 축 영구 종결
  · >= +15  ->  전체 학습 승격. 단 HGB 전례 경고 — 파트너 계열 자체의 2025 전이가
    나쁘면(HGB −68) 로컬 K 가 아무리 커도 무효. 승격해도 소량 가중으로만.

실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/83_ebm_probe.py   (~1..1.5시간)
"""

'''

BODY = '''
# ===========================================================================
log("")
log("=" * 88)
log("A. 서브샘플 짝지음 — 같은 40만 행에서 CatBoost vs EBM")
log("=" * 88)
N_SUB = 400_000
rng = np.random.default_rng(42)
idx = np.sort(rng.choice(len(Xtr), size=N_SUB, replace=False))
Xs = Xtr.iloc[idx].reset_index(drop=True)
ys = y_tr[idx]
log(f"  서브샘플 {N_SUB:,} / {len(Xtr):,}  성공률 {ys.mean():.4f}")

# ---- 프로브 키트 내보내기 (팀 4인 분산 측정용 — 전원 같은 행·같은 기준으로 잰다) ----
import os as _os
import shutil as _shutil
KIT = "probe_kit"
_os.makedirs(KIT, exist_ok=True)
Xs.to_parquet(f"{KIT}/X_train_sub.parquet")
np.save(f"{KIT}/y_train_sub.npy", ys.astype("int8"))
Xva.to_parquet(f"{KIT}/X_valid.parquet")
np.save(f"{KIT}/y_valid.npy", y_va.astype("int8"))
np.save(f"{KIT}/sub_idx.npy", idx)
_shutil.copy("lab/66_mc_lr04.npy", f"{KIT}/mc04_ref.npy")
_sz = sum(_os.path.getsize(f"{KIT}/{_f}") for _f in _os.listdir(KIT)) / 1024 ** 2
log(f"  ★ 프로브 키트 저장: {KIT}/ ({_sz:.0f} MB) — 팀원에게 통째로 전달, "
    f"채점은 score_probe.py")

r_ = float(y_va.mean())
DEN = r_ * (1 - r_)
pva = Pool(Xva, cat_features=list(s12.CAT))

t0 = time.time()
cb = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.04,
                        l2_leaf_reg=10.0, verbose=False, thread_count=NTHREAD,
                        allow_writing_files=False, random_seed=42)
cb.fit(Pool(Xs, ys, cat_features=list(s12.CAT)))
p_cb = cb.predict_proba(pva)[:, 1]
s_cb_sub = raw_score(p_cb, y_va)
ANCHOR_FULL = 830.9   # exp/65 d6_lr04 seed42 — 전체 122만 행, 같은 설정·같은 thread 8
scale_gain = ANCHOR_FULL - s_cb_sub
tick(f"CatBoost(40만) {s_cb_sub:8.1f}   [전체 122만 동일설정 {ANCHOR_FULL} "
     f"→ 데이터 3배의 값 {scale_gain:+.1f}]  ({time.time()-t0:.0f}s)")

# ---- EBM ----
from interpret.glassbox import ExplainableBoostingClassifier


def ebm_frame(X):
    d = X.copy()
    for c in d.columns:
        if c not in s12.CAT:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(-999.0)
    return d


FT = ["nominal" if c in s12.CAT else "continuous" for c in Xs.columns]
t0 = time.time()
ebm = ExplainableBoostingClassifier(interactions=20, outer_bags=4, max_bins=256,
                                    feature_types=FT, n_jobs=6, random_state=42)
ebm.fit(ebm_frame(Xs), ys)
tick(f"EBM 학습 완료 ({time.time()-t0:.0f}s)")
p_ebm = ebm.predict_proba(ebm_frame(Xva))[:, 1].astype("float64")
s_ebm_sub = raw_score(p_ebm, y_va)
np.save("lab/83_ebm_sub.npy", p_ebm.astype("float32"))
pen_e = 100000 * (p_ebm.mean() - r_) ** 2 / DEN
tick(f"EBM(40만, pairwise 20) {s_ebm_sub:8.1f}   변별력 {s_ebm_sub+pen_e:.1f} "
     f"벌점 {pen_e:.1f}")

gap = s_ebm_sub - s_cb_sub
log(f"")
log(f"  ★ 함수클래스 격차 (같은 40만 행): EBM − CatBoost = {gap:+.1f}")

log("")
log("=" * 88)
log("B. MC04 대비 파트너 값 — d 와 K 를 함께")
log("=" * 88)
MC04 = np.load("lab/66_mc_lr04.npy").astype("float64")
S_MC = raw_score(MC04, y_va)
D = float(np.mean((p_ebm - MC04) ** 2))
K = 100000 * D / DEN
log(f"  MC04 2시드 {S_MC:.1f}   D(EBM, MC04) = {D:.4e}   K_local = {K:.1f}")
log(f"  (참고 K_local: CB65↔CS79 191.3 / 같은 계열끼리 42~50 / OVA 11.7 / 시드간 26.8)")

s_ebm_opt = s_ebm_sub + scale_gain
rows = [("현재 (40만 학습 그대로)", s_ebm_sub - S_MC),
        ("낙관 (전체학습 = CB만큼 증가 가정)", s_ebm_opt - S_MC)]
log(f"")
log(f"  {'시나리오':34s} {'d':>8s} {'w*':>7s} {'로컬 혼합이득':>12s} {'K25 환산이득':>12s}")
best_opt = 0.0
for tag, d in rows:
    if K > 0 and d > -K:
        g = (d + K) ** 2 / (4 * K)
        w = min(max(0.5 + d / (2 * K), 0.0), 1.0)
    else:
        g, w = 0.0, 0.0
    K25 = K * 0.614
    g25 = (d + K25) ** 2 / (4 * K25) if (K25 > 0 and d > -K25) else 0.0
    log(f"  {tag:34s} {d:+8.1f} {w:7.3f} {g:+12.1f} {g25:+12.1f}")
    if "낙관" in tag:
        best_opt = g

log("")
log("=" * 88)
log("판정 (임계: 낙관 시나리오의 로컬 혼합이득 +15)")
log("=" * 88)
if best_opt >= 15:
    log(f"  ★통과 후보 — 낙관 혼합이득 {best_opt:+.1f}. 전체 학습 승격을 검토하라.")
    log("    ⚠ 단 HGB 전례: 파트너 계열의 2025 전이가 −68이면 로컬 K는 무효다.")
    log("    승격해도 가중은 w* 이하 소량으로, 그리고 반드시 별도 제출로 전이를 실측하라.")
else:
    log(f"  기각 — 낙관 상한조차 {best_opt:+.1f} < +15. EBM 축 영구 종결.")
    log("    가법(+쌍) 함수클래스는 이 문제의 고차 상호작용을 못 담는다는 뜻이다.")
    log("    이로써 '다른 함수클래스 파트너' 탐색은 EBM까지 소진:")
    log("    LightGBM(9번째 시드) / HGB(전이 −68) / OVA(같은 함수) / 트리구조(−83) / EBM")
log("=" * 88)
'''

io.open(DST, "w", encoding="utf-8", newline="").write(HEAD + prep + BODY)
import ast
ast.parse(io.open(DST, encoding="utf-8").read())
print(f"{DST} 생성 (syntax OK)")

# 준비 블록 동일성 확인
b = io.open(DST, encoding="utf-8").read()
assert "NTHREAD = 8" in b and 'tick(f"준비 완료' in b
assert "ExplainableBoostingClassifier" in b
print("준비 블록 재사용 확인 (exp/69 = exp/65 검증본)")
