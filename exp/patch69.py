# -*- coding: utf-8 -*-
"""exp/69 개정 — grow_policy 팔 추가 + exp/74·77 판정 프레임 적용."""
import io

P = "exp/69_drift_axis.py"
s = io.open(P, encoding="utf-8").read()

# ---------------------------------------------------------------- 1) 팔 추가
OLD = '''RARMS = [("l2_50", dict(l2_leaf_reg=50.0)),
         ("rsm06", dict(rsm=0.6)),
         ("mdl200", dict(min_data_in_leaf=200))]'''
NEW = '''RARMS = [("l2_50", dict(l2_leaf_reg=50.0)),
         ("rsm06", dict(rsm=0.6)),
         ("mdl200", dict(min_data_in_leaf=200))]
# ★ C상 (08-26 추가): 트리 구조 — 지금까지 건드린 적 없는 유일한 구조 축.
#   exp/78 이 "손실함수를 바꿔도 같은 함수로 수렴한다"(D(μ,μ)≈0)를 보였다.
#   손실함수는 무엇을 최소화할지만 바꾸고 **함수 클래스**는 그대로다.
#   CatBoost 기본 SymmetricTree 는 한 깊이의 모든 분기가 같은 조건을 쓰는 강한 제약이고
#   우리 모델 전부가 그 안에 있다. Depthwise/Lossguide 는 그 제약을 푼다
#   = 품질과 다양성 K 둘 다 움직일 수 있는 유일하게 남은 손잡이.
GARMS = [("depthwise", dict(grow_policy="Depthwise")),
         ("lossguide", dict(grow_policy="Lossguide", max_leaves=64))]'''
assert OLD in s
s = s.replace(OLD, NEW)

# ---------------------------------------------------------------- 2) 시드별 점수 수집
OLD_RUN = '''def run(name, prm_extra, weight):
    ptr = Pool(Xtr, y_tr, cat_features=list(s12.CAT), weight=weight)
    ps = []
    for sd in SEEDS:
        t0 = time.time()
        m = CatBoostClassifier(**{**BASE_CB, "random_seed": sd},
                               **BASE_PRM, **prm_extra).fit(ptr)
        p = m.predict_proba(pva)[:, 1]
        ps.append(p)
        tick(f"{name} seed={sd} {raw_score(p, y_va):8.1f}  ({time.time()-t0:.0f}s)")
        del m
    del ptr
    ens = np.mean(ps, axis=0)
    np.save(f"lab/69_{name}.npy", ens.astype("float32"))
    scv = raw_score(ens, y_va)
    pen = 100000 * (ens.mean() - r_) ** 2 / DEN
    res[name] = (scv, scv + pen, pen)
    log(f"  -> {name:9s} 2seed {scv:8.1f} (기준 {REF_LR04} 대비 {scv - REF_LR04:+.1f}) "
        f"| 변별력 {scv + pen:.1f} 벌점 {pen:.1f}")'''
NEW_RUN = '''# 기준 d6_lr04 의 시드별 점수 (exp/65 실측) — 편향/분산 분리에 필요
REF_SOLO = [830.9, 831.2]
REF_MEAN = sum(REF_SOLO) / 2
REF_P8 = 1.75 * REF_LR04 - 0.75 * REF_MEAN        # = 845.7 (exp/74 검증)
# 혼합 파트너 평가용 챔피언 벡터 (멀티클래스 MC04)
MC04 = np.load("lab/66_mc_lr04.npy").astype("float64")
RATIO_2025 = 0.614


def run(name, prm_extra, weight):
    ptr = Pool(Xtr, y_tr, cat_features=list(s12.CAT), weight=weight)
    ps, solo = [], []
    for sd in SEEDS:
        t0 = time.time()
        m = CatBoostClassifier(**{**BASE_CB, "random_seed": sd},
                               **BASE_PRM, **prm_extra).fit(ptr)
        p = m.predict_proba(pva)[:, 1]
        ps.append(p)
        solo.append(raw_score(p, y_va))
        tick(f"{name} seed={sd} {solo[-1]:8.1f}  ({time.time()-t0:.0f}s)")
        del m
    del ptr
    ens = np.mean(ps, axis=0)
    np.save(f"lab/69_{name}.npy", ens.astype("float32"))
    scv = raw_score(ens, y_va)
    pen = 100000 * (ens.mean() - r_) ** 2 / DEN
    mean_solo = float(np.mean(solo))
    p8 = 1.75 * scv - 0.75 * mean_solo            # exp/74 항등식
    # 혼합 파트너로서의 값 (exp/77 닫힌 해)
    s_mc = raw_score(MC04, y_va)
    D = float(np.mean((ens - MC04) ** 2))
    K = 100000 * D / DEN
    d = scv - s_mc
    bg = (d + K) ** 2 / (4 * K) if K > 0 else 0.0
    res[name] = (scv, scv + pen, pen, mean_solo, p8, K, d, bg)
    log(f"  -> {name:10s} 2seed {scv:7.1f} ({scv - REF_LR04:+6.1f})  "
        f"시드평균 {mean_solo:7.1f} ({mean_solo - REF_MEAN:+6.1f})  "
        f"proj8 {p8:7.1f} ({p8 - REF_P8:+6.1f})")
    log(f"     {'':10s} 변별력 {scv + pen:6.1f} 벌점 {pen:5.1f}  |  "
        f"MC04 대비 d={d:+7.1f} K={K:6.1f} → 혼합이득 {bg:+6.1f}")'''
assert OLD_RUN in s
s = s.replace(OLD_RUN, NEW_RUN)

# ---------------------------------------------------------------- 3) C상 실행
OLD_B = '''log("")
log("=" * 80)
log("B상: 정규화 — exp/58이 지목한 대안")
log("=" * 80)
for name, prm in RARMS:
    run(name, prm, None)'''
NEW_B = '''log("")
log("=" * 80)
log("B상: 정규화 — exp/58이 지목한 대안")
log("=" * 80)
for name, prm in RARMS:
    run(name, prm, None)

log("")
log("=" * 80)
log("C상: 트리 구조 — 대칭트리 제약을 푼다 (유일하게 남은 구조 축)")
log("=" * 80)
for name, prm in GARMS:
    run(name, prm, None)'''
assert OLD_B in s
s = s.replace(OLD_B, NEW_B)

# ---------------------------------------------------------------- 4) 판정부
OLD_J = s[s.index('log("")\nlog("=" * 80)\nlog("판정 (2024 폴드, 2시드'):]
NEW_J = '''log("")
log("=" * 96)
log("판정 — exp/74(시드 스케일링) · exp/77(혼합 닫힌 해) 프레임 적용")
log("=" * 96)
log("  판정 주축은 **시드평균 이득(편향)** 과 **proj8**. 2시드 숫자만 보면 lr 축처럼 속는다.")
log("  파트너 후보로는 혼합이득으로 본다 — 챔피언을 이길 필요가 없다.")
log("")
log(f"  {'팔':12s} {'2시드':>8s} {'시드평균':>9s} {'proj8':>8s} "
    f"{'벌점':>6s} {'K(MC04)':>8s} {'혼합이득':>8s}")
log(f"  {'기준 d6_lr04':12s} {REF_LR04:8.1f} {REF_MEAN:9.1f} {REF_P8:8.1f} "
    f"{'':>6s} {'':>8s} {'':>8s}")
for k, v in sorted(res.items(), key=lambda x: -x[1][4]):
    scv, disc, pen, mean_solo, p8, K, d, bg = v
    log(f"  {k:12s} {scv:8.1f} {mean_solo:9.1f} {p8:8.1f} {pen:6.1f} "
        f"{K:8.1f} {bg:8.1f}")
log("")
log("  ★ 채택 조건 (둘 중 하나)")
log(f"    (a) 품질:  proj8 이 기준 {REF_P8:.1f} 보다 +15 이상  → 챔피언 후보")
log("    (b) 파트너: MC04 대비 혼합이득 +15 이상            → 혼합 재료")
win_q = [k for k, v in res.items() if v[4] - REF_P8 >= 15]
win_p = [k for k, v in res.items() if v[7] >= 15]
log("")
log(f"  품질 통과: {win_q if win_q else '없음'}")
log(f"  파트너 통과: {win_p if win_p else '없음'}")
if not win_q and not win_p:
    log("")
    log("  → 전부 기각. 남은 것은 MC04/OVA 배포(proj8 857~860, LB 약 1038)뿐이다.")
log("=" * 96)
'''
s = s.replace(OLD_J, NEW_J)

io.open(P, "w", encoding="utf-8", newline="").write(s)
import ast
ast.parse(io.open(P, encoding="utf-8").read())
print("exp/69 개정 완료 (syntax OK) — 팔 8개 (시즌3 + 정규화3 + 구조2)")
