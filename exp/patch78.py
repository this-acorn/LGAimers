# -*- coding: utf-8 -*-
"""exp/66 -> exp/78 생성기 (MultiClassOneVsAll). 변경점을 명시적으로 남긴다."""
import io

SRC = "exp/66_mc_lr04.py"
DST = "exp/78_ova_lr04.py"
s = io.open(SRC, encoding="utf-8").read()

_open = s.index('"""')
head_end = s.index('"""', _open + 3) + 3
NEW_HEAD = '''# -*- coding: utf-8 -*-
"""
[78] MultiClassOneVsAll — 품질 후보가 아니라 **다양성 파트너**로서의 측정

왜 이걸 재는가 (exp/77 의 결론에서 직접 도출):
  우리는 사실상 단일모델이다. submit14 = CatBoost MultiClass 한 설정 × 시드 8개.
  2모델 혼합 이득에는 닫힌 해가 있다 —
      gain = (d + K)^2 / (4K),   d = S_파트너 − S_챔피언,  K = (1e5/DEN)·E[(p1−p2)^2]
  이 식이 스크리닝 기준을 바꾼다. 파트너는 **이길 필요가 없다**:
      K=117.5(계열 다름) 이면 d > −33.5 만 되어도 +15 이상
      K= 45.6(같은 계열) 이면 d > +6.7 이어야 +15
  따라서 후보마다 d 와 K 를 **함께** 재야 한다. d 만 보면 잘못 버린다.

OneVsAll 을 고른 이유:
  · 같은 5클래스·같은 79피처·같은 lr 인데 **손실함수만 다르다**
    (클래스별 독립 이진 학습 → 확률이 1로 합해지지 않는다)
  · 정보는 같고 오차 구조만 달라지므로, 품질은 비슷하고 K 는 클 가능성이 있다
    = 우리가 찾는 파트너의 정의 그 자체
  · exp/62·75(mc7), exp/51·72(CS reverse) 는 전부 '더 나은 챔피언'을 노렸다가 죽었다.
    이건 노리는 대상 자체가 다르다.

설계 (exp/66 과 단 한 곳만 다르다):
  loss_function: "MultiClass" -> "MultiClassOneVsAll"
  동일 유지: thread_count=14, SEEDS=[42,7], it500/d6/lr0.04/l2=10, 79피처, 같은 폴드
  P(성공) 은 두 가지로 낸다 — 정규화 전(raw)과 정규화 후(sum=1). OneVsAll 은 확률합이
  1이 아니므로 이 선택이 중심(calibration)을 바꾼다. 둘 다 재서 좋은 쪽을 쓴다.

기준 벡터가 디스크에 있다 — lab/66_mc_lr04.npy (MC04):
  시드42 842.3 / 시드7 848.9 / 시드평균 845.6 / 2시드 852.3 / K̄ 26.8 / 8시드추정 857.3

판정: 단독 점수가 아니라 **혼합 이득**으로 한다.
  통과: 로컬 혼합 이득 >= +15  ->  파트너 확보. 배포 학습 + 혼합 제출.
  기각: 그 미만  ->  혼합 축은 우리 손으로 만들 수 있는 범위에서 닫힌다.

실행: PYTHONIOENCODING=utf-8 python -u exp/78_ova_lr04.py   (~1.3시간)
"""'''
s = NEW_HEAD + s[head_end:]

OLD_LOSS = 'loss_function="MultiClass")'
NEW_LOSS = 'loss_function="MultiClassOneVsAll")'
assert s.count(OLD_LOSS) == 1, f"loss_function 블록 {s.count(OLD_LOSS)}개 발견"
s = s.replace(OLD_LOSS, NEW_LOSS)

OLD_TICK = '''    p_succ = proba[:, 0]                       # 클래스0 = 성공
    preds.append(p_succ)
    tick(f"seed={sd}  P(성공) 앙상블전 {raw_score(p_succ, y_va):8.1f}  "
         f"({time.time()-t0:.0f}s)")'''
NEW_TICK = '''    # OneVsAll 은 확률합이 1이 아니다 — 정규화 전/후를 둘 다 본다
    p_raw = proba[:, 0]
    p_nrm = proba[:, 0] / np.clip(proba.sum(axis=1), 1e-12, None)
    preds.append(p_raw)
    preds_n.append(p_nrm)
    solo.append(raw_score(p_raw, y_va))
    solo_n.append(raw_score(p_nrm, y_va))
    tick(f"seed={sd}  raw {solo[-1]:8.1f} / 정규화 {solo_n[-1]:8.1f}  "
         f"(확률합 평균 {proba.sum(axis=1).mean():.4f}, {time.time()-t0:.0f}s)")'''
assert OLD_TICK in s, "tick 블록 불일치"
s = s.replace(OLD_TICK, NEW_TICK)
s = s.replace("preds = []\nfor sd in SEEDS:",
              "preds, preds_n, solo, solo_n = [], [], [], []\nfor sd in SEEDS:")

OLD_JUDGE = s[s.index('ens = np.mean(preds, axis=0)'):]
NEW_JUDGE = '''ens_raw = np.mean(preds, axis=0)
ens_nrm = np.mean(preds_n, axis=0)
r_ = float(y_va.mean())
DEN = r_ * (1 - r_)

REF = np.load("lab/66_mc_lr04.npy").astype("float64")   # MC04 2시드
S_REF = raw_score(REF, y_va)
REF_SOLO = [842.3, 848.9]
REF_MEAN = sum(REF_SOLO) / 2
REF_K = 4 * (S_REF - REF_MEAN)
REF_P8 = 1.75 * S_REF - 0.75 * REF_MEAN
RATIO_2025 = 0.614          # K_2025 = 0.614 * K_local (exp/73, submit13 로 교정)
TAU = 1.12                  # 모델축 전이율 (submit14 실측)
LB_REF = 1033.99


def report(tag, ens, solos):
    mean_solo = float(np.mean(solos))
    e2 = raw_score(ens, y_va)
    K_self = 4 * (e2 - mean_solo)
    p8 = 1.75 * e2 - 0.75 * mean_solo
    pen = 100000 * (ens.mean() - r_) ** 2 / DEN
    D = float(np.mean((ens - REF) ** 2))
    K = 100000 * D / DEN
    d = e2 - S_REF
    g = (d + K) ** 2 / (4 * K) if K > 0 else 0.0
    w = min(max(0.5 + d / (2 * K), 0.0), 1.0) if K > 0 else 0.0
    log("")
    log(f"  [{tag}]")
    log(f"    시드 {' / '.join(f'{v:.1f}' for v in solos)}  시드평균 {mean_solo:.1f}  "
        f"2시드 {e2:.1f}  K̄(시드간) {K_self:.1f}  8시드추정 {p8:.1f}")
    log(f"    변별력 {e2 + pen:.1f} / 중심벌점 {pen:.1f} (예측평균 {ens.mean():.4f})")
    log(f"    ★ MC04 대비  d = {d:+.1f}   다양성 K_local = {K:.1f}  (D = {D:.4e})")
    log(f"      최적 가중 w*(파트너) = {w:.3f}   로컬 혼합 이득 = {g:+.1f}")
    K25 = K * RATIO_2025
    d25 = d * TAU
    g25 = (d25 + K25) ** 2 / (4 * K25) if K25 > 0 else 0.0
    log(f"      2025 환산(K×{RATIO_2025}, d×{TAU}): K={K25:.1f} d={d25:+.1f} "
        f"→ 혼합 이득 {g25:+.1f}  → LB 약 {LB_REF + g25:.0f}")
    return g, K, d


log("")
log("=" * 88)
log("판정 — MultiClassOneVsAll: 챔피언 교체가 아니라 혼합 파트너로서")
log("=" * 88)
log(f"  기준 MC04 (lab/66): 2시드 {S_REF:.1f}  시드평균 {REF_MEAN:.1f}  "
    f"K̄ {REF_K:.1f}  8시드추정 {REF_P8:.1f}")
g_raw, K_raw, d_raw = report("raw (정규화 전)", ens_raw, solo)
g_nrm, K_nrm, d_nrm = report("정규화 (합=1)", ens_nrm, solo_n)

best_tag, best_g, best_ens = (("raw", g_raw, ens_raw) if g_raw >= g_nrm
                              else ("정규화", g_nrm, ens_nrm))
np.save("lab/78_ova_lr04.npy", best_ens.astype("float32"))
log("")
log("=" * 88)
log(f"  승자: {best_tag}  로컬 혼합 이득 {best_g:+.1f}")
if best_g >= 15:
    log("  ★통과 — 혼합 파트너 확보. 배포 학습 후 가중 혼합 제출.")
    log("    ※ 파트너는 챔피언을 이길 필요가 없다. K 가 크면 져도 이득이다.")
else:
    log("  기각 — 손실함수만 바꾸는 것으로는 충분한 다양성이 안 나온다.")
    log("    혼합 축은 우리가 만들 수 있는 범위에서 닫힌다.")
    log("    (exp/77: 동급·최대다양성 파트너를 무한히 모아도 상한 +58.8)")
log("=" * 88)
'''
s = s.replace(OLD_JUDGE, NEW_JUDGE)

io.open(DST, "w", encoding="utf-8", newline="").write(s)
import ast
ast.parse(io.open(DST, encoding="utf-8").read())
print(f"{DST} 생성 완료 (syntax OK)")

a = io.open(SRC, encoding="utf-8").read().splitlines()
b = io.open(DST, encoding="utf-8").read().splitlines()
mark = "# ---- exp/48 검증 스테이지와 동일한 피처 준비 (79피처) ----"
ia, ib = a.index(mark), b.index(mark)
ja = next(i for i in range(ia, len(a)) if a[i].startswith("ptr = Pool"))
jb = next(i for i in range(ib, len(b)) if b[i].startswith("ptr = Pool"))
assert a[ia:ja] == b[ib:jb], "피처 준비 블록이 달라졌다 — 짝지음 무효"
print(f"피처 준비 블록 {ja-ia}행 동일: True")
# 라벨 블록도 동일해야 한다 (5클래스 그대로)
assert 'cls[ok & (y_bin == 0) & ~m_ & ~r_] = 4' in "\n".join(b), "5클래스 라벨 변형됨"
print("5클래스 라벨 블록 유지: True")
