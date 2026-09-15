# -*- coding: utf-8 -*-
"""exp/66 -> exp/86 : no-ID CatBoost 를 **전체 데이터**로 재서 d/K 를 확정한다."""
import io

s = io.open("exp/66_mc_lr04.py", encoding="utf-8").read()
_o = s.index('"""'); s = s[s.index('"""', _o+3)+3:]

HEAD = '''# -*- coding: utf-8 -*-
"""
[86] no-ID CatBoost — 혼합 파트너로서 d/K 확정 측정 (전체 데이터, 40만 프로브 아님)

왜 전체 데이터인가:
  40만 프로브는 d 에 데이터 페널티(약 -117)가 섞여 들어가 보정 논란이 남는다.
  no-ID 는 같은 CatBoost 라 exp/66 하네스를 그대로 쓸 수 있고 2시드 65분이면 끝난다.
  -> MC04(전체 2시드)와 **완전히 같은 조건**에서 d 와 K 를 잰다. 보정 불필요.

설계 (exp/66 과 단 한 곳만 다르다):
  FEATS 에서 pitcher_id, batter_id 두 개를 뺀다 (79 -> 77).
  동일 유지: MultiClass 5클래스, it500/d6/lr0.04/l2=10, thread 14, SEEDS[42,7], 같은 폴드.

왜 이게 마지막 후보인가 (08-27 프로브 결과):
    RealMLP    d=-491 K=635  로컬 +8.2  2025 +0.0
    ExtraTrees d=-332 K=446  로컬 +7.3  2025 +0.0
    FT 3종     d<-K          전부 0
    EBM        d=-413 K=315  0
  전부 d 가 너무 깊어 K 전이(0.614)를 못 견딘다. 승격 전선(2025 +15 기준):
    d=0 -> K_local 98 필요 / d=-20 -> 156 / d=-60 -> 256 / d=-330 -> 821
  no-ID 는 유일하게 d 가 얕을(-10~-30 예상) 후보다. 관건은 K 가 156 을 넘느냐.
  같은 계열 실측 K: 시드간 27 / 인접 피처셋 70 / 큰 피처셋 차이 191.
  ID 두 개는 중요도 최상위라 분할 구조가 크게 바뀔 수 있다 -> 잴 가치가 있다.

실행: PYTHONIOENCODING=utf-8 python -u exp/86_noid_full.py   (~65분)
"""'''
s = HEAD + s

OLD = "FEATS = BASE + ENG18 + s12.CS_FEATS + s12.PT_FEATS"
NEW = """FEATS_ALL = BASE + ENG18 + s12.CS_FEATS + s12.PT_FEATS
DROP_ID = ["pitcher_id", "batter_id"]
FEATS = [c for c in FEATS_ALL if c not in DROP_ID]
assert len(FEATS) == len(FEATS_ALL) - 2, "ID 컬럼 제거 실패"
log(f"no-ID: {len(FEATS_ALL)} -> {len(FEATS)} 피처 (제거 {DROP_ID})")"""
assert OLD in s
s = s.replace(OLD, NEW)

TAIL = '''ens = np.mean(preds, axis=0)
np.save("lab/86_noid_full.npy", ens.astype("float32"))
solo = [raw_score(p, y_va) for p in preds]

REF = np.load("lab/66_mc_lr04.npy").astype("float64")
S_REF = raw_score(REF, y_va)
r_ = float(y_va.mean())
DEN = r_ * (1 - r_)


def blend(d, K):
    if K <= 0:
        return 0.0, 0.0
    w = min(max(0.5 + d / (2 * K), 0.0), 1.0)
    return w, w * d + w * (1 - w) * K


s2 = raw_score(ens, y_va)
mean_solo = float(np.mean(solo))
p8 = 1.75 * s2 - 0.75 * mean_solo
pen = 100000 * (ens.mean() - r_) ** 2 / DEN
d = s2 - S_REF
D = float(np.mean((ens - REF) ** 2))
K = 100000 * D / DEN
w, g = blend(d, K)

log("")
log("=" * 84)
log("판정 — no-ID CatBoost, 전체 데이터 2시드 (MC04 와 완전 동일 조건)")
log("=" * 84)
log(f"  MC04 (79피처)   시드 842.3 / 848.9   2시드 {S_REF:.1f}   proj8 857.3")
log(f"  no-ID (77피처)  시드 {solo[0]:.1f} / {solo[1]:.1f}   2시드 {s2:.1f}   proj8 {p8:.1f}")
log(f"                  변별력 {s2 + pen:.1f} / 중심벌점 {pen:.1f}")
log("")
log(f"  d (격차)        {d:+8.1f}     <- 40만 보정 불필요, 같은 조건 직접 측정")
log(f"  K (다양성)      {K:8.1f}     (D = {D:.4e})")
log(f"  최적 가중 w*    {w:8.3f}")
log(f"  로컬 혼합 이득   {g:+8.1f}")
sens = []
for ratio in (0.4, 0.614, 0.8):
    _, gr = blend(d, K * ratio)
    sens.append(f"{ratio:g}배 {gr:+.1f}")
log("  2025 추정 이득   " + "   ".join(sens))
log("")
_, g614 = blend(d, K * 0.614)
if g614 >= 15:
    log("  ★통과 — 승격. 전체 배포 학습 후 단독 제출로 2025 전이 실측 -> 혼합 제출.")
elif g > 0:
    log("  △ 로컬은 양수지만 2025 환산에서 미달. K 전이율이 0.8 이상이어야 성립.")
    log("    이 경우 소량 가중 혼합을 1회 제출로 시험할지 결정 필요.")
else:
    log("  ✕ 기각 — 이종 파트너 축 소진. 남은 것은 submit14 확정.")
log("=" * 84)
'''
s = s[:s.index("ens = np.mean(preds, axis=0)")] + TAIL
io.open("exp/86_noid_full.py", "w", encoding="utf-8", newline="").write(s)
import ast; ast.parse(io.open("exp/86_noid_full.py", encoding="utf-8").read())
print("exp/86_noid_full.py 생성 (syntax OK)")
