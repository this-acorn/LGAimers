# -*- coding: utf-8 -*-
"""
프로브 채점기 v4 — 새 모델이 "혼합 파트너"로서 값이 있는지 d 와 K 로 판정한다.
결과는 화면 출력과 동시에 report_<이름>_<n>seed.txt 로 자동 저장된다 (복사 불필요).

사용법 (probe_kit 폴더 안에서):
    python score_probe.py preds_<모델>_seed42.npy 모델명                    # 예비 필터
    python score_probe.py preds_<모델>_seed42.npy preds_<모델>_seed7.npy 모델명  # ★최종 판정

.npy 를 여러 개 주면 평균(=시드 앙상블)으로 채점한다.

★ 원칙 1 — 모든 게이트는 "단독 점수"가 아니라 **혼합 이득** 기준이다.
★ 원칙 2 — d 와 K 는 반드시 **같은 두 예측벡터 쌍**에서 계산한다.
★ 원칙 3 — 단일 시드의 K 는 시드 노이즈만큼 부풀어 있다(같은 모델 시드간 K̄≈27).
  최종 판정은 2시드 평균으로. 단일 시드일 땐 보수 보정(K−27) 이득도 함께 출력된다.
★ 원칙 4 — 프로브를 대회에 제출하지 않는다. 보내는 것은 report_*.txt + npy 뿐이다.

혼합 이득 (닫힌 해, w* 는 0..1 클립):
    w* = clip(0.5 + d/(2K)),   gain = w*·d + w*(1−w*)·K
    d = 후보 − 챔피언(MC04),   K = (100000/DEN)·E[(p_후보 − p_챔피언)²]

2025 환산: 비율 0.614 는 CB65↔CS79 **한 쌍**에서 실측된 값. 다른 계열에는 '가정'이므로
민감도(0.4 / 0.614 / 0.8)를 병기한다. 최종 혼합비는 리더보드 실측으로만 재계산한다.
"""

import datetime
import hashlib
import re
import sys
import numpy as np

args = [a for a in sys.argv[1:] if a.endswith(".npy")]
name = next((a for a in sys.argv[1:] if not a.endswith(".npy")), None) \
    or args[0].replace(".npy", "")
if not args:
    print(__doc__)
    sys.exit(1)

_lines = []


def emit(msg=""):
    print(msg)
    _lines.append(msg)


def sha12(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


y = np.load("y_valid.npy").astype("float64")
ref = np.load("mc04_ref.npy").astype("float64")
r = y.mean()
DEN = r * (1 - r)


def S(q):
    return 100000 * (1 - np.mean((q - y) ** 2) / DEN)


def blend(d, K):
    """클립된 최적 가중과 이득. 같은 쌍의 (d, K)만 넣을 것."""
    if K <= 0:
        return 0.0, 0.0
    w = min(max(0.5 + d / (2 * K), 0.0), 1.0)
    return w, w * d + w * (1 - w) * K


emit("=" * 70)
emit(f"[{name}]  ({len(args)}시드 평균)   "
     f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
emit(f"  키트 지문: y_valid {sha12('y_valid.npy')} / mc04_ref {sha12('mc04_ref.npy')}")
ps = []
for a in args:
    p1 = np.load(a).astype("float64").ravel()
    assert p1.shape == y.shape, f"{a}: 길이 {p1.shape} != {y.shape} (253,507이어야 함)"
    assert np.isfinite(p1).all(), f"{a}: NaN/inf 포함"
    assert 0.0 <= p1.min() and p1.max() <= 1.0, f"{a}: 확률 범위 밖 [{p1.min()}, {p1.max()}]"
    ps.append(p1)
    emit(f"  {a}: 단독 {S(p1):8.1f}   sha256 {sha12(a)}")
p = np.mean(ps, axis=0)

s, s_ref = S(p), S(ref)
pen = 100000 * (p.mean() - r) ** 2 / DEN
d = s - s_ref
D = float(np.mean((p - ref) ** 2))
K = 100000 * D / DEN
w, g = blend(d, K)

emit("-" * 70)
emit(f"  앙상블 점수      {s:8.1f}   (변별력 {s+pen:.1f} / 중심벌점 {pen:.1f})")
emit(f"  예측 통계        평균 {p.mean():.4f}  std {p.std():.4f}  "
     f"범위 [{p.min():.4f}, {p.max():.4f}]")
emit(f"  챔피언 MC04      {s_ref:8.1f}")
emit(f"  d (격차)         {d:+8.1f}")
emit(f"  K (다양성)       {K:8.1f}   (D = {D:.4e}; 참고 — 시드간 27 / 인접 피처셋 70 / "
     f"큰 피처셋 차이 191)")
emit(f"  최적 가중 w*     {w:8.3f}")
emit(f"  ★ 로컬 혼합 이득  {g:+8.1f}   <- 모든 게이트는 이 숫자 기준")
sens = []
for ratio in (0.4, 0.614, 0.8):
    _, g25 = blend(d, K * ratio)
    sens.append(f"{ratio:g}배: {g25:+.1f}")
emit(f"  2025 추정 이득    " + "   ".join(sens) + "   (K 전이율 가정 — 0.614는 실측 1쌍)")
gc = None
if len(args) == 1:
    Kc = max(K - 27.0, 0.0)
    _, gc = blend(d, Kc)
    emit(f"  보수 보정(단일시드, K−27): 혼합 이득 {gc:+8.1f}")
emit("-" * 70)
if g >= 15:
    emit("  ★ 통과 후보 — report 파일 + npy 를 보내고, 시드 7을 추가해 재채점하세요.")
elif g >= 8:
    if gc is not None and gc >= 8:
        emit("  △ 통과 경계 — 보수 보정도 +8↑. 시드 7 확정 추가 후 재채점하세요.")
    elif gc is not None:
        emit("  △ 회색지대 — 원시만 +8↑(보수 보정 미달). 시간 남으면 시드 7 추가.")
    else:
        emit("  △ 경계 — 2시드 결과입니다. report 파일 + npy 를 보내주세요 (승격 검토).")
else:
    emit("  ✕ 기각 우도 높음 — 그래도 report 파일과 npy 는 보내주세요 (기록용).")
emit("  (40만 행 학습 페널티 — 같은 조건 CatBoost 기준 약 −117 — 는 취합 때 별도 보정)")
emit("=" * 70)

safe = re.sub(r"[^0-9A-Za-z가-힣_.-]", "_", name)
out = f"report_{safe}_{len(args)}seed.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(_lines) + "\n")
print(f"\n>> 저장됨: {out}  — 이 파일과 npy 를 단톡에 올려주세요 (복사 불필요)")
