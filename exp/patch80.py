# -*- coding: utf-8 -*-
"""exp/67 -> exp/80 생성기: 시즌 가중치 0.9^age 를 배포 학습에 적용."""
import io

SRC = "exp/67_train_mc04.py"
DST = "exp/80_train_mcw.py"
s = io.open(SRC, encoding="utf-8").read()

_open = s.index('"""')
head_end = s.index('"""', _open + 3) + 3
NEW_HEAD = '''# -*- coding: utf-8 -*-
"""
[80] 최종 배포 학습 — 멀티클래스 5클래스 + lr0.04 + **시즌 가중치 0.9^age**, 8시드

채택 근거 (exp/79, 2024 폴드 · 기준 MC04 무가중):
    w090   시드42 852.7 (+10.4)   시드7 859.8 (+10.9)
           paired 평균 +10.6   **paired 산포 0.27**
           2시드 862.7 (+10.4)  시드평균 856.2 (+10.6)  proj8 867.6 (+10.3)
           변별력 893.4 (기준 885.8)   중심벌점 30.7 (기준 33.5)
    w080   paired 평균 +1.3, 산포 4.12  -> 노이즈. 기각.

이 축이 채택된 이유 (네 가지가 전부 맞아떨어진다):
  1. 네 지표(2시드·시드평균·proj8·paired)가 +10.3~+10.6 에서 일치 = 순수 편향 이득.
     exp/74 의 lr 축(2시드 +15.3 -> 8시드 +3.8)처럼 앙상블에 먹히는 종류가 아니다.
  2. paired 산포 0.27. 고정 임계 +15 는 **비짝지음** σ=15.3 기준이라 여기 쓰면 안 된다.
  3. 변별력과 중심벌점이 **동시에** 개선됐다. 가중치가 예측 평균을 흔들어 벌점만
     줄어드는 경우를 경계했는데 반대로 나왔다.
  4. 냉동 표가 없다. 학습 표본에 가중치를 줄 뿐이라 2025 에 낡을 테이블이 생기지 않는다.
     -> 전이 위험 낮은 부류 (CS 1.03 / 멀티클래스 1.12 와 같은 계열)

감쇠율 0.9 가 정점인 근거: 멀티클래스에서 0.8 -> +1.3, 0.9 -> +10.6, 1.0 -> 0.
세 점에 2차식을 맞추면 정점이 0.897. 0.95 를 더 볼 필요 없다.
(이진에서는 정점이 0.8 이었다 — exp/69 A상 +7.7. 클래스가 늘면 유효표본이 더 필요하다.)

★ 검증과 배포의 유일한 차이: 학습 구간이 2019~2023(age 0~4) 에서
  2019~2024(age 0~5) 로 늘어난다. age 정의는 동일하게 `max(season) - season`.

실행: PYTHONIOENCODING=utf-8 python -u exp/80_train_mcw.py   (~5시간)
다음: exp/81 로 submit16.zip 빌드 + 가짜 서버 재현 테스트
"""'''
s = NEW_HEAD + s[head_end:]

# ---- 가중치 적용 ----
OLD = '''msk = tr["_cls"].to_numpy() >= 0
log(f"학습 행 (라벨 복원됨): {msk.sum():,} / {len(tr):,}")
Xtr = build_matrix(tr[msk], FEATS79)
ptr = Pool(Xtr, tr["_cls"].to_numpy()[msk], cat_features=list(CAT))
del tr, Xtr
tick("Pool 생성 완료")'''
NEW = '''msk = tr["_cls"].to_numpy() >= 0
log(f"학습 행 (라벨 복원됨): {msk.sum():,} / {len(tr):,}")
Xtr = build_matrix(tr[msk], FEATS79)

# ★ 유일한 변경점 — 시즌 가중치 0.9^age (exp/79 승자)
DECAY = 0.9
_season = tr["season"].to_numpy("float64")[msk]
_age = _season.max() - _season
assert len(_season) == len(Xtr), "가중치 정렬 불일치"
W = DECAY ** _age
W = W / W.mean()
_ess = (W.sum() ** 2) / (W ** 2).sum()
log(f"시즌 가중치 {DECAY}^age: 시즌 {int(_season.min())}~{int(_season.max())} "
    f"(age 0~{int(_age.max())}), 가중치 {W.min():.4f}~{W.max():.4f}")
log(f"  유효표본 {_ess:,.0f} / {len(W):,} ({_ess/len(W)*100:.0f}%)")
for _sv in sorted(set(_season)):
    _m = _season == _sv
    log(f"    {int(_sv)}  행 {_m.sum():>9,}  가중치 {W[_m][0]:.4f}")

ptr = Pool(Xtr, tr["_cls"].to_numpy()[msk], cat_features=list(CAT), weight=W)
del tr, Xtr
tick("Pool 생성 완료 (가중치 포함)")'''
assert OLD in s, "Pool 블록 불일치"
s = s.replace(OLD, NEW)

# ---- 번들 버전 표기 ----
s = s.replace('"version": "mc04"', '"version": "mc04w090", "season_decay": 0.9')

# ---- 마지막 안내 ----
s = s.replace("다음: exp/68 submit16.zip 빌드", "다음: exp/81 submit16.zip 빌드")

io.open(DST, "w", encoding="utf-8", newline="").write(s)
import ast
ast.parse(io.open(DST, encoding="utf-8").read())
print(f"{DST} 생성 완료 (syntax OK)")

# 가중치 블록 외에는 exp/67과 동일해야 한다
a = io.open(SRC, encoding="utf-8").read().splitlines()
b = io.open(DST, encoding="utf-8").read().splitlines()
ia = a.index('df["_cls"] = cls')
ib = b.index('df["_cls"] = cls')
ja = next(i for i in range(ia, len(a)) if a[i].startswith('msk = tr["_cls"]'))
jb = next(i for i in range(ib, len(b)) if b[i].startswith('msk = tr["_cls"]'))
assert a[ia:ja] == b[ib:jb], "피처 준비 블록이 달라졌다"
print(f"피처 준비 블록 {ja-ia}행 동일: True")
print("변경점:", [l for l in b if "DECAY = " in l or "weight=W" in l])
