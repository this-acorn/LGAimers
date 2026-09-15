# -*- coding: utf-8 -*-
"""exp/66 -> exp/75 생성기. 변경점을 명시적으로 남기기 위해 패치 형태로 작성한다."""
import io

SRC = "exp/66_mc_lr04.py"
DST = "exp/75_mc7_lr04.py"
s = io.open(SRC, encoding="utf-8").read()
n_before = len(s)

# ---------------------------------------------------------------- 1) 헤더
# 파일은 `# -*- coding -*-` 줄로 시작하므로 첫 """는 여는 따옴표다. 닫는 쪽을 찾는다.
_open = s.index('"""')
head_end = s.index('"""', _open + 3) + 3
NEW_HEAD = '''# -*- coding: utf-8 -*-
"""
[75] 7클래스 재측정 — 제구축 × 판정축, lr0.04, 스레드 정합 (exp/66과 완전 동일 조건)

★ 이건 새 축이 아니라 exp/62 재측정이다. HANDOFF §4-2 는 "7클래스 +10.9 — 임계 미달"로
  적어놨지만, 그 숫자는 **측정이 아니라 측정 인공물**이다:

    exp/62_mc7.py:34   NTHREAD = 8       <- mc7 을 스레드 8로 학습
    exp/53_multiclass.py:31 thread_count=14  <- 비교 기준 817.6 은 스레드 14로 학습
    lab/62_result.txt  판정 (2024 폴드, 시드42 **단일**)

  CatBoost는 thread_count가 바뀌면 결과가 바뀐다. 이 결함이 증류 실험에서 +37.3 짜리
  위양성을 만들었고 exp/64 가 스레드를 맞추자 +0.5 로 무너졌다(36.8점 인공물).
  거기에 시드 1개(σ=15.3)까지 겹쳐, +10.9 의 실제 오차범위는 ±29 수준이다.
  즉 이 축은 **기각된 적이 없다. 측정된 적이 없을 뿐이다.**

설계 (exp/66 과 단 두 곳만 다르다 — 나머지는 한 글자도 건드리지 않는다):
  · 라벨: 5클래스 -> 7클래스 (성공을 볼/스트라이크/기타로 세분, exp/62 와 동일 정의)
      0 성공&볼   1 성공&스트   2 성공&기타
      3 미들만    4 리버스만    5 미들∩리버스   6 빅미스
  · P(성공) = P0 + P1 + P2
  동일 유지: thread_count=14, SEEDS=[42,7], it500/d6/lr0.04/l2=10, 79피처, 같은 폴드

기준 벡터가 이미 디스크에 있다 — lab/66_mc_lr04.npy (MC04):
  시드42 842.3 / 시드7 848.9 / 시드평균 845.6 / 2시드 852.3 / K̄ 26.8 / 8시드추정 857.3

판정 (exp/74 의 교훈 반영 — 2시드 숫자만 보면 안 된다):
  ★ 시드평균 이득이 판정의 주축이다. 그것이 편향(=배포에서 살아남는 몫)이다.
    lr 축이 2시드에서 +15.3 로 보였다가 8시드에서 +3.8 로 사라진 게 그 증거다.
  proj8 = 1.75*S2 - 0.75*mean(solo)   (exp/74 에서 검증된 항등식)
  통과: 시드평균 이득 >= +10  ->  판정축은 살아있다, 배포 후보
  기각: 그 미만  ->  판정축 영구 종결. 9클래스 후속 금지.

실행: PYTHONIOENCODING=utf-8 python -u exp/75_mc7_lr04.py   (~1.3시간)
"""'''
s = NEW_HEAD + s[head_end:]

# ---------------------------------------------------------------- 2) 라벨 복원 4종 -> 6종
OLD_LABS = '''for k, col in [("lab_mid", "asof_pitcher_middle_rate"),
               ("lab_rev", "asof_pitcher_reverse_rate"),
               ("lab_fb", "asof_pitcher_fastball_rate"),
               ("lab_brk", "asof_pitcher_breaking_rate")]:'''
NEW_LABS = '''for k, col in [("lab_mid", "asof_pitcher_middle_rate"),
               ("lab_rev", "asof_pitcher_reverse_rate"),
               ("lab_fb", "asof_pitcher_fastball_rate"),
               ("lab_brk", "asof_pitcher_breaking_rate"),
               ("lab_ball", "asof_pitcher_ball_rate"),
               ("lab_str", "asof_pitcher_strike_rate")]:'''
assert OLD_LABS in s, "라벨 복원 블록 불일치"
s = s.replace(OLD_LABS, NEW_LABS)

OLD_REC = '''for k in ["lab_mid", "lab_rev", "lab_fb", "lab_brk"]:
    df[k] = rec[k]'''
NEW_REC = '''for k in ["lab_mid", "lab_rev", "lab_fb", "lab_brk", "lab_ball", "lab_str"]:
    df[k] = rec[k]'''
assert OLD_REC in s, "rec 대입 블록 불일치"
s = s.replace(OLD_REC, NEW_REC)

# ---------------------------------------------------------------- 3) 5클래스 -> 7클래스
OLD_CLS = '''# 5클래스: 0=성공 1=미들만 2=리버스만 3=미들∩리버스 4=빅미스
cls = np.full(len(df), -1, dtype="int8")
ok = df.lab_mid.notna().to_numpy() & df.lab_rev.notna().to_numpy()
m_ = df.lab_mid.to_numpy() == 1
r_ = df.lab_rev.to_numpy() == 1
cls[ok & (y_bin == 1)] = 0
cls[ok & (y_bin == 0) & m_ & ~r_] = 1
cls[ok & (y_bin == 0) & ~m_ & r_] = 2
cls[ok & (y_bin == 0) & m_ & r_] = 3
cls[ok & (y_bin == 0) & ~m_ & ~r_] = 4
df["_cls"] = cls
log(f"클래스 분포: " + " ".join(f"{c}:{np.mean(cls==c)*100:.1f}%" for c in range(5))
    + f"  미복원 {np.mean(cls==-1)*100:.2f}%")'''
NEW_CLS = '''# ★ 7클래스 (exp/62 와 동일 정의): 성공을 판정축으로 세분
#   0 성공&볼  1 성공&스트  2 성공&기타  3 미들만  4 리버스만  5 미들∩리버스  6 빅미스
cls = np.full(len(df), -1, dtype="int8")
ok = np.ones(len(df), dtype=bool)
for _k in ["lab_mid", "lab_rev", "lab_ball", "lab_str"]:
    ok &= df[_k].notna().to_numpy()
m_ = df.lab_mid.to_numpy() == 1
r_ = df.lab_rev.to_numpy() == 1
b_l = df.lab_ball.to_numpy() == 1
s_l = df.lab_str.to_numpy() == 1
cls[ok & (y_bin == 1) & b_l] = 0
cls[ok & (y_bin == 1) & s_l] = 1
cls[ok & (y_bin == 1) & ~b_l & ~s_l] = 2
cls[ok & (y_bin == 0) & m_ & ~r_] = 3
cls[ok & (y_bin == 0) & ~m_ & r_] = 4
cls[ok & (y_bin == 0) & m_ & r_] = 5
cls[ok & (y_bin == 0) & ~m_ & ~r_] = 6
df["_cls"] = cls
_names = ["성공&볼", "성공&스트", "성공&기타", "미들", "리버스", "미들∩리버스", "빅미스"]
log("7클래스 분포: " + "  ".join(f"{_names[c]} {np.mean(cls==c)*100:.1f}%"
                                for c in range(7))
    + f"  | 미복원 {np.mean(cls==-1)*100:.2f}%")
log(f"  정합 확인: P(cls 0~2) = {np.mean((cls >= 0) & (cls <= 2))*100:.2f}%"
    f"  vs 실제 성공률 {np.mean(y_bin == 1)*100:.2f}%")'''
assert OLD_CLS in s, "5클래스 블록 불일치"
s = s.replace(OLD_CLS, NEW_CLS)

# ---------------------------------------------------------------- 4) P(성공) 합산
OLD_P = '''    p_succ = proba[:, 0]                       # 클래스0 = 성공'''
NEW_P = '''    p_succ = proba[:, 0] + proba[:, 1] + proba[:, 2]   # 성공 = 클래스 0,1,2'''
assert OLD_P in s, "p_succ 블록 불일치"
s = s.replace(OLD_P, NEW_P)

# ---------------------------------------------------------------- 5) 시드 점수 수집
OLD_TICK = '''    preds.append(p_succ)
    tick(f"seed={sd}  P(성공) 앙상블전 {raw_score(p_succ, y_va):8.1f}  "
         f"({time.time()-t0:.0f}s)")'''
NEW_TICK = '''    preds.append(p_succ)
    solo.append(raw_score(p_succ, y_va))
    tick(f"seed={sd}  P(성공) 앙상블전 {solo[-1]:8.1f}  "
         f"({time.time()-t0:.0f}s)")'''
assert OLD_TICK in s, "tick 블록 불일치"
s = s.replace(OLD_TICK, NEW_TICK)
s = s.replace("preds = []\nfor sd in SEEDS:", "preds, solo = [], []\nfor sd in SEEDS:")

# ---------------------------------------------------------------- 6) 판정부 교체
OLD_JUDGE = s[s.index('ens = np.mean(preds, axis=0)'):]
NEW_JUDGE = '''ens = np.mean(preds, axis=0)
np.save("lab/75_mc7_lr04.npy", ens.astype("float32"))

# ---- exp/74 판정 프레임: 편향(시드평균) / 분산(앙상블) 분리 ----
# 기준 = MC04 (lab/66_mc_lr04.npy), 완전 동일 조건에서 측정된 값
REF = dict(name="MC04(5클래스)", solo=[842.3, 848.9], e2=852.3)
REF["mean"] = sum(REF["solo"]) / 2
REF["K"] = 4 * (REF["e2"] - REF["mean"])
REF["p8"] = 1.75 * REF["e2"] - 0.75 * REF["mean"]

mean_solo = float(np.mean(solo))
e2 = raw_score(ens, y_va)
K = 4 * (e2 - mean_solo)
p8 = 1.75 * e2 - 0.75 * mean_solo
s_inf = mean_solo + K / 2
r_ = float(y_va.mean())
DEN = r_ * (1 - r_)
pen = 100000 * (ens.mean() - r_) ** 2 / DEN

log("")
log("=" * 84)
log("판정 — mc7 @ lr0.04, 스레드/시드 정합 (exp/62 의 교란 제거)")
log("=" * 84)
log(f"  {'':16s} {'시드평균':>10s} {'2시드':>9s} {'K̄':>8s} "
    f"{'8시드추정':>10s} {'S_무한':>9s}")
log(f"  {REF['name']:16s} {REF['mean']:10.1f} {REF['e2']:9.1f} {REF['K']:8.1f} "
    f"{REF['p8']:10.1f} {REF['mean']+REF['K']/2:9.1f}")
log(f"  {'mc7(7클래스)':16s} {mean_solo:10.1f} {e2:9.1f} {K:8.1f} "
    f"{p8:10.1f} {s_inf:9.1f}")
log("")
g_mean = mean_solo - REF["mean"]
g_2 = e2 - REF["e2"]
g_8 = p8 - REF["p8"]
log(f"  ★ 시드평균 이득 {g_mean:+7.1f}   <- 편향. 배포에서 살아남는 몫. 이게 판정 주축.")
log(f"    2시드 이득    {g_2:+7.1f}   <- 이 숫자만 보면 lr 축처럼 속는다")
log(f"    8시드추정 이득 {g_8:+7.1f}")
log(f"    변별력 {e2 + pen:.1f} / 중심벌점 {pen:.1f} (예측평균 {ens.mean():.4f}, r={r_:.4f})")
log("")
if g_mean >= 10:
    log(f"  ★통과 — 판정축(볼/스트라이크)에 실제 신호가 있다.")
    log(f"    다음: 8시드 배포 학습. 전이율 1.12 가정 시 LB 약 "
        f"{1033.99 + 1.12 * g_8:.0f} (8시드추정 경로)")
else:
    log(f"  기각 — 판정축 영구 종결. 9클래스 후속 금지.")
    log(f"    exp/62 의 +10.9 는 스레드 교란 인공물이었음이 확정된다.")
log("=" * 84)
'''
s = s.replace(OLD_JUDGE, NEW_JUDGE)

io.open(DST, "w", encoding="utf-8", newline="").write(s)
import ast
ast.parse(io.open(DST, encoding="utf-8").read())
print(f"{DST} 생성 완료 (syntax OK)  {n_before} -> {len(s)} chars")

# 준비 블록이 exp/66과 동일한지 확인 (라벨/클래스 블록 제외)
a = io.open(SRC, encoding="utf-8").read().splitlines()
b = io.open(DST, encoding="utf-8").read().splitlines()
ia = a.index("# ---- exp/48 검증 스테이지와 동일한 피처 준비 (79피처) ----")
ib = b.index("# ---- exp/48 검증 스테이지와 동일한 피처 준비 (79피처) ----")
ja = next(i for i in range(ia, len(a)) if a[i].startswith("ptr = Pool"))
jb = next(i for i in range(ib, len(b)) if b[i].startswith("ptr = Pool"))
same = a[ia:ja] == b[ib:jb]
print(f"피처 준비 블록 {ja-ia}행 동일: {same}")
assert same, "피처 준비 블록이 달라졌다 — 짝지음 무효"
