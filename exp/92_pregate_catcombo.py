# -*- coding: utf-8 -*-
"""
[92] 학습 없는 사전 분석 3건 (사용자 계획 08-28 밤) — 챔피언 cat5(lab/89) 2024 폴드 기준

  ① 저카디널 categorical 조합 5종의 카디널리티 / 2024 미출현 비율 / 셀 크기
     + 챔피언 잔차의 셀 평균 구조 (in-sample, 귀무 기대치 차감) — 값싼 신호 게이트
  ② 5클래스 혼동 구조 → 미들∩리버스(3)를 어느 클래스에 병합할지 데이터로 결정
     (2024 홀드아웃 예측 = OOF 보다 깨끗한 out-of-fold. lab/89 5클래스 확률 사용)
  ③ game_type R/F 점수 분해 — 전체/R/F Brier, 세그먼트 스킬, 중심벌점, SSE 점유율

입력: lab/90_analysis_2024.parquet (2024 폴드 253,507행, exp/90 정렬),
      lab/89_cat5_probs_seed{42,7}.npy (5클래스 확률), data/train.csv (카디널리티용 일부 컬럼)
출력: lab/92_result.txt (stdout), lab/92_merge.json (M4 병합 대상)
실행: PYTHONIOENCODING=utf-8 python -u exp/92_pregate_catcombo.py   (~1분)
"""

import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "exp")
from common import log, raw_score, decompose

va = pd.read_parquet("lab/90_analysis_2024.parquet")
y = va["y"].to_numpy("float64")
y5 = va["y5"].to_numpy("int64")
N = len(va)
assert N == 253507
r = y.mean()
DEN = r * (1 - r)

P = [np.load(f"lab/89_cat5_probs_seed{sd}.npy").astype("float64") for sd in (42, 7)]
P5 = (P[0] + P[1]) / 2
p = P5[:, 0]
sc = raw_score(p, y)
log(f"챔피언 cat5 2시드 재현 {sc:.1f} (기대 873.9)   시드별 {raw_score(P[0][:,0], y):.1f} / {raw_score(P[1][:,0], y):.1f}")
assert abs(sc - 873.9) < 0.6, "lab/89 정렬 불일치"
resid = y - p


def _s(v):
    if pd.api.types.is_numeric_dtype(v):
        return v.fillna(-1).astype("int64").astype(str)
    return v.astype(str)


def combos(d):
    pt, bt = _s(d.pitcher_team_id), _s(d.batter_team_id)
    tb, gt = _s(d.top_bottom), _s(d.game_type)
    b, s = _s(d.balls_before), _s(d.strikes_before)
    ph, bh = _s(d.pitcher_hand), _s(d.batter_hand)
    bs, o = _s(d.base_state), _s(d.outs_before)
    return pd.DataFrame({
        "c_team_matchup": pt + "_" + bt,
        "c_pteam_role": pt + "_" + tb + "_" + gt,
        "c_bteam_role": bt + "_" + tb + "_" + gt,
        "c_count_hand": b + s + "_" + ph + bh,
        "c_base_out": bs + "_" + o,
    }, index=d.index)


# =====================================================================
# ① 카디널리티 / 커버리지 / 잔차 셀 구조
# =====================================================================
log("\n" + "=" * 88)
log("① categorical 조합 — 카디널리티·2024 커버리지·잔차 셀 구조 (챔피언 잔차, in-sample)")
log("=" * 88)
COLS = ["season", "pitcher_team_id", "batter_team_id", "top_bottom", "game_type",
        "balls_before", "strikes_before", "pitcher_hand", "batter_hand",
        "base_state", "outs_before"]
tr = pd.read_csv("data/train.csv", encoding="utf-8-sig", usecols=lambda c: c.replace("﻿", "").strip() in COLS)
tr.columns = [c.replace("﻿", "").strip() for c in tr.columns]
log(f"train 컬럼 로드 {tr.shape}  dtypes: " + ", ".join(f"{c}:{tr[c].dtype}" for c in COLS[1:]))
log("  원시 카디널리티(전체): " + ", ".join(f"{c}={tr[c].nunique()}" for c in COLS[1:]))
ctr = combos(tr)
cva = combos(va)
hist = ctr[tr.season <= 2023]
h24 = ctr[tr.season == 2024]
assert len(h24) == N
log(f"\n  {'조합':16s} {'≤2023 레벨':>10s} {'2024 레벨':>9s} {'2024 미출현행':>12s} {'셀 최소':>8s} {'셀 중앙':>8s} {'잔차구조 raw':>12s} {'귀무기대':>8s} {'순신호':>8s}")
var_r = resid.var()
gate = {}
for c in ctr.columns:
    lv_h = set(hist[c].unique())
    lv_24 = h24[c].unique()
    unseen = (~h24[c].isin(lv_h)).mean() * 100
    cnt = hist[c].value_counts()
    # 잔차 셀 평균 구조 (2024, in-sample): SSE 감소량 = Σ n_g·mean_g²
    g = pd.DataFrame({"c": cva[c].to_numpy(), "r": resid}).groupby("c")["r"].agg(["size", "mean"])
    sse_red = float((g["size"] * g["mean"] ** 2).sum())
    raw_gain = 100000.0 * (sse_red / N) / DEN
    null_gain = 100000.0 * ((len(g) - 1) * var_r / N) / DEN
    gate[c] = raw_gain - null_gain
    log(f"  {c:16s} {len(lv_h):10d} {len(lv_24):9d} {unseen:11.3f}% {cnt.min():8d} {int(cnt.median()):8d} "
        f"{raw_gain:+12.1f} {null_gain:8.1f} {raw_gain-null_gain:+8.1f}")
log("  ※ 잔차구조 = 2024 in-sample 셀 평균으로 잡히는 점수 상당량(상한). 귀무기대 = 셀 수만큼의 잡음 흡수.")
log("     순신호가 +5 미만이면 챔피언이 이미 그 조합을 흡수한 것. 학습 없이 '못 흡수한 구조'만 본다.")
# 참고: 원시 팀ID 단독 / 팀×시즌 출현
log("\n  참고 — 팀 코드 × game_type 시즌별 출현 (2019~2024):")
pres = tr.groupby(["pitcher_team_id", "game_type"])["season"].agg(lambda s: "".join(str(x)[-1] for x in sorted(s.unique())))
log("   " + pres.to_string().replace("\n", "\n   "))
for c in ["pitcher_team_id", "batter_team_id", "base_state", "top_bottom", "game_type"]:
    g = pd.DataFrame({"c": _s(va[c]).to_numpy(), "r": resid}).groupby("c")["r"].agg(["size", "mean"])
    sse_red = float((g["size"] * g["mean"] ** 2).sum())
    raw_gain = 100000.0 * (sse_red / N) / DEN
    null_gain = 100000.0 * ((len(g) - 1) * var_r / N) / DEN
    log(f"  (원시) {c:16s} 레벨 {len(g):3d}  잔차구조 raw {raw_gain:+6.1f}  귀무 {null_gain:4.1f}  순 {raw_gain-null_gain:+6.1f}")

# =====================================================================
# ② 5클래스 혼동 구조 — 클래스 3(미들∩리버스) 병합 대상
# =====================================================================
log("\n" + "=" * 88)
log("② 5클래스 혼동 구조 (2024 홀드아웃, lab/89 cat5 2시드 평균 확률)")
log("=" * 88)
NAMES = {0: "성공", 1: "미들만", 2: "리버스만", 3: "미들∩리버스", 4: "빅미스"}
ok = y5 >= 0
prior = np.array([np.mean(y5[ok] == c) for c in range(5)])
pm = P5[ok].mean(0)
log("  클래스 사전분포(2024): " + "  ".join(f"{NAMES[c]} {prior[c]*100:5.1f}%" for c in range(5)))
log("  예측 평균확률       : " + "  ".join(f"{NAMES[c]} {pm[c]*100:5.1f}%" for c in range(5)))
log("\n  (a) 진짜 클래스별 평균 예측확률 (행=진짜, 열=예측클래스)  [ ]=사전분포 대비 리프트")
hdr = f"  {'진짜\\예측':12s}" + "".join(f"{NAMES[c]:>16s}" for c in range(5))
log(hdr)
M = np.zeros((5, 5))
for t in range(5):
    m = ok & (y5 == t)
    M[t] = P5[m].mean(0)
    log(f"  {NAMES[t]:12s}" + "".join(f"{M[t,c]*100:7.1f}% [{M[t,c]/prior[c]:4.2f}]" for c in range(5)))
log("\n  (b) argmax 혼동행렬 (행 정규화 %) — 사전확률이 작은 클래스는 argmax 로 거의 안 뽑힘(참고용)")
am = P5.argmax(1)
log(f"  {'진짜\\예측':12s}" + "".join(f"{NAMES[c]:>12s}" for c in range(5)))
for t in range(5):
    m = ok & (y5 == t)
    row = [np.mean(am[m] == c) * 100 for c in range(5)]
    log(f"  {NAMES[t]:12s}" + "".join(f"{v:11.1f}%" for v in row))
# 실패 클래스 내부 정규화: 진짜 3 행에서 실패 클래스 {1,2,4} 중 어디에 확률이 쏠리는가
m3 = ok & (y5 == 3)
fail_cols = [1, 2, 4]
sub = P5[m3][:, fail_cols]
sub = sub / sub.sum(1, keepdims=True)
sub_all = P5[ok][:, fail_cols]
sub_all = sub_all / sub_all.sum(1, keepdims=True)
log("\n  (c) 진짜 3(미들∩리버스) 행: 실패클래스 {1,2,4} 내부 정규화 평균확률 vs 전체행")
lift = {}
for i, c in enumerate(fail_cols):
    lift[c] = sub[:, i].mean() / sub_all[:, i].mean()
    log(f"      {NAMES[c]:10s}  진짜3행 {sub[:,i].mean()*100:5.1f}%   전체행 {sub_all[:,i].mean()*100:5.1f}%   리프트 {lift[c]:.3f}")
# 반대 방향: 클래스 c 행에서 P(3) 리프트 — 3과 '서로' 혼동되는 정도
log("  (d) 각 진짜 클래스 행에서 P(3) 리프트 (상호 혼동):  "
    + "  ".join(f"{NAMES[c]} {M[c,3]/prior[3]:.2f}" for c in [1, 2, 4]))
# 코사인 유사도: 클래스별 평균 예측벡터끼리 (3 vs 1/2/4)
def cos(a, b):
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))
log("  (e) 평균 예측벡터 코사인 유사도 (3 vs c):  "
    + "  ".join(f"{NAMES[c]} {cos(M[3], M[c]):.4f}" for c in [1, 2, 4]))
# 결정: (c) 리프트 최대 = 3 이 실패클래스 중 가장 닮은 곳. (d)/(e) 는 확인용.
merge_c = max(lift, key=lift.get)
alt_d = max([1, 2, 4], key=lambda c: M[c, 3] / prior[3])
alt_e = max([1, 2, 4], key=lambda c: cos(M[3], M[c]))
log(f"\n  ★ 병합 대상 결정: 클래스 3 → {merge_c} ({NAMES[merge_c]})   [기준 (c) 리프트 {lift[merge_c]:.3f};"
    f" (d) 기준이면 {NAMES[alt_d]}, (e) 기준이면 {NAMES[alt_e]}]")
json.dump({"merge_into": int(merge_c), "lift": {str(k): float(v) for k, v in lift.items()},
           "alt_d": int(alt_d), "alt_e": int(alt_e)},
          open("lab/92_merge.json", "w"), indent=1)
# 5클래스 → 4클래스로 사후 병합해도 P(성공)은 그대로 — 학습 효과만 남는다는 점 확인
log("  (참고) 확률을 사후 병합해도 P(성공)=클래스0 은 불변 → M4 의 이득/손실은 순수 학습 효과")

# =====================================================================
# ③ game_type R/F 점수 분해
# =====================================================================
log("\n" + "=" * 88)
log("③ game_type 분해 (2024 폴드, 챔피언 cat5 2시드)")
log("=" * 88)
gt = va["game_type"].astype(str).to_numpy()
se = (p - y) ** 2
log(f"  전체   n {N:7d}  r {r:.4f}  Brier {se.mean():.5f}  점수 {sc:7.1f}  중심벌점 {decompose(p, y)['중심벌점']:5.1f}  (d={p.mean()-r:+.4f})")
log(f"  {'seg':4s} {'n':>7s} {'행비율':>7s} {'r_seg':>7s} {'Brier':>8s} {'SSE비율':>7s} {'세그스킬':>8s} {'세그벌점':>8s} {'d_seg':>8s} {'전체점수기여':>12s}")
for g in sorted(set(gt)):
    m = gt == g
    n = m.sum()
    rs = y[m].mean()
    bs = se[m].mean()
    z = decompose(p[m], y[m])
    contrib = 100000.0 * (se[m].sum() / N) / DEN   # 전체 점수에서 이 세그먼트가 깎는 양 (Brier 기여)
    log(f"  {g:4s} {n:7d} {m.mean()*100:6.1f}% {rs:7.4f} {bs:8.5f} {se[m].sum()/se.sum()*100:6.1f}% "
        f"{z['총점']:8.1f} {z['중심벌점']:8.1f} {z['d']:+8.4f} {100000 - contrib:12.1f}")
log("  ※ 세그스킬 = 세그먼트 자체 r 기준 스킬 점수. SSE비율 > 행비율 이면 그 세그먼트가 평균보다 더 틀림.")
log("     '전체점수기여' = 100000 − (세그 SSE / N) / DEN×1e5 : 세그먼트별 손실을 전체 척도로.")
# 세그먼트 × 시드: 안정성
for i, sd in enumerate((42, 7)):
    pp = P[i][:, 0]
    log(f"  시드{sd}: " + "  ".join(f"{g} {decompose(pp[gt==g], y[gt==g])['총점']:7.1f}" for g in sorted(set(gt))))
# 세그먼트 × 클래스 보정
log("\n  세그먼트별 5클래스 실제빈도 vs 예측평균 (보정 점검):")
for g in sorted(set(gt)):
    m = (gt == g) & ok
    fr = np.array([np.mean(y5[m] == c) for c in range(5)])
    pm_ = P5[m].mean(0)
    log(f"  {g}: " + "  ".join(f"{NAMES[c]} {fr[c]*100:4.1f}/{pm_[c]*100:4.1f}" for c in range(5)))
# F 안에서 무엇이 틀리나: 투수 경험(asof_pitcher_n) 구간별
pn = va["asof_pitcher_n"].fillna(0).to_numpy("float64")
bins = [0, 50, 300, 1000, 3000, 1e9]
log("\n  game_type × 투수 경험(asof_pitcher_n) 구간별 세그스킬 / n:")
for g in sorted(set(gt)):
    parts = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (gt == g) & (pn >= lo) & (pn < hi)
        if m.sum() > 500:
            parts.append(f"[{int(lo)},{int(hi) if hi < 1e9 else '∞'}) {decompose(p[m], y[m])['총점']:6.1f} (n{m.sum()})")
    log(f"  {g}: " + "  ".join(parts))
# 판정 규칙 (사용자): F 에 오류가 집중될 때만 specialist 진행
mF, mR = gt == "F", gt == "R"
ratio = se[mF].mean() / se[mR].mean()
log(f"\n  ★ F/R Brier 비 {ratio:.3f}   F 세그스킬 {decompose(p[mF], y[mF])['총점']:.1f} vs R {decompose(p[mR], y[mR])['총점']:.1f}"
    f"   F 벌점 {decompose(p[mF], y[mF])['중심벌점']:.1f} vs R {decompose(p[mR], y[mR])['중심벌점']:.1f}")
log("  판정: F 가 R 보다 뚜렷이 나쁘고(세그스킬 격차 큼) 벌점도 F 에 몰리면 3순위(F-specialist) 진행. 아니면 보류.")
log("\n저장: lab/92_merge.json")
