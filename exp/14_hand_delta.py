"""
[14] 투수별 좌/우 타자 상대 이력 delta — f_same_hand(+107)의 개인화 확장

실행:  PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 py -3.12 -u exp/14_hand_delta.py

아이디어 (사용자 제안):
  f_same_hand(투수손==타자손)는 리그 평균적인 매치업 효과만 잡는다.
  '이 투수가 같은손/다른손 타자를 상대로 유독 강한가/약한가'는
  행 집계가 필요해서 트리가 구조적으로 못 만든다 (HANDOFF §5-2의 가치 기준).
  주최측 asof_*에도 손 조건부 성공률은 없다 → 순수하게 새로운 정보.

  압박 상황(3볼·만루·고LI) delta는 이번에 넣지 않는다 — 클러치는 야구 연구에서
  안정적 스킬이 아니라고 알려져 있어, 가장 가망 있는 가설 하나만 정확히 잰다.

피처 정의 (f_hand_delta, 1개):
  각 행에 대해 '그 행 시즌 이전 시즌들'의 본인 기록으로
      p_over = (통산 성공 + prior×200) / (통산 투구수 + 200)          ← 전체 스무딩
      p_cond = (해당 손 상대 성공 + p_over×200) / (해당 손 투구수 + 200) ← 본인 쪽으로 스무딩
      f_hand_delta = p_cond − p_over     (이전 시즌 기록이 전혀 없으면 NaN)
  ★ delta 형태인 이유: 시즌 드리프트(r 0.56→0.49)가 원값 rate를 오염시키지만
    같은 기간의 본인 통산을 빼면 드리프트가 상쇄된다.

시간 안전성:
  train에 경기 날짜가 없어 투구 단위 as-of는 불가 → 시즌 단위 as-of로 만든다.
  (pid, season, sh) 키의 값은 반드시 season '이전' 시즌들의 합계만 사용
  (cumsum − 현재시즌). 따라서 2024 검증 행에는 ≤2023 기록만 들어간다.
  실전 test(2025) 행은 (pid, sh)별 ≤2024 전체 집계를 조인하면 되고,
  이는 여기 (pid, 2024… 아닌 2025 가상 키) 방식과 동일한 값이다.

합법성:
  집계는 전부 train.csv 기록으로 사전 계산 → 각 test 행에 자기 pid/손 조합만
  독립 조회. test 행 간 정보 교환 없음 (HANDOFF §6 허용 방식).

판정: 4시드 앙상블, ±15점 (기준 대비)
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "exp")
from common import (log, tick, decompose, load_train, add_features,
                    fit_encoder, to_matrix, ALL_ENG, HGB_FAST, LB_TOP)

SEEDS = [42, 7, 123, 2024]
THR = 15.0
A1 = 200.0   # 통산을 전체 prior 쪽으로 당기는 스무딩 (common.py의 a=200과 동일)
A2 = 200.0   # 손 조건부를 본인 통산 쪽으로 당기는 스무딩

log("train.csv 로딩 중...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS = BASE + ALL_ENG

tr_raw = df[df["season"] <= 2023].reset_index(drop=True)
va_raw = df[df["season"] == 2024].reset_index(drop=True)
y_tr = tr_raw["control_success"].to_numpy()
y_va = va_raw["control_success"].to_numpy()
PRIOR = float(y_tr.mean())


# =====================================================================
# 피처 테이블 — (pid, season, sh) → 그 시즌 '이전'까지의 delta
# =====================================================================
def hand_history_table(src, prior):
    """
    src 전체(2019~2024)로 만들어도 안전하다:
    season=S 키의 값은 S '이전' 시즌 합계만 쓰므로(cumsum−현재분),
    2024 키에는 ≤2023 기록만 들어간다. 2024의 y가 들어갈 자리(2025 키)는 없다.
    """
    t = pd.DataFrame({
        "pid": src["pitcher_id"].to_numpy(),
        "season": src["season"].to_numpy(),
        "sh": (src["pitcher_hand"] == src["batter_hand"]).astype("int8").to_numpy(),
        "y": src["control_success"].to_numpy(),
    })
    # 손 조건부: (pid, sh)별 시즌 누적에서 현재 시즌 몫을 뺀다 → 과거 시즌만
    cond = (t.groupby(["pid", "season", "sh"])["y"]
             .agg(succ="sum", n="size").reset_index()
             .sort_values(["pid", "sh", "season"]))
    g = cond.groupby(["pid", "sh"])
    cond["pc_succ"] = g["succ"].cumsum() - cond["succ"]
    cond["pc_n"] = g["n"].cumsum() - cond["n"]

    # 본인 통산(손 무관): 같은 방식
    over = (t.groupby(["pid", "season"])["y"]
             .agg(succ="sum", n="size").reset_index()
             .sort_values(["pid", "season"]))
    go = over.groupby("pid")
    over["po_succ"] = go["succ"].cumsum() - over["succ"]
    over["po_n"] = go["n"].cumsum() - over["n"]

    tbl = cond.merge(over[["pid", "season", "po_succ", "po_n"]],
                     on=["pid", "season"], how="left")
    p_over = (tbl["po_succ"] + prior * A1) / (tbl["po_n"] + A1)
    p_cond = (tbl["pc_succ"] + p_over * A2) / (tbl["pc_n"] + A2)
    tbl["f_hand_delta"] = np.where(tbl["po_n"] > 0,
                                   (p_cond - p_over), np.nan).astype("float32")
    return tbl[["pid", "season", "sh", "f_hand_delta"]]


def attach(frame, tbl):
    """행마다 자기 (pid, season, 현재 매치업 손)만 조회 — 행 독립"""
    d = frame.copy()
    key = pd.DataFrame({
        "pid": d["pitcher_id"].to_numpy(),
        "season": d["season"].to_numpy(),
        "sh": d["f_same_hand"].to_numpy(),
    })
    d["f_hand_delta"] = key.merge(tbl, on=["pid", "season", "sh"],
                                  how="left")["f_hand_delta"].to_numpy()
    return d


TBL = hand_history_table(df, PRIOR)
tr = attach(add_features(tr_raw, PRIOR), TBL)
va = attach(add_features(va_raw, PRIOR), TBL)
tick(f"준비 완료 — 학습 {len(tr):,} / 검증 {len(va):,}")

# 피처 진단
for nm, d_ in [("학습", tr), ("검증2024", va)]:
    v = d_["f_hand_delta"]
    log(f"  [{nm}] 커버리지 {v.notna().mean()*100:5.1f}%  "
        f"평균 {v.mean():+.4f}  표준편차 {v.std():.4f}  "
        f"p1/p99 {v.quantile(0.01):+.4f}/{v.quantile(0.99):+.4f}")

# 신호 유무 눈대중: 검증에서 delta 십분위 → 실제 성공률 (진단 출력 전용)
v = va["f_hand_delta"]
m = v.notna()
q = pd.qcut(v[m], 10, labels=False, duplicates="drop")
log("\n  delta 십분위별 실제 성공률 (검증 2024, 진단용):")
log(f"    {'십분위':>4s} {'행수':>9s} {'delta평균':>10s} {'실제r':>8s}")
for b in range(int(q.max()) + 1):
    mb = q == b
    idx = v[m][mb].index
    log(f"    {b:4d} {mb.sum():9,d} {v[m][mb].mean():+10.4f} "
        f"{va.loc[idx, 'control_success'].mean():8.4f}")
log("")


# =====================================================================
# 본 실험 — 기준 vs +f_hand_delta
# =====================================================================
def run(name, feats):
    enc = fit_encoder(tr, feats)
    Xtr, Xva = to_matrix(tr, feats, enc), to_matrix(va, feats, enc)
    acc = np.zeros(len(va))
    for i, s in enumerate(SEEDS, 1):
        tick(f"{name} seed={s} 학습 중 ({i}/{len(SEEDS)})...")
        m = HistGradientBoostingClassifier(**{**HGB_FAST, "random_state": s}).fit(Xtr, y_tr)
        acc += m.predict_proba(Xva)[:, 1]
    z = decompose(acc / len(SEEDS), y_va)
    log(f"  {name:24s} 총점 {z['총점']:7.1f}  변별력 {z['변별력']:7.1f}  "
        f"벌점 {z['중심벌점']:5.1f}")
    return z


log("=" * 88)
log("기준(65피처) vs +f_hand_delta(66피처) — 4시드 앙상블")
log("=" * 88)
z_base = run("기준", FEATS)
z_hand = run("+f_hand_delta", FEATS + ["f_hand_delta"])

gain = z_hand["총점"] - z_base["총점"]
gain_d = z_hand["변별력"] - z_base["변별력"]
log("\n" + "=" * 88)
log("판정")
log("=" * 88)
log(f"  총점 이득   {gain:+8.1f}   (임계 {THR:.0f})")
log(f"  변별력 이득 {gain_d:+8.1f}")
if gain >= THR:
    log(f"\n  [채택] f_hand_delta를 제출 피처에 추가할 것")
    log(f"     다음 후보: 타자 쪽 대칭 피처(타자별 같은손 투수 상대 delta)")
elif gain <= -THR:
    log(f"\n  [기각] 유해 — 추가하지 말 것")
else:
    log(f"\n  [보류] 노이즈 범위 — 좌우 이력 delta로는 신호가 안 잡힌다")
    log(f"     압박 상황 delta(더 약한 가설)도 같이 접는 것을 권함")
log(f"  참고: 1위 {LB_TOP:.1f} / 현재 제출 리더보드 830.32")
log("=" * 88)
