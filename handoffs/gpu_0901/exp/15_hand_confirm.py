"""
[15] hand delta 확증 실험 — 독립 시드 8개 재측정 + 타자 쪽 대칭 피처 확장

실행:  PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 py -3.12 -u exp/15_hand_confirm.py

배경:
  exp/14(시드 42,7,123,2024)에서 f_hand_delta가 총점 +13.9 / 변별력 +21.0.
  임계 15에 1.1 모자라는 회색지대 — 선택에 쓴 시드로 단정하면 승자의 저주 위험.
  → 선택에 쓰지 않은 새 시드 8개로 독립 재측정한다 (XGB 검증 때와 같은 절차).
  겸사겸사 타자 쪽 대칭 피처(f_bhand_delta)도 한 팔 추가해서 한 번에 결판.

판정 기준:
  8시드 앙상블 차이의 노이즈: σ_diff ≈ √2 × 15.3/√8 ≈ 7.7 → 2σ ≈ 15
  단, exp/14의 +13.9가 독립 재현되면 두 측정을 합쳐 판단한다
  (독립 두 번 다 +14 근처면 우연일 확률은 각각 회색지대라도 결합 시 유의미).

피처 (exp/14와 동일한 시즌 단위 as-of, 검증된 cumsum-현재시즌 로직 재사용):
  f_hand_delta  = 투수별 [해당 손 타자 상대 과거 성공률 − 본인 통산 과거 성공률]
  f_bhand_delta = 타자별 [해당 손 투수 상대 과거 성공률 − 본인 통산 과거 성공률]
                  (타자 성공률 = 그 타자가 상대한 투구의 제구 성공률, asof_batter_*와 동일 관점)
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "exp")
from common import (log, tick, decompose, load_train, add_features,
                    fit_encoder, to_matrix, ALL_ENG, HGB_FAST, LB_TOP)

SEEDS = [11, 77, 314, 999, 1234, 5678, 20250823, 424242]   # exp/14와 겹치지 않는 새 시드
THR = 15.0
PREV_GAIN = 13.9   # exp/14 (선택 시드) 총점 이득
A1 = 200.0
A2 = 200.0

log("train.csv 로딩 중...")
df = load_train()
BASE = [c for c in df.columns if c not in ("row_id", "control_success")]
FEATS = BASE + ALL_ENG

tr_raw = df[df["season"] <= 2023].reset_index(drop=True)
va_raw = df[df["season"] == 2024].reset_index(drop=True)
y_tr = tr_raw["control_success"].to_numpy()
y_va = va_raw["control_success"].to_numpy()
PRIOR = float(y_tr.mean())


def hand_history_table(src, prior, id_col):
    """exp/14에서 토이 실행으로 누수 없음이 검증된 로직. id_col만 일반화."""
    t = pd.DataFrame({
        "eid": src[id_col].to_numpy(),
        "season": src["season"].to_numpy(),
        "sh": (src["pitcher_hand"] == src["batter_hand"]).astype("int8").to_numpy(),
        "y": src["control_success"].to_numpy(),
    })
    cond = (t.groupby(["eid", "season", "sh"])["y"]
             .agg(succ="sum", n="size").reset_index()
             .sort_values(["eid", "sh", "season"]))
    g = cond.groupby(["eid", "sh"])
    cond["pc_succ"] = g["succ"].cumsum() - cond["succ"]
    cond["pc_n"] = g["n"].cumsum() - cond["n"]

    over = (t.groupby(["eid", "season"])["y"]
             .agg(succ="sum", n="size").reset_index()
             .sort_values(["eid", "season"]))
    go = over.groupby("eid")
    over["po_succ"] = go["succ"].cumsum() - over["succ"]
    over["po_n"] = go["n"].cumsum() - over["n"]

    tbl = cond.merge(over[["eid", "season", "po_succ", "po_n"]],
                     on=["eid", "season"], how="left")
    p_over = (tbl["po_succ"] + prior * A1) / (tbl["po_n"] + A1)
    p_cond = (tbl["pc_succ"] + p_over * A2) / (tbl["pc_n"] + A2)
    tbl["delta"] = np.where(tbl["po_n"] > 0,
                            (p_cond - p_over), np.nan).astype("float32")
    return tbl[["eid", "season", "sh", "delta"]]


def attach(frame, tbl, id_col, feat_name):
    d = frame.copy()
    key = pd.DataFrame({
        "eid": d[id_col].to_numpy(),
        "season": d["season"].to_numpy(),
        "sh": d["f_same_hand"].to_numpy(),
    })
    d[feat_name] = key.merge(tbl, on=["eid", "season", "sh"],
                             how="left")["delta"].to_numpy()
    return d


TBL_P = hand_history_table(df, PRIOR, "pitcher_id")
TBL_B = hand_history_table(df, PRIOR, "batter_id")
tr = attach(add_features(tr_raw, PRIOR), TBL_P, "pitcher_id", "f_hand_delta")
tr = attach(tr, TBL_B, "batter_id", "f_bhand_delta")
va = attach(add_features(va_raw, PRIOR), TBL_P, "pitcher_id", "f_hand_delta")
va = attach(va, TBL_B, "batter_id", "f_bhand_delta")
tick(f"준비 완료 — 학습 {len(tr):,} / 검증 {len(va):,}")

for feat in ["f_hand_delta", "f_bhand_delta"]:
    for nm, d_ in [("학습", tr), ("검증", va)]:
        v = d_[feat]
        log(f"  [{feat} {nm}] 커버리지 {v.notna().mean()*100:5.1f}%  "
            f"표준편차 {v.std():.4f}  p1/p99 {v.quantile(0.01):+.4f}/{v.quantile(0.99):+.4f}")
log("")


def run(name, feats):
    enc = fit_encoder(tr, feats)
    Xtr, Xva = to_matrix(tr, feats, enc), to_matrix(va, feats, enc)
    acc = np.zeros(len(va))
    for i, s in enumerate(SEEDS, 1):
        tick(f"{name} seed={s} ({i}/{len(SEEDS)})...")
        m = HistGradientBoostingClassifier(**{**HGB_FAST, "random_state": s}).fit(Xtr, y_tr)
        acc += m.predict_proba(Xva)[:, 1]
    z = decompose(acc / len(SEEDS), y_va)
    log(f"  {name:26s} 총점 {z['총점']:7.1f}  변별력 {z['변별력']:7.1f}  "
        f"벌점 {z['중심벌점']:5.1f}")
    return z


log("=" * 88)
log("새 시드 8개 — 기준 / +투수 delta / +투수+타자 delta")
log("=" * 88)
z0 = run("기준", FEATS)
z1 = run("+투수 hand delta", FEATS + ["f_hand_delta"])
z2 = run("+투수+타자 hand delta", FEATS + ["f_hand_delta", "f_bhand_delta"])

g1, g2 = z1["총점"] - z0["총점"], z2["총점"] - z0["총점"]
g1d, g2d = z1["변별력"] - z0["변별력"], z2["변별력"] - z0["변별력"]
log("\n" + "=" * 88)
log("판정")
log("=" * 88)
log(f"  +투수 delta        총점 {g1:+7.1f}  변별력 {g1d:+7.1f}   (exp/14 선택시드: +13.9/+21.0)")
log(f"  +투수+타자 delta   총점 {g2:+7.1f}  변별력 {g2d:+7.1f}")
log(f"  타자 delta 한계기여 총점 {g2-g1:+7.1f}  변별력 {g2d-g1d:+7.1f}")
log(f"  임계 {THR:.0f} (8시드 diff 2σ≈15)\n")

mean_p = (g1 + PREV_GAIN) / 2
log(f"  투수 delta 독립 2회 측정: exp/14 {PREV_GAIN:+.1f} / 이번 {g1:+.1f} → 평균 {mean_p:+.1f}")
if g1 >= THR or (g1 >= THR * 0.6 and mean_p >= THR * 0.8):
    log(f"  [채택 후보] 두 독립 측정이 일관되게 양수 — 재현되는 실제 개선으로 판단")
elif g1 <= 0:
    log(f"  [기각] 새 시드에서 사라짐 — exp/14의 +13.9는 시드 운이었다")
else:
    log(f"  [회색] 판단 유보 — 추가 증거 필요")
if g2 - g1 >= THR:
    log(f"  타자 delta: 유의미한 추가 기여 — 같이 채택 후보")
elif g2 - g1 <= -THR:
    log(f"  타자 delta: 유해 — 제외")
else:
    log(f"  타자 delta: 한계기여 노이즈 범위")
log(f"  참고: 1위 {LB_TOP:.1f} / 현재 제출 리더보드 830.32")
log("=" * 88)
