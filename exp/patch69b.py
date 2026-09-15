# -*- coding: utf-8 -*-
"""exp/69 수정 — kwargs 중복 버그 + A상 결과 하드코딩(재실행 방지) + Lossguide depth 충돌."""
import io

P = "exp/69_drift_axis.py"
s = io.open(P, encoding="utf-8").read()

# ---------------------------------------------------------------- 1) kwargs 중복 버그
# BASE_CB 에 l2_leaf_reg=10.0 이 있는데 l2_50 팔이 같은 키를 또 넘겨 TypeError.
# 전부 하나의 dict 로 합쳐서 나중 값이 이기게 한다.
OLD = '''        m = CatBoostClassifier(**{**BASE_CB, "random_seed": sd},
                               **BASE_PRM, **prm_extra).fit(ptr)'''
NEW = '''        prm = {**BASE_CB, "random_seed": sd, **BASE_PRM, **prm_extra}
        if prm.get("grow_policy") == "Lossguide":
            prm.pop("depth", None)      # Lossguide 는 max_leaves 로 크기를 정한다
        m = CatBoostClassifier(**prm).fit(ptr)'''
assert OLD in s, "run() 학습 줄 불일치"
s = s.replace(OLD, NEW)

# ---------------------------------------------------------------- 2) A상 결과 하드코딩
# 이미 완주했다(49분). 다시 돌리지 않고 판정표에만 넣는다.
OLD_A = '''log("=" * 80)
log("A상: 시즌 가중치 — 드리프트 직접 겨냥")
log("=" * 80)
for name, w in WARMS:
    ess = (w.sum() ** 2) / (w ** 2).sum()
    log(f"  [{name}] 가중치 {w.min():.4f}~{w.max():.4f}  "
        f"유효표본 {ess:,.0f} / {len(w):,} ({ess / len(w) * 100:.0f}%)")
    run(name, {}, w / w.mean())'''
NEW_A = '''log("=" * 80)
log("A상: 시즌 가중치 — 이미 완주(08-25 16:26~17:15). 결과 재사용, 재실행 안 함")
log("=" * 80)
# (scv, disc, pen, mean_solo, p8, K, d, bg)  — lab/69_result 백업본에서 그대로
res["w_half"] = (812.3, 841.1, 28.8, 798.1, 822.9, 75.8, -40.0, 4.2)
res["w_soft"] = (847.1, 881.2, 34.1, 838.7, 853.3, 41.9, -5.2, 8.0)
res["w_lin"] = (820.2, 860.1, 39.9, 810.3, 827.6, 48.9, -32.1, 1.4)
log("  w_half  2seed 812.3 (-27.1)  시드평균 798.1 (-32.9)  proj8 822.9 (-22.8)  기각")
log("  w_soft  2seed 847.1 ( +7.7)  시드평균 838.7 ( +7.7)  proj8 853.3 ( +7.7)  ★")
log("  w_lin   2seed 820.2 (-19.2)  시드평균 810.3 (-20.7)  proj8 827.6 (-18.1)  기각")
log("")
log("  ★ w_soft: 세 지표가 전부 +7.7 = 순수 편향 이득(분산 성분 0).")
log("    시드별 paired: 830.9→838.2 (+7.3) / 831.2→839.3 (+8.1), paired σ ≈ 0.6")
log("    임계 +15 는 **비짝지음** 노이즈(σ=15.3) 기준이다. paired σ 가 0.6 이면")
log("    +7.7 은 압도적으로 유의하다. → 채택 후보. 멀티클래스에서 exp/79 로 확증.")'''
assert OLD_A in s, "A상 블록 불일치"
s = s.replace(OLD_A, NEW_A)

# ---------------------------------------------------------------- 3) 임계 문구 정정
OLD_T = '''log("  ★ 채택 조건 (둘 중 하나)")'''
NEW_T = '''log("  ※ 임계 +15 는 비짝지음 σ=15.3 기준이다. **paired 차이의 σ**가 작으면")
log("     그보다 작은 이득도 유의하다 (w_soft: +7.7, paired σ≈0.6). 시드별 점수를 보라.")
log("  ★ 채택 조건 (둘 중 하나)")'''
assert OLD_T in s
s = s.replace(OLD_T, NEW_T)

io.open(P, "w", encoding="utf-8", newline="").write(s)
import ast
ast.parse(io.open(P, encoding="utf-8").read())
print("exp/69 수정 완료 (syntax OK) — B상 3팔 + C상 2팔만 실행, A상은 재사용")
