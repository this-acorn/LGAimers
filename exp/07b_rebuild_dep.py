"""
[07b] model.pkl 배포 테이블 재생성 — 감사 [중간-1] 수정 반영

실행 (★ venv311로):
  PYTHONIOENCODING=utf-8 <venv311>/Scripts/python.exe -u exp/07b_rebuild_dep.py

배경: exp/07이 백그라운드로 도는 중에 deployment_table의 엣지 수정
  (train에 있는 선수의 미경험 sh → NaN이 아니라 delta=0, as-of 수식과 일치)이
  파일에 반영됐다. 돌던 프로세스는 수정 전 코드를 메모리에 들고 있었으므로,
  저장된 model.pkl의 테이블만 수정판으로 교체한다. 모델 가중치는 그대로다.

안전장치: 기존 테이블과 공통 키에서 delta가 완전 일치해야만 저장한다
  (수정은 '없던 키 추가'뿐이어야 하므로).
※ deployment_table 함수는 exp/07의 수정판과 동일해야 한다 (일회성 유지보수 스크립트).
"""

import numpy as np
import pandas as pd
import joblib

A1 = 200.0
A2 = 200.0


def deployment_table(src, prior, id_col):
    t = pd.DataFrame({
        "eid": src[id_col].to_numpy(),
        "sh": (src["pitcher_hand"] == src["batter_hand"]).astype("int8").to_numpy(),
        "y": src["control_success"].to_numpy()})
    cond = t.groupby(["eid", "sh"])["y"].agg(succ="sum", n="size").reset_index()
    over = (t.groupby("eid")["y"].agg(po_succ="sum", po_n="size").reset_index())
    grid = pd.MultiIndex.from_product(
        [over["eid"].to_numpy(), np.array([0, 1], dtype="int8")],
        names=["eid", "sh"]).to_frame(index=False)
    tbl = grid.merge(cond, on=["eid", "sh"], how="left")
    tbl[["succ", "n"]] = tbl[["succ", "n"]].fillna(0.0)
    tbl = tbl.merge(over, on="eid", how="left")
    p_over = (tbl["po_succ"] + prior * A1) / (tbl["po_n"] + A1)
    p_cond = (tbl["succ"] + p_over * A2) / (tbl["n"] + A2)
    tbl["delta"] = (p_cond - p_over).astype("float32")
    return tbl[["eid", "sh", "delta"]]


df = pd.read_csv("data/train.csv", encoding="utf-8-sig",
                 usecols=["pitcher_id", "batter_id", "pitcher_hand",
                          "batter_hand", "control_success"])
df.columns = [c.replace("﻿", "").strip() for c in df.columns]

b = joblib.load("submit/model/model.pkl")
prior = b["prior"]
print(f"로드 완료 — prior={prior:.6f}, 기존 테이블 "
      f"투수 {len(b['hand_tbl_p'])}행 / 타자 {len(b['hand_tbl_b'])}행")

dep_p = deployment_table(df, prior, "pitcher_id")
dep_b = deployment_table(df, prior, "batter_id")

for old, new, nm in [(b["hand_tbl_p"], dep_p, "투수"),
                     (b["hand_tbl_b"], dep_b, "타자")]:
    mg = old.merge(new, on=["eid", "sh"], suffixes=("_o", "_n"))
    mx = float(np.abs(mg["delta_o"] - mg["delta_n"]).max()) if len(mg) else 0.0
    added = len(new) - len(mg)
    print(f"  [{nm}] 공통 키 {len(mg)} (delta 최대차 {mx:.2e}) / "
          f"추가된 키 {added} (미경험 sh → delta=0)")
    assert mx < 1e-6, "공통 키 delta 불일치 — 수정 로직 점검 필요"
    assert len(mg) == len(old), "기존 키가 사라짐 — 수정 로직 점검 필요"

b["hand_tbl_p"], b["hand_tbl_b"] = dep_p, dep_b
joblib.dump(b, "submit/model/model.pkl", compress=3)
print("재저장 완료: submit/model/model.pkl")
