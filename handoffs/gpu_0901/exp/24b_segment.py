"""game_type 세그먼트 분해 → eda_stats.json 에 추가"""
import json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "lab", "eda_stats.json")
R = json.load(open(P, encoding="utf-8"))

df = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                 usecols=["season", "game_type", "pitcher_team_id", "batter_team_id",
                          "control_success", "pitcher_id", "asof_pitcher_n"])

seg = []
for (gt, s), sub in df.groupby(["game_type", "season"]):
    seg.append({"type": str(gt), "season": int(s), "n": int(len(sub)),
                "rate": float(sub.control_success.mean())})
R["segment_season"] = seg

# 드리프트 분해: 2019→2024 전체 변화 중 각 세그먼트 기여
tot = {}
for s, sub in df.groupby("season"):
    tot[int(s)] = {"n": int(len(sub)), "rate": float(sub.control_success.mean())}
first, last = min(tot), max(tot)
decomp = []
for gt in sorted(df.game_type.unique()):
    a = df[(df.game_type == gt) & (df.season == first)]
    b = df[(df.game_type == gt) & (df.season == last)]
    w = len(b) / tot[last]["n"]
    decomp.append({"type": str(gt),
                   "rate_first": float(a.control_success.mean()),
                   "rate_last": float(b.control_success.mean()),
                   "share_last": float(w),
                   "contrib": float(w * (b.control_success.mean() - a.control_success.mean()))})
R["drift_decomp"] = {"first": first, "last": last,
                     "total_change": tot[last]["rate"] - tot[first]["rate"],
                     "parts": decomp}

# 팀 x game_type
tt = df.groupby(["game_type", "pitcher_team_id"]).size().reset_index(name="n")
R["type_team"] = [{"type": str(r.game_type), "team": int(r.pitcher_team_id), "n": int(r.n)}
                  for _, r in tt.iterrows()]

# 투수 겹침
pf = set(df[df.game_type == "F"].pitcher_id); pr = set(df[df.game_type == "R"].pitcher_id)
R["type_overlap"] = {"F_only": len(pf - pr), "R_only": len(pr - pf), "both": len(pf & pr),
                     "F_asof_n_median": float(df[df.game_type == "F"].asof_pitcher_n.median()),
                     "R_asof_n_median": float(df[df.game_type == "R"].asof_pitcher_n.median())}

json.dump(R, open(P, "w", encoding="utf-8"), ensure_ascii=False)
print("total change", R["drift_decomp"]["total_change"])
for p in decomp: print(p)
print(R["type_overlap"])
