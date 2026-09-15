"""
train.csv 전체 현황 통계 → JSON 덤프 (시각화 대시보드용)
실행: PYTHONIOENCODING=utf-8 py -3.12 -u exp/24_eda_stats.py
출력: lab/eda_stats.json
"""
import json, os, time
import numpy as np
import pandas as pd

T0 = time.time()
def log(*a): print(*a, flush=True)
def tick(m): log(f"  [{time.time()-T0:6.1f}s] {m}")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "lab", "eda_stats.json")

df = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
tick(f"train loaded {df.shape}")
y = df["control_success"].to_numpy()

R = {}

# ---------------------------------------------------------------- 1. meta
R["meta"] = {
    "rows": int(len(df)),
    "cols": int(df.shape[1]),
    "target_mean": float(y.mean()),
    "n_pos": int(y.sum()),
    "n_neg": int((1 - y).sum()),
    "seasons": sorted(int(s) for s in df["season"].unique()),
    "n_pitcher": int(df["pitcher_id"].nunique()),
    "n_batter": int(df["batter_id"].nunique()),
    "n_pitcher_team": int(df["pitcher_team_id"].nunique()),
    "mem_mb": float(df.memory_usage(deep=True).sum() / 1e6),
    "baseline_brier": float(y.mean() * (1 - y.mean())),
}

# ---------------------------------------------------------------- 2. 컬럼 프로파일
GROUPS = {
    "row_id": "id", "season": "game", "game_month": "game", "game_dayofweek": "game",
    "inning": "game", "top_bottom": "game", "game_type": "game",
    "balls_before": "count", "strikes_before": "count", "outs_before": "count",
    "run_top_before": "count", "run_bot_before": "count", "run_total_before": "count",
    "score_diff_home": "count", "score_diff_pitcher_team": "count",
    "runner_on_1b": "runner", "runner_on_2b": "runner", "runner_on_3b": "runner",
    "num_runners_on": "runner", "base_state": "runner",
    "home_win_expectancy": "runner", "away_win_expectancy": "runner", "li": "runner",
    "pitcher_id": "player", "batter_id": "player", "pitcher_hand": "player",
    "batter_hand": "player", "pitcher_team_id": "player", "batter_team_id": "player",
    "control_success": "target",
}
def grp(c):
    if c in GROUPS: return GROUPS[c]
    if c.startswith("asof_pitcher_prev"): return "asof_form"
    if "pitchmix" in c or c.endswith(("fastball_rate", "breaking_rate", "offspeed_rate")): return "asof_mix"
    if c.startswith("asof_batter"): return "asof_batter"
    if c.startswith("asof_pitcher"): return "asof_pitcher"
    return "other"

cols = []
for c in df.columns:
    s = df[c]
    miss = float(s.isna().mean())
    rec = {"name": c, "group": grp(c), "dtype": str(s.dtype),
           "missing": miss, "nunique": int(s.nunique(dropna=True))}
    if pd.api.types.is_numeric_dtype(s) and c != "row_id":
        v = s.dropna()
        rec.update({"min": float(v.min()), "max": float(v.max()),
                    "mean": float(v.mean()), "std": float(v.std()),
                    "p50": float(v.median())})
        if c != "control_success":
            m = s.notna().to_numpy()
            if m.sum() > 1000:
                rec["corr"] = float(np.corrcoef(s.to_numpy()[m], y[m])[0, 1])
    cols.append(rec)
R["columns"] = cols
tick("column profile")

# ---------------------------------------------------------------- 3. 시즌 드리프트
g = df.groupby("season")["control_success"]
R["season"] = [{"season": int(k), "n": int(v), "rate": float(m)}
               for k, v, m in zip(g.count().index, g.count().values, g.mean().values)]

# 시즌×월
sm = df.groupby(["season", "game_month"])["control_success"].agg(["count", "mean"]).reset_index()
R["season_month"] = [{"season": int(r.season), "month": int(r.game_month),
                      "n": int(r["count"]), "rate": float(r["mean"])}
                     for _, r in sm.iterrows()]

# 시즌×팀
st = df.groupby(["season", "pitcher_team_id"])["control_success"].agg(["count", "mean"]).reset_index()
R["season_team"] = [{"season": int(r.season), "team": int(r.pitcher_team_id),
                     "n": int(r["count"]), "rate": float(r["mean"])}
                    for _, r in st.iterrows() if r["count"] >= 500]
tick("season drift")

# ---------------------------------------------------------------- 4. 카테고리별 성공률
def breakdown(key, label=None, minn=200, cast=int):
    g = df.groupby(key)["control_success"].agg(["count", "mean"])
    out = []
    for k, r in g.iterrows():
        if r["count"] < minn: continue
        out.append({"key": cast(k) if cast else k, "n": int(r["count"]), "rate": float(r["mean"])})
    return out

R["by"] = {
    "balls": breakdown("balls_before"),
    "strikes": breakdown("strikes_before"),
    "outs": breakdown("outs_before"),
    "inning": breakdown("inning"),
    "month": breakdown("game_month"),
    "dow": breakdown("game_dayofweek"),
    "num_runners": breakdown("num_runners_on"),
    "base_state": breakdown("base_state", cast=str),
    "top_bottom": breakdown("top_bottom", cast=str),
    "game_type": breakdown("game_type", cast=str),
    "pitcher_hand": breakdown("pitcher_hand"),
    "batter_hand": breakdown("batter_hand"),
    "pitcher_team": breakdown("pitcher_team_id"),
}

# 볼카운트 히트맵
cm = df.groupby(["balls_before", "strikes_before"])["control_success"].agg(["count", "mean"]).reset_index()
R["count_matrix"] = [{"b": int(r.balls_before), "s": int(r.strikes_before),
                      "n": int(r["count"]), "rate": float(r["mean"])}
                     for _, r in cm.iterrows()]

# 손 매치업
hm = df.groupby(["pitcher_hand", "batter_hand"])["control_success"].agg(["count", "mean"]).reset_index()
R["hand_matrix"] = [{"p": int(r.pitcher_hand), "b": int(r.batter_hand),
                     "n": int(r["count"]), "rate": float(r["mean"])}
                    for _, r in hm.iterrows()]
tick("breakdowns")

# ---------------------------------------------------------------- 5. 수치형 구간별 성공률 (신호 곡선)
NUMS = [c for c in df.columns
        if c.startswith("asof_") or c in ("li", "home_win_expectancy", "score_diff_pitcher_team",
                                          "run_total_before", "inning")]
curves = {}
for c in NUMS:
    s = df[c]
    m = s.notna().to_numpy()
    if m.sum() < 5000: continue
    v = s.to_numpy()[m]; yy = y[m]
    try:
        q = pd.qcut(v, 12, duplicates="drop", labels=False)
    except Exception:
        continue
    tmp = pd.DataFrame({"q": q, "v": v, "y": yy}).groupby("q").agg(
        n=("y", "size"), rate=("y", "mean"), lo=("v", "min"), hi=("v", "max"), mid=("v", "median"))
    curves[c] = {
        "missing": float(1 - m.mean()),
        "bins": [{"n": int(r.n), "rate": float(r.rate), "lo": float(r.lo),
                  "hi": float(r.hi), "mid": float(r.mid)} for _, r in tmp.iterrows()]
    }
R["curves"] = curves
tick("signal curves")

# ---------------------------------------------------------------- 6. cold-start / 결측 구조
n_pn = df["asof_pitcher_n"].fillna(0).to_numpy()
n_bn = df["asof_batter_n"].fillna(0).to_numpy()
def bucket(n):
    edges = [0, 1, 10, 50, 200, 1000, 5000, 1e18]
    labels = ["0", "1-9", "10-49", "50-199", "200-999", "1k-5k", "5k+"]
    idx = np.digitize(n, edges[1:-1], right=False)
    return np.array(labels)[idx]
for key, arr in (("pitcher_n", n_pn), ("batter_n", n_bn)):
    b = bucket(arr)
    t = pd.DataFrame({"b": b, "y": y}).groupby("b").agg(n=("y", "size"), rate=("y", "mean"))
    order = ["0", "1-9", "10-49", "50-199", "200-999", "1k-5k", "5k+"]
    R.setdefault("coldstart", {})[key] = [
        {"bucket": o, "n": int(t.loc[o, "n"]), "rate": float(t.loc[o, "rate"])}
        for o in order if o in t.index]

# 결측 동시발생 (asof rate 컬럼들)
rate_cols = [c for c in df.columns if c.startswith("asof_") and c.endswith("_rate")]
miss_any = df[rate_cols].isna().any(axis=1)
R["missing_summary"] = {
    "rate_cols": len(rate_cols),
    "any_missing_frac": float(miss_any.mean()),
    "rate_when_missing": float(y[miss_any.to_numpy()].mean()) if miss_any.any() else None,
    "rate_when_full": float(y[~miss_any.to_numpy()].mean()),
    "per_col": sorted([{"name": c, "missing": float(df[c].isna().mean())} for c in rate_cols],
                      key=lambda d: -d["missing"]),
}
tick("coldstart")

# ---------------------------------------------------------------- 7. 선수 규모 분포
pv = df.groupby("pitcher_id").agg(n=("control_success", "size"), rate=("control_success", "mean"))
bv = df.groupby("batter_id").agg(n=("control_success", "size"), rate=("control_success", "mean"))
def dist(t, name):
    n = t["n"].to_numpy()
    order = np.sort(n)[::-1]
    cum = np.cumsum(order) / order.sum()
    return {
        "n_entities": int(len(t)),
        "quantiles": {q: float(np.quantile(n, q / 100)) for q in [10, 25, 50, 75, 90, 99]},
        "max": int(n.max()),
        "lorenz": [{"top_frac": float((i + 1) / len(order)), "vol_frac": float(cum[i])}
                   for i in np.unique(np.linspace(0, len(order) - 1, 60).astype(int))],
        # 표본 500+ 인 선수들의 성공률 분포 (개인차 크기)
        "rate_hist": np.histogram(t.loc[t.n >= 500, "rate"], bins=30, range=(0.3, 0.75))[0].tolist(),
        "rate_hist_range": [0.3, 0.75],
        "rate_std": float(t.loc[t.n >= 500, "rate"].std()),
        "n_qualified": int((t.n >= 500).sum()),
    }
R["entities"] = {"pitcher": dist(pv, "pitcher"), "batter": dist(bv, "batter")}
tick("entity dist")

# ---------------------------------------------------------------- 8. trackman 요약
try:
    tm = pd.read_csv(os.path.join(ROOT, "data", "trackman_history.csv"),
                     usecols=["season", "pitch_type_group", "rel_speed", "spin_rate",
                              "induced_vert_break", "horz_break", "rel_height", "rel_side",
                              "extension", "pitcher_trackman_id", "pitcher_team"])
    tmr = {
        "rows": int(len(tm)),
        "seasons": sorted(int(s) for s in tm["season"].dropna().unique()),
        "n_pitcher": int(tm["pitcher_trackman_id"].nunique()),
        "by_season": [{"season": int(k), "n": int(v)} for k, v in tm["season"].value_counts().sort_index().items()],
        "pitch_mix": [{"group": str(k), "n": int(v)} for k, v in tm["pitch_type_group"].value_counts().items()],
        "speed_by_group": [],
        "missing": {c: float(tm[c].isna().mean()) for c in tm.columns},
    }
    for gname, sub in tm.groupby("pitch_type_group"):
        if len(sub) < 5000: continue
        tmr["speed_by_group"].append({
            "group": str(gname), "n": int(len(sub)),
            "speed_mean": float(sub.rel_speed.mean()), "speed_std": float(sub.rel_speed.std()),
            "spin_mean": float(sub.spin_rate.mean()),
            "ivb_mean": float(sub.induced_vert_break.mean()),
            "hb_mean": float(sub.horz_break.mean()),
        })
    R["trackman"] = tmr
    del tm
except Exception as e:
    R["trackman"] = {"error": str(e)}
tick("trackman")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False)
tick(f"saved {OUT}  ({os.path.getsize(OUT)/1e3:.0f} KB)")
