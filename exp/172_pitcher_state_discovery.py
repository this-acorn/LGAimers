# -*- coding: utf-8 -*-
"""Discovery-only test of a persistent pitcher residual state.

Learns a game-type-centered pitcher residual from 2022 current-anchor OOF and
applies it to 2023.  This is the cheapest strict-forward proxy for the still
unresolved pitcher AR(1) idea.  No 2024 target/result is evaluated.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "lab" / "172_pitcher_state_discovery.json"
OUT_TXT = ROOT / "lab" / "172_pitcher_state_discovery.txt"
ALPHAS = (100.0, 300.0, 600.0, 1200.0)
SCALES = (0.25, 0.50, 0.75, 1.00)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def score(p, y):
    r = float(np.mean(y))
    return 100000.0 * (1.0 - float(np.mean((p-y)**2))/(r*(1-r)))


def gain(base, cand, y, mask):
    return score(cand[mask], y[mask]) - score(base[mask], y[mask])


def effect_for_alpha(src, resid, valid, alpha):
    work = src[["pitcher_id", "game_type"]].copy()
    work["r"] = resid
    # Strip source-year global calibration so this is only relative pitcher state.
    work["r"] -= work.groupby("game_type", observed=True)["r"].transform("mean")
    tab = work.groupby(["pitcher_id", "game_type"], observed=True, sort=False)["r"].agg(["sum", "size"]).reset_index()
    tab["e"] = tab["sum"] / (tab["size"] + alpha)
    left = valid[["pitcher_id", "game_type"]].copy()
    left["_i"] = np.arange(len(left))
    got = left.merge(tab[["pitcher_id", "game_type", "e", "size"]],
                     on=["pitcher_id", "game_type"], how="left", sort=False, validate="m:1").sort_values("_i")
    return got["e"].fillna(0.0).to_numpy(float), float(got["size"].notna().mean())


def bootstrap(base, cand, y, pitcher, draws=5000):
    row = (base-y)**2 - (cand-y)**2
    g = pd.DataFrame({"p":pitcher, "s":row, "n":1}).groupby("p", sort=False).agg({"s":"sum","n":"sum"})
    s, n = g["s"].to_numpy(), g["n"].to_numpy()
    rng = np.random.default_rng(172)
    vals = np.empty(draws)
    for i in range(draws):
        k = rng.integers(0, len(g), len(g))
        vals[i] = s[k].sum()/n[k].sum()
    vals *= 100000.0/(float(y.mean())*(1-float(y.mean())))
    return {"clusters":int(len(g)), "p025":float(np.quantile(vals,.025)),
            "median":float(np.median(vals)), "p975":float(np.quantile(vals,.975)),
            "prob_positive":float(np.mean(vals>0))}


def main():
    e158 = load_module(ROOT/"exp"/"158_robust_conditional_tensor_eb.py", "e158_172")
    e155 = load_module(ROOT/"exp"/"155_cause_runner_currentblend_gate.py", "e155_172")
    cols = ["season","game_month","game_type","pitcher_id","control_success"]
    f = pd.read_csv(ROOT/"data"/"train.csv", usecols=cols, encoding="utf-8-sig", low_memory=False)
    f = f.loc[f["season"].isin([2022,2023])].reset_index(drop=True)
    f["game_type"] = f["game_type"].astype(str)
    rows = {z:f.loc[f.season.eq(z)].reset_index(drop=True) for z in (2022,2023)}
    y = {z:rows[z].control_success.to_numpy(float) for z in rows}
    base = {z:e158.load_current_baseline(e155, rows[z], z) for z in rows}
    resid = y[2022]-base[2022]
    gt = rows[2023].game_type.to_numpy()
    mo = rows[2023].game_month.to_numpy()
    allmask = np.ones(len(rows[2023]), dtype=bool)
    results=[]
    effects={}
    for alpha in ALPHAS:
        effect, coverage = effect_for_alpha(rows[2022], resid, rows[2023], alpha)
        effects[alpha]=effect
        for scale in SCALES:
            cand=np.clip(base[2023]+scale*effect,0,1)
            results.append({"alpha":alpha,"scale":scale,"coverage":coverage,
                "gain":gain(base[2023],cand,y[2023],allmask),
                "F":gain(base[2023],cand,y[2023],gt=="F"),
                "R":gain(base[2023],cand,y[2023],gt=="R"),
                "early":gain(base[2023],cand,y[2023],mo<=6),
                "late":gain(base[2023],cand,y[2023],mo>6),
                "effect_std":float((scale*effect).std())})
    best=max(results,key=lambda z:z["gain"])
    cand=np.clip(base[2023]+best["scale"]*effects[best["alpha"]],0,1)
    boot=bootstrap(base[2023],cand,y[2023],rows[2023].pitcher_id.to_numpy())
    passed=bool(best["gain"]>=8 and best["F"]>0 and best["R"]>0 and best["early"]>0
                and best["late"]>0 and boot["p025"]>0)
    payload={"experiment":172,"discovery":"2022 current-anchor residual -> 2023",
             "reads_2024":False,"results":results,"best":best,"bootstrap":boot,
             "gate_pass":passed,"status":"PASS_MAY_OPEN_2024" if passed else "FAIL_DO_NOT_OPEN_2024"}
    OUT_JSON.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["EXP172 centered pitcher season-state discovery (2022 -> 2023)",
           f"best={best}",f"bootstrap={boot}",f"FINAL: {payload['status']}"]
    OUT_TXT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines),flush=True)


if __name__ == "__main__":
    main()
