# -*- coding: utf-8 -*-
"""[109] 우리가 '피처로는 죽였던' 신호 5개를 V18형 보호 잔차로 재검사 — exp/107 프로토콜 그대로 (Claude 소유)

기존 correction(.30×(pitcher_bhand+pressure_bhand))은 고정. 아래 5개 축을 사전 선언(키·parent·k·빈 경계 고정,
탐색 없음)하고 한 번에 하나씩만 얹어 2021~2023 leave-one-year-out(HGB OOF) → Holm α=.05 → 생존 1개만
exact CAT5(2022/2023/2024)로 확인한다. 학습·test·zip·제출 없음. exp/107 과의 차이는 AXES 정의뿐.

축 (전부 우리 장부에서 '피처 투입'으로 사망했거나 미검사인 신호):
  team_match    (game_type, p팀, b팀)            parent=root       k=700   ← exp/93 팔A cat 피처 −20.2 / exp/92 진단 +42
  rookie_srate  (game_type, 신인여부, 성공률빈)    parent=신인그룹    k=200   ← P01 피처 산포 28.8 사망 / 진단 +5~15
  batter_count  (game_type, batter_id, 카운트)    parent=batter     k=280   ← 공개 HIER 미검사 자식
  mid_bin       (game_type, 미들률빈)             parent=root       k=300   ← f_mid_idx 계열 (진단 약양성)
  b_early       (game_type, 타자 n 버킷)          parent=root       k=300   ← f_b_early (18축 P3)
신인여부 = (그 투수의 데이터 내 첫 시즌 == 행의 시즌) & 시즌>=2020 (2019 는 절단이라 제외) — 행 시점 정보만.
빈 경계는 아래 상수로 고정 선언 (데이터 보고 고르지 않음).
실행: PYTHONIOENCODING=utf-8 python -u exp/109_our_residuals.py   (~5분, 학습 0)
"""

from __future__ import annotations

_PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path(str(_PROJECT_ROOT))
EXPERIMENT = 109
DATA = ROOT / "data/train.csv"
OOF_PATH = ROOT / "lab/24b_preds.npz"
OUT_TXT = ROOT / "lab/109_our_residuals.txt"
OUT_JSON = ROOT / "lab/109_our_residuals.json"
DISCOVERY = (2021, 2022, 2023)
CONFIRMATION = (2022, 2023, 2024)
WEIGHTS = np.round(np.arange(0.0, 0.501, 0.05), 2)
BASE_GAMMA = 0.30
BOOTSTRAP_DRAWS = 5000
ALPHA = 0.05

SRATE_EDGES = [0.0, 0.40, 0.44, 0.48, 0.52, 0.56, 1.01]   # asof_pitcher_success_rate 고정 빈
MID_EDGES = [0.0, 0.10, 0.13, 0.16, 0.20, 1.01]           # asof_pitcher_middle_rate 고정 빈
BN_EDGES = [-0.5, 0.5, 30.5, 100.5, 400.5, 10**9]         # asof_batter_n 버킷 {0, 1-30, 31-100, 101-400, 400+}

AXES = {
    "team_match": {"keys": ["game_type", "pitcher_team_id", "batter_team_id"], "parent": "root", "k": 700.0},
    "rookie_srate": {"keys": ["game_type", "_rookie", "_srate_bin"], "parent": "rookie_grp", "k": 200.0},
    "batter_count": {"keys": ["game_type", "batter_id", "_count_state"], "parent": "batter", "k": 280.0},
    "mid_bin": {"keys": ["game_type", "_mid_bin"], "parent": "root", "k": 300.0},
    "b_early": {"keys": ["game_type", "_bn_bucket"], "parent": "root", "k": 300.0},
}


def load_v18_module():
    path = ROOT / "exp/103_v18_residual.py"
    spec = importlib.util.spec_from_file_location("v18_probe_103_ours", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v18 = load_v18_module()


def score(p, y):
    r = float(y.mean())
    return 100000.0 * (1.0 - float(np.mean((p - y) ** 2)) / (r * (1.0 - r)))


def shape_gain(base, cand, y):
    return score(cand - cand.mean() + base.mean(), y) - score(base, y)


def contribution(base, cand, y, mask):
    r = float(y.mean())
    imp = (base - y) ** 2 - (cand - y) ** 2
    return float(100000.0 * imp[mask].sum() / (len(y) * r * (1.0 - r)))


def hierarchy_effects(history, rows):
    root_keys = ["game_type"]
    pitcher_keys = ["game_type", "pitcher_id"]
    batter_keys = ["game_type", "batter_id"]
    rookie_keys = ["game_type", "_rookie"]

    root = v18.grouped(history, root_keys)
    batter = v18.grouped(history, batter_keys)
    rookie_g = v18.grouped(history, rookie_keys)

    root_n = v18.lookup(root, rows, root_keys, "n")
    root_s = v18.lookup(root, rows, root_keys, "success")
    root_rate = (root_s + v18.K_ROOT * v18.ROOT_PRIOR) / (root_n + v18.K_ROOT)
    batter_n = v18.lookup(batter, rows, batter_keys, "n")
    batter_s = v18.lookup(batter, rows, batter_keys, "success")
    batter_rate = (batter_s + 300.0 * root_rate) / (batter_n + 300.0)
    rk_n = v18.lookup(rookie_g, rows, rookie_keys, "n")
    rk_s = v18.lookup(rookie_g, rows, rookie_keys, "success")
    rookie_rate = (rk_s + 300.0 * root_rate) / (rk_n + 300.0)

    parents = {"root": root_rate, "batter": batter_rate, "rookie_grp": rookie_rate}
    effects, diagnostics = {}, {}
    for name, d in AXES.items():
        keys, strength = d["keys"], float(d["k"])
        table = v18.grouped(history, keys)
        n = v18.lookup(table, rows, keys, "n")
        s = v18.lookup(table, rows, keys, "success")
        parent = parents[d["parent"]]
        post = (s + strength * parent) / (n + strength)
        eff = (n / (n + strength)) * (post - parent)
        effects[name] = eff
        diagnostics[name] = {"coverage": float(np.mean(n > 0)), "n_mean": float(n.mean()),
                             "mean": float(eff.mean()), "std": float(eff.std())}
    return effects, diagnostics


def weighted_candidate(base, cur, new, w):
    q0 = np.clip(base + BASE_GAMMA * cur, 0.0, 1.0)
    q1 = np.clip(q0 + float(w) * new, 0.0, 1.0)
    return q0, q1


def paired(base, cur, new, y, w):
    q0, q1 = weighted_candidate(base, cur, new, w)
    return {"raw_gain": float(score(q1, y) - score(q0, y)),
            "shape_gain": float(shape_gain(q0, q1, y)),
            "mean_shift": float(q1.mean() - q0.mean())}


def select_weight(records, axis, years):
    rows = []
    for w in WEIGHTS:
        yearly = {str(yr): paired(records[yr]["hgb"], records[yr]["current"],
                                  records[yr]["axes"][axis], records[yr]["target"], w)["raw_gain"]
                  for yr in years}
        rows.append({"weight": float(w), "yearly_gain": yearly,
                     "min_gain": float(min(yearly.values())),
                     "mean_gain": float(np.mean(list(yearly.values())))})
    return max(rows, key=lambda r: (r["min_gain"], r["mean_gain"], -r["weight"]))


def bootstrap(base, cand, y, pitcher, draws, seed):
    work = pd.DataFrame({"pitcher": pitcher, "n": np.ones(len(y), dtype=np.int32), "y": y,
                         "base_se": (base - y) ** 2, "cand_se": (cand - y) ** 2})
    g = work.groupby("pitcher", sort=False, observed=True).agg(
        n=("n", "sum"), y=("y", "sum"), base_se=("base_se", "sum"), cand_se=("cand_se", "sum"))
    values = g[["n", "y", "base_se", "cand_se"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    gains = np.empty(draws, dtype=float)
    cursor, groups = 0, len(values)
    while cursor < draws:
        size = min(250, draws - cursor)
        idx = rng.integers(0, groups, size=(size, groups))
        sm = values[idx].sum(axis=1)
        n = sm[:, 0]
        rate = sm[:, 1] / n
        gains[cursor:cursor + size] = 100000.0 * (sm[:, 2] - sm[:, 3]) / (n * rate * (1.0 - rate))
        cursor += size
    return {"median": float(np.median(gains)), "p025": float(np.quantile(gains, 0.025)),
            "p975": float(np.quantile(gains, 0.975)), "prob_positive": float(np.mean(gains > 0.0)),
            "p_one_sided": float((1 + np.count_nonzero(gains <= 0.0)) / (draws + 1))}


def exact_seed_bases(year):
    if year == 2024:
        return {42: np.load(ROOT / "lab/89_cat5_probs_seed42.npy").astype(float)[:, 0],
                7: np.load(ROOT / "lab/89_cat5_probs_seed7.npy").astype(float)[:, 0]}
    return {sd: np.load(ROOT / f"lab/104_cat5_y{year}_probs_seed{sd}.npy").astype(float)[:, 0]
            for sd in (42, 7)}


def main():
    t0 = time.time()
    lines = [f"=== exp/{EXPERIMENT} our-signals-as-protected-residuals audit (exp/107 protocol) ===",
             f"{len(AXES)} axes predeclared (keys/parent/k/bin edges fixed); weight grid only."]
    cols = ["season", "game_type", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
            "pitcher_team_id", "batter_team_id", "balls_before", "strikes_before",
            "base_state", "control_success",
            "asof_pitcher_success_rate", "asof_pitcher_middle_rate", "asof_batter_n"]
    frame = pd.read_csv(DATA, encoding="utf-8-sig", usecols=cols, low_memory=False)
    for c in ("game_type", "batter_hand", "base_state"):
        frame[c] = frame[c].astype("string").fillna("__MISSING__").astype(str)
    first_seen = frame.groupby("pitcher_id")["season"].transform("min")
    frame["_rookie"] = ((frame["season"] == first_seen) & (frame["season"] >= 2020)).astype(int)
    sr = pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce")
    frame["_srate_bin"] = pd.cut(sr, SRATE_EDGES, labels=False).fillna(-1).astype(int)
    mr = pd.to_numeric(frame["asof_pitcher_middle_rate"], errors="coerce")
    frame["_mid_bin"] = pd.cut(mr, MID_EDGES, labels=False).fillna(-1).astype(int)
    bn = pd.to_numeric(frame["asof_batter_n"], errors="coerce").fillna(0)
    frame["_bn_bucket"] = pd.cut(bn, BN_EDGES, labels=False).fillna(0).astype(int)
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0).astype(int)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(0).astype(int)
    frame["_count_state"] = balls.astype(str) + "_" + strikes.astype(str)
    frame["_pressure"] = v18.pressure_code(frame)
    oof = np.load(OOF_PATH)

    records = {}
    for year in (2021, 2022, 2023, 2024):
        validation = frame[frame["season"] == year]
        history = frame[frame["season"] < year]
        target = validation["control_success"].to_numpy(float)
        hgb = oof[f"{year}_base65"].astype(float)
        if len(hgb) != len(target) or not np.array_equal(target, oof[f"{year}_y"].astype(float)):
            raise AssertionError(f"OOF alignment failed for {year}")
        current, _ = v18.make_effect(history, validation)
        axes, diagnostics = hierarchy_effects(history, validation)
        records[year] = {"target": target, "hgb": hgb, "current": current, "axes": axes,
                         "pitcher": validation["pitcher_id"].to_numpy(),
                         "game_type": validation["game_type"].to_numpy(),
                         "diagnostics": diagnostics}
        rk_share = float(np.mean(validation["_rookie"].to_numpy()))
        lines.append(f"{year}: rows={len(target):,} rookie_share={rk_share:.3f} effects ready")

    loo = {}
    for axis in AXES:
        folds = {}
        for heldout in DISCOVERY:
            train_years = tuple(y for y in DISCOVERY if y != heldout)
            sel = select_weight(records, axis, train_years)
            ev = paired(records[heldout]["hgb"], records[heldout]["current"],
                        records[heldout]["axes"][axis], records[heldout]["target"], sel["weight"])
            q0, q1 = weighted_candidate(records[heldout]["hgb"], records[heldout]["current"],
                                        records[heldout]["axes"][axis], sel["weight"])
            boot = bootstrap(q0, q1, records[heldout]["target"], records[heldout]["pitcher"],
                             BOOTSTRAP_DRAWS, seed=10900 + heldout)
            folds[str(heldout)] = {"train_years": list(train_years),
                                   "selected_weight": sel["weight"], **ev, "bootstrap": boot}
        full = select_weight(records, axis, DISCOVERY)
        preliminary = bool(all(r["selected_weight"] > 0 for r in folds.values())
                           and all(r["raw_gain"] > 0 and r["shape_gain"] > 0 for r in folds.values()))
        p_value = float(max(r["bootstrap"]["p_one_sided"] for r in folds.values()))
        loo[axis] = {"folds": folds, "full_discovery": full,
                     "preliminary_pass": preliminary, "p_value": p_value, "holm_pass": False}

    ordered = sorted(loo, key=lambda a: loo[a]["p_value"])
    holm_open = True
    for rank, axis in enumerate(ordered):
        threshold = ALPHA / (len(ordered) - rank)
        passed = bool(holm_open and loo[axis]["p_value"] <= threshold)
        loo[axis]["holm_threshold"] = threshold
        loo[axis]["holm_pass"] = passed
        if not passed:
            holm_open = False

    survivors = [a for a in AXES if loo[a]["preliminary_pass"] and loo[a]["holm_pass"]]
    lines += ["", "LOO/Holm 결과", "axis            weights        raw heldout          shape heldout        p(max)  Holm"]
    for axis in AXES:
        f = loo[axis]["folds"]
        w_str = "/".join("%.2f" % f[str(y)]["selected_weight"] for y in DISCOVERY)
        r_str = "/".join("%+.2f" % f[str(y)]["raw_gain"] for y in DISCOVERY)
        s_str = "/".join("%+.2f" % f[str(y)]["shape_gain"] for y in DISCOVERY)
        lines.append(f"{axis:15s} {w_str:13s} {r_str:20s} {s_str:20s} {loo[axis]['p_value']:.4f}  {loo[axis]['holm_pass']}")

    winner, confirmation, gate_pass = None, None, False
    if survivors:
        winner = max(survivors, key=lambda a: (loo[a]["full_discovery"]["min_gain"],
                                               loo[a]["full_discovery"]["mean_gain"],
                                               -loo[a]["full_discovery"]["weight"]))
        weight = float(loo[winner]["full_discovery"]["weight"])
        lines += ["", f"FROZEN WINNER: {winner} weight={weight:.2f}", "Exact CAT5 confirmation"]
        confirmation = {}
        for year in CONFIRMATION:
            row = records[year]
            seed_bases = exact_seed_bases(year)
            seed_results = {str(sd): paired(base, row["current"], row["axes"][winner], row["target"], weight)
                            for sd, base in seed_bases.items()}
            ensemble = np.mean(list(seed_bases.values()), axis=0)
            result = paired(ensemble, row["current"], row["axes"][winner], row["target"], weight)
            q0, q1 = weighted_candidate(ensemble, row["current"], row["axes"][winner], weight)
            result["F_contribution"] = contribution(q0, q1, row["target"], row["game_type"] == "F")
            result["R_contribution"] = contribution(q0, q1, row["target"], row["game_type"] == "R")
            result["bootstrap"] = bootstrap(q0, q1, row["target"], row["pitcher"], BOOTSTRAP_DRAWS, 109 + year)
            result["seed_results"] = seed_results
            confirmation[str(year)] = result
            lines.append(f"{year}: raw={result['raw_gain']:+.3f} shape={result['shape_gain']:+.3f} "
                         f"seeds={seed_results['42']['raw_gain']:+.3f}/{seed_results['7']['raw_gain']:+.3f} "
                         f"F/R={result['F_contribution']:+.3f}/{result['R_contribution']:+.3f} "
                         f"boot95=[{result['bootstrap']['p025']:+.3f},{result['bootstrap']['p975']:+.3f}]")
        raws = [confirmation[str(y)]["raw_gain"] for y in CONFIRMATION]
        shapes = [confirmation[str(y)]["shape_gain"] for y in CONFIRMATION]
        seeds_ok = all(confirmation[str(y)]["seed_results"][str(sd)]["raw_gain"] >= 0
                       for y in CONFIRMATION for sd in (42, 7))
        y24 = confirmation["2024"]
        gate_pass = bool(min(raws) >= 3.0 and np.mean(raws) >= 6.0 and min(shapes) > 0.0 and seeds_ok
                         and y24["raw_gain"] >= 8.0 and y24["shape_gain"] >= 5.0
                         and y24["F_contribution"] >= 0.0 and y24["R_contribution"] >= 0.0
                         and y24["bootstrap"]["p025"] > 0.0)
    else:
        lines += ["", "생존자 0 — exact 확인 생략 (설계상)."]

    lines += ["", f"FINAL GATE {'PASS' if gate_pass else 'FAIL'}",
              "PASS 여도 이 스크립트는 zip/제출을 만들지 않는다 (별도 감사 빌더에서).",
              f"elapsed={time.time()-t0:.1f}s"]
    out = {"experiment": EXPERIMENT, "baseline_gamma": BASE_GAMMA, "axes": AXES,
           "bin_edges": {"srate": SRATE_EDGES, "mid": MID_EDGES, "bn": BN_EDGES},
           "selection": "2021-2023 LOO; worst-fold p; Holm alpha=.05; top1 only (exp/107 동일)",
           "loo": loo, "survivors": survivors, "winner": winner,
           "confirmation": confirmation, "gate_pass": gate_pass,
           "elapsed_seconds": time.time() - t0}
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
