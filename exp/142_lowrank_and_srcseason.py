# -*- coding: utf-8 -*-
"""[142] mk-isos 의 두 '구조 기술'을 우리 챔피언 위에서 검증 (Claude 소유, 학습 0)

공개 저장소(mk-isos) 상승분의 정체는 새 정보가 아니라 **정규화·선택 방식**이었다. 우리가 안 쓴 두 가지:

  ① SRC  source-season 평균 — 냉동 표를 과거 전체에서 한 번 추정하지 않고, 과거 **각 시즌에서 따로 추정해
         균등 평균**한다. 레짐 변화에 덜 흔들린다. (그들 EXP-019 Team EB 방식)
  ② LR   저랭크 공유 — `투수 × 24문맥(카운트 12 × 타자손 2)` 잔차 행렬을 만들어 **rank-r SVD** 로 압축.
         얇은 셀을 개별 추정하는 대신 투수들이 소수의 공통 '문맥 반응 패턴'을 공유한다. (그들 EXP-020, rank 6)
         우리가 얇은 셀 때문에 15전 1승이었던 자리를, 키를 늘리는 대신 **모수를 줄여** 접근한다.

기준선은 현 챔피언의 V18 보정(γ=0.30) 적용 후이며, 두 기술 모두 **그 위에 얹는 추가 항**으로 검증한다.
게이트는 exp/107·109 와 동일: 2021~2023 leave-one-year-out 으로 강도(및 rank)를 잠그고, 통과 시에만
exact CAT5 2022/2023/2024 로 확인. 학습·zip·제출 없음.

실행: PYTHONIOENCODING=utf-8 python -u exp/142_lowrank_and_srcseason.py   (~5분)
출력: lab/142_lowrank_srcseason.txt / .json
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
DATA = ROOT / "data/train.csv"
OOF = ROOT / "lab/24b_preds.npz"
OUT_TXT = ROOT / "lab/142_lowrank_srcseason.txt"
OUT_JSON = ROOT / "lab/142_lowrank_srcseason.json"

DISCOVERY = (2021, 2022, 2023)
CONFIRMATION = (2022, 2023, 2024)
BASE_GAMMA = 0.30                      # 기존 V18 강도 (고정)
WEIGHTS = np.round(np.arange(0.0, 0.501, 0.05), 2)   # 추가 항의 강도 후보
RANKS = (2, 4, 6, 8, 12)               # 공개 EXP-020 과 동일한 후보 집합
K_CONTEXT = 300.0                      # 공개 EXP-020 의 문맥 수축 상수
BOOTSTRAP = 5000
ALPHA = 0.05

lines: list[str] = []


def log(m: str = "") -> None:
    print(m, flush=True)
    lines.append(m)


def load_v18():
    spec = importlib.util.spec_from_file_location("v18_142", ROOT / "exp/103_v18_residual.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v18 = load_v18()


def score(p, y):
    r = float(y.mean())
    return 100000.0 * (1.0 - float(np.mean((p - y) ** 2)) / (r * (1.0 - r)))


def shape_gain(base, cand, y):
    return score(cand - cand.mean() + base.mean(), y) - score(base, y)


def contribution(base, cand, y, mask):
    r = float(y.mean())
    imp = (base - y) ** 2 - (cand - y) ** 2
    return float(100000.0 * imp[mask].sum() / (len(y) * r * (1.0 - r)))


# =====================================================================
# ① SRC — source-season 평균판 V18 효과
#    기존: 과거 전체를 한 덩어리로 집계해 한 번 추정
#    변경: 과거 각 시즌 s 에서 따로 효과를 만들고 균등 평균 (그들 'all_prior' 방식)
# =====================================================================
def effect_source_season(history: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    seasons = sorted(history["season"].unique())
    if not seasons:
        return np.zeros(len(validation))
    acc = np.zeros(len(validation), dtype=np.float64)
    for s in seasons:
        acc += v18.make_effect(history[history["season"] == s], validation)[0]
    return acc / len(seasons)


# =====================================================================
# ② LR — 투수 × 24문맥 잔차 행렬의 rank-r SVD 공유
#    셀 값 = EB 수축된 (투수의 그 문맥 성공률 − 투수 전체 성공률).  V18 과 같은 원재료를 쓰되
#    계층 수축이 아니라 **저랭크 근사**로 규제한다. 결측 셀은 0(=투수 평균과 같음)으로 둔다.
# =====================================================================
def context_index(frame: pd.DataFrame) -> np.ndarray:
    balls = frame["balls_before"].fillna(0).to_numpy(np.int64)
    strikes = frame["strikes_before"].fillna(0).to_numpy(np.int64)
    hand = (frame["batter_hand"].astype(str).to_numpy() == "R").astype(np.int64)
    return (balls * 3 + strikes) * 2 + hand          # 12 카운트 × 2 손 = 24


def lowrank_effects(history: pd.DataFrame, ranks=RANKS):
    """history 로 (투수 × 24) 효과 행렬을 만들고 rank 별 저랭크 근사를 반환."""
    pid = history["pitcher_id"].to_numpy()
    ctx = context_index(history)
    y = history["control_success"].to_numpy(np.float64)
    pitchers = np.unique(pid)
    pmap = {p: i for i, p in enumerate(pitchers)}
    pi = np.fromiter((pmap[p] for p in pid), dtype=np.int64, count=len(pid))
    P, C = len(pitchers), 24
    n = np.zeros((P, C)); s = np.zeros((P, C))
    np.add.at(n, (pi, ctx), 1.0)
    np.add.at(s, (pi, ctx), y)
    p_n = n.sum(1, keepdims=True); p_s = s.sum(1, keepdims=True)
    p_rate = np.divide(p_s, p_n, out=np.full_like(p_s, float(y.mean())), where=p_n > 0)
    # EB 수축: 셀 성공률을 투수 평균 쪽으로 K_CONTEXT 만큼 끌어당긴 뒤 차이를 취한다
    cell_rate = (s + K_CONTEXT * p_rate) / (n + K_CONTEXT)
    M = (n / (n + K_CONTEXT)) * (cell_rate - p_rate)      # 신뢰도 가중 편차
    M = M - M.mean(0, keepdims=True)                       # 문맥 주효과 제거(모델이 이미 가짐)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    out = {}
    for r in ranks:
        rr = min(r, len(S))
        out[r] = (U[:, :rr] * S[:rr]) @ Vt[:rr]
    return pmap, out, {"pitchers": P, "cells_nonzero": float((n > 0).mean()),
                       "sv_top": [float(x) for x in S[:12]]}


def apply_lowrank(pmap, mat, validation: pd.DataFrame) -> np.ndarray:
    pid = validation["pitcher_id"].to_numpy()
    ctx = context_index(validation)
    idx = np.fromiter((pmap.get(p, -1) for p in pid), dtype=np.int64, count=len(pid))
    eff = np.zeros(len(validation), dtype=np.float64)
    ok = idx >= 0
    eff[ok] = mat[idx[ok], ctx[ok]]
    return eff


# =====================================================================
def paired(base, cur, new, y, w):
    q0 = np.clip(base + BASE_GAMMA * cur, 0.0, 1.0)
    q1 = np.clip(q0 + float(w) * new, 0.0, 1.0)
    return q0, q1, {"raw_gain": float(score(q1, y) - score(q0, y)),
                    "shape_gain": float(shape_gain(q0, q1, y)),
                    "mean_shift": float(q1.mean() - q0.mean())}


def bootstrap(base, cand, y, pitcher, draws=BOOTSTRAP, seed=142):
    w = pd.DataFrame({"p": pitcher, "n": 1.0, "y": y,
                      "b": (base - y) ** 2, "c": (cand - y) ** 2})
    g = w.groupby("p", sort=False, observed=True).agg(n=("n", "sum"), y=("y", "sum"),
                                                      b=("b", "sum"), c=("c", "sum"))
    v = g[["n", "y", "b", "c"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    gains = np.empty(draws)
    cur, G = 0, len(v)
    while cur < draws:
        k = min(250, draws - cur)
        sm = v[rng.integers(0, G, size=(k, G))].sum(1)
        rate = sm[:, 1] / sm[:, 0]
        gains[cur:cur + k] = 100000.0 * (sm[:, 2] - sm[:, 3]) / (sm[:, 0] * rate * (1 - rate))
        cur += k
    return {"median": float(np.median(gains)), "p025": float(np.quantile(gains, .025)),
            "p975": float(np.quantile(gains, .975)),
            "p_one_sided": float((1 + np.count_nonzero(gains <= 0)) / (draws + 1))}


def exact_bases(year):
    if year == 2024:
        return {s: np.load(ROOT / f"lab/89_cat5_probs_seed{s}.npy").astype(float)[:, 0] for s in (42, 7)}
    return {s: np.load(ROOT / f"lab/104_cat5_y{year}_probs_seed{s}.npy").astype(float)[:, 0] for s in (42, 7)}


def main():
    t0 = time.time()
    log("=== exp/142 — source-season 평균(SRC) · 저랭크 공유(LR) 검증 (학습 0) ===")
    log(f"기준: 챔피언 + V18 γ={BASE_GAMMA}.  게이트: 2021~2023 LOO 로 강도·rank 잠금 → exact CAT5 확인")
    cols = ["season", "game_type", "pitcher_id", "batter_hand",
            "balls_before", "strikes_before", "control_success"]
    frame = pd.read_csv(DATA, encoding="utf-8-sig", usecols=cols, low_memory=False)
    for c in ("game_type", "batter_hand"):
        frame[c] = frame[c].astype("string").fillna("__MISSING__").astype(str)
    frame["_pressure"] = v18.pressure_code(frame)
    oof = np.load(OOF)

    rec = {}
    for year in (2021, 2022, 2023, 2024):
        va = frame[frame["season"] == year]
        hi = frame[frame["season"] < year]
        y = va["control_success"].to_numpy(float)
        hgb = oof[f"{year}_base65"].astype(float)
        assert len(hgb) == len(y) and np.array_equal(y, oof[f"{year}_y"].astype(float)), f"OOF 정렬 {year}"
        cur, _ = v18.make_effect(hi, va)
        src = effect_source_season(hi, va)
        pmap, mats, diag = lowrank_effects(hi)
        lr = {r: apply_lowrank(pmap, m, va) for r, m in mats.items()}
        rec[year] = {"y": y, "hgb": hgb, "cur": cur, "SRC": src - cur, "LR": lr,
                     "pitcher": va["pitcher_id"].to_numpy(),
                     "gt": va["game_type"].to_numpy()}
        log(f"  {year}: rows {len(y):,}  투수 {diag['pitchers']}  셀채움 {diag['cells_nonzero']*100:.1f}%  "
            f"특이값 상위6 {np.round(diag['sv_top'][:6], 4).tolist()}")

    # 후보: SRC(추가항 = src−cur, 즉 '평균 방식으로 바꾸는 변화량') + LR rank 별
    cands = {"SRC(source-season 평균으로 교체)": lambda r, yr: r[yr]["SRC"]}
    for rk in RANKS:
        cands[f"LR rank={rk}"] = (lambda rk_: (lambda r, yr: r[yr]["LR"][rk_]))(rk)

    def sel(name, fn, years):
        rows = []
        for w in WEIGHTS:
            g = {yr: paired(rec[yr]["hgb"], rec[yr]["cur"], fn(rec, yr), rec[yr]["y"], w)[2]["raw_gain"]
                 for yr in years}
            rows.append({"weight": float(w), "yearly": g,
                         "min": float(min(g.values())), "mean": float(np.mean(list(g.values())))})
        return max(rows, key=lambda x: (x["min"], x["mean"], -x["weight"]))

    log("")
    log("LOO(2021~2023) — 강도는 나머지 두 해에서만 고르고 남은 해로 평가")
    log(f"{'후보':30s} {'선택강도':>14s} {'held-out raw':>22s} {'p(max)':>8s}")
    loo = {}
    for name, fn in cands.items():
        folds = {}
        for ho in DISCOVERY:
            tr = tuple(y for y in DISCOVERY if y != ho)
            s_ = sel(name, fn, tr)
            q0, q1, ev = paired(rec[ho]["hgb"], rec[ho]["cur"], fn(rec, ho), rec[ho]["y"], s_["weight"])
            folds[ho] = {"w": s_["weight"], **ev,
                         "boot": bootstrap(q0, q1, rec[ho]["y"], rec[ho]["pitcher"], seed=14200 + ho)}
        full = sel(name, fn, DISCOVERY)
        pv = max(f["boot"]["p_one_sided"] for f in folds.values())
        prelim = all(f["w"] > 0 for f in folds.values()) and all(
            f["raw_gain"] > 0 and f["shape_gain"] > 0 for f in folds.values())
        loo[name] = {"folds": {str(k): v for k, v in folds.items()}, "full": full,
                     "prelim": bool(prelim), "p": float(pv)}
        log(f"{name:30s} {'/'.join('%.2f' % folds[y]['w'] for y in DISCOVERY):>14s} "
            f"{'/'.join('%+.2f' % folds[y]['raw_gain'] for y in DISCOVERY):>22s} {pv:8.4f}")

    order = sorted(loo, key=lambda k: loo[k]["p"])
    open_ = True
    for i, k in enumerate(order):
        thr = ALPHA / (len(order) - i)
        loo[k]["holm"] = bool(open_ and loo[k]["p"] <= thr)
        if not loo[k]["holm"]:
            open_ = False
    surv = [k for k in cands if loo[k]["prelim"] and loo[k]["holm"]]
    log("")
    log(f"LOO+Holm 생존: {surv if surv else '없음'}")

    conf, gate = None, False
    if surv:
        win = max(surv, key=lambda k: (loo[k]["full"]["min"], loo[k]["full"]["mean"], -loo[k]["full"]["weight"]))
        w = float(loo[win]["full"]["weight"])
        log(f"FROZEN: {win}  강도 {w:.2f} → exact CAT5 확인")
        conf = {}
        for yr in CONFIRMATION:
            b = exact_bases(yr)
            per = {str(s): paired(v_, rec[yr]["cur"], cands[win](rec, yr), rec[yr]["y"], w)[2] for s, v_ in b.items()}
            ens = np.mean(list(b.values()), axis=0)
            q0, q1, ev = paired(ens, rec[yr]["cur"], cands[win](rec, yr), rec[yr]["y"], w)
            ev["F"] = contribution(q0, q1, rec[yr]["y"], rec[yr]["gt"] == "F")
            ev["R"] = contribution(q0, q1, rec[yr]["y"], rec[yr]["gt"] == "R")
            ev["boot"] = bootstrap(q0, q1, rec[yr]["y"], rec[yr]["pitcher"], seed=142 + yr)
            ev["seeds"] = per
            conf[str(yr)] = ev
            log(f"  {yr}: raw {ev['raw_gain']:+.3f} shape {ev['shape_gain']:+.3f} "
                f"seeds {per['42']['raw_gain']:+.3f}/{per['7']['raw_gain']:+.3f} "
                f"F/R {ev['F']:+.3f}/{ev['R']:+.3f} boot95 [{ev['boot']['p025']:+.3f},{ev['boot']['p975']:+.3f}]")
        raws = [conf[str(y)]["raw_gain"] for y in CONFIRMATION]
        shp = [conf[str(y)]["shape_gain"] for y in CONFIRMATION]
        y24 = conf["2024"]
        gate = bool(min(raws) >= 3 and np.mean(raws) >= 6 and min(shp) > 0
                    and all(conf[str(y)]["seeds"][str(s)]["raw_gain"] >= 0 for y in CONFIRMATION for s in (42, 7))
                    and y24["raw_gain"] >= 8 and y24["shape_gain"] >= 5
                    and y24["F"] >= 0 and y24["R"] >= 0 and y24["boot"]["p025"] > 0)
    else:
        log("생존자 0 — exact 확인 생략(설계상).")
        win = None

    log("")
    log(f"FINAL GATE {'PASS' if gate else 'FAIL'}   (통과해도 이 스크립트는 zip·제출을 만들지 않는다)")
    log(f"elapsed {time.time()-t0:.1f}s")
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps({"base_gamma": BASE_GAMMA, "ranks": list(RANKS),
                                    "k_context": K_CONTEXT, "weights": WEIGHTS.tolist(),
                                    "loo": loo, "survivors": surv, "winner": win,
                                    "confirmation": conf, "gate_pass": gate},
                                   ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
