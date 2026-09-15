# -*- coding: utf-8 -*-
"""Clean-room fine-pitch selection x failure-mechanism forward screen.

High-level hypothesis: even though the current pitch type is unavailable at
inference, a pitcher's historical pitch selection in count/hand contexts can
change the expected mix of failure mechanisms.  All estimates use official
train/TrackMan rows strictly before the validation season.  Validation rows'
actual pitch types are never used.

This implementation is original to this workspace.  It reads no third-party
code and creates no model, test prediction, ZIP, submission, or LB probe.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train.csv"
TRACKMAN = ROOT / "data" / "trackman_history.csv"
OUT_JSON = ROOT / "lab" / "156_pitchtype_failure_prior_clean_screen.json"
OUT_TXT = ROOT / "lab" / "156_pitchtype_failure_prior_clean_screen.txt"

TEAM_MAP = {
    12: "DOO_BEA", 13: "LG_TWI", 14: "KIW_HER", 15: "LOT_GIA",
    16: "KIA_TIG", 17: "HAN_EAG", 18: "SAM_LIO", 19: "NC_DIN",
    20: "KT_WIZ", 21: "SSG_LAN",
}
JOIN_KEY = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom",
    "balls_before", "strikes_before", "outs_before", "ph", "bh", "pt", "bt",
]
OUTCOME_K = 100.0
SELECTION_K = 300.0
PITCHER_K = 220.0
RIDGES = (0.01, 0.1, 1.0, 10.0)
STRUCTURES = {
    "success": (0,),
    "reverse_bigmiss": (2, 4),
    "middle_reverse_bigmiss": (1, 2, 4),
    "no_overlap": (0, 1, 2, 4),
    "all5": (0, 1, 2, 3, 4),
}
CLASS_NAMES = ("success", "middle", "reverse", "overlap", "bigmiss")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def align_fine_pitch(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, object]]:
    tm_columns = [
        "season", "game_month", "game_dayofweek", "inning", "top_bottom",
        "balls_before", "strikes_before", "outs_before", "pitcher_hand",
        "batter_hand", "pitcher_team", "batter_team", "auto_pitch_type",
    ]
    trackman = pd.read_csv(
        TRACKMAN, encoding="utf-8-sig", usecols=tm_columns, low_memory=False
    )
    trackman.columns = [str(column).replace("\ufeff", "").strip() for column in trackman]
    hand = {2: "Right", 1: "Left"}
    left = frame[[
        "season", "game_month", "game_dayofweek", "inning", "top_bottom",
        "balls_before", "strikes_before", "outs_before", "pitcher_hand",
        "batter_hand", "pitcher_team_id", "batter_team_id",
    ]].copy()
    left["ph"] = left["pitcher_hand"].map(hand)
    left["bh"] = left["batter_hand"].map(hand)
    left["pt"] = left["pitcher_team_id"].map(TEAM_MAP)
    left["bt"] = left["batter_team_id"].map(TEAM_MAP)
    left["top_bottom"] = left["top_bottom"].astype(str).str[0]
    left["_row"] = np.arange(len(left), dtype=np.int64)

    trackman["pt"] = trackman["pitcher_team"].replace({"SK_WYV": "SSG_LAN"})
    trackman["bt"] = trackman["batter_team"].replace({"SK_WYV": "SSG_LAN"})
    trackman["ph"] = trackman["pitcher_hand"].astype(str)
    trackman["bh"] = trackman["batter_hand"].astype(str)
    trackman["top_bottom"] = trackman["top_bottom"].astype(str).str[0]
    for column in (
        "season", "game_month", "game_dayofweek", "inning", "balls_before",
        "strikes_before", "outs_before",
    ):
        left[column] = pd.to_numeric(left[column], errors="coerce").astype("Int64")
        trackman[column] = pd.to_numeric(trackman[column], errors="coerce").astype("Int64")

    left_ok = left.dropna(subset=["pt", "bt", "ph", "bh"])
    tm_ok = trackman[
        trackman["pt"].isin(TEAM_MAP.values())
        & trackman["bt"].isin(TEAM_MAP.values())
    ]
    left_count = left_ok.groupby(JOIN_KEY, observed=True).size()
    right_count = tm_ok.groupby(JOIN_KEY, observed=True).size()
    one_to_one = left_count[left_count.eq(1)].index.intersection(
        right_count[right_count.eq(1)].index
    )
    right = tm_ok.set_index(JOIN_KEY).loc[one_to_one, ["auto_pitch_type"]].reset_index()
    matched = left_ok[JOIN_KEY + ["_row"]].merge(
        right, on=JOIN_KEY, how="inner", validate="many_to_one"
    )
    if not matched["_row"].is_unique:
        raise AssertionError("TrackMan alignment is not one-to-one")
    fine = np.full(len(frame), "UNMATCHED", dtype=object)
    value = matched["auto_pitch_type"].fillna("OTHER").astype(str).str.strip()
    value = value.where(~value.isin(["", "nan", "None"]), "OTHER")
    fine[matched["_row"].to_numpy(np.int64)] = value.to_numpy(object)
    report = {
        "matched_rows": int(len(matched)),
        "matched_rate": float(len(matched) / len(frame)),
        "unique_join_keys": int(len(one_to_one)),
        "fine_types": sorted(pd.unique(value).tolist()),
    }
    return fine, report


def context_code(frame: pd.DataFrame) -> np.ndarray:
    balls = frame["balls_before"].to_numpy(np.int16)
    strikes = frame["strikes_before"].to_numpy(np.int16)
    hand = frame["batter_hand"].to_numpy(np.int16)
    return ((balls * 3 + strikes) * 2 + (hand - 1)).astype(np.int16)


def pitchtype_delta(
    frame: pd.DataFrame,
    fine: np.ndarray,
    source_before: int,
    validation: pd.DataFrame,
    vocabulary: list[str],
) -> tuple[np.ndarray, dict[str, object]]:
    source_mask = (
        (frame["season"].to_numpy() < source_before)
        & (frame["_target5"].to_numpy() >= 0)
        & (fine != "UNMATCHED")
    )
    source = frame.loc[source_mask]
    source_fine = fine[source_mask]
    all_pitchers = np.sort(frame["pitcher_id"].unique().astype(np.int64))
    pitcher_position = {int(value): index for index, value in enumerate(all_pitchers)}
    type_position = {value: index for index, value in enumerate(vocabulary)}
    p = np.fromiter(
        (pitcher_position[int(value)] for value in source["pitcher_id"]),
        dtype=np.int32, count=len(source),
    )
    t = np.fromiter(
        (type_position.get(str(value), type_position["OTHER"]) for value in source_fine),
        dtype=np.int16, count=len(source),
    )
    c = context_code(source)
    y5 = source["_target5"].to_numpy(np.int8)
    npitcher, ntype = len(all_pitchers), len(vocabulary)
    outcome = np.zeros((npitcher, ntype, 5), dtype=np.float64)
    selection = np.zeros((npitcher, 24, ntype), dtype=np.float64)
    np.add.at(outcome, (p, t, y5), 1.0)
    np.add.at(selection, (p, c, t), 1.0)

    global_counts = outcome.sum(axis=(0, 1))
    global_prior = global_counts / global_counts.sum()
    pitcher_counts = outcome.sum(axis=1)
    pitcher_n = pitcher_counts.sum(axis=1)
    pitcher_prior = (pitcher_counts + PITCHER_K * global_prior) / (
        pitcher_n[:, None] + PITCHER_K
    )
    outcome_n = outcome.sum(axis=2)
    outcome_probability = (
        outcome + OUTCOME_K * pitcher_prior[:, None, :]
    ) / (outcome_n[:, :, None] + OUTCOME_K)

    overall_type_count = selection.sum(axis=1)
    overall_n = overall_type_count.sum(axis=1)
    global_type = overall_type_count.sum(axis=0)
    global_type = global_type / global_type.sum()
    overall_type = (overall_type_count + SELECTION_K * global_type) / (
        overall_n[:, None] + SELECTION_K
    )
    context_n = selection.sum(axis=2)
    context_type = (
        selection + SELECTION_K * overall_type[:, None, :]
    ) / (context_n[:, :, None] + SELECTION_K)
    expected_context = np.einsum("pct,ptk->pck", context_type, outcome_probability)
    expected_parent = np.einsum("pt,ptk->pk", overall_type, outcome_probability)
    table_delta = expected_context - expected_parent[:, None, :]

    validation_pitcher = validation["pitcher_id"].to_numpy(np.int64)
    positions = np.searchsorted(all_pitchers, validation_pitcher)
    safe = np.minimum(positions, len(all_pitchers) - 1)
    seen = (positions < len(all_pitchers)) & (all_pitchers[safe] == validation_pitcher)
    code = context_code(validation)
    delta = np.zeros((len(validation), 5), dtype=np.float64)
    delta[seen] = table_delta[safe[seen], code[seen]]
    # No matched source history means exactly no correction.
    source_seen = np.zeros(npitcher, dtype=bool)
    source_seen[np.unique(p)] = True
    usable = seen & source_seen[safe]
    delta[~usable] = 0.0
    if not np.isfinite(delta).all() or np.max(np.abs(delta.sum(axis=1))) > 1e-9:
        raise AssertionError("invalid fine-pitch semantic delta")
    return delta, {
        "source_rows": int(len(source)),
        "source_pitchers": int(source_seen.sum()),
        "validation_seen_rate": float(usable.mean()),
        "delta_mean_abs": float(np.abs(delta).mean()),
        "delta_max_abs": float(np.abs(delta).max()),
    }


def main() -> None:
    started = time.time()
    e153 = load_module(ROOT / "exp" / "153_cause_runner_tensor_screen.py", "e153")
    e155 = load_module(ROOT / "exp" / "155_cause_runner_currentblend_gate.py", "e155")
    columns = [
        "season", "game_month", "game_dayofweek", "inning", "top_bottom",
        "balls_before", "strikes_before", "outs_before", "pitcher_hand",
        "batter_hand", "pitcher_team_id", "batter_team_id", "pitcher_id",
        "asof_pitcher_n", "asof_pitcher_middle_rate", "asof_pitcher_reverse_rate",
        "control_success", "runner_on_1b", "runner_on_2b", "runner_on_3b",
        "num_runners_on", "game_type",
    ]
    frame = pd.read_csv(TRAIN, encoding="utf-8-sig", usecols=columns, low_memory=False)
    frame = e153.add_context(e153.restore_target5(frame))
    fine, match_report = align_fine_pitch(frame)
    observed = pd.Series(fine[fine != "UNMATCHED"]).astype(str)
    vocabulary = sorted(observed.unique().tolist())
    if "OTHER" not in vocabulary:
        vocabulary.append("OTHER")

    rows = {
        year: frame.loc[frame["season"].eq(year)].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    targets = {year: rows[year]["control_success"].to_numpy(float) for year in rows}
    baseline = {}
    delta = {}
    fold_report = {}
    for year in (2022, 2023, 2024):
        n = len(rows[year])
        saved_target = e155.load_vector(e155.EXP021_TARGET[year], n)
        if not np.array_equal(saved_target, targets[year]):
            raise AssertionError(f"target/order mismatch {year}")
        cat = e155.load_probability(e155.CAT5[year], n)
        v18 = e155.load_vector(e155.V18[year], n)
        external = e155.load_probability(e155.EXP021[year], n)
        champion = e155.champion_probability(cat, v18)
        baseline[year] = e155.CHAMPION_WEIGHT * champion + e155.EXP021_WEIGHT * external
        delta[year], fold_report[str(year)] = pitchtype_delta(
            frame, fine, year, rows[year], vocabulary
        )
        fold_report[str(year)]["baseline_score"] = e155.score(baseline[year], targets[year])

    discovery = []
    for structure, used in STRUCTURES.items():
        x22, x23 = delta[2022][:, used], delta[2023][:, used]
        for ridge in RIDGES:
            coefficient = e155.ridge_fit(x22, targets[2022] - baseline[2022], ridge)
            result = e155.metrics(
                baseline[2023], x23 @ coefficient, targets[2023],
                rows[2023]["game_type"].astype(str).to_numpy(),
            )
            discovery.append({
                "structure": structure, "ridge": ridge,
                "columns": [CLASS_NAMES[index] for index in used],
                "coefficient_fit_2022": coefficient.tolist(),
                "transfer_2023": result,
            })
    eligible = [
        row for row in discovery
        if row["transfer_2023"]["raw_gain"] > 0
        and row["transfer_2023"]["equal_mean_shape_gain"] > 0
        and row["transfer_2023"]["F_contribution"] >= 0
        and row["transfer_2023"]["R_contribution"] >= 0
    ]
    selected = max(
        eligible or discovery,
        key=lambda row: (
            min(row["transfer_2023"]["F_contribution"], row["transfer_2023"]["R_contribution"]),
            row["transfer_2023"]["raw_gain"],
        ),
    )
    used = STRUCTURES[str(selected["structure"])]
    xfit = np.vstack([delta[2022][:, used], delta[2023][:, used]])
    residual = np.concatenate([
        targets[2022] - baseline[2022], targets[2023] - baseline[2023]
    ])
    coefficient = e155.ridge_fit(xfit, residual, float(selected["ridge"]))
    effect24 = delta[2024][:, used] @ coefficient
    confirmation = e155.metrics(
        baseline[2024], effect24, targets[2024],
        rows[2024]["game_type"].astype(str).to_numpy(),
    )
    oracle = e155.scalar_oracle(baseline[2024], effect24, targets[2024])
    passed = bool(
        eligible
        and confirmation["raw_gain"] >= 10
        and confirmation["equal_mean_shape_gain"] >= 8
        and confirmation["F_contribution"] >= 0
        and confirmation["R_contribution"] >= 0
    )
    status = "PASS_FOR_BUILD" if passed else "FAIL_NO_SUBMISSION"
    payload = {
        "experiment": 156, "status": status, "analysis_only": True,
        "reads_test": False, "reads_third_party_code": False, "uses_lb": False,
        "match": match_report, "vocabulary": vocabulary,
        "protocol": {
            "outcome_k": OUTCOME_K, "selection_k": SELECTION_K,
            "pitcher_k": PITCHER_K,
            "validation_pitch_type_used": False,
            "source_rule": "official matched rows with season < validation year",
            "fit": "2022 -> select on 2023 -> refit 2022+2023 -> untouched 2024",
        },
        "folds": fold_report, "discovery": discovery, "selected": selected,
        "refit_coefficient": coefficient.tolist(), "confirmation_2024": confirmation,
        "same_fold_2024_scalar_oracle_diagnostic_only": oracle,
        "elapsed_seconds": float(time.time() - started),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "=== exp156 clean-room fine-pitch failure prior ===",
        "DIAGNOSTIC ONLY: no current validation pitch type/test/model/ZIP/LB",
        f"match={match_report}",
        f"selected on 2023: {selected['structure']} ridge={selected['ridge']}",
        f"2023 transfer: {selected['transfer_2023']}",
        f"refit coefficients: {coefficient.tolist()}",
        f"untouched 2024: {confirmation}",
        f"2024 scalar oracle (diagnostic only): {oracle}",
        f"FINAL: {status}",
        f"elapsed={payload['elapsed_seconds']:.2f}s",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
