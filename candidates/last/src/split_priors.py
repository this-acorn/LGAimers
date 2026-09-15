"""League-relative within-entity split priors.

Each table stores how much an entity (pitcher/batter) deviates from *its own*
level inside a subgroup, measured on league-centred outcomes. Because the splits
are re-centred per entity they are zero-sum by construction, so applying them can
reallocate probability mass between rows without moving the global level - the
one thing that has to stay untouched when the season environment is unknown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# entity, subgroup, shrinkage, weight. Selected greedily on worst-of-2023/2024
# forward gain and re-checked unchanged against four different base models.
SPLIT_SPECS = [
    ("pitcher_id", "batter_hand", 200, 0.50),
    ("pitcher_id", "count_ahead", 1600, 0.70),
    ("batter_id", "count_ahead", 1600, 0.35),
    ("pitcher_id", "count_state", 1600, 0.35),
    ("pitcher_id", "inning_bucket", 1600, 0.25),
    ("batter_id", "pitcher_hand", 1600, 0.15),
]


def add_split_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Row-local subgroup keys; nothing here depends on other rows."""
    out = pd.DataFrame(index=df.index)
    balls = pd.to_numeric(df["balls_before"], errors="coerce").fillna(0)
    strikes = pd.to_numeric(df["strikes_before"], errors="coerce").fillna(0)
    inning = pd.to_numeric(df["inning"], errors="coerce").fillna(1).clip(1, 10)
    out["pitcher_id"] = df["pitcher_id"].astype(str)
    out["batter_id"] = df["batter_id"].astype(str)
    out["pitcher_hand"] = df["pitcher_hand"].astype(str)
    out["batter_hand"] = df["batter_hand"].astype(str)
    out["count_state"] = balls.astype(int).astype(str) + "-" + strikes.astype(int).astype(str)
    out["count_ahead"] = np.where(strikes > balls, "a", "b")
    out["inning_bucket"] = inning.astype(int).astype(str)
    return out


def build_split_tables(history: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fit every configured split table on labelled seasons only."""
    keys = add_split_keys(history)
    league = history.groupby("season")["control_success"].transform("mean")
    keys["rel"] = history["control_success"].to_numpy(float) - league.to_numpy(float)
    keys["season"] = history["season"].to_numpy()

    tables: dict[str, pd.DataFrame] = {}
    for entity, subgroup, _, _ in SPLIT_SPECS:
        cell = keys.groupby([entity, subgroup], observed=True)["rel"].agg(["sum", "size"])
        own = keys.groupby(entity, observed=True)["rel"].agg(["sum", "size"])
        cell = cell.join(own, rsuffix="_entity")
        cell["split"] = cell["sum"] / cell["size"] - cell["sum_entity"] / cell["size_entity"]
        tables[f"{entity}|{subgroup}"] = cell[["split", "size"]]
    return tables


def apply_split_priors(raw: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> np.ndarray:
    """Additive probability correction for one frame of rows."""
    keys = add_split_keys(raw)
    correction = np.zeros(len(raw), dtype=float)
    for entity, subgroup, shrink, weight in SPLIT_SPECS:
        table = tables.get(f"{entity}|{subgroup}")
        if table is None:
            continue
        index = pd.MultiIndex.from_arrays([keys[entity], keys[subgroup]])
        split = np.nan_to_num(index.map(table["split"]).to_numpy(float))
        seen = np.nan_to_num(index.map(table["size"]).to_numpy(float))
        correction += weight * split * seen / (seen + shrink)
    return correction
