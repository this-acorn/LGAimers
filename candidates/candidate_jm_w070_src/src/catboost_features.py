"""Domain features for ordered boosting; all transforms are row-local."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.deep_preprocessing import add_row_features
from src.features import ID_COL, TARGET_COL


CAT_COLS = [
    "top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "pitcher_id", "batter_id",
    "count_state", "hand_matchup",
]


def build_catboost_features(df: pd.DataFrame, prior: float) -> pd.DataFrame:
    x = add_row_features(df.drop(columns=[ID_COL, TARGET_COL], errors="ignore"), prior)
    x["count_state"] = x["balls_before"].astype(str) + "-" + x["strikes_before"].astype(str)
    x["hand_matchup"] = x["pitcher_hand"].astype(str) + "-" + x["batter_hand"].astype(str)
    x["score_abs"] = x["score_diff_pitcher_team"].abs()
    x["close_game"] = (x["score_abs"] <= 1).astype("int8")
    x["pressure_index"] = (
        (x["balls_before"] + 1) / (x["strikes_before"] + 1)
        * (1 + x["num_runners_on"]) * np.log1p(x["li"].clip(lower=0))
    )
    x["pitcher_uncertainty"] = 1 / np.sqrt(x["asof_pitcher_n"].clip(lower=0) + 1)
    x["batter_uncertainty"] = 1 / np.sqrt(x["asof_batter_n"].clip(lower=0) + 1)
    x["recent_slope_1_5"] = (
        x["asof_pitcher_prev1_game_success_rate"]
        - x["asof_pitcher_prev5_game_success_rate"]
    )
    x["middle_slope_1_5"] = (
        x["asof_pitcher_prev1_game_middle_rate"]
        - x["asof_pitcher_prev5_game_middle_rate"]
    )
    rates = x[["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]]
    safe = rates.clip(lower=1e-7)
    x["pitchmix_entropy"] = -(safe * np.log(safe)).sum(axis=1, min_count=1)
    for col in CAT_COLS:
        x[col] = x[col].astype("string").fillna("__MISSING__").astype(str)
    numeric = [c for c in x.columns if c not in CAT_COLS]
    x[numeric] = x[numeric].replace([np.inf, -np.inf], np.nan)
    return x


def attach_trackman_features(x: pd.DataFrame, table_path: str) -> pd.DataFrame:
    """Left join precomputed prior-season features without changing row order."""
    table = pd.read_csv(table_path, dtype={"pitcher_id": str})
    out = x.copy(); out["_row_order"] = np.arange(len(out))
    out["pitcher_id"] = out["pitcher_id"].astype(str)
    out = out.merge(table, on=["pitcher_id", "season"], how="left", sort=False)
    return out.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)
