"""Shared, row-wise-only feature engineering for training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"

# IDs are arbitrary labels. The supplied as-of statistics carry the useful player
# history without teaching a tree a fake numerical ordering between identifiers.
DROP_COLS = [ID_COL, TARGET_COL, "pitcher_id", "batter_id"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features using one row at a time (safe under the test-data rules)."""
    x = df.drop(columns=DROP_COLS, errors="ignore").copy()

    # Small, fixed-domain categorical variables. Factorization is deliberately
    # avoided because deriving categories from the full test set is prohibited.
    mappings = {
        "top_bottom": {"T": 0, "B": 1},
        "game_type": {"R": 0, "P": 1, "E": 2},
        "base_state": {
            "___": 0, "1__": 1, "_2_": 2, "__3": 3,
            "12_": 4, "1_3": 5, "_23": 6, "123": 7,
        },
    }
    for col, mapping in mappings.items():
        if col in x:
            x[col] = x[col].map(mapping).fillna(-1).astype("int16")

    # All remaining columns are numeric codes or measurements in the official
    # schema. Unknown/missing values stay NaN; HistGradientBoosting handles them.
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    # Context interactions, all computable independently from the current row.
    x["count_pressure"] = x["balls_before"] - x["strikes_before"]
    x["late_inning"] = (x["inning"] >= 7).astype("int8")
    x["high_leverage"] = (x["li"] >= 2.0).astype("int8")
    x["runners_x_li"] = x["num_runners_on"] * x["li"]
    x["pitcher_log_n"] = np.log1p(x["asof_pitcher_n"].clip(lower=0))
    x["batter_log_n"] = np.log1p(x["asof_batter_n"].clip(lower=0))
    x["pitchmix_log_n"] = np.log1p(x["asof_pitcher_pitchmix_n"].clip(lower=0))

    recent = [
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]
    x["recent_success_mean"] = x[recent].mean(axis=1)
    x["recent_vs_career"] = (
        x["recent_success_mean"] - x["asof_pitcher_success_rate"]
    )
    return x.astype("float32")

