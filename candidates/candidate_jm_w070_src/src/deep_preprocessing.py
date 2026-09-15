"""Leakage-safe preprocessing for the neural mixture-of-experts model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features import ID_COL, TARGET_COL


CATEGORICAL_COLS = [
    "top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "pitcher_id", "batter_id",
]

# Expert interpretation is only an inductive bias; the three types are not labeled.
FEATURE_GROUPS = {
    "context": [
        "inning", "balls_before", "strikes_before", "outs_before",
        "score_diff_pitcher_team", "num_runners_on", "home_win_expectancy", "li",
    ],
    "career": [
        "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    ],
    "recent": [c for k in (1, 3, 5) for c in (
        f"asof_pitcher_prev{k}_game_success_rate",
        f"asof_pitcher_prev{k}_game_middle_rate",
    )],
    "batter": ["asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate"],
    "pitchmix": [
        "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    ],
}


def add_row_features(df: pd.DataFrame, prior: float = 0.5) -> pd.DataFrame:
    """Only row-local transformations; never aggregates evaluation rows."""
    x = df.copy()
    x["count_pressure"] = x["balls_before"] - x["strikes_before"]
    x["two_strike"] = (x["strikes_before"] == 2).astype(np.int8)
    x["three_ball"] = (x["balls_before"] == 3).astype(np.int8)
    x["late_inning"] = (x["inning"] >= 7).astype(np.int8)
    x["scoring_position"] = ((x["runner_on_2b"] == 1) | (x["runner_on_3b"] == 1)).astype(np.int8)
    x["runners_x_li"] = x["num_runners_on"] * x["li"]
    x["recent_success_mean"] = x[[
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]].mean(axis=1)
    x["recent_vs_career"] = x["recent_success_mean"] - x["asof_pitcher_success_rate"]
    # Empirical-Bayes shrinkage: tiny samples move toward a train-only prior.
    # This distinguishes a 1/1 history from a stable 1000-pitch history.
    for prefix, n_col, strength in (
        ("pitcher", "asof_pitcher_n", 100.0),
        ("batter", "asof_batter_n", 100.0),
    ):
        rate_col = f"asof_{prefix}_success_rate"
        n = pd.to_numeric(x[n_col], errors="coerce").fillna(0).clip(lower=0)
        rate = pd.to_numeric(x[rate_col], errors="coerce").fillna(prior)
        x[f"{prefix}_success_smoothed"] = (rate * n + prior * strength) / (n + strength)
    return x


@dataclass
class TabularPreprocessor:
    categorical_cols: list[str] | None = None
    numeric_cols: list[str] | None = None
    category_maps: dict[str, dict[str, int]] | None = None
    medians: dict[str, float] | None = None
    means: dict[str, float] | None = None
    stds: dict[str, float] | None = None
    target_prior: float = 0.5

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        if TARGET_COL in df:
            self.target_prior = float(df[TARGET_COL].mean())
        x = add_row_features(df, self.target_prior)
        self.categorical_cols = [c for c in CATEGORICAL_COLS if c in x]
        excluded = set(self.categorical_cols + [ID_COL, TARGET_COL])
        self.numeric_cols = [c for c in x.columns if c not in excluded]
        self.category_maps = {}
        for col in self.categorical_cols:
            values = x[col].astype("string").fillna("__MISSING__")
            # Vocabulary is learned from labeled training rows only. 0 is unknown.
            self.category_maps[col] = {v: i + 1 for i, v in enumerate(sorted(values.unique()))}

        num = x[self.numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        self.medians = num.median().fillna(0.0).to_dict()
        filled = num.fillna(self.medians)
        self.means = filled.mean().to_dict()
        std = filled.std().replace(0, 1).fillna(1.0)
        self.stds = std.to_dict()
        return self

    @property
    def cardinalities(self) -> list[int]:
        assert self.categorical_cols and self.category_maps
        return [len(self.category_maps[c]) + 1 for c in self.categorical_cols]

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.numeric_cols is None or self.category_maps is None:
            raise RuntimeError("preprocessor is not fitted")
        x = add_row_features(df, self.target_prior)
        num = x[self.numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing = num.isna().to_numpy(dtype=np.float32)
        num = num.fillna(self.medians)
        num = (num - pd.Series(self.means)) / pd.Series(self.stds)
        # Missing indicators preserve cold-start information explicitly.
        numeric = np.concatenate([num.to_numpy(np.float32), missing], axis=1)

        encoded = []
        for col in self.categorical_cols or []:
            values = x[col].astype("string").fillna("__MISSING__")
            encoded.append(values.map(self.category_maps[col]).fillna(0).to_numpy(np.int64))
        categorical = np.stack(encoded, axis=1)
        return numeric, categorical
