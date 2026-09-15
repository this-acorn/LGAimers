"""Production inference helpers for the stable minimax correction channels."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.context_adjusted_psych import attach_saved_context_psych
from src.context_pressure_features import build_context_pressure_features

PLATOON_COLS = [
    'same_hand_matchup','opposite_hand_matchup','left_pitcher','rare_left_opposite',
    'fastball_rate','breaking_rate','offspeed_rate','pitchmix_reliability',
    'fastball_threeball','breaking_twostrike','offspeed_twostrike',
    'breaking_platoon','offspeed_platoon',
]

def context_ridge_correction(raw, profile, asset_path):
    x=attach_saved_context_psych(raw,profile);a=np.load(asset_path,allow_pickle=True)
    x=x.reindex(columns=a['columns'].tolist(),fill_value=0);z=np.nan_to_num((x.to_numpy(float)-a['mean'])/a['std'])
    return z@a['coef']*float(a['scale'])

def platoon_correction(raw, model, scale):
    x=build_context_pressure_features(raw).reindex(columns=PLATOON_COLS)
    return model.predict(x)*float(scale)
