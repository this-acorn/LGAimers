# -*- coding: utf-8 -*-
"""Evaluate two additional hierarchical residual configurations.
"""

from __future__ import annotations

_PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

import importlib.util
from pathlib import Path


ROOT = Path(str(_PROJECT_ROOT))
SOURCE = ROOT / "exp/107_v18_extension_probe.py"
spec = importlib.util.spec_from_file_location("v18_extension_107", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {SOURCE}")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

probe.EXPERIMENT = 108
probe.OUT_TXT = ROOT / "lab/108_v18_missing_public_axes.txt"
probe.OUT_JSON = ROOT / "lab/108_v18_missing_public_axes.json"
probe.AXES = {
    "batter_phand": {
        "keys": ["game_type", "batter_id", "pitcher_hand"],
        "parent": "batter",
        "k": 200.0,
    },
    "pitcher_stint": {
        "keys": ["game_type", "pitcher_id", "pitcher_team_id"],
        "parent": "pitcher",
        "k": 180.0,
    },
}


if __name__ == "__main__":
    probe.main()
