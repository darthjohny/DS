"""Константы и соглашения для router EDA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FEATURES: list[str] = ["teff_gspphot", "logg_gspphot", "radius_gspphot"]
SPEC_CLASSES: list[str] = ["A", "B", "F", "G", "K", "M", "O"]
EVOLUTION_STAGES: list[str] = ["dwarf", "evolved"]
FEATURE_LABELS: dict[str, str] = {
    "teff_gspphot": "Эффективная температура, K",
    "logg_gspphot": "Логарифм поверхностной гравитации, dex",
    "radius_gspphot": "Радиус, R☉",
}
EVOLUTION_STAGE_LABELS: dict[str, str] = {
    "dwarf": "Карлики",
    "evolved": "Эволюционировавшие",
}
ROUTER_LAYER_LABELS: dict[str, str] = {
    "ALL": "Все объекты",
    "DWARFS": "Карлики",
    "EVOLVED": "Эволюционировавшие",
}
ROUTER_VIEW = "lab.v_gaia_router_training"
OUTPUT_DIR = Path("data/eda/router")
PLOTS_DIR = OUTPUT_DIR / "plots"
SNAPSHOT_PATH = OUTPUT_DIR / "router_training_snapshot.csv"
CLASS_COUNTS_PATH = OUTPUT_DIR / "router_class_counts.csv"
GAUSS_STATS_PATH = OUTPUT_DIR / "router_gaussian_readiness.csv"
ROUTER_SHRINK_ALPHA = 0.15
ROUTER_SCORE_MODE = "gaussian_log_posterior_v1"
ROUTER_PRIOR_MODE = "uniform"
MIN_READY_CLASS_SIZE = 5

FloatArray = npt.NDArray[np.floating[Any]]
