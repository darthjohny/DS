"""Константы и соглашения для host EDA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

CLASS_ORDER: list[str] = ["M", "K", "G", "F"]
FEATURES: list[str] = ["teff_gspphot", "logg_gspphot", "radius_gspphot"]
OUTPUT_DIR = Path("data/eda/host")
PLOTS_DIR = OUTPUT_DIR / "plots"
LOGG_DWARF_MIN = 4.0
FEATURE_LABELS: dict[str, str] = {
    "teff_gspphot": "Эффективная температура, K",
    "logg_gspphot": "Логарифм поверхностной гравитации, dex",
    "radius_gspphot": "Радиус, R☉",
}
LAYER_LABELS: dict[str, str] = {
    "all": "Все MKGF",
    "dwarfs": "Карлики",
    "evolved": "Эволюционировавшие",
    "nb_all": "Все MKGF",
    "nb_dwarfs": "Карлики",
    "nb_evolved": "Эволюционировавшие",
}

M_EARLY_MIN = 3500.0
M_EARLY_MAX = 4000.0
M_MID_MIN = 3200.0

CONTRASTIVE_GAUSS_STATS_PATH = OUTPUT_DIR / "contrastive_gaussian_readiness.csv"
EVOLVED_SNAPSHOT_PATH = OUTPUT_DIR / "evolved_stars_snapshot.csv"
ABO_TOP20_PATH = OUTPUT_DIR / "abo_top20_by_teff.csv"
CONTRASTIVE_MODEL_VERSION = "gaussian_host_field_v1"
CONTRASTIVE_SCORE_MODE = "host_vs_field_log_lr_v1"
CONTRASTIVE_SHRINK_ALPHA = 0.15
CONTRASTIVE_MIN_POPULATION_SIZE = 2
CONTRASTIVE_VIEW_ENV = "CONTRASTIVE_HOST_FIELD_VIEW"

QUERY_ALL_MKGF = """
SELECT
    spec_class,
    teff_gspphot,
    logg_gspphot,
    radius_gspphot
FROM lab.v_nasa_gaia_train_classified
WHERE spec_class IN ('M','K','G','F');
"""

QUERY_DWARFS_MKGF = """
SELECT
    spec_class,
    teff_gspphot,
    logg_gspphot,
    radius_gspphot
FROM lab.v_nasa_gaia_train_dwarfs
WHERE spec_class IN ('M','K','G','F');
"""

QUERY_EVOLVED_MKGF = """
SELECT
    spec_class,
    teff_gspphot,
    logg_gspphot,
    radius_gspphot
FROM lab.v_nasa_gaia_train_evolved
WHERE spec_class IN ('M','K','G','F');
"""

QUERY_ABO_REF = """
SELECT
    spec_class,
    teff_gspphot,
    logg_gspphot,
    radius_gspphot
FROM lab.v_gaia_ref_abo_training;
"""

FloatArray = npt.NDArray[np.floating[Any]]
