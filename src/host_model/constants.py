"""Общие константы пакета host-модели.

Здесь собраны:

- базовые признаки для обучения и scoring;
- поддерживаемые dwarf-классы;
- имена стандартных DB-источников для contrastive обучения;
- физические пороги для разделения M-подклассов.
"""

from __future__ import annotations

FEATURES: list[str] = ["teff_gspphot", "logg_gspphot", "radius_gspphot"]
DWARF_CLASSES: list[str] = ["M", "K", "G", "F"]

CONTRASTIVE_POPULATION_COLUMN = "is_host"
CONTRASTIVE_VIEW_ENV = "CONTRASTIVE_HOST_FIELD_VIEW"
DEFAULT_CONTRASTIVE_HOST_VIEW = "lab.v_nasa_gaia_train_dwarfs"
DEFAULT_CONTRASTIVE_FIELD_VIEW = "lab.v_gaia_ref_mkgf_dwarfs"

LOGG_DWARF_MIN = 4.0
M_EARLY_MIN = 3500.0
M_EARLY_MAX = 4000.0
M_MID_MIN = 3200.0
M_MID_MAX = 3500.0
M_LATE_MAX = 3200.0
EPS = 1e-12

__all__ = [
    "CONTRASTIVE_POPULATION_COLUMN",
    "CONTRASTIVE_VIEW_ENV",
    "DEFAULT_CONTRASTIVE_FIELD_VIEW",
    "DEFAULT_CONTRASTIVE_HOST_VIEW",
    "DWARF_CLASSES",
    "EPS",
    "FEATURES",
    "LOGG_DWARF_MIN",
    "M_EARLY_MAX",
    "M_EARLY_MIN",
    "M_LATE_MAX",
    "M_MID_MAX",
    "M_MID_MIN",
]
