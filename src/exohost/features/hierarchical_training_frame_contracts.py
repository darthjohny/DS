# Файл `hierarchical_training_frame_contracts.py` слоя `features`.
#
# Этот файл отвечает только за:
# - подготовку признаков и training frame-слой;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `features` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

COARSE_NUMERIC_COLUMNS: tuple[str, ...] = (
    "ra",
    "dec",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "bp_rp",
    "parallax",
    "parallax_over_error",
    "ruwe",
    "radius_feature",
    "radius_flame",
    "radius_gspphot",
    "lum_flame",
)

REFINEMENT_NUMERIC_COLUMNS: tuple[str, ...] = (
    "ra",
    "dec",
    "phot_g_mean_mag",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "bp_rp",
    "parallax",
    "parallax_over_error",
    "ruwe",
    "radius_flame",
    "radius_gspphot",
    "lum_flame",
    "xmatch_separation_arcsec",
)

ID_OOD_NUMERIC_COLUMNS: tuple[str, ...] = (
    "ra",
    "dec",
    "phot_g_mean_mag",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "bp_rp",
    "parallax",
    "parallax_over_error",
    "ruwe",
    "radius_flame",
    "selector_score_1",
    "selector_score_2",
)

DOMAIN_TARGET_VALUES: tuple[str, ...] = ("id", "ood")
SPECTRAL_SUBCLASS_DIGITS: tuple[str, ...] = tuple(str(index) for index in range(10))
MIN_REFINEMENT_SUBCLASS_SUPPORT = 15


__all__ = [
    "COARSE_NUMERIC_COLUMNS",
    "DOMAIN_TARGET_VALUES",
    "ID_OOD_NUMERIC_COLUMNS",
    "MIN_REFINEMENT_SUBCLASS_SUPPORT",
    "REFINEMENT_NUMERIC_COLUMNS",
    "SPECTRAL_SUBCLASS_DIGITS",
]
