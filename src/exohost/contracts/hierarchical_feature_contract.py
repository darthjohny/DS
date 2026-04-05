# Файл `hierarchical_feature_contract.py` слоя `contracts`.
#
# Этот файл отвечает только за:
# - контракты колонок, датасетов и ролей правил;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `contracts` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

GAIA_ID_COARSE_FEATURES: tuple[str, ...] = (
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "bp_rp",
    "parallax",
    "parallax_over_error",
    "ruwe",
    "radius_feature",
)

GAIA_MK_REFINEMENT_FEATURES: tuple[str, ...] = (
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "bp_rp",
    "parallax",
    "parallax_over_error",
    "ruwe",
    "radius_flame",
    "lum_flame",
    "phot_g_mean_mag",
)

GAIA_ID_OOD_FEATURES: tuple[str, ...] = (
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "bp_rp",
    "parallax",
    "parallax_over_error",
    "ruwe",
    "phot_g_mean_mag",
)
