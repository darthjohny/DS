# Файл `refinement_family_feature_contract.py` слоя `contracts`.
#
# Этот файл отвечает только за:
# - контракты колонок, датасетов и ролей правил;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `contracts` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

REFINEMENT_FAMILY_FEATURES: tuple[str, ...] = (
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "bp_rp",
    "parallax",
    "parallax_over_error",
    "ruwe",
    "radius_flame",
    "lum_flame",
    "evolstage_flame",
    "phot_g_mean_mag",
)
