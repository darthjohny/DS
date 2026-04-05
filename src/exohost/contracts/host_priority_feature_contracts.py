# Файл `host_priority_feature_contracts.py` слоя `contracts`.
#
# Этот файл отвечает только за:
# - контракты колонок, датасетов и ролей правил;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `contracts` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from exohost.contracts.feature_contract import unique_columns

HOST_PRIORITY_CANONICAL_RADIUS_COLUMN = "radius_flame"

HOST_PRIORITY_CORE_FEATURES: tuple[str, ...] = (
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "bp_rp",
    "parallax",
    "parallax_over_error",
    "ruwe",
    HOST_PRIORITY_CANONICAL_RADIUS_COLUMN,
)

HOST_PRIORITY_OBSERVABILITY_FEATURES: tuple[str, ...] = (
    "ra",
    "dec",
    "phot_g_mean_mag",
)

HOST_PRIORITY_CONTEXT_FEATURES: tuple[str, ...] = (
    "spec_class",
    "evolution_stage",
    "spec_subclass",
)

HOST_PRIORITY_OPTIONAL_PHYSICAL_FEATURES: tuple[str, ...] = (
    "lum_flame",
    "evolstage_flame",
    "radius_gspphot",
)

HOST_PRIORITY_ALL_FEATURES: tuple[str, ...] = unique_columns(
    HOST_PRIORITY_CORE_FEATURES,
    HOST_PRIORITY_OBSERVABILITY_FEATURES,
    HOST_PRIORITY_CONTEXT_FEATURES,
    HOST_PRIORITY_OPTIONAL_PHYSICAL_FEATURES,
)
