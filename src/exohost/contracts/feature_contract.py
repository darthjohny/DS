# Файл `feature_contract.py` слоя `contracts`.
#
# Этот файл отвечает только за:
# - контракты колонок, датасетов и ролей правил;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `contracts` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

IDENTIFIER_COLUMNS: tuple[str, ...] = ("source_id",)

# Базовые физические признаки для классификации звезд.
BASE_STELLAR_FEATURES: tuple[str, ...] = (
    "teff_gspphot",
    "logg_gspphot",
    "radius_gspphot",
)

# Дополнительные физические признаки, которые могут усиливать модель,
# но не должны ломать базовую классификацию.
ADDITIONAL_PHYSICAL_FEATURES: tuple[str, ...] = (
    "mh_gspphot",
    "bp_rp",
)

# Признаки качества и надежности астрометрии.
QUALITY_FEATURES: tuple[str, ...] = (
    "parallax_over_error",
    "ruwe",
    "validation_factor",
)

# Признаки, которые пригодятся для наблюдательной пригодности
# и downstream ranking-логики.
OBSERVABILITY_FEATURES: tuple[str, ...] = (
    "ra",
    "dec",
    "parallax",
    "phot_g_mean_mag",
)

# Базовый набор признаков для router-слоя.
ROUTER_FEATURES: tuple[str, ...] = (
    *BASE_STELLAR_FEATURES,
    "parallax",
    "parallax_over_error",
    "ruwe",
    "bp_rp",
    "mh_gspphot",
)


def unique_columns(*column_groups: tuple[str, ...]) -> tuple[str, ...]:
    # Сохраняем порядок первого появления и убираем дубли.
    seen: set[str] = set()
    ordered_columns: list[str] = []

    for group in column_groups:
        for column_name in group:
            if column_name in seen:
                continue
            seen.add(column_name)
            ordered_columns.append(column_name)

    return tuple(ordered_columns)
