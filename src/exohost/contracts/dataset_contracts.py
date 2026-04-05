# Файл `dataset_contracts.py` слоя `contracts`.
#
# Этот файл отвечает только за:
# - контракты колонок, датасетов и ролей правил;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `contracts` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from dataclasses import dataclass

from exohost.contracts.feature_contract import (
    ADDITIONAL_PHYSICAL_FEATURES,
    BASE_STELLAR_FEATURES,
    IDENTIFIER_COLUMNS,
    QUALITY_FEATURES,
)
from exohost.contracts.host_priority_feature_contracts import (
    HOST_PRIORITY_CANONICAL_RADIUS_COLUMN,
)


@dataclass(frozen=True, slots=True)
class DatasetContract:
    # Имя relation в формате schema.table.
    relation_name: str

    # Колонки, без которых датасет не считается пригодным.
    required_columns: tuple[str, ...]

    # Колонки, которые полезны, но не обязательны для первой версии loader.
    optional_columns: tuple[str, ...] = ()


ROUTER_TRAINING_CONTRACT = DatasetContract(
    relation_name="lab.v_gaia_router_training",
    required_columns=(
        *IDENTIFIER_COLUMNS,
        "ra",
        "dec",
        *BASE_STELLAR_FEATURES,
        "spec_class",
        "evolution_stage",
    ),
    optional_columns=(
        "spec_subclass",
        "parallax",
        "parallax_over_error",
        "ruwe",
        *ADDITIONAL_PHYSICAL_FEATURES,
        "source_type",
        "random_index",
    ),
)

HOST_TRAINING_CONTRACT = DatasetContract(
    relation_name="lab.nasa_gaia_host_training_enriched",
    required_columns=(
        *IDENTIFIER_COLUMNS,
        "teff_gspphot",
        "logg_gspphot",
        HOST_PRIORITY_CANONICAL_RADIUS_COLUMN,
        "spec_class",
        "evolution_stage",
    ),
    optional_columns=(
        "hostname",
        "spec_subclass",
        "ra_gaia",
        "dec_gaia",
        "dist_arcsec",
        "parallax",
        "parallax_over_error",
        "ruwe",
        "phot_g_mean_mag",
        "bp_rp",
        "mh_gspphot",
        "radius_gspphot",
        "lum_flame",
        "evolstage_flame",
        "non_single_star",
        "classprob_dsc_combmod_star",
        *QUALITY_FEATURES,
    ),
)


def missing_contract_columns(
    contract: DatasetContract,
    available_columns: set[str],
) -> tuple[str, ...]:
    # Возвращаем только действительно отсутствующие обязательные колонки.
    return tuple(
        column_name
        for column_name in contract.required_columns
        if column_name not in available_columns
    )


def select_contract_columns(
    contract: DatasetContract,
    available_columns: set[str],
) -> tuple[str, ...]:
    # Собираем порядок колонок строго по контракту:
    # сначала обязательные, потом доступные необязательные.
    selected_columns: list[str] = []
    seen_columns: set[str] = set()

    for column_name in contract.required_columns:
        if column_name in seen_columns:
            continue
        seen_columns.add(column_name)
        selected_columns.append(column_name)

    for column_name in contract.optional_columns:
        if column_name not in available_columns or column_name in seen_columns:
            continue
        seen_columns.add(column_name)
        selected_columns.append(column_name)

    return tuple(selected_columns)
