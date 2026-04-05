# Файл `load_host_training_dataset.py` слоя `datasets`.
#
# Этот файл отвечает только за:
# - loader-слой и сборку рабочих dataframe из relation-слоя;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `datasets` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from exohost.contracts.dataset_contracts import (
    HOST_TRAINING_CONTRACT,
    DatasetContract,
    missing_contract_columns,
    select_contract_columns,
)
from exohost.contracts.host_priority_feature_contracts import (
    HOST_PRIORITY_CANONICAL_RADIUS_COLUMN,
)
from exohost.contracts.label_contract import SOURCE_EVOLUTION_STAGES, SPECTRAL_CLASSES
from exohost.db.relations import relation_columns, relation_exists, split_relation_name


def build_host_training_query(
    relation_name: str,
    selected_columns: tuple[str, ...],
    *,
    limit: int | None = None,
) -> str:
    # Собираем SQL для выборки host training source без лишней логики.
    schema_name, table_name = split_relation_name(
        relation_name,
        validate_identifiers=True,
    )
    limit_sql = f"LIMIT {int(limit)}" if limit is not None else ""
    selected_columns_sql = ",\n        ".join(selected_columns)
    supported_classes_sql = ", ".join(f"'{label}'" for label in SPECTRAL_CLASSES)
    supported_stages_sql = ", ".join(f"'{label}'" for label in SOURCE_EVOLUTION_STAGES)

    return f"""
    SELECT
        {selected_columns_sql}
    FROM {schema_name}.{table_name}
    WHERE source_id IS NOT NULL
      AND teff_gspphot IS NOT NULL
      AND logg_gspphot IS NOT NULL
      AND {HOST_PRIORITY_CANONICAL_RADIUS_COLUMN} IS NOT NULL
      AND spec_class IN ({supported_classes_sql})
      AND evolution_stage IN ({supported_stages_sql})
    ORDER BY source_id ASC
    {limit_sql};
    """


def load_host_training_dataset(
    engine: Engine,
    *,
    contract: DatasetContract = HOST_TRAINING_CONTRACT,
    limit: int | None = None,
) -> pd.DataFrame:
    # Загружаем host training relation по явному контракту.
    if not relation_exists(engine, contract.relation_name, validate_identifiers=True):
        raise RuntimeError(
            f"Host training source does not exist: {contract.relation_name}"
        )

    available_columns = set(
        relation_columns(
            engine,
            contract.relation_name,
            validate_identifiers=True,
        )
    )
    missing_columns = missing_contract_columns(contract, available_columns)
    if missing_columns:
        missing_columns_sql = ", ".join(missing_columns)
        raise RuntimeError(
            "Host training source is missing required columns: "
            f"{missing_columns_sql}"
        )

    selected_columns = select_contract_columns(contract, available_columns)
    query = build_host_training_query(
        contract.relation_name,
        selected_columns,
        limit=limit,
    )
    return pd.read_sql(query, engine).reset_index(drop=True)
