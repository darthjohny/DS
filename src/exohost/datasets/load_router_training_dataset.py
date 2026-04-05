# Файл `load_router_training_dataset.py` слоя `datasets`.
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
    ROUTER_TRAINING_CONTRACT,
    DatasetContract,
    missing_contract_columns,
    select_contract_columns,
)
from exohost.contracts.label_contract import EVOLUTION_STAGES, SPECTRAL_CLASSES
from exohost.db.relations import relation_columns, relation_exists, split_relation_name


def build_router_training_query(
    relation_name: str,
    selected_columns: tuple[str, ...],
    *,
    limit: int | None = None,
) -> str:
    # Собираем простой и явный SQL для router training source.
    schema_name, table_name = split_relation_name(
        relation_name,
        validate_identifiers=True,
    )
    limit_sql = f"LIMIT {int(limit)}" if limit is not None else ""
    selected_columns_sql = ",\n        ".join(selected_columns)
    supported_classes_sql = ", ".join(f"'{label}'" for label in SPECTRAL_CLASSES)
    supported_stages_sql = ", ".join(f"'{label}'" for label in EVOLUTION_STAGES)
    order_by_sql = "random_index ASC, source_id ASC" if limit is not None else "source_id ASC"

    return f"""
    SELECT
        {selected_columns_sql}
    FROM {schema_name}.{table_name}
    WHERE source_id IS NOT NULL
      AND teff_gspphot IS NOT NULL
      AND logg_gspphot IS NOT NULL
      AND radius_gspphot IS NOT NULL
      AND spec_class IN ({supported_classes_sql})
      AND evolution_stage IN ({supported_stages_sql})
    ORDER BY {order_by_sql}
    {limit_sql};
    """


def load_router_training_dataset(
    engine: Engine,
    *,
    contract: DatasetContract = ROUTER_TRAINING_CONTRACT,
    limit: int | None = None,
) -> pd.DataFrame:
    # Загружаем и валидируем router training source по зафиксированному контракту.
    if not relation_exists(engine, contract.relation_name, validate_identifiers=True):
        raise RuntimeError(
            f"Router training source does not exist: {contract.relation_name}"
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
            "Router training source is missing required columns: "
            f"{missing_columns_sql}"
        )

    selected_columns = select_contract_columns(contract, available_columns)
    query = build_router_training_query(
        contract.relation_name,
        selected_columns,
        limit=limit,
    )
    return pd.read_sql(query, engine).reset_index(drop=True)
