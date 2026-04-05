# Файл `load_model_scoring_dataset.py` слоя `datasets`.
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
    DatasetContract,
    missing_contract_columns,
    select_contract_columns,
)
from exohost.contracts.feature_contract import (
    IDENTIFIER_COLUMNS,
    OBSERVABILITY_FEATURES,
    QUALITY_FEATURES,
    unique_columns,
)
from exohost.db.relations import relation_columns, relation_exists, split_relation_name

MODEL_SCORING_OPTIONAL_COLUMNS: tuple[str, ...] = unique_columns(
    OBSERVABILITY_FEATURES,
    QUALITY_FEATURES,
    ("spec_class", "evolution_stage", "spec_subclass", "source_type", "random_index"),
)


def build_model_scoring_contract(
    relation_name: str,
    *,
    feature_columns: tuple[str, ...],
) -> DatasetContract:
    # Собираем узкий контракт для model-scoring relation.
    return DatasetContract(
        relation_name=relation_name,
        required_columns=unique_columns(IDENTIFIER_COLUMNS, feature_columns),
        optional_columns=MODEL_SCORING_OPTIONAL_COLUMNS,
    )


def build_model_scoring_query(
    relation_name: str,
    selected_columns: tuple[str, ...],
    *,
    feature_columns: tuple[str, ...],
    limit: int | None = None,
) -> str:
    # Собираем явный SQL для inference-выборки под сохраненную модель.
    schema_name, table_name = split_relation_name(
        relation_name,
        validate_identifiers=True,
    )
    selected_columns_sql = ",\n        ".join(selected_columns)
    limit_sql = f"LIMIT {int(limit)}" if limit is not None else ""
    order_by_sql = "random_index ASC, source_id ASC" if "random_index" in selected_columns else "source_id ASC"
    non_null_checks_sql = "\n      AND ".join(f"{column_name} IS NOT NULL" for column_name in feature_columns)

    return f"""
    SELECT
        {selected_columns_sql}
    FROM {schema_name}.{table_name}
    WHERE source_id IS NOT NULL
      AND {non_null_checks_sql}
    ORDER BY {order_by_sql}
    {limit_sql};
    """


def load_model_scoring_dataset(
    engine: Engine,
    *,
    relation_name: str,
    feature_columns: tuple[str, ...],
    limit: int | None = None,
) -> pd.DataFrame:
    # Загружаем inference-датасет для применения сохраненной модели.
    contract = build_model_scoring_contract(
        relation_name,
        feature_columns=feature_columns,
    )
    if not relation_exists(engine, contract.relation_name, validate_identifiers=True):
        raise RuntimeError(
            f"Model scoring source does not exist: {contract.relation_name}"
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
            "Model scoring source is missing required columns: "
            f"{missing_columns_sql}"
        )

    selected_columns = select_contract_columns(contract, available_columns)
    query = build_model_scoring_query(
        contract.relation_name,
        selected_columns,
        feature_columns=feature_columns,
        limit=limit,
    )
    return pd.read_sql(query, engine).reset_index(drop=True)
