# Файл `load_gaia_mk_refinement_family_training_dataset.py` слоя `datasets`.
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
    missing_contract_columns,
    select_contract_columns,
)
from exohost.contracts.refinement_family_dataset_contracts import (
    build_gaia_mk_refinement_family_training_contract,
    validate_refinement_family_class,
)
from exohost.db.relations import relation_columns, relation_exists, split_relation_name


def build_gaia_mk_refinement_family_training_query(
    relation_name: str,
    selected_columns: tuple[str, ...],
    *,
    limit: int | None = None,
) -> str:
    # Собираем явный SQL для одного family-view без повторной task-логики.
    schema_name, table_name = split_relation_name(
        relation_name,
        validate_identifiers=True,
    )
    selected_columns_sql = ",\n        ".join(selected_columns)
    limit_sql = f"LIMIT {int(limit)}" if limit is not None else ""
    order_by_sql = (
        "random_index ASC, source_id ASC" if "random_index" in selected_columns else "source_id ASC"
    )
    return f"""
    SELECT
        {selected_columns_sql}
    FROM {schema_name}.{table_name}
    WHERE source_id IS NOT NULL
      AND spectral_subclass IS NOT NULL
    ORDER BY {order_by_sql}
    {limit_sql};
    """


def load_gaia_mk_refinement_family_training_dataset(
    engine: Engine,
    *,
    spectral_class: str,
    limit: int | None = None,
) -> pd.DataFrame:
    # Загружаем один family-view по зафиксированному dataset contract.
    normalized_class = validate_refinement_family_class(spectral_class)
    contract = build_gaia_mk_refinement_family_training_contract(normalized_class)
    if not relation_exists(engine, contract.relation_name, validate_identifiers=True):
        raise RuntimeError(
            f"Refinement family training source does not exist: {contract.relation_name}"
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
            "Refinement family training source is missing required columns: "
            f"{missing_columns_sql}"
        )

    selected_columns = select_contract_columns(contract, available_columns)
    query = build_gaia_mk_refinement_family_training_query(
        contract.relation_name,
        selected_columns,
        limit=limit,
    )
    return pd.read_sql(query, engine).reset_index(drop=True)
