# Файл `load_quality_gate_audit_dataset.py` слоя `datasets`.
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
from exohost.contracts.quality_gate_dataset_contracts import (
    GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT,
)
from exohost.db.relations import relation_columns, relation_exists, split_relation_name


def build_quality_gate_audit_query(
    relation_name: str,
    selected_columns: tuple[str, ...],
    *,
    limit: int | None = None,
) -> str:
    # Собираем устойчивый SQL для quality-gate audit relation без notebook-join логики.
    schema_name, table_name = split_relation_name(
        relation_name,
        validate_identifiers=True,
    )
    selected_columns_sql = ",\n        ".join(selected_columns)
    order_by_sql = (
        "random_index ASC, quality_state ASC, source_id ASC"
        if "random_index" in selected_columns
        else "quality_state ASC, source_id ASC"
    )
    limit_sql = f"LIMIT {int(limit)}" if limit is not None else ""

    return f"""
    SELECT
        {selected_columns_sql}
    FROM {schema_name}.{table_name}
    WHERE source_id IS NOT NULL
      AND quality_state IS NOT NULL
      AND ood_state IS NOT NULL
    ORDER BY {order_by_sql}
    {limit_sql};
    """


def load_quality_gate_audit_dataset(
    engine: Engine,
    *,
    contract: DatasetContract = GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT,
    limit: int | None = None,
) -> pd.DataFrame:
    # Загружаем quality-gate audit source по явному relation contract.
    if not relation_exists(engine, contract.relation_name, validate_identifiers=True):
        raise RuntimeError(
            f"Quality-gate audit source does not exist: {contract.relation_name}"
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
            "Quality-gate audit source is missing required columns: "
            f"{missing_columns_sql}"
        )

    selected_columns = select_contract_columns(contract, available_columns)
    query = build_quality_gate_audit_query(
        contract.relation_name,
        selected_columns,
        limit=limit,
    )
    return pd.read_sql(query, engine).reset_index(drop=True)
