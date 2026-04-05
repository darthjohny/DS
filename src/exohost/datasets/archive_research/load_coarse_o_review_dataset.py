# Loader narrow `O`-class source для coarse rare-tail review.

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from exohost.contracts.dataset_contracts import (
    DatasetContract,
    select_contract_columns,
)
from exohost.contracts.quality_gate_dataset_contracts import (
    GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT,
)
from exohost.db.relations import relation_columns, relation_exists, split_relation_name


def build_coarse_o_review_query(
    relation_name: str,
    selected_columns: tuple[str, ...],
    *,
    quality_state: str | None = None,
    limit: int | None = None,
) -> str:
    # Собираем устойчивый SQL только для true `O` rows из quality-gated source.
    schema_name, table_name = split_relation_name(
        relation_name,
        validate_identifiers=True,
    )
    selected_columns_sql = ",\n        ".join(selected_columns)
    quality_filter_sql = (
        f"AND quality_state = '{quality_state}'" if quality_state is not None else ""
    )
    order_by_sql = (
        "random_index ASC, source_id ASC" if "random_index" in selected_columns else "source_id ASC"
    )
    limit_sql = f"LIMIT {int(limit)}" if limit is not None else ""

    return f"""
    SELECT
        {selected_columns_sql}
    FROM {schema_name}.{table_name}
    WHERE source_id IS NOT NULL
      AND spectral_class = 'O'
      {quality_filter_sql}
    ORDER BY {order_by_sql}
    {limit_sql};
    """


def load_coarse_o_review_dataset(
    engine: Engine,
    *,
    contract: DatasetContract = GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT,
    quality_state: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    # Загружаем только `O`-rows из live quality-gated relation для rare-tail review.
    if not relation_exists(engine, contract.relation_name, validate_identifiers=True):
        raise RuntimeError(f"Coarse O review source does not exist: {contract.relation_name}")

    available_columns = set(
        relation_columns(
            engine,
            contract.relation_name,
            validate_identifiers=True,
        )
    )
    required_columns = set(contract.required_columns) | {"spectral_class"}
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        missing_columns_sql = ", ".join(missing_columns)
        raise RuntimeError(
            "Coarse O review source is missing required columns: "
            f"{missing_columns_sql}"
        )

    selected_columns = select_contract_columns(
        contract,
        available_columns | {"spectral_class"},
    )
    if "spectral_class" not in selected_columns:
        selected_columns = (*selected_columns, "spectral_class")

    query = build_coarse_o_review_query(
        contract.relation_name,
        selected_columns,
        quality_state=quality_state,
        limit=limit,
    )
    return pd.read_sql(query, engine).reset_index(drop=True)
