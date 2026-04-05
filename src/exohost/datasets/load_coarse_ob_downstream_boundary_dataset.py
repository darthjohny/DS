# Файл `load_coarse_ob_downstream_boundary_dataset.py` слоя `datasets`.
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

from exohost.contracts.dataset_contracts import DatasetContract, select_contract_columns
from exohost.contracts.quality_gate_dataset_contracts import (
    GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT,
)
from exohost.db.relations import relation_columns, relation_exists, split_relation_name


def build_coarse_ob_downstream_boundary_query(
    relation_name: str,
    selected_columns: tuple[str, ...],
    *,
    quality_state: str = "pass",
    teff_min_k: float = 10_000.0,
    limit: int | None = None,
) -> str:
    # Собираем SQL для downstream hot pass `O/B` boundary source.
    schema_name, table_name = split_relation_name(
        relation_name,
        validate_identifiers=True,
    )
    selected_columns_sql = ",\n        ".join(selected_columns)
    order_by_sql = (
        "random_index ASC, source_id ASC" if "random_index" in selected_columns else "source_id ASC"
    )
    limit_sql = f"LIMIT {int(limit)}" if limit is not None else ""

    return f"""
    SELECT
        {selected_columns_sql}
    FROM {schema_name}.{table_name}
    WHERE source_id IS NOT NULL
      AND spectral_class IN ('O', 'B')
      AND quality_state = '{quality_state}'
      AND teff_gspphot >= {teff_min_k:.1f}
    ORDER BY {order_by_sql}
    {limit_sql};
    """


def load_coarse_ob_downstream_boundary_dataset(
    engine: Engine,
    *,
    contract: DatasetContract = GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT,
    quality_state: str = "pass",
    teff_min_k: float = 10_000.0,
    limit: int | None = None,
) -> pd.DataFrame:
    # Загружаем downstream hot pass `O/B` boundary source из quality-gated relation.
    if not relation_exists(engine, contract.relation_name, validate_identifiers=True):
        raise RuntimeError(
            "Coarse O/B downstream boundary source does not exist: "
            f"{contract.relation_name}"
        )

    available_columns = set(
        relation_columns(
            engine,
            contract.relation_name,
            validate_identifiers=True,
        )
    )
    required_columns = set(contract.required_columns) | {"spectral_class", "teff_gspphot"}
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        missing_columns_sql = ", ".join(missing_columns)
        raise RuntimeError(
            "Coarse O/B downstream boundary source is missing required columns: "
            f"{missing_columns_sql}"
        )

    selected_columns = select_contract_columns(contract, available_columns | {"spectral_class"})
    if "spectral_class" not in selected_columns:
        selected_columns = (*selected_columns, "spectral_class")

    query = build_coarse_ob_downstream_boundary_query(
        contract.relation_name,
        selected_columns,
        quality_state=quality_state,
        teff_min_k=teff_min_k,
        limit=limit,
    )
    return pd.read_sql(query, engine).reset_index(drop=True)
