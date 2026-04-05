# Файл `load_gaia_id_ood_training_dataset.py` слоя `datasets`.
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
from exohost.contracts.hierarchical_dataset_contracts import (
    GAIA_ID_OOD_TRAINING_CONTRACT,
)
from exohost.db.relations import relation_columns, relation_exists, split_relation_name


def build_gaia_id_ood_training_query(
    relation_name: str,
    selected_columns: tuple[str, ...],
    *,
    limit: int | None = None,
) -> str:
    # Собираем SQL для binary ID/OOD view без implicit фильтров.
    schema_name, table_name = split_relation_name(
        relation_name,
        validate_identifiers=True,
    )
    selected_columns_sql = ",\n        ".join(selected_columns)
    limit_sql = f"LIMIT {int(limit)}" if limit is not None else ""
    order_by_sql = (
        "random_index ASC, domain_target ASC, source_id ASC"
        if "random_index" in selected_columns
        else "source_id ASC"
    )

    return f"""
    SELECT
        {selected_columns_sql}
    FROM {schema_name}.{table_name}
    WHERE source_id IS NOT NULL
      AND domain_target IN ('id', 'ood')
      AND teff_gspphot IS NOT NULL
      AND logg_gspphot IS NOT NULL
      AND mh_gspphot IS NOT NULL
      AND bp_rp IS NOT NULL
      AND parallax IS NOT NULL
      AND parallax_over_error IS NOT NULL
      AND ruwe IS NOT NULL
    ORDER BY {order_by_sql}
    {limit_sql};
    """


def load_gaia_id_ood_training_dataset(
    engine: Engine,
    *,
    contract: DatasetContract = GAIA_ID_OOD_TRAINING_CONTRACT,
    limit: int | None = None,
) -> pd.DataFrame:
    # Загружаем ID-vs-OOD training view по зафиксированному contract.
    if not relation_exists(engine, contract.relation_name, validate_identifiers=True):
        raise RuntimeError(
            f"ID/OOD training source does not exist: {contract.relation_name}"
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
            "ID/OOD training source is missing required columns: "
            f"{missing_columns_sql}"
        )

    selected_columns = select_contract_columns(contract, available_columns)
    query = build_gaia_id_ood_training_query(
        contract.relation_name,
        selected_columns,
        limit=limit,
    )
    return pd.read_sql(query, engine).reset_index(drop=True)
