# Файл `load_final_decision_input_dataset.py` слоя `datasets`.
#
# Этот файл отвечает только за:
# - loader-слой и сборку рабочих dataframe из relation-слоя;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `datasets` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sqlalchemy.engine import Engine

from exohost.contracts.dataset_contracts import (
    DatasetContract,
)
from exohost.contracts.feature_contract import (
    IDENTIFIER_COLUMNS,
    OBSERVABILITY_FEATURES,
    QUALITY_FEATURES,
    unique_columns,
)
from exohost.db.relations import relation_columns, relation_exists, split_relation_name

FINAL_DECISION_REQUIRED_COLUMNS: tuple[str, ...] = ("quality_state",)
FINAL_DECISION_OPTIONAL_COLUMNS: tuple[str, ...] = unique_columns(
    OBSERVABILITY_FEATURES,
    QUALITY_FEATURES,
    (
        "spec_class",
        "evolution_stage",
        "spec_subclass",
        "source_type",
        "radius_flame",
        "has_core_features",
        "has_flame_features",
        "has_missing_core_features",
        "has_missing_flame_features",
        "has_high_ruwe",
        "has_low_parallax_snr",
        "random_index",
        "quality_reason",
        "review_bucket",
        "ood_state",
        "ood_reason",
        "quality_gate_version",
        "quality_gated_at_utc",
    ),
)
FINAL_DECISION_FEATURE_COMPATIBILITY_SOURCES: dict[str, tuple[str, ...]] = {
    "radius_feature": ("radius_flame", "radius_gspphot"),
    "radius_gspphot": ("radius_flame",),
}


def build_final_decision_input_contract(
    relation_name: str,
    *,
    feature_columns: tuple[str, ...],
) -> DatasetContract:
    # Собираем узкий contract для decision pipeline input relation.
    return DatasetContract(
        relation_name=relation_name,
        required_columns=unique_columns(
            IDENTIFIER_COLUMNS,
            FINAL_DECISION_REQUIRED_COLUMNS,
            feature_columns,
        ),
        optional_columns=FINAL_DECISION_OPTIONAL_COLUMNS,
    )


def build_final_decision_input_query(
    relation_name: str,
    selected_columns: tuple[str, ...],
    *,
    limit: int | None = None,
) -> str:
    # Собираем явный SQL для decision input relation.
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
      AND quality_state IS NOT NULL
    ORDER BY {order_by_sql}
    {limit_sql};
    """


def resolve_final_decision_missing_columns(
    *,
    required_columns: tuple[str, ...],
    feature_columns: tuple[str, ...],
    available_columns: set[str],
) -> tuple[str, ...]:
    # Проверяем contract c учетом явных compatibility aliases для feature names.
    missing_columns: list[str] = []
    feature_column_names = set(feature_columns)
    for column_name in required_columns:
        if column_name in available_columns:
            continue
        if (
            column_name in feature_column_names
            and resolve_final_decision_feature_source_column(
                column_name,
                available_columns=available_columns,
            )
            is not None
        ):
            continue
        missing_columns.append(column_name)
    return tuple(missing_columns)


def resolve_final_decision_feature_source_column(
    column_name: str,
    *,
    available_columns: set[str],
) -> str | None:
    # Ищем физический source-column для feature с учетом compatibility policy.
    if column_name in available_columns:
        return column_name
    for source_column in FINAL_DECISION_FEATURE_COMPATIBILITY_SOURCES.get(
        column_name,
        (),
    ):
        if source_column in available_columns:
            return source_column
    return None


def build_final_decision_selected_columns(
    *,
    contract: DatasetContract,
    available_columns: set[str],
    feature_columns: tuple[str, ...],
) -> tuple[str, ...]:
    # Выбираем relation columns и при необходимости добавляем alias-source поля.
    selected_columns: list[str] = []
    selected_column_names: set[str] = set()
    feature_column_names = set(feature_columns)

    for column_name in contract.required_columns:
        if column_name in feature_column_names and column_name not in available_columns:
            continue
        if column_name not in available_columns or column_name in selected_column_names:
            continue
        selected_columns.append(column_name)
        selected_column_names.add(column_name)

    for column_name in contract.optional_columns:
        if column_name not in available_columns or column_name in selected_column_names:
            continue
        selected_columns.append(column_name)
        selected_column_names.add(column_name)

    for feature_column in feature_columns:
        source_column = resolve_final_decision_feature_source_column(
            feature_column,
            available_columns=available_columns,
        )
        if source_column is None or source_column in selected_column_names:
            continue
        selected_columns.append(source_column)
        selected_column_names.add(source_column)
    return tuple(selected_columns)


def apply_final_decision_feature_aliases(
    df: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    # Собираем legacy feature aliases поверх канонических relation columns.
    if df.empty:
        return df.copy()

    result = df.copy()
    for feature_column in feature_columns:
        if feature_column in result.columns:
            continue
        source_column = _resolve_alias_source_from_frame(
            result.columns,
            feature_column=feature_column,
        )
        if source_column is None:
            continue
        result[feature_column] = result[source_column]
    return result


def _resolve_alias_source_from_frame(
    available_columns: Iterable[object],
    *,
    feature_column: str,
) -> str | None:
    available_column_names = {str(column_name) for column_name in available_columns}
    return resolve_final_decision_feature_source_column(
        feature_column,
        available_columns=available_column_names,
    )


def load_final_decision_input_dataset(
    engine: Engine,
    *,
    relation_name: str,
    feature_columns: tuple[str, ...],
    limit: int | None = None,
) -> pd.DataFrame:
    # Загружаем input frame для final decision pipeline.
    contract = build_final_decision_input_contract(
        relation_name,
        feature_columns=feature_columns,
    )
    if not relation_exists(engine, contract.relation_name, validate_identifiers=True):
        raise RuntimeError(
            f"Final decision source does not exist: {contract.relation_name}"
        )

    available_columns = set(
        relation_columns(
            engine,
            contract.relation_name,
            validate_identifiers=True,
        )
    )
    missing_columns = resolve_final_decision_missing_columns(
        required_columns=contract.required_columns,
        feature_columns=feature_columns,
        available_columns=available_columns,
    )
    if missing_columns:
        missing_columns_sql = ", ".join(missing_columns)
        raise RuntimeError(
            "Final decision source is missing required columns: "
            f"{missing_columns_sql}"
        )

    selected_columns = build_final_decision_selected_columns(
        contract=contract,
        available_columns=available_columns,
        feature_columns=feature_columns,
    )
    query = build_final_decision_input_query(
        contract.relation_name,
        selected_columns,
        limit=limit,
    )
    loaded_df = pd.read_sql(query, engine).reset_index(drop=True)
    return apply_final_decision_feature_aliases(
        loaded_df,
        feature_columns=feature_columns,
    )
