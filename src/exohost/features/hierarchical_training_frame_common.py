# Файл `hierarchical_training_frame_common.py` слоя `features`.
#
# Этот файл отвечает только за:
# - подготовку признаков и training frame-слой;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `features` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from typing import Any

import pandas as pd

from exohost.contracts.dataset_contracts import DatasetContract, select_contract_columns
from exohost.contracts.label_contract import normalize_spectral_subclass
from exohost.features.hierarchical_training_frame_contracts import (
    DOMAIN_TARGET_VALUES,
    SPECTRAL_SUBCLASS_DIGITS,
)


def select_available_columns(
    df: pd.DataFrame,
    *,
    contract: DatasetContract,
) -> pd.DataFrame:
    # Сохраняем только колонки, входящие в contract, без скрытых перестановок.
    selected_columns = select_contract_columns(
        contract,
        set(str(column_name) for column_name in df.columns),
    )
    return df.loc[:, [name for name in selected_columns if name in df.columns]].copy()


def normalize_optional_text(value: object) -> object:
    # Сохраняем nullable-текст как есть, но без лишних пробелов.
    if value is None or value is pd.NA:
        return pd.NA
    if isinstance(value, float) and pd.isna(value):
        return pd.NA
    normalized_value = str(value).strip()
    return normalized_value if normalized_value else pd.NA


def normalize_domain_target(value: object) -> str:
    # Приводим binary domain target к каноническому lower-case виду.
    return str(value).strip().lower()


def normalize_refinement_subclass(spec_class: object, subclass_value: object) -> object:
    # Приводим refinement subclass к каноническому виду:
    # либо уже полный код вида G2, либо digit-only код из view + spec_class.
    if subclass_value is None or subclass_value is pd.NA:
        return pd.NA
    normalized_class = str(spec_class).strip().upper()
    normalized_subclass = str(subclass_value).strip().upper()
    if not normalized_subclass:
        return pd.NA
    if normalized_subclass in SPECTRAL_SUBCLASS_DIGITS:
        return f"{normalized_class}{normalized_subclass}"
    return normalize_spectral_subclass(normalized_subclass)


def filter_minimum_label_support(
    df: pd.DataFrame,
    *,
    target_column: str,
    min_count: int,
) -> pd.DataFrame:
    # Убираем редкие метки, которые не выдерживают first-wave split/CV policy.
    target_counts = df.loc[:, target_column].astype(str).value_counts()
    supported_targets = set(
        target_counts.loc[target_counts >= min_count].index.astype(str).tolist()
    )
    return (
        df.loc[df.loc[:, target_column].astype(str).isin(supported_targets)]
        .reset_index(drop=True)
        .copy()
    )


def validate_domain_target_values(df: pd.DataFrame, *, frame_name: str) -> None:
    invalid_targets = sorted(
        {
            str(value)
            for value in df["domain_target"].dropna().unique().tolist()
            if str(value) not in DOMAIN_TARGET_VALUES
        }
    )
    if invalid_targets:
        sample = ", ".join(invalid_targets[:5])
        raise ValueError(
            f"{frame_name} contains unsupported domain_target values: {sample}"
        )


def bool_to_evolution_stage(value: object) -> str:
    # Для coarse source boolean-флаг переводим в legacy-compatible stage.
    return "evolved" if bool(value) else "dwarf"


def to_optional_float(value: object) -> float | None:
    # Приводим scalar к float без implicit pandas promotion.
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            return None
        try:
            return float(stripped_value)
        except ValueError:
            return None
    return None


def max_numeric_value(values: pd.Series | object) -> float | None:
    # Приводим selector/count values к scalar max без pandas-магии в типах.
    raw_values = values.tolist() if isinstance(values, pd.Series) else [values]
    numeric_values = [
        numeric_value
        for numeric_value in (to_optional_float(value) for value in raw_values)
        if numeric_value is not None
    ]
    if not numeric_values:
        return None
    return max(numeric_values)


def normalize_record_mapping(row: Mapping[Hashable, object]) -> dict[str, Any]:
    return {str(key): value for key, value in row.items()}


__all__ = [
    "bool_to_evolution_stage",
    "filter_minimum_label_support",
    "max_numeric_value",
    "normalize_domain_target",
    "normalize_optional_text",
    "normalize_record_mapping",
    "normalize_refinement_subclass",
    "select_available_columns",
    "to_optional_float",
    "validate_domain_target_values",
]
