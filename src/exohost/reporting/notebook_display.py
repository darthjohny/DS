# Файл `notebook_display.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pandas as pd

# Для display-слоя notebook намеренно ослабляем тип внутренних словарей.
# Снаружи сюда приходят обычные `dict[str, str]`, `dict[bool, str]`
# и похожие пользовательские маппинги из notebook. Это динамическая UI-граница,
# поэтому держим вход как Mapping[str, object] и валидируем его в рантайме,
# чтобы не тащить ложные ошибки Pylance/Pyright в исследовательский слой.
DisplayLabelMapping = Mapping[Any, object]
DisplayValueMappings = Mapping[str, object]


def rename_frame_for_display(
    df: pd.DataFrame,
    *,
    column_mapping: Mapping[str, str],
    value_mapping: DisplayValueMappings | None = None,
) -> pd.DataFrame:
    # Переименовываем колонки и при необходимости переводим значения для вывода.
    result = df.copy()
    if value_mapping is not None:
        for column_name, raw_label_mapping in value_mapping.items():
            if column_name in result.columns:
                label_mapping = _require_display_mapping(raw_label_mapping)
                result[column_name] = result[column_name].map(
                    lambda value, current_mapping=label_mapping: _map_display_value(
                        value,
                        current_mapping,
                    )
                )
    return result.rename(columns=dict(column_mapping))


def scalar_to_int(value: object) -> int:
    # Безопасно приводим pandas/numpy-скаляры к обычному целому значению.
    item_method = getattr(value, "item", None)
    scalar = item_method() if callable(item_method) else value
    if isinstance(scalar, bool):
        return int(scalar)
    if isinstance(scalar, int):
        return scalar
    if isinstance(scalar, float):
        return int(scalar)
    if isinstance(scalar, str):
        return int(scalar)
    raise TypeError(f"Expected integer-like scalar, got {type(value)!r}.")


def _map_display_value(
    value: object,
    label_mapping: DisplayLabelMapping,
) -> object:
    # Возвращаем подпись для значения, если она определена; иначе оставляем исходное.
    return label_mapping.get(value, value)


def _require_display_mapping(value: object) -> DisplayLabelMapping:
    # На notebook-границе принимаем только mapping-подобные словари подписей.
    if not isinstance(value, Mapping):
        raise TypeError("value_mapping values must be mapping-like objects.")
    return cast(DisplayLabelMapping, value)


__all__ = [
    "DisplayLabelMapping",
    "DisplayValueMappings",
    "rename_frame_for_display",
    "scalar_to_int",
]
