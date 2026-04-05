# Тестовый файл `test_notebook_display.py` домена `reporting`.
#
# Этот файл проверяет только:
# - проверку логики домена: review-helper, сводки и notebook display-слой;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `reporting` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import numpy as np
import pandas as pd

from exohost.reporting.notebook_display import (
    rename_frame_for_display,
    scalar_to_int,
)


def test_rename_frame_for_display_renames_columns_and_values() -> None:
    df = pd.DataFrame(
        {
            "flag": [True, False],
            "group": ["a", "b"],
            "value": [1, 2],
        }
    )

    result = rename_frame_for_display(
        df,
        column_mapping={
            "flag": "Флаг",
            "group": "Группа",
            "value": "Значение",
        },
        value_mapping={
            "flag": {True: "Да", False: "Нет"},
            "group": {"a": "Группа A", "b": "Группа B"},
        },
    )

    assert list(result.columns) == ["Флаг", "Группа", "Значение"]
    assert result["Флаг"].tolist() == ["Да", "Нет"]
    assert result["Группа"].tolist() == ["Группа A", "Группа B"]
    assert result["Значение"].tolist() == [1, 2]


def test_rename_frame_for_display_ignores_missing_mapping_column() -> None:
    df = pd.DataFrame({"name": ["x"], "value": [1]})

    result = rename_frame_for_display(
        df,
        column_mapping={"name": "Имя", "value": "Значение"},
        value_mapping={"missing": {"x": "ignored"}},
    )

    assert list(result.columns) == ["Имя", "Значение"]
    assert result["Имя"].tolist() == ["x"]


def test_rename_frame_for_display_accepts_regular_string_and_boolean_dicts() -> None:
    df = pd.DataFrame({"state": ["pass", "reject"], "flag": [True, False]})

    result = rename_frame_for_display(
        df,
        column_mapping={"state": "Состояние", "flag": "Флаг"},
        value_mapping={
            "state": {"pass": "Допуск", "reject": "Отклонено"},
            "flag": {True: "Да", False: "Нет"},
        },
    )

    assert result["Состояние"].tolist() == ["Допуск", "Отклонено"]
    assert result["Флаг"].tolist() == ["Да", "Нет"]


def test_scalar_to_int_accepts_numpy_and_python_scalars() -> None:
    assert scalar_to_int(3) == 3
    assert scalar_to_int(np.int64(4)) == 4
    assert scalar_to_int(np.float64(5.9)) == 5
    assert scalar_to_int(True) == 1
