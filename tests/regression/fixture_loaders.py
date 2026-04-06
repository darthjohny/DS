# Helpers чтения frozen fixtures для регресс-слоя.
#
# Этот файл отвечает только за:
# - чтение маленьких CSV и JSON fixtures из `tests/regression/fixtures`;
# - минимальную валидацию путей и форматов перед использованием в тестах.
#
# Следующий слой:
# - regression-тесты доменов `decision`, `posthoc` и `reporting`;
# - assertions и testkit этого же регресс-контура.

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd


def load_regression_csv_fixture(csv_path: Path) -> pd.DataFrame:
    # Загружаем маленький CSV fixture и сразу проверяем его расширение и наличие файла.
    _require_existing_path(csv_path, expected_suffix=".csv")
    return pd.read_csv(csv_path)


def load_regression_json_fixture(json_path: Path) -> Mapping[str, Any]:
    # Загружаем маленький JSON fixture для expected-структур и schema-check.
    _require_existing_path(json_path, expected_suffix=".json")
    loaded_payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(loaded_payload, dict):
        raise TypeError("Regression JSON fixture must contain a top-level object.")
    return loaded_payload


def _require_existing_path(path: Path, *, expected_suffix: str) -> None:
    if path.suffix != expected_suffix:
        raise ValueError(f"Expected {expected_suffix} fixture, got: {path.name}")
    if not path.exists():
        raise FileNotFoundError(path)


__all__ = [
    "load_regression_csv_fixture",
    "load_regression_json_fixture",
]
