# Тестовый файл `test_priority_score_scalars.py` домена `ranking`.
#
# Этот файл проверяет только:
# - проверку логики домена: priority- и observability-логики;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `ranking` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import math

import pandas as pd
import pytest

from exohost.ranking.priority_score import coerce_optional_float, is_missing_scalar


def test_is_missing_scalar_recognizes_standard_missing_values() -> None:
    assert is_missing_scalar(None) is True
    assert is_missing_scalar(pd.NA) is True
    assert is_missing_scalar(float("nan")) is True
    assert is_missing_scalar("1.5") is False


def test_coerce_optional_float_converts_supported_values() -> None:
    assert coerce_optional_float("1.5") == 1.5
    assert coerce_optional_float(2) == 2.0
    assert coerce_optional_float(True) == 1.0
    assert coerce_optional_float(float("nan")) is None


def test_coerce_optional_float_rejects_unsupported_value() -> None:
    with pytest.raises(TypeError, match="convertible to float"):
        coerce_optional_float(object())


def test_coerce_optional_float_preserves_non_missing_real_number() -> None:
    result = coerce_optional_float(3.25)

    assert result is not None
    assert math.isclose(result, 3.25)
