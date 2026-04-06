# Тесты helper-слоя регресс-контура.
#
# Этот файл отвечает только за:
# - проверку loaders и assertions, на которых строятся regression-тесты;
# - защиту от хрупких ошибок в самом testkit нового слоя.
#
# Следующий слой:
# - доменные regression-тесты в `decision`, `posthoc` и `reporting`;
# - helper-модули `fixture_loaders.py` и `assertions.py`.

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.regression.assertions import (
    assert_required_columns,
    assert_small_frame_equal,
    require_float_scalar,
    require_int_scalar,
)
from tests.regression.conftest import QUALITY_GATE_SMALL_FIXTURE_PATH
from tests.regression.fixture_loaders import (
    load_regression_csv_fixture,
    load_regression_json_fixture,
)


def test_load_regression_csv_fixture_reads_small_fixture() -> None:
    fixture_df = load_regression_csv_fixture(QUALITY_GATE_SMALL_FIXTURE_PATH)

    assert list(fixture_df["source_id"]) == [1, 2, 3, 4]
    assert list(fixture_df["quality_state"]) == ["pass", "unknown", "unknown", "reject"]


def test_load_regression_json_fixture_rejects_non_json_path(tmp_path: Path) -> None:
    csv_path = tmp_path / "payload.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected .json fixture"):
        load_regression_json_fixture(csv_path)


def test_assert_required_columns_accepts_complete_frame() -> None:
    fixture_df = load_regression_csv_fixture(QUALITY_GATE_SMALL_FIXTURE_PATH)

    assert_required_columns(
        fixture_df,
        required_columns=("source_id", "quality_state", "quality_reason"),
    )


def test_assert_required_columns_raises_for_missing_columns() -> None:
    fixture_df = load_regression_csv_fixture(QUALITY_GATE_SMALL_FIXTURE_PATH)

    with pytest.raises(AssertionError, match="missing required columns"):
        assert_required_columns(
            fixture_df,
            required_columns=("source_id", "missing_column"),
        )


def test_require_int_scalar_accepts_integer_like_values() -> None:
    value = pd.Series([4], dtype="int64").iloc[0]

    assert require_int_scalar(value) == 4


def test_require_float_scalar_accepts_real_like_values() -> None:
    value = pd.Series([0.75], dtype="float64").iloc[0]

    assert require_float_scalar(value) == 0.75


def test_assert_small_frame_equal_accepts_same_rows_with_different_column_order() -> None:
    left = pd.DataFrame([{"a": 1, "b": "x"}])
    right = pd.DataFrame([{"b": "x", "a": 1}])

    assert_small_frame_equal(left, right)
