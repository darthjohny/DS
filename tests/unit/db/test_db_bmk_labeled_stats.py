# Тестовый файл `test_db_bmk_labeled_stats.py` домена `db`.
#
# Этот файл проверяет только:
# - проверку логики домена: relation-layer, materialization и SQL-helper;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `db` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import pytest

from exohost.db.bmk_labeled_stats import fetch_single_count


class DummyCursor:
    # Минимальный cursor-stub для проверки COUNT-helper без реальной БД.

    def __init__(self, result: tuple[object, ...] | None) -> None:
        self._result = result
        self.last_sql: str | None = None

    def execute(self, operation: str) -> object:
        self.last_sql = operation
        return None

    def fetchone(self) -> tuple[object, ...] | None:
        return self._result


def test_fetch_single_count_returns_integer_like_value() -> None:
    cursor = DummyCursor((42,))

    result = fetch_single_count(cursor, "SELECT COUNT(*)")

    assert result == 42
    assert cursor.last_sql == "SELECT COUNT(*)"


def test_fetch_single_count_rejects_non_integer_like_value() -> None:
    cursor = DummyCursor((object(),))

    with pytest.raises(TypeError, match="non-integer-like value"):
        fetch_single_count(cursor, "SELECT COUNT(*)")


def test_fetch_single_count_rejects_missing_row() -> None:
    cursor = DummyCursor(None)

    with pytest.raises(RuntimeError, match="returned no rows"):
        fetch_single_count(cursor, "SELECT COUNT(*)")
