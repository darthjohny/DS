# Тестовый файл `test_db_bmk_parser_sync_scalars.py` домена `db`.
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

from exohost.db.bmk_parser_sync_scalars import cursor_rowcount, scalar_to_int


class DummyCursor:
    # Минимальный stub для rowcount-проверок без реального DB-курсора.

    def __init__(self, rowcount: object) -> None:
        self.rowcount = rowcount


def test_cursor_rowcount_normalizes_supported_values() -> None:
    assert cursor_rowcount(DummyCursor(7)) == 7
    assert cursor_rowcount(DummyCursor(-3)) == 0
    assert cursor_rowcount(DummyCursor("12")) == 12
    assert cursor_rowcount(DummyCursor(4.8)) == 4


def test_cursor_rowcount_returns_zero_for_unsupported_value() -> None:
    assert cursor_rowcount(DummyCursor(object())) == 0


def test_scalar_to_int_accepts_integer_like_scalars() -> None:
    assert scalar_to_int(3, relation_name="lab.test") == 3
    assert scalar_to_int("4", relation_name="lab.test") == 4
    assert scalar_to_int(5.9, relation_name="lab.test") == 5


def test_scalar_to_int_rejects_unsupported_value() -> None:
    with pytest.raises(RuntimeError, match="Unable to convert parser sync scalar"):
        scalar_to_int(object(), relation_name="lab.test")
