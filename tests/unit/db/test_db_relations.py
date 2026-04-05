# Тестовый файл `test_db_relations.py` домена `db`.
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

from exohost.db.relations import quote_relation_name, split_relation_name, validate_identifier


def test_split_relation_name_uses_public_by_default() -> None:
    # Если схема не указана, по умолчанию используем public.
    assert split_relation_name("stars") == ("public", "stars")


def test_split_relation_name_keeps_explicit_schema() -> None:
    # Явно заданную схему не меняем.
    assert split_relation_name("lab.v_gaia_router_training") == (
        "lab",
        "v_gaia_router_training",
    )


def test_split_relation_name_rejects_invalid_identifiers() -> None:
    # Ловим имена с недопустимыми символами, чтобы не собирать сырой SQL.
    with pytest.raises(ValueError, match="Invalid schema name"):
        split_relation_name("lab-1.table", validate_identifiers=True)


def test_validate_identifier_rejects_invalid_name() -> None:
    # Отдельная валидация identifier нужна для COPY и DDL helpers.
    with pytest.raises(ValueError, match="Invalid identifier"):
        validate_identifier("bad-name")


def test_quote_relation_name_wraps_schema_and_table() -> None:
    # SQL helper должен собирать безопасное schema-qualified имя relation.
    assert quote_relation_name("lab.gaia_mk_external_raw", validate_identifiers=True) == (
        '"lab"."gaia_mk_external_raw"'
    )
