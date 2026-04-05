# Тестовый файл `test_db_engine.py` домена `db`.
#
# Этот файл проверяет только:
# - проверку логики домена: relation-layer, materialization и SQL-helper;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `db` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import os
from pathlib import Path

import pytest

from exohost.db.engine import (
    build_connect_args,
    build_pg_dsn_from_env,
    build_write_connect_args,
    load_dotenv_local,
    looks_like_placeholder_dsn,
    make_read_only_engine,
    make_write_engine,
)


def clear_db_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Чистим только те переменные, которые влияют на bootstrap DB.
    for env_name in (
        "DATABASE_URL",
        "PGHOST",
        "PGPORT",
        "PGDATABASE",
        "PGUSER",
        "PGPASSWORD",
        "KEEP_ME",
        "NEW_VALUE",
    ):
        monkeypatch.delenv(env_name, raising=False)


def test_load_dotenv_local_sets_only_missing_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Проверяем, что локальный .env не перетирает уже заданное окружение.
    clear_db_env(monkeypatch)
    monkeypatch.setenv("KEEP_ME", "outer")
    dotenv_path = tmp_path / "sample.env"
    dotenv_path.write_text(
        "KEEP_ME=inner\nNEW_VALUE=from_file\n",
        encoding="utf-8",
    )

    load_dotenv_local(str(dotenv_path))

    assert os.getenv("KEEP_ME") == "outer"
    assert os.getenv("NEW_VALUE") == "from_file"


def test_build_pg_dsn_from_env_uses_pg_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    # Проверяем сборку DSN из PG* переменных.
    clear_db_env(monkeypatch)
    monkeypatch.setenv("PGHOST", "localhost")
    monkeypatch.setenv("PGPORT", "5433")
    monkeypatch.setenv("PGDATABASE", "stars")
    monkeypatch.setenv("PGUSER", "reader")
    monkeypatch.setenv("PGPASSWORD", "secret")

    dsn = build_pg_dsn_from_env()

    assert dsn == "postgresql+psycopg2://reader:secret@localhost:5433/stars"


def test_build_connect_args_enables_read_only_mode() -> None:
    # Проверяем базовый набор connect_args для read-only режима.
    connect_args = build_connect_args(connect_timeout=7)

    assert connect_args["options"] == "-c default_transaction_read_only=on"
    assert connect_args["connect_timeout"] == 7


def test_build_write_connect_args_keeps_only_timeout() -> None:
    # Write-connect args не должны включать read-only options.
    connect_args = build_write_connect_args(connect_timeout=9)

    assert connect_args == {"connect_timeout": 9}


def test_looks_like_placeholder_dsn_detects_template() -> None:
    # Проверяем защиту от незаполненного DSN-шаблона.
    placeholder_dsn = "postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME"

    assert looks_like_placeholder_dsn(placeholder_dsn) is True


def test_make_read_only_engine_rejects_placeholder_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Проверяем, что placeholder-DSN не проходит в production bootstrap.
    clear_db_env(monkeypatch)
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME",
    )

    with pytest.raises(RuntimeError, match="placeholder"):
        make_read_only_engine()


def test_make_read_only_engine_uses_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Проверяем, что engine собирается из DATABASE_URL без подключения.
    clear_db_env(monkeypatch)
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg2://reader:secret@localhost:5432/stars",
    )

    engine = make_read_only_engine()

    assert engine.url.drivername == "postgresql+psycopg2"
    assert engine.url.username == "reader"
    assert engine.url.host == "localhost"


def test_make_write_engine_uses_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Write-engine тоже собирается из DATABASE_URL без подключения.
    clear_db_env(monkeypatch)
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg2://writer:secret@localhost:5432/stars",
    )

    engine = make_write_engine()

    assert engine.url.drivername == "postgresql+psycopg2"
    assert engine.url.username == "writer"
    assert engine.url.host == "localhost"
