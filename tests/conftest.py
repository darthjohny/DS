"""Общие настройки тестов проекта."""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _validate_identifier(identifier: str) -> str:
    """Проверить, что тестовый SQL-идентификатор безопасен для DDL."""
    if not IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"Небезопасный SQL-идентификатор: {identifier}")
    return identifier


@pytest.fixture(scope="session")
def postgres_test_engine() -> Iterator[Engine]:
    """Вернуть engine для DB-backed тестов или пропустить их без конфига."""
    from infra.db import make_engine_from_env

    try:
        engine = make_engine_from_env(
            dotenv_path=str(PROJECT_ROOT / ".env"),
            connect_timeout=3,
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(
            "DB-backed тесты требуют доступный Postgres через .env или "
            f"переменные окружения: {exc}"
        )

    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture()
def temp_pg_schema(postgres_test_engine: Engine) -> Iterator[str]:
    """Создать и затем удалить временную схему Postgres для интеграционного теста."""
    schema_name = _validate_identifier(f"test_it_{uuid4().hex[:10]}")
    with postgres_test_engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema_name}"))

    try:
        yield schema_name
    finally:
        with postgres_test_engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
