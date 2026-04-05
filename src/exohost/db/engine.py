# Файл `engine.py` слоя `db`.
#
# Этот файл отвечает только за:
# - relation-layer, materialization и SQL-обвязку;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `db` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

PLACEHOLDER_DSN_TOKENS: tuple[str, ...] = ("HOST", "USER", "PASSWORD", "DBNAME")
DEFAULT_MISSING_DB_MESSAGE = (
    "Database connection is missing. Set DATABASE_URL or PG* variables."
)


def load_dotenv_local(dotenv_path: str = ".env") -> None:
    # Локальный .env используем только как fallback и не перетираем
    # уже переданные переменные окружения.
    path = Path(dotenv_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        cleaned_value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), cleaned_value)


def build_pg_dsn_from_env() -> str | None:
    # Собираем DSN из набора PG* переменных, если DATABASE_URL не задан.
    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT", "5432")
    database_name = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")

    if not all([host, database_name, user, password]):
        return None

    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database_name}"


def build_connect_args(connect_timeout: int | None = None) -> dict[str, Any]:
    # Включаем read-only режим на уровне сессии Postgres и при желании
    # добавляем таймаут подключения.
    connect_args: dict[str, Any] = {
        "options": "-c default_transaction_read_only=on",
    }

    if connect_timeout is not None:
        connect_args["connect_timeout"] = connect_timeout

    return connect_args


def build_write_connect_args(connect_timeout: int | None = None) -> dict[str, Any]:
    # Для write-engine не включаем read-only режим, но сохраняем таймаут.
    connect_args: dict[str, Any] = {}
    if connect_timeout is not None:
        connect_args["connect_timeout"] = connect_timeout
    return connect_args


def make_read_only_engine(
    *,
    dotenv_path: str = ".env",
    reject_placeholder_url: bool = True,
    connect_timeout: int | None = None,
    missing_message: str = DEFAULT_MISSING_DB_MESSAGE,
) -> Engine:
    # Создаем единый read-only engine для всех V2 loaders.
    load_dotenv_local(dotenv_path)

    database_url = os.getenv("DATABASE_URL") or build_pg_dsn_from_env()
    if not database_url:
        raise RuntimeError(missing_message)

    if reject_placeholder_url and looks_like_placeholder_dsn(database_url):
        raise RuntimeError(
            "DATABASE_URL looks like a placeholder. Provide a real DSN."
        )

    engine = create_engine(
        database_url,
        connect_args=build_connect_args(connect_timeout),
        pool_pre_ping=True,
    )

    @event.listens_for(engine, "connect")
    def _enforce_read_only(dbapi_connection: Any, _: Any) -> None:
        # Дублируем read-only режим через SQL на случай, если драйвер
        # проигнорирует libpq options.
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("SET default_transaction_read_only = on")
        finally:
            cursor.close()

    return engine


def make_write_engine(
    *,
    dotenv_path: str = ".env",
    reject_placeholder_url: bool = True,
    connect_timeout: int | None = None,
    missing_message: str = DEFAULT_MISSING_DB_MESSAGE,
) -> Engine:
    # Создаем write-capable engine для controlled ingestion-сценариев.
    load_dotenv_local(dotenv_path)

    database_url = os.getenv("DATABASE_URL") or build_pg_dsn_from_env()
    if not database_url:
        raise RuntimeError(missing_message)

    if reject_placeholder_url and looks_like_placeholder_dsn(database_url):
        raise RuntimeError(
            "DATABASE_URL looks like a placeholder. Provide a real DSN."
        )

    return create_engine(
        database_url,
        connect_args=build_write_connect_args(connect_timeout),
        pool_pre_ping=True,
    )


def looks_like_placeholder_dsn(database_url: str) -> bool:
    # Отсекаем явно незаполненные DSN-шаблоны.
    return any(token in database_url for token in PLACEHOLDER_DSN_TOKENS)
