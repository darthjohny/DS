"""Общие helpers для bootstrap-подключения к Postgres.

Что делает модуль:
    - читает локальный `.env`, если он есть;
    - собирает SQLAlchemy engine из `DATABASE_URL` или набора `PG*`;
    - отбрасывает очевидно шаблонные DSN, если это запрошено вызывающим кодом.

Где используется:
    - в CLI-скриптах production-контура;
    - во входной валидации датасета;
    - в EDA- и devtools-утилитах, которым нужен единый способ подключения.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

PLACEHOLDER_DSN_TOKENS: tuple[str, ...] = ("HOST", "USER", "PASSWORD", "DBNAME")
DEFAULT_MISSING_DB_MESSAGE = (
    "Database connection is missing. Set DATABASE_URL or PG* variables."
)


def load_dotenv_local(dotenv_path: str = ".env") -> None:
    """Загрузить переменные из локального `.env` в текущее окружение.

    Функция читает файл построчно и добавляет только те переменные,
    которых ещё нет в `os.environ`. Это позволяет использовать локальный
    `.env` как fallback, не перетирая значения, уже переданные снаружи.
    """
    path = Path(dotenv_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cleaned = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), cleaned)


def make_engine_from_env(
    *,
    dotenv_path: str = ".env",
    reject_placeholder_url: bool = False,
    connect_timeout: int | None = None,
    missing_message: str = DEFAULT_MISSING_DB_MESSAGE,
) -> Engine:
    """Собрать SQLAlchemy engine из `DATABASE_URL` или набора `PG*`.

    Источник конфигурации
    ---------------------
    Сначала функция пытается дочитать локальный `.env`, затем ищет
    `DATABASE_URL`. Если он не задан, используется набор переменных
    `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`.

    Параметры
    ---------
    reject_placeholder_url
        Если `True`, функция отклоняет DSN, которые всё ещё содержат
        шаблонные токены вроде `HOST` или `PASSWORD`.
    connect_timeout
        Таймаут подключения к Postgres в секундах, если его нужно явно
        прокинуть в `create_engine`.
    missing_message
        Сообщение для `RuntimeError`, если конфигурации подключения нет.

    Возвращает
    ----------
    Engine
        SQLAlchemy engine для Postgres.

    Исключения
    ----------
    RuntimeError
        Если параметры подключения отсутствуют или DSN выглядит как
        не заполненный шаблон.
    """
    load_dotenv_local(dotenv_path)

    connect_args: dict[str, Any] | None = None
    if connect_timeout is not None:
        connect_args = {"connect_timeout": connect_timeout}

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        if reject_placeholder_url and _looks_like_placeholder_dsn(database_url):
            raise RuntimeError(
                "DATABASE_URL looks like a placeholder. Provide a real DSN."
            )
        if connect_args is not None:
            return create_engine(database_url, connect_args=connect_args)
        return create_engine(database_url)

    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT", "5432")
    dbname = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")

    if all([host, dbname, user, password]):
        dsn = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
        if connect_args is not None:
            return create_engine(dsn, connect_args=connect_args)
        return create_engine(dsn)

    raise RuntimeError(missing_message)


def _looks_like_placeholder_dsn(database_url: str) -> bool:
    """Проверить, содержит ли DSN незаполненные шаблонные токены."""
    return any(token in database_url for token in PLACEHOLDER_DSN_TOKENS)
