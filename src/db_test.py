"""Простой smoke-test подключения к Postgres для проекта ВКР.

Что делает файл:
- загружает переменные окружения из локального `.env`, если он есть;
- создаёт SQLAlchemy engine для Postgres;
- проверяет базовое подключение;
- проверяет доступность схемы `lab`;
- проверяет, что ключевая view для NASA host-train читается.

Зачем нужен:
- быстро убедиться, что Python окружение и БД настроены корректно;
- иметь воспроизводимый тест подключения вне DBeaver;
- не смешивать этот технический smoke-test с основным ML пайплайном.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def _load_dotenv_local(path: str = ".env") -> None:
    """Подгружает переменные из локального `.env`, не затирая уже заданные."""
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def make_engine_from_env() -> Engine:
    """Создаёт SQLAlchemy engine из `DATABASE_URL` или набора `PG*`."""
    _load_dotenv_local(".env")

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return create_engine(
            database_url,
            connect_args={"connect_timeout": 5},
        )

    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT", "5432")
    dbname = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")

    if all([host, dbname, user, password]):
        return create_engine(
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}",
            connect_args={"connect_timeout": 5},
        )

    raise RuntimeError(
        "Параметры подключения к БД не найдены. "
        "Задай DATABASE_URL или набор PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD."
    )


def run_db_smoke_test(engine: Engine) -> None:
    """Выполняет минимальный набор SQL-проверок для БД проекта."""
    with engine.connect() as conn:
        print("=== CONNECTION SANITY ===")
        print(conn.execute(text("select 1")).fetchone())
        print(
            conn.execute(
                text(
                    "select current_database(), current_user, current_schema()"
                )
            ).fetchone()
        )

        print("\n=== SCHEMA CHECK ===")
        print(
            conn.execute(
                text(
                    "select schema_name "
                    "from information_schema.schemata "
                    "where schema_name = 'lab'"
                )
            ).fetchall()
        )

        print("\n=== VIEW CHECK ===")
        print(
            conn.execute(
                text("select count(*) from lab.v_nasa_gaia_train_classified")
            ).fetchone()
        )


def main() -> None:
    """Точка входа для ручного запуска DB smoke-test."""
    engine = make_engine_from_env()
    run_db_smoke_test(engine)


if __name__ == "__main__":
    main()
