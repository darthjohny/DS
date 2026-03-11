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

from sqlalchemy import text
from sqlalchemy.engine import Engine

from infra.db import make_engine_from_env as _make_engine_from_env


def make_engine_from_env() -> Engine:
    """Создать engine для короткой smoke-проверки БД.

    Функция использует общий bootstrap из `infra.db`, но задаёт более
    жёсткий `connect_timeout` и человекочитаемое сообщение об ошибке
    именно для devtools-сценария.
    """
    return _make_engine_from_env(
        connect_timeout=5,
        missing_message=(
            "Параметры подключения к БД не найдены. "
            "Задай DATABASE_URL или набор PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD."
        ),
    )


def run_db_smoke_test(engine: Engine) -> None:
    """Выполнить минимальный набор SQL-проверок для БД проекта.

    Что проверяется
    ---------------
    - базовое подключение и текущий контекст пользователя;
    - существование схемы `lab`;
    - чтение ключевой view `lab.v_nasa_gaia_train_classified`.

    Побочные эффекты
    ----------------
    Ничего не пишет в БД, но печатает результаты проверок в stdout.
    """
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
    """Запустить smoke-проверку подключения к БД из CLI."""
    engine = make_engine_from_env()
    run_db_smoke_test(engine)


if __name__ == "__main__":
    main()
