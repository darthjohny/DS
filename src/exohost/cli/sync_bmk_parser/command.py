# Файл `command.py` слоя `cli`.
#
# Этот файл отвечает только за:
# - CLI-команды и orchestration entrypoints;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - CLI-команды или support-модули этого же домена;
# - пользовательский запуск через `python -m exohost.cli.main`.

from __future__ import annotations

import argparse

from exohost.db.bmk_parser_sync import sync_bmk_parser_fields_downstream
from exohost.db.coarse_ob_provenance_refresh import (
    refresh_gaia_ob_hot_provenance_relations,
)
from exohost.db.engine import make_write_engine

from .support import (
    print_bmk_parser_sync_summary,
    print_coarse_ob_provenance_refresh_summary,
    print_sync_bmk_parser_stage,
)


def register_sync_bmk_parser_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    # Регистрируем CLI-команду для проталкивания parser-fix в downstream relations.
    sync_parser = subparsers.add_parser(
        "sync-bmk-parser",
        help="Синхронизировать parser-derived поля в downstream MK relations.",
    )
    sync_parser.add_argument(
        "--dotenv-path",
        default=".env",
        help="Путь к .env файлу с параметрами подключения.",
    )
    sync_parser.add_argument(
        "--connect-timeout",
        type=int,
        default=10,
        help="Таймаут подключения к БД в секундах.",
    )
    sync_parser.set_defaults(handler=handle_sync_bmk_parser)


def handle_sync_bmk_parser(namespace: argparse.Namespace) -> int:
    # Выполняем downstream sync parser-fix и refresh provenance audit слоя.
    print_sync_bmk_parser_stage("sync downstream parser-derived labels")
    engine = make_write_engine(
        dotenv_path=namespace.dotenv_path,
        connect_timeout=namespace.connect_timeout,
    )
    try:
        parser_sync_summary = sync_bmk_parser_fields_downstream(engine)
        provenance_summary = refresh_gaia_ob_hot_provenance_relations(engine)
    finally:
        engine.dispose()

    print_bmk_parser_sync_summary(parser_sync_summary)
    print_coarse_ob_provenance_refresh_summary(provenance_summary)
    return 0
