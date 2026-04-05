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

from exohost.db.bmk_upload import (
    B_MK_GAIA_UPLOAD_SOURCE_RELATION_NAME,
    export_bmk_gaia_upload_csv,
)
from exohost.db.engine import make_read_only_engine

from .support import (
    DEFAULT_GAIA_UPLOAD_OUTPUT_DIR,
    print_gaia_upload_export_summary,
    print_prepare_upload_stage,
    resolve_gaia_upload_output_path,
)


def register_prepare_upload_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    # Регистрируем локальную prepare-upload команду перед будущим Gaia этапом.
    prepare_upload_parser = subparsers.add_parser(
        "prepare-upload",
        help="Собрать локальный upload CSV для Gaia Archive из БД.",
    )
    prepare_upload_parser.add_argument(
        "--relation-name",
        default=B_MK_GAIA_UPLOAD_SOURCE_RELATION_NAME,
        help="Relation в БД, из которой собирается upload layer.",
    )
    prepare_upload_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_GAIA_UPLOAD_OUTPUT_DIR),
        help="Каталог для сохранения локального Gaia upload CSV.",
    )
    prepare_upload_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Необязательный лимит строк для локального export прогона.",
    )
    prepare_upload_parser.add_argument(
        "--dotenv-path",
        default=".env",
        help="Путь к .env файлу с параметрами подключения.",
    )
    prepare_upload_parser.add_argument(
        "--connect-timeout",
        type=int,
        default=10,
        help="Таймаут подключения к БД в секундах.",
    )
    prepare_upload_parser.set_defaults(handler=handle_prepare_upload)


def handle_prepare_upload(namespace: argparse.Namespace) -> int:
    # Выполняем локальный export upload table из БД без обращения к Gaia.
    output_csv_path = resolve_gaia_upload_output_path(namespace)

    print_prepare_upload_stage(f"export relation={namespace.relation_name}")
    engine = make_read_only_engine(
        dotenv_path=namespace.dotenv_path,
        connect_timeout=namespace.connect_timeout,
    )
    try:
        export_summary = export_bmk_gaia_upload_csv(
            engine,
            output_csv_path=output_csv_path,
            relation_name=str(namespace.relation_name),
            limit=namespace.limit,
        )
    finally:
        engine.dispose()

    print_gaia_upload_export_summary(export_summary)
    return 0
