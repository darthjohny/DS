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

from exohost.db.bmk_ingestion import load_bmk_exports_into_db
from exohost.db.engine import make_write_engine
from exohost.ingestion.bmk.pipeline import build_bmk_export_bundle

from .support import (
    DEFAULT_INGEST_OUTPUT_DIR,
    build_bmk_catalog_source,
    print_db_load_summary,
    print_export_paths,
    print_import_summary,
    print_ingest_stage,
    print_primary_filter_summary,
    resolve_ingest_output_dir,
)


def register_ingest_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    # Регистрируем ingest-команду для B/mk parser/export/load шага.
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Загрузка staging-слоев B/mk в локальную БД.",
    )
    ingest_parser.add_argument(
        "--readme-path",
        required=True,
        help="Путь к официальному ReadMe каталога B/mk.",
    )
    ingest_parser.add_argument(
        "--data-path",
        required=True,
        help="Путь к mktypes.dat каталога B/mk.",
    )
    ingest_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_INGEST_OUTPUT_DIR),
        help="Каталог для staging CSV одного ingest-прогона.",
    )
    ingest_parser.add_argument(
        "--skip-db-load",
        action="store_true",
        help="Только собрать staging CSV без загрузки в Postgres.",
    )
    ingest_parser.add_argument(
        "--dotenv-path",
        default=".env",
        help="Путь к .env файлу с параметрами подключения.",
    )
    ingest_parser.add_argument(
        "--connect-timeout",
        type=int,
        default=10,
        help="Таймаут подключения к БД в секундах.",
    )
    ingest_parser.set_defaults(handler=handle_ingest)


def handle_ingest(namespace: argparse.Namespace) -> int:
    # Выполняем B/mk ingest: parser/export и при необходимости DB-load.
    source = build_bmk_catalog_source(namespace)
    output_dir = resolve_ingest_output_dir(namespace)

    print_ingest_stage("read and export bmk")
    export_bundle = build_bmk_export_bundle(
        source,
        output_dir=output_dir,
    )
    print_import_summary(export_bundle.import_summary)
    print_primary_filter_summary(export_bundle.primary_filter_summary)
    print_export_paths(export_bundle)

    if namespace.skip_db_load:
        return 0

    print_ingest_stage("load csv to postgres")
    engine = make_write_engine(
        dotenv_path=namespace.dotenv_path,
        connect_timeout=namespace.connect_timeout,
    )
    try:
        load_summary = load_bmk_exports_into_db(
            engine,
            export_bundle.export_paths,
        )
    finally:
        engine.dispose()

    print_db_load_summary(load_summary)
    return 0
