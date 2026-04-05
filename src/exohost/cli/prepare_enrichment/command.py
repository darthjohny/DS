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

from exohost.db.bmk_enrichment import (
    B_MK_GAIA_ENRICHMENT_SOURCE_RELATION_NAME,
    export_bmk_gaia_enrichment_batches,
)
from exohost.db.engine import make_read_only_engine

from .support import (
    DEFAULT_GAIA_ENRICHMENT_OUTPUT_DIR,
    print_gaia_enrichment_export_summary,
    print_prepare_enrichment_stage,
    resolve_gaia_enrichment_output_dir,
)


def register_prepare_enrichment_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    # Регистрируем локальную prepare-enrichment команду перед повторным Gaia шагом.
    prepare_enrichment_parser = subparsers.add_parser(
        "prepare-enrichment",
        help="Собрать source_id batch-артефакты для chunk-wise Gaia enrichment.",
    )
    prepare_enrichment_parser.add_argument(
        "--relation-name",
        default=B_MK_GAIA_ENRICHMENT_SOURCE_RELATION_NAME,
        help="Relation в БД, из которой собираются source_id batch-ы.",
    )
    prepare_enrichment_parser.add_argument(
        "--xmatch-batch-id",
        required=True,
        help="Stable batch id для conflict-free enrichment source.",
    )
    prepare_enrichment_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_GAIA_ENRICHMENT_OUTPUT_DIR),
        help="Каталог для сохранения локальных Gaia enrichment batch-ов.",
    )
    prepare_enrichment_parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Размер одного source_id batch для ручного Gaia upload.",
    )
    prepare_enrichment_parser.add_argument(
        "--include-conflicts",
        action="store_true",
        help="Включить source_id с конфликтами по внешним labels.",
    )
    prepare_enrichment_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Необязательный лимит строк для локального debug-прогона.",
    )
    prepare_enrichment_parser.add_argument(
        "--dotenv-path",
        default=".env",
        help="Путь к .env файлу с параметрами подключения.",
    )
    prepare_enrichment_parser.add_argument(
        "--connect-timeout",
        type=int,
        default=10,
        help="Таймаут подключения к БД в секундах.",
    )
    prepare_enrichment_parser.set_defaults(handler=handle_prepare_enrichment)


def handle_prepare_enrichment(namespace: argparse.Namespace) -> int:
    # Выполняем локальный export source_id batch-ов без обращения к Gaia.
    output_dir = resolve_gaia_enrichment_output_dir(namespace)
    print_prepare_enrichment_stage(
        f"export relation={namespace.relation_name} xmatch_batch_id={namespace.xmatch_batch_id}"
    )
    engine = make_read_only_engine(
        dotenv_path=namespace.dotenv_path,
        connect_timeout=namespace.connect_timeout,
    )
    try:
        export_summary = export_bmk_gaia_enrichment_batches(
            engine,
            output_dir=output_dir,
            relation_name=str(namespace.relation_name),
            xmatch_batch_id=str(namespace.xmatch_batch_id),
            batch_size=namespace.batch_size,
            only_conflict_free=not bool(namespace.include_conflicts),
            limit=namespace.limit,
        )
    finally:
        engine.dispose()

    print_gaia_enrichment_export_summary(export_summary)
    return 0
