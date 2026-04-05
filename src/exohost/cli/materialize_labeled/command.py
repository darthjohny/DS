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

from exohost.db.bmk_labeled import (
    B_MK_EXTERNAL_LABELED_RELATION_NAME,
    B_MK_EXTERNAL_LABELED_SOURCE_CROSSMATCH_RELATION_NAME,
    B_MK_EXTERNAL_LABELED_SOURCE_FILTERED_RELATION_NAME,
    materialize_bmk_external_labeled_relation,
)
from exohost.db.engine import make_write_engine

from .support import (
    print_bmk_external_labeled_load_summary,
    print_materialize_labeled_stage,
)


def register_materialize_labeled_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    # Регистрируем post-Gaia materialization команду для normalized labeled layer.
    materialize_labeled_parser = subparsers.add_parser(
        "materialize-labeled",
        help="Построить lab.gaia_mk_external_labeled из filtered и selected crossmatch.",
    )
    materialize_labeled_parser.add_argument(
        "--filtered-relation",
        default=B_MK_EXTERNAL_LABELED_SOURCE_FILTERED_RELATION_NAME,
        help="Relation с локальным filtered B/mk слоем.",
    )
    materialize_labeled_parser.add_argument(
        "--crossmatch-relation",
        default=B_MK_EXTERNAL_LABELED_SOURCE_CROSSMATCH_RELATION_NAME,
        help="Relation с canonical crossmatch слоем.",
    )
    materialize_labeled_parser.add_argument(
        "--target-relation",
        default=B_MK_EXTERNAL_LABELED_RELATION_NAME,
        help="Canonical relation для normalized labeled слоя.",
    )
    materialize_labeled_parser.add_argument(
        "--xmatch-batch-id",
        required=True,
        help="Stable batch id post-Gaia materialization шага.",
    )
    materialize_labeled_parser.add_argument(
        "--chunksize",
        type=int,
        default=50_000,
        help="Размер chunk-а при чтении selected join из БД.",
    )
    materialize_labeled_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Необязательный лимит строк для локального debug-прогона.",
    )
    materialize_labeled_parser.add_argument(
        "--dotenv-path",
        default=".env",
        help="Путь к .env файлу с параметрами подключения.",
    )
    materialize_labeled_parser.add_argument(
        "--connect-timeout",
        type=int,
        default=10,
        help="Таймаут подключения к БД в секундах.",
    )
    materialize_labeled_parser.set_defaults(handler=handle_materialize_labeled)


def handle_materialize_labeled(namespace: argparse.Namespace) -> int:
    # Выполняем local DB materialization normalized labeled layer.
    print_materialize_labeled_stage(
        f"materialize filtered={namespace.filtered_relation} "
        f"crossmatch={namespace.crossmatch_relation} "
        f"target={namespace.target_relation}"
    )
    engine = make_write_engine(
        dotenv_path=namespace.dotenv_path,
        connect_timeout=namespace.connect_timeout,
    )
    try:
        load_summary = materialize_bmk_external_labeled_relation(
            engine,
            xmatch_batch_id=str(namespace.xmatch_batch_id),
            filtered_relation_name=str(namespace.filtered_relation),
            crossmatch_relation_name=str(namespace.crossmatch_relation),
            target_relation_name=str(namespace.target_relation),
            chunksize=namespace.chunksize,
            limit=namespace.limit,
        )
    finally:
        engine.dispose()

    print_bmk_external_labeled_load_summary(load_summary)
    return 0
