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

from exohost.db.bmk_crossmatch import (
    B_MK_EXTERNAL_CROSSMATCH_RELATION_NAME,
    B_MK_GAIA_XMATCH_RAW_SOURCE_RELATION_NAME,
    materialize_bmk_crossmatch_relation,
)
from exohost.db.engine import make_write_engine

from .support import (
    print_bmk_crossmatch_materialization_summary,
    print_materialize_crossmatch_stage,
)


def register_materialize_crossmatch_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    # Регистрируем post-Gaia materialization команду для canonical crossmatch layer.
    materialize_crossmatch_parser = subparsers.add_parser(
        "materialize-crossmatch",
        help="Построить lab.gaia_mk_external_crossmatch из wide Gaia export в БД.",
    )
    materialize_crossmatch_parser.add_argument(
        "--source-relation",
        default=B_MK_GAIA_XMATCH_RAW_SOURCE_RELATION_NAME,
        help="Wide raw landing relation с Gaia xmatch export.",
    )
    materialize_crossmatch_parser.add_argument(
        "--target-relation",
        default=B_MK_EXTERNAL_CROSSMATCH_RELATION_NAME,
        help="Canonical relation для узкого crossmatch слоя.",
    )
    materialize_crossmatch_parser.add_argument(
        "--xmatch-batch-id",
        required=True,
        help="Stable batch id для materialized xmatch результата.",
    )
    materialize_crossmatch_parser.add_argument(
        "--dotenv-path",
        default=".env",
        help="Путь к .env файлу с параметрами подключения.",
    )
    materialize_crossmatch_parser.add_argument(
        "--connect-timeout",
        type=int,
        default=10,
        help="Таймаут подключения к БД в секундах.",
    )
    materialize_crossmatch_parser.set_defaults(handler=handle_materialize_crossmatch)


def handle_materialize_crossmatch(namespace: argparse.Namespace) -> int:
    # Выполняем post-Gaia materialization canonical crossmatch relation.
    print_materialize_crossmatch_stage(
        f"materialize source={namespace.source_relation} target={namespace.target_relation}"
    )
    engine = make_write_engine(
        dotenv_path=namespace.dotenv_path,
        connect_timeout=namespace.connect_timeout,
    )
    try:
        load_summary = materialize_bmk_crossmatch_relation(
            engine,
            source_relation_name=str(namespace.source_relation),
            target_relation_name=str(namespace.target_relation),
            xmatch_batch_id=str(namespace.xmatch_batch_id),
        )
    finally:
        engine.dispose()

    print_bmk_crossmatch_materialization_summary(load_summary)
    return 0
