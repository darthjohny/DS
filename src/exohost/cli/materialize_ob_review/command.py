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

from exohost.db.coarse_ob_review_pool import (
    GAIA_OB_REVIEW_POOL_SOURCE_RELATION_NAME,
    materialize_coarse_ob_review_pool_relations,
)
from exohost.db.engine import make_write_engine

from .support import (
    print_coarse_ob_review_pool_summary,
    print_materialize_ob_review_stage,
)


def register_materialize_ob_review_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    # Регистрируем CLI-команду materialization O/B review-pool.
    materialize_parser = subparsers.add_parser(
        "materialize-ob-review",
        help="Построить явный review-pool для O/B boundary subset.",
    )
    materialize_parser.add_argument(
        "--source-relation",
        default=GAIA_OB_REVIEW_POOL_SOURCE_RELATION_NAME,
        help="Relation с O/B boundary subset для review-pool.",
    )
    materialize_parser.add_argument(
        "--dotenv-path",
        default=".env",
        help="Путь к .env файлу с параметрами подключения.",
    )
    materialize_parser.add_argument(
        "--connect-timeout",
        type=int,
        default=10,
        help="Таймаут подключения к БД в секундах.",
    )
    materialize_parser.set_defaults(handler=handle_materialize_ob_review)


def handle_materialize_ob_review(namespace: argparse.Namespace) -> int:
    # Выполняем materialization O/B review-pool relations.
    print_materialize_ob_review_stage(
        f"materialize source={namespace.source_relation}"
    )
    engine = make_write_engine(
        dotenv_path=namespace.dotenv_path,
        connect_timeout=namespace.connect_timeout,
    )
    try:
        summary = materialize_coarse_ob_review_pool_relations(
            engine,
            source_relation_name=str(namespace.source_relation),
        )
    finally:
        engine.dispose()

    print_coarse_ob_review_pool_summary(summary)
    return 0
