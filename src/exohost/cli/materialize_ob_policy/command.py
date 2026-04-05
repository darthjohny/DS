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

from exohost.db.coarse_ob_boundary_policy import (
    GAIA_OB_POLICY_SOURCE_RELATION_NAME,
    materialize_coarse_ob_boundary_policy_relations,
)
from exohost.db.engine import make_write_engine

from .support import (
    print_coarse_ob_boundary_policy_summary,
    print_materialize_ob_policy_stage,
)


def register_materialize_ob_policy_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    # Регистрируем CLI-команду materialization O/B boundary policy relation.
    materialize_parser = subparsers.add_parser(
        "materialize-ob-policy",
        help="Построить secure O-like и O/B boundary subsets.",
    )
    materialize_parser.add_argument(
        "--source-relation",
        default=GAIA_OB_POLICY_SOURCE_RELATION_NAME,
        help="Relation с hot-star provenance source для O/B policy.",
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
    materialize_parser.set_defaults(handler=handle_materialize_ob_policy)


def handle_materialize_ob_policy(namespace: argparse.Namespace) -> int:
    # Выполняем materialization secure O-like и O/B boundary relations.
    print_materialize_ob_policy_stage(
        f"materialize source={namespace.source_relation}"
    )
    engine = make_write_engine(
        dotenv_path=namespace.dotenv_path,
        connect_timeout=namespace.connect_timeout,
    )
    try:
        summary = materialize_coarse_ob_boundary_policy_relations(
            engine,
            source_relation_name=str(namespace.source_relation),
        )
    finally:
        engine.dispose()

    print_coarse_ob_boundary_policy_summary(summary)
    return 0
