# Файл `main.py` слоя `cli`.
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
from collections.abc import Callable, Sequence
from typing import cast

from exohost.cli.benchmark import register_benchmark_parser
from exohost.cli.decide import register_decide_parser
from exohost.cli.ingest import register_ingest_parser
from exohost.cli.materialize_crossmatch import register_materialize_crossmatch_parser
from exohost.cli.materialize_labeled import register_materialize_labeled_parser
from exohost.cli.materialize_ob_policy import register_materialize_ob_policy_parser
from exohost.cli.materialize_ob_review import register_materialize_ob_review_parser
from exohost.cli.prepare_enrichment import register_prepare_enrichment_parser
from exohost.cli.prepare_upload import register_prepare_upload_parser
from exohost.cli.prioritize import register_prioritize_parser
from exohost.cli.report import register_report_parser
from exohost.cli.score import register_score_parser
from exohost.cli.sync_bmk_parser import register_sync_bmk_parser_parser
from exohost.cli.train import register_train_parser


def build_parser() -> argparse.ArgumentParser:
    # Собираем единый корневой парсер команд верхнего уровня.
    parser = argparse.ArgumentParser(
        prog="exohost",
        description="CLI новой версии проекта ExoHost.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_ingest_parser(subparsers)
    register_prepare_upload_parser(subparsers)
    register_materialize_crossmatch_parser(subparsers)
    register_materialize_labeled_parser(subparsers)
    register_materialize_ob_policy_parser(subparsers)
    register_materialize_ob_review_parser(subparsers)
    register_prepare_enrichment_parser(subparsers)
    register_train_parser(subparsers)
    register_benchmark_parser(subparsers)
    register_decide_parser(subparsers)
    register_score_parser(subparsers)
    register_prioritize_parser(subparsers)
    register_report_parser(subparsers)
    register_sync_bmk_parser_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Выполняем CLI-команду верхнего уровня через зарегистрированный handler.
    namespace = build_parser().parse_args(argv)
    raw_handler = getattr(namespace, "handler", None)
    if raw_handler is None:
        raise RuntimeError(f"CLI handler is not registered for command: {namespace.command}")

    handler = cast(Callable[[argparse.Namespace], int], raw_handler)
    return int(handler(namespace))


if __name__ == "__main__":
    raise SystemExit(main())
