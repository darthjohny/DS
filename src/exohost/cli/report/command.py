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

from .support import build_report, print_report_stage


def register_report_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    # Регистрируем report-команду для сохраненных run_dir.
    report_parser = subparsers.add_parser(
        "report",
        help="Сборка отчетов и артефактов V2.",
    )
    report_parser.add_argument(
        "--kind",
        choices=("benchmark", "ranking", "scoring"),
        required=True,
        help="Тип run_dir, для которого собирается markdown-отчет.",
    )
    report_parser.add_argument(
        "--run-dir",
        required=True,
        help="Каталог существующего benchmark или ranking прогона.",
    )
    report_parser.add_argument(
        "--output-path",
        default=None,
        help="Необязательный путь для итогового markdown-отчета.",
    )
    report_parser.add_argument(
        "--top-rows",
        type=int,
        default=10,
        help="Сколько строк показывать в ranking-отчете.",
    )
    report_parser.add_argument(
        "--ranking-run-dir",
        default=None,
        help="Необязательный ranking run_dir для scoring-отчета с goal-alignment блоком.",
    )
    report_parser.set_defaults(handler=handle_report)
def handle_report(namespace: argparse.Namespace) -> int:
    # Выполняем report-команду и сохраняем markdown-отчет.
    print_report_stage(f"build kind={namespace.kind}")
    report_path = build_report(namespace)
    print_report_stage(f"saved_to={report_path}")
    return 0
