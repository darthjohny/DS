# Файл `support.py` слоя `cli`.
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
from pathlib import Path
from typing import Literal

from exohost.reporting.benchmark_report import save_benchmark_report
from exohost.reporting.ranking_report import save_ranking_report
from exohost.reporting.scoring_report import save_scoring_report

ReportKind = Literal["benchmark", "ranking", "scoring"]


def print_report_stage(message: str) -> None:
    # Печатаем короткий статус report-команды.
    print(f"[report] {message}")


def build_report(namespace: argparse.Namespace) -> Path:
    # Собираем markdown-отчет по типу run_dir.
    report_kind = namespace.kind
    if report_kind == "benchmark":
        return save_benchmark_report(
            namespace.run_dir,
            output_path=namespace.output_path,
        )
    if report_kind == "ranking":
        return save_ranking_report(
            namespace.run_dir,
            output_path=namespace.output_path,
            top_rows=namespace.top_rows,
        )
    return save_scoring_report(
        namespace.run_dir,
        ranking_run_dir=namespace.ranking_run_dir,
        output_path=namespace.output_path,
        top_rows=namespace.top_rows,
    )
