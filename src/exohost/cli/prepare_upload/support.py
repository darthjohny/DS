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
from datetime import UTC, datetime
from pathlib import Path

from exohost.db.bmk_upload import (
    B_MK_GAIA_UPLOAD_CSV_FILENAME,
    BmkGaiaUploadExportSummary,
)

DEFAULT_GAIA_UPLOAD_OUTPUT_DIR = Path("artifacts/gaia_upload")


def print_prepare_upload_stage(message: str) -> None:
    # Печатаем короткий статус локального Gaia upload export шага.
    print(f"[prepare-upload] {message}")


def resolve_gaia_upload_output_path(namespace: argparse.Namespace) -> Path:
    # Создаем отдельный run_dir под один local Gaia upload export.
    base_dir = Path(namespace.output_dir)
    run_stamp = datetime.now(UTC).strftime("%Y_%m_%d_%H%M%S_%f")
    return base_dir / f"bmk_gaia_upload__{run_stamp}" / B_MK_GAIA_UPLOAD_CSV_FILENAME


def print_gaia_upload_export_summary(summary: BmkGaiaUploadExportSummary) -> None:
    # Печатаем путь и row count локального upload artifact.
    print(
        "[prepare-upload] export_summary "
        f"relation={summary.relation_name} "
        f"rows_exported={summary.rows_exported}"
    )
    print(f"[artifacts] gaia_upload_csv={summary.output_csv_path}")
