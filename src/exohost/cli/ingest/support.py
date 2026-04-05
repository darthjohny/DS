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

from exohost.db.bmk_ingestion import BmkDatabaseLoadSummary
from exohost.ingestion.bmk.contracts import (
    BmkCatalogSource,
    BmkExportBundle,
    BmkImportSummary,
    BmkPrimaryFilterSummary,
)

DEFAULT_INGEST_OUTPUT_DIR = Path("artifacts/ingestion")


def print_ingest_stage(message: str) -> None:
    # Печатаем короткий статус ingest-команды.
    print(f"[ingest] {message}")


def build_bmk_catalog_source(namespace: argparse.Namespace) -> BmkCatalogSource:
    # Собираем и валидируем локальные пути к файлам B/mk.
    readme_path = Path(namespace.readme_path)
    data_path = Path(namespace.data_path)
    if not readme_path.exists():
        raise FileNotFoundError(f"B/mk ReadMe does not exist: {readme_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"B/mk data file does not exist: {data_path}")
    return BmkCatalogSource(
        readme_path=readme_path,
        data_path=data_path,
    )


def resolve_ingest_output_dir(namespace: argparse.Namespace) -> Path:
    # Создаем отдельный run_dir под один ingest-прогон.
    base_dir = Path(namespace.output_dir)
    run_stamp = datetime.now(UTC).strftime("%Y_%m_%d_%H%M%S_%f")
    return base_dir / f"bmk_ingestion__{run_stamp}"


def print_import_summary(summary: BmkImportSummary) -> None:
    # Печатаем ключевые метрики сырого импорта B/mk.
    print(
        "[ingest] raw_summary "
        f"total_rows={summary.total_rows} "
        f"rows_with_coordinates={summary.rows_with_coordinates} "
        f"rows_with_raw_sptype={summary.rows_with_raw_sptype} "
        f"rows_with_supported_spectral_prefix={summary.rows_with_supported_spectral_prefix} "
        f"exported_rows={summary.exported_rows}"
    )


def print_primary_filter_summary(summary: BmkPrimaryFilterSummary) -> None:
    # Печатаем ключевые метрики primary-filter слоя.
    print(
        "[ingest] filter_summary "
        f"total_rows={summary.total_rows} "
        f"filtered_rows={summary.filtered_rows} "
        f"rejected_rows={summary.rejected_rows} "
        f"rows_ready_for_gaia_crossmatch={summary.rows_ready_for_gaia_crossmatch} "
        f"rejected_missing_coordinates={summary.rejected_missing_coordinates} "
        f"rejected_missing_raw_sptype={summary.rejected_missing_raw_sptype} "
        f"rejected_unsupported_spectral_prefix={summary.rejected_unsupported_spectral_prefix}"
    )


def print_export_paths(bundle: BmkExportBundle) -> None:
    # Печатаем пути к трем staging CSV.
    print(f"[artifacts] raw_csv={bundle.export_paths.raw_csv_path}")
    print(f"[artifacts] filtered_csv={bundle.export_paths.filtered_csv_path}")
    print(f"[artifacts] rejected_csv={bundle.export_paths.rejected_csv_path}")


def print_db_load_summary(summary: BmkDatabaseLoadSummary) -> None:
    # Печатаем фактические row counts после загрузки в БД.
    print(
        "[ingest] db_summary "
        f"raw_relation={summary.raw_relation_name} "
        f"raw_rows_loaded={summary.raw_rows_loaded} "
        f"filtered_relation={summary.filtered_relation_name} "
        f"filtered_rows_loaded={summary.filtered_rows_loaded} "
        f"rejected_relation={summary.rejected_relation_name} "
        f"rejected_rows_loaded={summary.rejected_rows_loaded}"
    )
