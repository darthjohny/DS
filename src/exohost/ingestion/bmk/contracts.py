# Файл `contracts.py` слоя `ingestion`.
#
# Этот файл отвечает только за:
# - разбор и нормализацию внешних B/mk-меток;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `ingestion` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pandas as pd

B_MK_CATALOG_NAME = "bmk"
B_MK_RAW_CSV_FILENAME = "bmk_external_raw.csv"
B_MK_FILTERED_CSV_FILENAME = "bmk_external_filtered.csv"
B_MK_REJECTED_CSV_FILENAME = "bmk_external_rejected.csv"
B_MK_RAW_COLUMNS: tuple[str, ...] = (
    "external_row_id",
    "external_catalog_name",
    "external_object_id",
    "ra_deg",
    "dec_deg",
    "raw_sptype",
    "raw_magnitude",
    "raw_source_bibcode",
    "raw_notes",
)
B_MK_FILTERED_COLUMNS: tuple[str, ...] = (
    "external_row_id",
    "external_catalog_name",
    "external_object_id",
    "ra_deg",
    "dec_deg",
    "raw_sptype",
    "raw_magnitude",
    "raw_source_bibcode",
    "raw_notes",
    "spectral_prefix",
    "spectral_class",
    "spectral_subclass",
    "luminosity_class",
    "parse_status",
    "parse_note",
    "has_supported_prefix",
    "has_coordinates",
    "has_raw_sptype",
    "ready_for_gaia_crossmatch",
)
B_MK_REJECTED_COLUMNS: tuple[str, ...] = (
    "external_row_id",
    "external_catalog_name",
    "external_object_id",
    "ra_deg",
    "dec_deg",
    "raw_sptype",
    "raw_magnitude",
    "raw_source_bibcode",
    "raw_notes",
    "spectral_prefix",
    "reject_reason",
)
SUPPORTED_SPECTRAL_PREFIXES: tuple[str, ...] = ("O", "B", "A", "F", "G", "K", "M")
type RejectReason = Literal["missing_coordinates", "missing_raw_sptype", "unsupported_spectral_prefix"]


@dataclass(frozen=True)
class BmkCatalogSource:
    # Локальные пути к исходным файлам каталога B/mk.
    readme_path: Path
    data_path: Path


@dataclass(frozen=True)
class BmkImportSummary:
    # Сводка качества импорта внешнего B/mk-каталога.
    total_rows: int
    rows_with_coordinates: int
    rows_with_raw_sptype: int
    rows_with_supported_spectral_prefix: int
    exported_rows: int


@dataclass(frozen=True)
class BmkPrimaryFilterSummary:
    # Сводка первичной фильтрации после чтения B/mk.
    total_rows: int
    filtered_rows: int
    rejected_rows: int
    rows_ready_for_gaia_crossmatch: int
    rejected_missing_coordinates: int
    rejected_missing_raw_sptype: int
    rejected_unsupported_spectral_prefix: int


@dataclass(frozen=True)
class BmkExportPaths:
    # Пути к трем staging CSV одного B/mk прогона.
    raw_csv_path: Path
    filtered_csv_path: Path
    rejected_csv_path: Path


@dataclass(frozen=True)
class BmkExportBundle:
    # Итог одного parser/export прогона без DB-write логики.
    source: BmkCatalogSource | None
    export_paths: BmkExportPaths
    import_summary: BmkImportSummary
    primary_filter_summary: BmkPrimaryFilterSummary


@dataclass(frozen=True)
class BmkTransformBundle:
    # Результат одного полного прохода по CDS-таблице B/mk.
    raw_frame: pd.DataFrame
    filtered_frame: pd.DataFrame
    rejected_frame: pd.DataFrame
    import_summary: BmkImportSummary
    primary_filter_summary: BmkPrimaryFilterSummary
