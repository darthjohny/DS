# Файл `bmk_labeled_contracts.py` слоя `db`.
#
# Этот файл отвечает только за:
# - relation-layer, materialization и SQL-обвязку;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `db` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from exohost.db.bmk_crossmatch import B_MK_EXTERNAL_CROSSMATCH_RELATION_NAME
from exohost.db.bmk_ingestion import B_MK_FILTERED_RELATION_NAME

B_MK_EXTERNAL_LABELED_RELATION_NAME = "lab.gaia_mk_external_labeled"
B_MK_EXTERNAL_LABELED_SOURCE_FILTERED_RELATION_NAME = B_MK_FILTERED_RELATION_NAME
B_MK_EXTERNAL_LABELED_SOURCE_CROSSMATCH_RELATION_NAME = B_MK_EXTERNAL_CROSSMATCH_RELATION_NAME
B_MK_EXTERNAL_LABELED_CSV_FILENAME = "bmk_external_labeled.csv"
B_MK_EXTERNAL_LABELED_COLUMNS: tuple[str, ...] = (
    "xmatch_batch_id",
    "source_id",
    "external_row_id",
    "external_catalog_name",
    "external_object_id",
    "raw_sptype",
    "spectral_class",
    "spectral_subclass",
    "luminosity_class",
    "peculiarity_suffix",
    "label_parse_status",
    "label_parse_notes",
    "xmatch_separation_arcsec",
    "has_source_conflict",
    "source_conflict_count",
)
B_MK_EXTERNAL_LABELED_REQUIRED_FILTERED_COLUMNS: tuple[str, ...] = (
    "external_row_id",
    "external_catalog_name",
    "external_object_id",
    "raw_sptype",
)
B_MK_EXTERNAL_LABELED_REQUIRED_CROSSMATCH_COLUMNS: tuple[str, ...] = (
    "xmatch_batch_id",
    "source_id",
    "external_row_id",
    "xmatch_separation_arcsec",
    "xmatch_selected",
)
NULLABLE_INTEGER_LABELED_COLUMNS: tuple[str, ...] = ("spectral_subclass",)


@dataclass(frozen=True, slots=True)
class BmkExternalLabeledExportSummary:
    # Сводка локального export перед COPY в canonical labeled relation.
    filtered_relation_name: str
    crossmatch_relation_name: str
    output_csv_path: Path
    xmatch_batch_id: str
    rows_exported: int


@dataclass(frozen=True, slots=True)
class BmkExternalLabeledLoadSummary:
    # Фактическая сводка materialized labeled relation после DB-load.
    filtered_relation_name: str
    crossmatch_relation_name: str
    target_relation_name: str
    xmatch_batch_id: str
    rows_loaded: int
    distinct_external_rows: int
    distinct_source_ids: int
    duplicate_source_ids: int
    parsed_rows: int
    partial_rows: int
    unsupported_rows: int
    empty_rows: int
    rows_without_luminosity_class: int
