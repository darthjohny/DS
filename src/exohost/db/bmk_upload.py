# Файл `bmk_upload.py` слоя `db`.
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

import pandas as pd
from sqlalchemy.engine import Engine

from exohost.db.bmk_ingestion import B_MK_FILTERED_RELATION_NAME
from exohost.db.relations import (
    quote_identifier,
    quote_relation_name,
    validate_identifier,
)

B_MK_GAIA_UPLOAD_CSV_FILENAME = "bmk_gaia_upload.csv"
B_MK_GAIA_UPLOAD_SOURCE_RELATION_NAME = B_MK_FILTERED_RELATION_NAME
B_MK_GAIA_UPLOAD_COLUMNS: tuple[str, ...] = (
    "external_row_id",
    "external_catalog_name",
    "external_object_id",
    "ra_deg",
    "dec_deg",
    "raw_sptype",
)


@dataclass(frozen=True, slots=True)
class BmkGaiaUploadExportSummary:
    # Сводка локального export шага для будущего Gaia upload.
    relation_name: str
    output_csv_path: Path
    rows_exported: int


def build_bmk_gaia_upload_query(
    relation_name: str = B_MK_GAIA_UPLOAD_SOURCE_RELATION_NAME,
    *,
    limit: int | None = None,
) -> str:
    # Собираем минимальный reproducible SELECT для Gaia upload table.
    relation_sql = quote_relation_name(
        relation_name,
        validate_identifiers=True,
    )
    selected_columns_sql = ",\n        ".join(
        quote_identifier(validate_identifier(column_name))
        for column_name in B_MK_GAIA_UPLOAD_COLUMNS
    )
    limit_sql = f"LIMIT {int(limit)}" if limit is not None else ""
    return f"""
    SELECT
        {selected_columns_sql}
    FROM {relation_sql}
    WHERE "ready_for_gaia_crossmatch" IS TRUE
      AND "ra_deg" IS NOT NULL
      AND "dec_deg" IS NOT NULL
      AND "raw_sptype" IS NOT NULL
    ORDER BY "external_row_id" ASC
    {limit_sql};
    """


def load_bmk_gaia_upload_frame(
    engine: Engine,
    *,
    relation_name: str = B_MK_GAIA_UPLOAD_SOURCE_RELATION_NAME,
    limit: int | None = None,
) -> pd.DataFrame:
    # Загружаем upload-ready frame из локальной БД по зафиксированному contract.
    query = build_bmk_gaia_upload_query(
        relation_name=relation_name,
        limit=limit,
    )
    return pd.read_sql(query, engine).reset_index(drop=True)


def export_bmk_gaia_upload_csv(
    engine: Engine,
    *,
    output_csv_path: Path,
    relation_name: str = B_MK_GAIA_UPLOAD_SOURCE_RELATION_NAME,
    limit: int | None = None,
) -> BmkGaiaUploadExportSummary:
    # Выгружаем upload-ready слой из БД в reproducible CSV.
    upload_frame = load_bmk_gaia_upload_frame(
        engine,
        relation_name=relation_name,
        limit=limit,
    )
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    upload_frame.to_csv(
        output_csv_path,
        index=False,
        na_rep="",
        encoding="utf-8",
        lineterminator="\n",
    )
    return BmkGaiaUploadExportSummary(
        relation_name=relation_name,
        output_csv_path=output_csv_path,
        rows_exported=int(upload_frame.shape[0]),
    )
