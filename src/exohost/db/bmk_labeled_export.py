# Файл `bmk_labeled_export.py` слоя `db`.
#
# Этот файл отвечает только за:
# - relation-layer, materialization и SQL-обвязку;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `db` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pandas as pd
from sqlalchemy.engine import Engine

from exohost.db.bmk_labeled_contracts import (
    B_MK_EXTERNAL_LABELED_COLUMNS,
    B_MK_EXTERNAL_LABELED_SOURCE_CROSSMATCH_RELATION_NAME,
    B_MK_EXTERNAL_LABELED_SOURCE_FILTERED_RELATION_NAME,
    NULLABLE_INTEGER_LABELED_COLUMNS,
    BmkExternalLabeledExportSummary,
)
from exohost.db.bmk_labeled_sql import build_bmk_external_labeled_source_query
from exohost.ingestion.mk_label_parser import parse_mk_label


def export_bmk_external_labeled_csv(
    engine: Engine,
    *,
    output_csv_path: Path,
    xmatch_batch_id: str,
    filtered_relation_name: str = B_MK_EXTERNAL_LABELED_SOURCE_FILTERED_RELATION_NAME,
    crossmatch_relation_name: str = B_MK_EXTERNAL_LABELED_SOURCE_CROSSMATCH_RELATION_NAME,
    chunksize: int = 50_000,
    limit: int | None = None,
) -> BmkExternalLabeledExportSummary:
    # Экспортируем normalized labeled layer во временный CSV перед COPY в Postgres.
    query = build_bmk_external_labeled_source_query(
        filtered_relation_name=filtered_relation_name,
        crossmatch_relation_name=crossmatch_relation_name,
        xmatch_batch_id=xmatch_batch_id,
        limit=limit,
    )
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows_exported = 0
    write_header = True

    for source_chunk in iter_bmk_external_labeled_source_chunks(
        engine,
        query=query,
        chunksize=chunksize,
    ):
        labeled_chunk = build_bmk_external_labeled_chunk(source_chunk)
        if labeled_chunk.empty:
            continue

        prepared_chunk = prepare_bmk_external_labeled_export_frame(labeled_chunk)
        prepared_chunk.to_csv(
            output_csv_path,
            mode="w" if write_header else "a",
            index=False,
            header=write_header,
            encoding="utf-8",
            lineterminator="\n",
            na_rep="",
        )
        write_header = False
        rows_exported += int(prepared_chunk.shape[0])

    if write_header:
        pd.DataFrame(columns=B_MK_EXTERNAL_LABELED_COLUMNS).to_csv(
            output_csv_path,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )

    return BmkExternalLabeledExportSummary(
        filtered_relation_name=filtered_relation_name,
        crossmatch_relation_name=crossmatch_relation_name,
        output_csv_path=output_csv_path,
        xmatch_batch_id=xmatch_batch_id,
        rows_exported=rows_exported,
    )


def iter_bmk_external_labeled_source_chunks(
    engine: Engine,
    *,
    query: str,
    chunksize: int,
) -> Iterator[pd.DataFrame]:
    # Идем по selected join chunk-ами, чтобы не тянуть весь labeled source в память.
    chunk_iter = pd.read_sql(query, engine, chunksize=chunksize)
    return iter(chunk_iter)


def build_bmk_external_labeled_chunk(
    source_chunk: pd.DataFrame,
) -> pd.DataFrame:
    # Преобразуем selected join chunk в normalized labeled records через канонический parser.
    labeled_rows: list[dict[str, object]] = []
    for source_row in source_chunk.to_dict(orient="records"):
        raw_sptype = str(source_row["raw_sptype"])
        parse_result = parse_mk_label(raw_sptype)
        source_conflict_count = int(source_row["source_conflict_count"])
        labeled_rows.append(
            {
                "xmatch_batch_id": str(source_row["xmatch_batch_id"]),
                "source_id": int(source_row["source_id"]),
                "external_row_id": int(source_row["external_row_id"]),
                "external_catalog_name": str(source_row["external_catalog_name"]),
                "external_object_id": source_row["external_object_id"],
                "raw_sptype": raw_sptype,
                "spectral_class": parse_result.spectral_class,
                "spectral_subclass": parse_result.spectral_subclass,
                "luminosity_class": parse_result.luminosity_class,
                "peculiarity_suffix": parse_result.peculiarity_suffix,
                "label_parse_status": parse_result.parse_status,
                "label_parse_notes": parse_result.parse_note,
                "xmatch_separation_arcsec": float(source_row["xmatch_separation_arcsec"]),
                "has_source_conflict": source_conflict_count > 1,
                "source_conflict_count": source_conflict_count,
            }
        )

    return pd.DataFrame(labeled_rows, columns=B_MK_EXTERNAL_LABELED_COLUMNS)


def prepare_bmk_external_labeled_export_frame(frame: pd.DataFrame) -> pd.DataFrame:
    # Фиксируем порядок колонок и nullable integer-поля перед CSV export/COPY.
    prepared_frame = frame.reindex(columns=B_MK_EXTERNAL_LABELED_COLUMNS).copy()
    for column_name in NULLABLE_INTEGER_LABELED_COLUMNS:
        if column_name in prepared_frame.columns:
            normalized_column = pd.to_numeric(
                prepared_frame[column_name],
                errors="coerce",
            )
            prepared_frame[column_name] = pd.Series(
                cast(Any, normalized_column),
                dtype="Int64",
            )
    return prepared_frame
