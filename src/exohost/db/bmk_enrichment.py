# Файл `bmk_enrichment.py` слоя `db`.
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
from typing import Any, cast

import pandas as pd
from sqlalchemy.engine import Engine

from exohost.db.bmk_crossmatch import validate_xmatch_batch_id
from exohost.db.bmk_labeled import B_MK_EXTERNAL_LABELED_RELATION_NAME
from exohost.db.relations import quote_relation_name, relation_columns

B_MK_GAIA_ENRICHMENT_SOURCE_RELATION_NAME = B_MK_EXTERNAL_LABELED_RELATION_NAME
B_MK_GAIA_ENRICHMENT_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "xmatch_batch_id",
    "source_id",
    "has_source_conflict",
)
B_MK_GAIA_ENRICHMENT_COLUMNS: tuple[str, ...] = ("source_id",)
B_MK_GAIA_ENRICHMENT_MANIFEST_COLUMNS: tuple[str, ...] = (
    "batch_index",
    "csv_filename",
    "rows_exported",
    "source_id_min",
    "source_id_max",
)
B_MK_GAIA_ENRICHMENT_MANIFEST_FILENAME = "bmk_gaia_enrichment_manifest.csv"
B_MK_GAIA_ENRICHMENT_QUERY_TEMPLATE_FILENAME = "gaia_radius_flame_enrichment_template.sql"
DEFAULT_GAIA_ENRICHMENT_USER_TABLE_PLACEHOLDER = "user_<login>.<uploaded_source_id_batch>"


@dataclass(frozen=True, slots=True)
class BmkGaiaEnrichmentBatchSummary:
    # Сводка по одному upload-ready source_id batch для ручного Gaia enrichment.
    batch_index: int
    output_csv_path: Path
    rows_exported: int
    source_id_min: int | None
    source_id_max: int | None


@dataclass(frozen=True, slots=True)
class BmkGaiaEnrichmentExportSummary:
    # Общая сводка локального export шага перед chunk-wise Gaia enrichment.
    relation_name: str
    output_dir: Path
    manifest_path: Path
    query_template_path: Path
    xmatch_batch_id: str
    only_conflict_free: bool
    total_rows_exported: int
    total_batches: int
    batch_size: int


def build_bmk_gaia_enrichment_source_query(
    *,
    relation_name: str = B_MK_GAIA_ENRICHMENT_SOURCE_RELATION_NAME,
    xmatch_batch_id: str,
    only_conflict_free: bool = True,
    limit: int | None = None,
) -> str:
    # Собираем минимальный SELECT с `source_id` для повторного Gaia enrichment.
    relation_sql = quote_relation_name(
        relation_name,
        validate_identifiers=True,
    )
    batch_id_sql = _quote_text_literal(validate_xmatch_batch_id(xmatch_batch_id))
    conflict_sql = (
        '\n  AND "has_source_conflict" IS FALSE'
        if only_conflict_free
        else ""
    )
    limit_sql = f"\nLIMIT {int(limit)}" if limit is not None else ""
    return f"""
    SELECT DISTINCT
        "source_id"
    FROM {relation_sql}
    WHERE "xmatch_batch_id" = {batch_id_sql}
      AND "source_id" IS NOT NULL{conflict_sql}
    ORDER BY "source_id" ASC
    {limit_sql};
    """


def load_bmk_gaia_enrichment_source_frame(
    engine: Engine,
    *,
    relation_name: str = B_MK_GAIA_ENRICHMENT_SOURCE_RELATION_NAME,
    xmatch_batch_id: str,
    only_conflict_free: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    # Загружаем source_id frame для batch-wise Gaia enrichment.
    _validate_required_source_columns(engine, relation_name=relation_name)
    source_query = build_bmk_gaia_enrichment_source_query(
        relation_name=relation_name,
        xmatch_batch_id=xmatch_batch_id,
        only_conflict_free=only_conflict_free,
        limit=limit,
    )
    source_frame = pd.read_sql(source_query, engine).reset_index(drop=True)
    prepared_frame = source_frame.reindex(columns=B_MK_GAIA_ENRICHMENT_COLUMNS).copy()
    normalized_source_id = pd.to_numeric(
        prepared_frame["source_id"],
        errors="coerce",
    )
    prepared_frame["source_id"] = pd.Series(
        cast(Any, normalized_source_id),
        dtype="Int64",
    )
    prepared_frame = prepared_frame.dropna(subset=["source_id"]).reset_index(drop=True)
    if not prepared_frame.empty:
        prepared_frame["source_id"] = prepared_frame["source_id"].astype("int64")
    return prepared_frame


def build_gaia_radius_flame_query_template(
    *,
    user_table_name: str = DEFAULT_GAIA_ENRICHMENT_USER_TABLE_PLACEHOLDER,
) -> str:
    # Строим минимальный ADQL template для вытягивания `radius_flame` по source_id.
    return f"""
SELECT
    u.source_id,
    ap.radius_flame
FROM {user_table_name} AS u
LEFT JOIN gaiadr3.astrophysical_parameters AS ap
    ON ap.source_id = u.source_id
ORDER BY u.source_id ASC;
""".strip()


def export_bmk_gaia_enrichment_batches(
    engine: Engine,
    *,
    output_dir: Path,
    xmatch_batch_id: str,
    relation_name: str = B_MK_GAIA_ENRICHMENT_SOURCE_RELATION_NAME,
    batch_size: int = 50_000,
    only_conflict_free: bool = True,
    limit: int | None = None,
) -> BmkGaiaEnrichmentExportSummary:
    # Выгружаем conflict-free source_id в набор маленьких CSV batch-файлов.
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    source_frame = load_bmk_gaia_enrichment_source_frame(
        engine,
        relation_name=relation_name,
        xmatch_batch_id=xmatch_batch_id,
        only_conflict_free=only_conflict_free,
        limit=limit,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_summaries: list[BmkGaiaEnrichmentBatchSummary] = []
    for batch_index, batch_frame in enumerate(
        _iter_source_id_batches(source_frame, batch_size=batch_size),
        start=1,
    ):
        output_csv_path = output_dir / _build_batch_filename(batch_index)
        batch_frame.to_csv(
            output_csv_path,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        batch_summaries.append(
            BmkGaiaEnrichmentBatchSummary(
                batch_index=batch_index,
                output_csv_path=output_csv_path,
                rows_exported=int(batch_frame.shape[0]),
                source_id_min=_get_batch_source_id_min(batch_frame),
                source_id_max=_get_batch_source_id_max(batch_frame),
            )
        )

    manifest_path = output_dir / B_MK_GAIA_ENRICHMENT_MANIFEST_FILENAME
    _write_enrichment_manifest(batch_summaries, manifest_path=manifest_path)

    query_template_path = output_dir / B_MK_GAIA_ENRICHMENT_QUERY_TEMPLATE_FILENAME
    query_template_path.write_text(
        build_gaia_radius_flame_query_template(),
        encoding="utf-8",
    )

    return BmkGaiaEnrichmentExportSummary(
        relation_name=relation_name,
        output_dir=output_dir,
        manifest_path=manifest_path,
        query_template_path=query_template_path,
        xmatch_batch_id=xmatch_batch_id,
        only_conflict_free=only_conflict_free,
        total_rows_exported=int(source_frame.shape[0]),
        total_batches=len(batch_summaries),
        batch_size=batch_size,
    )


def _iter_source_id_batches(
    source_frame: pd.DataFrame,
    *,
    batch_size: int,
) -> list[pd.DataFrame]:
    # Режем deterministic source_id frame на небольшие upload-ready батчи.
    return [
        source_frame.iloc[start_index : start_index + batch_size].copy()
        for start_index in range(0, int(source_frame.shape[0]), batch_size)
    ]


def _build_batch_filename(batch_index: int) -> str:
    # Имена batch-файлов делаем стабильными и сортируемыми.
    return f"bmk_gaia_enrichment_batch_{batch_index:04d}.csv"


def _write_enrichment_manifest(
    batch_summaries: list[BmkGaiaEnrichmentBatchSummary],
    *,
    manifest_path: Path,
) -> None:
    # Сохраняем локальный manifest, чтобы руками не считать batch boundaries.
    manifest_rows = [
        {
            "batch_index": batch_summary.batch_index,
            "csv_filename": batch_summary.output_csv_path.name,
            "rows_exported": batch_summary.rows_exported,
            "source_id_min": batch_summary.source_id_min,
            "source_id_max": batch_summary.source_id_max,
        }
        for batch_summary in batch_summaries
    ]
    pd.DataFrame(
        manifest_rows,
        columns=B_MK_GAIA_ENRICHMENT_MANIFEST_COLUMNS,
    ).to_csv(
        manifest_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )


def _get_batch_source_id_min(batch_frame: pd.DataFrame) -> int | None:
    # Возвращаем нижнюю границу source_id для audit manifest.
    if batch_frame.empty:
        return None
    return int(batch_frame["source_id"].iloc[0])


def _get_batch_source_id_max(batch_frame: pd.DataFrame) -> int | None:
    # Возвращаем верхнюю границу source_id для audit manifest.
    if batch_frame.empty:
        return None
    return int(batch_frame["source_id"].iloc[-1])


def _validate_required_source_columns(
    engine: Engine,
    *,
    relation_name: str,
) -> None:
    # Проверяем, что source relation годится для построения enrichment batches.
    available_columns = set(relation_columns(engine, relation_name))
    missing_columns = [
        column_name
        for column_name in B_MK_GAIA_ENRICHMENT_REQUIRED_SOURCE_COLUMNS
        if column_name not in available_columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required columns for B/mk Gaia enrichment batch export: "
            + ", ".join(missing_columns)
        )


def _quote_text_literal(value: str) -> str:
    # Экранируем текстовый literal для узкого SQL-контура batch export шага.
    return "'" + value.replace("'", "''") + "'"
