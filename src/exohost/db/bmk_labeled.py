# Файл `bmk_labeled.py` слоя `db`.
#
# Этот файл отвечает только за:
# - relation-layer, materialization и SQL-обвязку;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `db` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy.engine import Engine

from exohost.db.bmk_crossmatch import validate_xmatch_batch_id
from exohost.db.bmk_ingestion import build_copy_from_stdin_sql
from exohost.db.bmk_labeled_contracts import (
    B_MK_EXTERNAL_LABELED_COLUMNS,
    B_MK_EXTERNAL_LABELED_CSV_FILENAME,
    B_MK_EXTERNAL_LABELED_RELATION_NAME,
    B_MK_EXTERNAL_LABELED_SOURCE_CROSSMATCH_RELATION_NAME,
    B_MK_EXTERNAL_LABELED_SOURCE_FILTERED_RELATION_NAME,
    BmkExternalLabeledExportSummary,
    BmkExternalLabeledLoadSummary,
)
from exohost.db.bmk_labeled_export import export_bmk_external_labeled_csv
from exohost.db.bmk_labeled_sql import (
    build_bmk_external_labeled_schema_sql,
    build_bmk_external_labeled_source_query,
    build_delete_bmk_external_labeled_batch_sql,
)
from exohost.db.bmk_labeled_stats import (
    count_by_parse_status,
    count_distinct_external_rows,
    count_distinct_source_ids,
    count_duplicate_source_ids,
    count_labeled_rows,
    count_without_luminosity_class,
)
from exohost.db.bmk_labeled_validation import validate_required_bmk_labeled_source_columns


def materialize_bmk_external_labeled_relation(
    engine: Engine,
    *,
    xmatch_batch_id: str,
    filtered_relation_name: str = B_MK_EXTERNAL_LABELED_SOURCE_FILTERED_RELATION_NAME,
    crossmatch_relation_name: str = B_MK_EXTERNAL_LABELED_SOURCE_CROSSMATCH_RELATION_NAME,
    target_relation_name: str = B_MK_EXTERNAL_LABELED_RELATION_NAME,
    chunksize: int = 50_000,
    limit: int | None = None,
) -> BmkExternalLabeledLoadSummary:
    # Строим canonical labeled relation из selected crossmatch и filtered B/mk layer.
    validate_xmatch_batch_id(xmatch_batch_id)
    validate_required_bmk_labeled_source_columns(
        engine,
        filtered_relation_name=filtered_relation_name,
        crossmatch_relation_name=crossmatch_relation_name,
    )
    with TemporaryDirectory(prefix="bmk_external_labeled__") as temp_dir:
        # Сначала выгружаем канонический CSV-срез, а уже потом загружаем его в БД.
        # Это держит export и load в одном воспроизводимом контуре и упрощает отладку.
        export_summary = export_bmk_external_labeled_csv(
            engine,
            output_csv_path=Path(temp_dir) / B_MK_EXTERNAL_LABELED_CSV_FILENAME,
            xmatch_batch_id=xmatch_batch_id,
            filtered_relation_name=filtered_relation_name,
            crossmatch_relation_name=crossmatch_relation_name,
            chunksize=chunksize,
            limit=limit,
        )
        dbapi_connection = engine.raw_connection()
        cursor = dbapi_connection.cursor()
        try:
            # Схему таблицы обновляем перед каждым запуском materialization, чтобы
            # не зависеть от ручного состояния relation в локальной базе.
            for statement in build_bmk_external_labeled_schema_sql(target_relation_name):
                cursor.execute(statement)

            # Удаляем только текущую партию `xmatch_batch_id`, а не весь relation.
            # Это позволяет безопасно пересобирать отдельный батч без полного drop/load.
            cursor.execute(
                build_delete_bmk_external_labeled_batch_sql(
                    target_relation_name,
                    xmatch_batch_id=xmatch_batch_id,
                )
            )
            copy_sql = build_copy_from_stdin_sql(
                target_relation_name,
                columns=B_MK_EXTERNAL_LABELED_COLUMNS,
            )
            with export_summary.output_csv_path.open("r", encoding="utf-8", newline="") as input_file:
                cursor.copy_expert(copy_sql, input_file)

            # После загрузки сразу считаем основные контрольные метрики relation.
            # Эти числа потом используются и для QA, и для review в документации.
            load_summary = BmkExternalLabeledLoadSummary(
                filtered_relation_name=filtered_relation_name,
                crossmatch_relation_name=crossmatch_relation_name,
                target_relation_name=target_relation_name,
                xmatch_batch_id=xmatch_batch_id,
                rows_loaded=count_labeled_rows(
                    cursor,
                    relation_name=target_relation_name,
                    xmatch_batch_id=xmatch_batch_id,
                ),
                distinct_external_rows=count_distinct_external_rows(
                    cursor,
                    relation_name=target_relation_name,
                    xmatch_batch_id=xmatch_batch_id,
                ),
                distinct_source_ids=count_distinct_source_ids(
                    cursor,
                    relation_name=target_relation_name,
                    xmatch_batch_id=xmatch_batch_id,
                ),
                duplicate_source_ids=count_duplicate_source_ids(
                    cursor,
                    relation_name=target_relation_name,
                    xmatch_batch_id=xmatch_batch_id,
                ),
                parsed_rows=count_by_parse_status(
                    cursor,
                    relation_name=target_relation_name,
                    xmatch_batch_id=xmatch_batch_id,
                    parse_status="parsed",
                ),
                partial_rows=count_by_parse_status(
                    cursor,
                    relation_name=target_relation_name,
                    xmatch_batch_id=xmatch_batch_id,
                    parse_status="partial",
                ),
                unsupported_rows=count_by_parse_status(
                    cursor,
                    relation_name=target_relation_name,
                    xmatch_batch_id=xmatch_batch_id,
                    parse_status="unsupported",
                ),
                empty_rows=count_by_parse_status(
                    cursor,
                    relation_name=target_relation_name,
                    xmatch_batch_id=xmatch_batch_id,
                    parse_status="empty",
                ),
                rows_without_luminosity_class=count_without_luminosity_class(
                    cursor,
                    relation_name=target_relation_name,
                    xmatch_batch_id=xmatch_batch_id,
                ),
            )
            dbapi_connection.commit()
            return load_summary
        except Exception:
            # Любая ошибка должна откатить и схему загрузки, и промежуточный batch.
            dbapi_connection.rollback()
            raise
        finally:
            cursor.close()
            dbapi_connection.close()


__all__ = [
    "B_MK_EXTERNAL_LABELED_COLUMNS",
    "B_MK_EXTERNAL_LABELED_CSV_FILENAME",
    "B_MK_EXTERNAL_LABELED_RELATION_NAME",
    "B_MK_EXTERNAL_LABELED_SOURCE_CROSSMATCH_RELATION_NAME",
    "B_MK_EXTERNAL_LABELED_SOURCE_FILTERED_RELATION_NAME",
    "BmkExternalLabeledExportSummary",
    "BmkExternalLabeledLoadSummary",
    "build_bmk_external_labeled_schema_sql",
    "build_bmk_external_labeled_source_query",
    "build_delete_bmk_external_labeled_batch_sql",
    "export_bmk_external_labeled_csv",
    "materialize_bmk_external_labeled_relation",
]
