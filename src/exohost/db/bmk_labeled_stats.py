# Файл `bmk_labeled_stats.py` слоя `db`.
#
# Этот файл отвечает только за:
# - relation-layer, materialization и SQL-обвязку;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `db` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from typing import Protocol, SupportsInt, cast

from exohost.db.bmk_labeled_sql import quote_text_literal
from exohost.db.relations import quote_relation_name


class BmkLabeledCursorProtocol(Protocol):
    # Минимальный DB-API cursor contract для COUNT-запросов в labeled-слое.
    def execute(self, operation: str) -> object: ...

    def fetchone(self) -> tuple[object, ...] | None: ...


def count_labeled_rows(
    cursor: BmkLabeledCursorProtocol,
    *,
    relation_name: str,
    xmatch_batch_id: str,
) -> int:
    # Считаем строки materialized labeled batch-а.
    return fetch_single_count(
        cursor,
        f"""
SELECT COUNT(*)
FROM {quote_relation_name(relation_name, validate_identifiers=True)}
WHERE "xmatch_batch_id" = {quote_text_literal(xmatch_batch_id)}
""".strip(),
    )


def count_distinct_external_rows(
    cursor: BmkLabeledCursorProtocol,
    *,
    relation_name: str,
    xmatch_batch_id: str,
) -> int:
    # Считаем уникальные external_row_id внутри labeled batch-а.
    return fetch_single_count(
        cursor,
        f"""
SELECT COUNT(DISTINCT "external_row_id")
FROM {quote_relation_name(relation_name, validate_identifiers=True)}
WHERE "xmatch_batch_id" = {quote_text_literal(xmatch_batch_id)}
""".strip(),
    )


def count_distinct_source_ids(
    cursor: BmkLabeledCursorProtocol,
    *,
    relation_name: str,
    xmatch_batch_id: str,
) -> int:
    # Считаем число различных Gaia source_id в labeled batch-е.
    return fetch_single_count(
        cursor,
        f"""
SELECT COUNT(DISTINCT "source_id")
FROM {quote_relation_name(relation_name, validate_identifiers=True)}
WHERE "xmatch_batch_id" = {quote_text_literal(xmatch_batch_id)}
""".strip(),
    )


def count_duplicate_source_ids(
    cursor: BmkLabeledCursorProtocol,
    *,
    relation_name: str,
    xmatch_batch_id: str,
) -> int:
    # Считаем, сколько source_id имеют больше одной labeled строки внутри batch-а.
    return fetch_single_count(
        cursor,
        f"""
SELECT COUNT(*)
FROM (
    SELECT "source_id"
    FROM {quote_relation_name(relation_name, validate_identifiers=True)}
    WHERE "xmatch_batch_id" = {quote_text_literal(xmatch_batch_id)}
    GROUP BY "source_id"
    HAVING COUNT(*) > 1
) AS duplicate_source_rows
""".strip(),
    )


def count_by_parse_status(
    cursor: BmkLabeledCursorProtocol,
    *,
    relation_name: str,
    xmatch_batch_id: str,
    parse_status: str,
) -> int:
    # Считаем labeled строки конкретного parse_status внутри batch-а.
    return fetch_single_count(
        cursor,
        f"""
SELECT COUNT(*)
FROM {quote_relation_name(relation_name, validate_identifiers=True)}
WHERE "xmatch_batch_id" = {quote_text_literal(xmatch_batch_id)}
  AND "label_parse_status" = {quote_text_literal(parse_status)}
""".strip(),
    )


def count_without_luminosity_class(
    cursor: BmkLabeledCursorProtocol,
    *,
    relation_name: str,
    xmatch_batch_id: str,
) -> int:
    # Считаем labeled строки без `luminosity_class`, чтобы видеть неполные MK labels.
    return fetch_single_count(
        cursor,
        f"""
SELECT COUNT(*)
FROM {quote_relation_name(relation_name, validate_identifiers=True)}
WHERE "xmatch_batch_id" = {quote_text_literal(xmatch_batch_id)}
  AND "luminosity_class" IS NULL
""".strip(),
    )


def fetch_single_count(
    cursor: BmkLabeledCursorProtocol,
    sql: str,
) -> int:
    # Выполняем COUNT-запрос и возвращаем целое значение.
    cursor.execute(sql)
    result = cursor.fetchone()
    if result is None:
        raise RuntimeError("COUNT query returned no rows")
    return _normalize_count_value(result[0])


def _normalize_count_value(value: object) -> int:
    # Держим явную границу типов на DB-ответе, а не надеемся на неявный int().
    if not hasattr(value, "__int__"):
        raise TypeError(
            "COUNT query returned a non-integer-like value: "
            f"{type(value).__name__}",
        )
    return int(cast(SupportsInt, value))
