"""Входной слой пайплайна.

Назначение файла:
    - валидировать входной датасет перед запуском основного пайплайна;
    - не давать `star_orchestrator.py` стартовать на битых или неполных
      данных;
    - фиксировать статус датасета в БД отдельной registry-таблицей.

Почему статус хранится отдельно:
    - саму таблицу `public.gaia_dr3_training` менять под служебные нужды
      не стоит;
    - статус относится ко всему набору данных, а не к каждой строке;
    - registry-слой удобнее для воспроизводимости и последующей отладки.

Что делает этот файл:
    1. Проверяет существование relation в БД.
    2. Проверяет схему входного набора.
    3. Считает базовые метрики качества:
       - количество строк;
       - NULL по обязательным полям;
       - число дублей source_id;
       - базовые sanity-check по диапазонам.
    4. Записывает verdict в `lab.input_dataset_registry`.
    5. При успешной валидации может пометить датасет как `READY`.

Что не делает:
    - не запускает модели;
    - не считает final_score;
    - не пишет результаты распознавания или приоритизации.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TypedDict

from sqlalchemy import text
from sqlalchemy.engine import Engine

from infra.db import (
    load_dotenv_local as _load_dotenv_local,
)
from infra.db import (
    make_engine_from_env as _make_engine_from_env,
)
from infra.relations import (
    relation_columns as _relation_columns,
)
from infra.relations import (
    relation_exists as _relation_exists,
)
from infra.relations import (
    split_relation_name as _split_relation_name,
)

DEFAULT_INPUT_RELATION = "public.gaia_dr3_training"
REGISTRY_TABLE = "lab.input_dataset_registry"

REQUIRED_COLUMNS: tuple[str, ...] = (
    "source_id",
    "ra",
    "dec",
    "teff_gspphot",
    "logg_gspphot",
    "radius_gspphot",
    "mh_gspphot",
    "parallax",
    "parallax_over_error",
    "ruwe",
)

OPTIONAL_COLUMNS: tuple[str, ...] = (
    "bp_rp",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "validation_factor",
)


class DatasetStatus(StrEnum):
    """Поддерживаемые статусы входного датасета."""

    LOADING = "LOADING"
    VALIDATED = "VALIDATED"
    READY = "READY"
    FAILED = "FAILED"


@dataclass(frozen=True)
class DatasetSummary:
    """Агрегированная сводка по входному relation перед запуском pipeline.

    Хранит SQL-агрегаты, которые нужны для принятия verdict по датасету:
    размер набора, число NULL в обязательных колонках, количество дублей
    `source_id` и минимальные sanity-check значения по ключевым полям.
    """

    relation_name: str
    row_count: int
    n_source_id_null: int
    n_coords_null: int
    n_teff_null: int
    n_logg_null: int
    n_radius_null: int
    n_mh_null: int
    n_parallax_null: int
    n_plx_err_null: int
    n_ruwe_null: int
    n_duplicate_source_ids: int
    min_teff: float | None
    min_radius: float | None
    min_ruwe: float | None


@dataclass(frozen=True)
class DatasetValidationResult:
    """Полный результат валидации входного датасета.

    Объединяет вычисленную сводку, итоговый статус, списки ошибок и
    предупреждений, а также метку времени, с которой результат будет
    записан в registry-таблицу.
    """

    relation_name: str
    source_name: str
    status: DatasetStatus
    summary: DatasetSummary
    missing_required_columns: tuple[str, ...]
    missing_optional_columns: tuple[str, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    validated_at: datetime


class RegistryPayload(TypedDict):
    """Типизированный payload для upsert в `lab.input_dataset_registry`.

    Структура повторяет схему registry-таблицы и используется перед
    записью результата валидации через параметризованный SQL-запрос.
    """

    relation_name: str
    source_name: str
    status: str
    row_count: int
    n_source_id_null: int
    n_coords_null: int
    n_teff_null: int
    n_logg_null: int
    n_radius_null: int
    n_mh_null: int
    n_parallax_null: int
    n_plx_err_null: int
    n_ruwe_null: int
    n_duplicate_source_ids: int
    min_teff: float | None
    min_radius: float | None
    min_ruwe: float | None
    validated_at: datetime
    notes: str


def load_dotenv_local(dotenv_path: str = ".env") -> None:
    """Загрузить `.env` через общий helper из `infra.db`.

    Функция сохранена в `input_layer` как совместимая точка доступа,
    чтобы CLI-слой валидации не зависел от деталей внутреннего импорта.
    """
    _load_dotenv_local(dotenv_path)


def make_engine_from_env() -> Engine:
    """Создать SQLAlchemy engine для сценария входной валидации.

    Источник подключения
    --------------------
    Использует общий bootstrap из `infra.db` и читает `DATABASE_URL`
    либо набор переменных `PG*` с fallback на локальный `.env`.
    """
    return _make_engine_from_env(
        missing_message=(
            "Database connection is missing. "
            "Set DATABASE_URL or PG* variables."
        ),
    )


def parse_relation_name(relation_name: str) -> tuple[str, str]:
    """Разделить relation на `schema.table` с валидацией идентификаторов.

    Функция делегирует разбор в `infra.relations`, но включает строгую
    проверку идентификаторов, потому что имя relation далее подставляется
    в SQL-текст валидационных запросов.
    """
    return _split_relation_name(
        relation_name,
        validate_identifiers=True,
    )


def relation_exists(engine: Engine, relation_name: str) -> bool:
    """Проверить, что входной relation существует именно как таблица.

    Для input-layer view не считаются допустимым источником, поэтому
    вызов к `infra.relations.relation_exists()` делается с
    `include_views=False`.
    """
    return _relation_exists(
        engine,
        relation_name,
        include_views=False,
        validate_identifiers=True,
    )


def relation_columns(engine: Engine, relation_name: str) -> tuple[str, ...]:
    """Получить список колонок входного relation через `infra.relations`."""
    return _relation_columns(
        engine,
        relation_name,
        validate_identifiers=True,
    )


def missing_columns(
    available_columns: Sequence[str],
    required_columns: Iterable[str],
) -> tuple[str, ...]:
    """Вернуть обязательные или опциональные колонки, которых нет во входе."""
    available = set(available_columns)
    return tuple(
        column for column in required_columns if column not in available
    )


def collect_dataset_summary(
    engine: Engine,
    relation_name: str,
    available_columns: Sequence[str] | None = None,
) -> DatasetSummary:
    """Собрать агрегированную SQL-сводку по входному датасету.

    Источник данных
    ---------------
    Читает только агрегаты из указанного relation в Postgres: число
    строк, число NULL по обязательным колонкам, количество дублей
    `source_id` и минимальные значения для простых sanity-check.

    Если часть обязательных колонок уже отсутствует в схеме relation,
    функция не падает на SQL, а подставляет для таких полей нейтральные
    агрегаты. Сам факт отсутствия колонок обрабатывается отдельно на
    уровне `validate_dataset()`.
    """
    schema_name, table_name = parse_relation_name(relation_name)
    columns = set(available_columns or relation_columns(engine, relation_name))

    def count_null_expr(column_name: str, alias: str) -> str:
        if column_name in columns:
            return (
                "COUNT(*) FILTER ("
                f" WHERE {column_name} IS NULL"
                f") AS {alias}"
            )
        return f"0 AS {alias}"

    def min_expr(column_name: str, alias: str) -> str:
        if column_name in columns:
            return f"MIN({column_name}) AS {alias}"
        return f"NULL AS {alias}"

    if {"ra", "dec"}.issubset(columns):
        coords_expr = (
            "COUNT(*) FILTER ("
            " WHERE ra IS NULL OR dec IS NULL"
            ") AS n_coords_null"
        )
    else:
        coords_expr = "0 AS n_coords_null"

    duplicate_expr = (
        "COUNT(*) - COUNT(DISTINCT source_id) AS n_duplicate_source_ids"
        if "source_id" in columns
        else "0 AS n_duplicate_source_ids"
    )

    query = f"""
    SELECT
        COUNT(*) AS row_count,
        {count_null_expr("source_id", "n_source_id_null")},
        {coords_expr},
        {count_null_expr("teff_gspphot", "n_teff_null")},
        {count_null_expr("logg_gspphot", "n_logg_null")},
        {count_null_expr("radius_gspphot", "n_radius_null")},
        {count_null_expr("mh_gspphot", "n_mh_null")},
        {count_null_expr("parallax", "n_parallax_null")},
        {count_null_expr("parallax_over_error", "n_plx_err_null")},
        {count_null_expr("ruwe", "n_ruwe_null")},
        {duplicate_expr},
        {min_expr("teff_gspphot", "min_teff")},
        {min_expr("radius_gspphot", "min_radius")},
        {min_expr("ruwe", "min_ruwe")}
    FROM {schema_name}.{table_name};
    """
    with engine.connect() as conn:
        row = conn.execute(text(query)).one()

    return DatasetSummary(
        relation_name=relation_name,
        row_count=int(row.row_count),
        n_source_id_null=int(row.n_source_id_null),
        n_coords_null=int(row.n_coords_null),
        n_teff_null=int(row.n_teff_null),
        n_logg_null=int(row.n_logg_null),
        n_radius_null=int(row.n_radius_null),
        n_mh_null=int(row.n_mh_null),
        n_parallax_null=int(row.n_parallax_null),
        n_plx_err_null=int(row.n_plx_err_null),
        n_ruwe_null=int(row.n_ruwe_null),
        n_duplicate_source_ids=int(row.n_duplicate_source_ids),
        min_teff=(
            None if row.min_teff is None else float(row.min_teff)
        ),
        min_radius=(
            None if row.min_radius is None else float(row.min_radius)
        ),
        min_ruwe=(
            None if row.min_ruwe is None else float(row.min_ruwe)
        ),
    )


def validate_dataset(
    engine: Engine,
    relation_name: str,
    source_name: str,
    mark_ready: bool = True,
) -> DatasetValidationResult:
    """Проверить входной relation и вернуть итоговый verdict.

    Что проверяется
    ---------------
    - существование relation;
    - наличие всех обязательных колонок;
    - базовые агрегатные проблемы качества данных;
    - предупреждения по дублям `source_id` и отсутствующим optional fields.

    Возвращает
    ----------
    DatasetValidationResult
        Полный результат проверки, который затем можно записать
        в registry-таблицу и использовать в decision о статусе
        `VALIDATED` или `READY`.
    """
    if not relation_exists(engine, relation_name):
        empty_summary = DatasetSummary(
            relation_name=relation_name,
            row_count=0,
            n_source_id_null=0,
            n_coords_null=0,
            n_teff_null=0,
            n_logg_null=0,
            n_radius_null=0,
            n_mh_null=0,
            n_parallax_null=0,
            n_plx_err_null=0,
            n_ruwe_null=0,
            n_duplicate_source_ids=0,
            min_teff=None,
            min_radius=None,
            min_ruwe=None,
        )
        return DatasetValidationResult(
            relation_name=relation_name,
            source_name=source_name,
            status=DatasetStatus.FAILED,
            summary=empty_summary,
            missing_required_columns=REQUIRED_COLUMNS,
            missing_optional_columns=OPTIONAL_COLUMNS,
            errors=(f"Relation does not exist: {relation_name}",),
            warnings=(),
            validated_at=datetime.now(UTC),
        )

    columns = relation_columns(engine, relation_name)
    missing_required = missing_columns(columns, REQUIRED_COLUMNS)
    missing_optional = missing_columns(columns, OPTIONAL_COLUMNS)

    summary = collect_dataset_summary(
        engine,
        relation_name,
        available_columns=columns,
    )
    errors: list[str] = []
    warnings: list[str] = []

    if missing_required:
        errors.append(
            "Missing required columns: " + ", ".join(missing_required)
        )
    if summary.row_count <= 0:
        errors.append("Dataset is empty.")
    if summary.n_source_id_null > 0:
        errors.append("source_id contains NULL values.")
    if summary.n_coords_null > 0:
        errors.append("Coordinates contain NULL values.")
    if summary.n_teff_null > 0:
        errors.append("teff_gspphot contains NULL values.")
    if summary.n_logg_null > 0:
        errors.append("logg_gspphot contains NULL values.")
    if summary.n_radius_null > 0:
        errors.append("radius_gspphot contains NULL values.")
    if summary.n_mh_null > 0:
        errors.append("mh_gspphot contains NULL values.")
    if summary.n_parallax_null > 0:
        errors.append("parallax contains NULL values.")
    if summary.n_plx_err_null > 0:
        errors.append("parallax_over_error contains NULL values.")
    if summary.n_ruwe_null > 0:
        errors.append("ruwe contains NULL values.")
    if summary.min_teff is not None and summary.min_teff <= 0:
        errors.append("teff_gspphot contains non-positive values.")
    if summary.min_radius is not None and summary.min_radius <= 0:
        errors.append("radius_gspphot contains non-positive values.")
    if summary.min_ruwe is not None and summary.min_ruwe <= 0:
        errors.append("ruwe contains non-positive values.")

    if summary.n_duplicate_source_ids > 0:
        warnings.append(
            "source_id contains duplicates; "
            "orchestrator will select one row deterministically."
        )
    if missing_optional:
        warnings.append(
            "Missing optional columns: " + ", ".join(missing_optional)
        )

    status = DatasetStatus.READY if mark_ready else DatasetStatus.VALIDATED
    if errors:
        status = DatasetStatus.FAILED

    return DatasetValidationResult(
        relation_name=relation_name,
        source_name=source_name,
        status=status,
        summary=summary,
        missing_required_columns=missing_required,
        missing_optional_columns=missing_optional,
        errors=tuple(errors),
        warnings=tuple(warnings),
        validated_at=datetime.now(UTC),
    )


def ensure_registry_table(engine: Engine) -> None:
    """Создать schema и registry-таблицу для статусов входных датасетов.

    Побочные эффекты
    ----------------
    Выполняет DDL в Postgres:
    - создаёт схему `lab`, если её ещё нет;
    - создаёт таблицу `lab.input_dataset_registry`;
    - создаёт индексы по статусу и времени валидации.
    """
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS lab;

    CREATE TABLE IF NOT EXISTS {REGISTRY_TABLE} (
        relation_name TEXT PRIMARY KEY,
        source_name TEXT NOT NULL,
        status TEXT NOT NULL CHECK (
            status IN ('LOADING', 'VALIDATED', 'READY', 'FAILED')
        ),
        row_count BIGINT NOT NULL,
        n_source_id_null BIGINT NOT NULL,
        n_coords_null BIGINT NOT NULL,
        n_teff_null BIGINT NOT NULL,
        n_logg_null BIGINT NOT NULL,
        n_radius_null BIGINT NOT NULL,
        n_mh_null BIGINT NOT NULL,
        n_parallax_null BIGINT NOT NULL,
        n_plx_err_null BIGINT NOT NULL,
        n_ruwe_null BIGINT NOT NULL,
        n_duplicate_source_ids BIGINT NOT NULL,
        min_teff DOUBLE PRECISION,
        min_radius DOUBLE PRECISION,
        min_ruwe DOUBLE PRECISION,
        validated_at TIMESTAMPTZ NOT NULL,
        notes TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_input_dataset_registry_status
        ON {REGISTRY_TABLE} (status);

    CREATE INDEX IF NOT EXISTS idx_input_dataset_registry_validated_at
        ON {REGISTRY_TABLE} (validated_at DESC);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def build_registry_notes(result: DatasetValidationResult) -> str:
    """Собрать текстовое поле `notes` для registry-таблицы.

    В одну строковую сводку упаковываются ошибки и предупреждения,
    чтобы их можно было просматривать без повторного запуска валидации.
    """
    parts: list[str] = []
    if result.errors:
        parts.append("errors: " + " | ".join(result.errors))
    if result.warnings:
        parts.append("warnings: " + " | ".join(result.warnings))
    return "\n".join(parts)


def register_dataset_result(
    engine: Engine,
    result: DatasetValidationResult,
) -> None:
    """Записать результат валидации в `lab.input_dataset_registry`.

    Побочные эффекты
    ----------------
    - при необходимости создаёт registry-таблицу;
    - выполняет `INSERT ... ON CONFLICT DO UPDATE`;
    - обновляет последнюю известную запись по `relation_name`.
    """
    ensure_registry_table(engine)

    query = text(
        f"""
        INSERT INTO {REGISTRY_TABLE} (
            relation_name,
            source_name,
            status,
            row_count,
            n_source_id_null,
            n_coords_null,
            n_teff_null,
            n_logg_null,
            n_radius_null,
            n_mh_null,
            n_parallax_null,
            n_plx_err_null,
            n_ruwe_null,
            n_duplicate_source_ids,
            min_teff,
            min_radius,
            min_ruwe,
            validated_at,
            notes
        )
        VALUES (
            :relation_name,
            :source_name,
            :status,
            :row_count,
            :n_source_id_null,
            :n_coords_null,
            :n_teff_null,
            :n_logg_null,
            :n_radius_null,
            :n_mh_null,
            :n_parallax_null,
            :n_plx_err_null,
            :n_ruwe_null,
            :n_duplicate_source_ids,
            :min_teff,
            :min_radius,
            :min_ruwe,
            :validated_at,
            :notes
        )
        ON CONFLICT (relation_name) DO UPDATE SET
            source_name = EXCLUDED.source_name,
            status = EXCLUDED.status,
            row_count = EXCLUDED.row_count,
            n_source_id_null = EXCLUDED.n_source_id_null,
            n_coords_null = EXCLUDED.n_coords_null,
            n_teff_null = EXCLUDED.n_teff_null,
            n_logg_null = EXCLUDED.n_logg_null,
            n_radius_null = EXCLUDED.n_radius_null,
            n_mh_null = EXCLUDED.n_mh_null,
            n_parallax_null = EXCLUDED.n_parallax_null,
            n_plx_err_null = EXCLUDED.n_plx_err_null,
            n_ruwe_null = EXCLUDED.n_ruwe_null,
            n_duplicate_source_ids = EXCLUDED.n_duplicate_source_ids,
            min_teff = EXCLUDED.min_teff,
            min_radius = EXCLUDED.min_radius,
            min_ruwe = EXCLUDED.min_ruwe,
            validated_at = EXCLUDED.validated_at,
            notes = EXCLUDED.notes;
        """
    )

    payload: RegistryPayload = {
        "relation_name": result.relation_name,
        "source_name": result.source_name,
        "status": result.status.value,
        "row_count": result.summary.row_count,
        "n_source_id_null": result.summary.n_source_id_null,
        "n_coords_null": result.summary.n_coords_null,
        "n_teff_null": result.summary.n_teff_null,
        "n_logg_null": result.summary.n_logg_null,
        "n_radius_null": result.summary.n_radius_null,
        "n_mh_null": result.summary.n_mh_null,
        "n_parallax_null": result.summary.n_parallax_null,
        "n_plx_err_null": result.summary.n_plx_err_null,
        "n_ruwe_null": result.summary.n_ruwe_null,
        "n_duplicate_source_ids": result.summary.n_duplicate_source_ids,
        "min_teff": result.summary.min_teff,
        "min_radius": result.summary.min_radius,
        "min_ruwe": result.summary.min_ruwe,
        "validated_at": result.validated_at,
        "notes": build_registry_notes(result),
    }
    parameters: Mapping[str, object] = payload

    with engine.begin() as conn:
        conn.execute(query, parameters)


def print_validation_result(result: DatasetValidationResult) -> None:
    """Напечатать короткий CLI-отчёт по результату валидации."""
    print("=== INPUT DATASET VALIDATION ===")
    print(f"Relation: {result.relation_name}")
    print(f"Source name: {result.source_name}")
    print(f"Status: {result.status.value}")
    print(f"Rows: {result.summary.row_count}")
    print(f"Duplicate source_id: {result.summary.n_duplicate_source_ids}")

    if result.errors:
        print("Errors:")
        for item in result.errors:
            print(f"- {item}")

    if result.warnings:
        print("Warnings:")
        for item in result.warnings:
            print(f"- {item}")

    if not result.errors and not result.warnings:
        print("Validation passed without warnings.")


def parse_args() -> Namespace:
    """Разобрать аргументы CLI для сценария валидации входного датасета."""
    parser = ArgumentParser(
        description="Валидирует входной датасет и регистрирует его статус."
    )
    parser.add_argument(
        "--relation",
        default=DEFAULT_INPUT_RELATION,
        help="Полное имя relation, например public.gaia_dr3_training.",
    )
    parser.add_argument(
        "--source-name",
        default="Gaia DR3 random 20k sample",
        help="Человеческое название источника данных.",
    )
    parser.add_argument(
        "--validated-only",
        action="store_true",
        help="Не ставить READY, а записать только VALIDATED.",
    )
    return parser.parse_args()


def main() -> None:
    """Запустить полный CLI-сценарий валидации и регистрации датасета.

    Сценарий выполняет bootstrap подключения к БД, проверяет relation,
    записывает verdict в registry и печатает короткую сводку в stdout.
    """
    args = parse_args()
    engine = make_engine_from_env()
    result = validate_dataset(
        engine=engine,
        relation_name=args.relation,
        source_name=args.source_name,
        mark_ready=not args.validated_only,
    )
    register_dataset_result(engine, result)
    print_validation_result(result)


if __name__ == "__main__":
    main()
