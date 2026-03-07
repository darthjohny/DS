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
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
import os
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine

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

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DatasetStatus(StrEnum):
    """Поддерживаемые статусы входного датасета."""

    LOADING = "LOADING"
    VALIDATED = "VALIDATED"
    READY = "READY"
    FAILED = "FAILED"


@dataclass(frozen=True)
class DatasetSummary:
    """Сводка по входному датасету."""

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
    """Результат проверки входного датасета."""

    relation_name: str
    source_name: str
    status: DatasetStatus
    summary: DatasetSummary
    missing_required_columns: tuple[str, ...]
    missing_optional_columns: tuple[str, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    validated_at: datetime


def load_dotenv_local(dotenv_path: str = ".env") -> None:
    """Загрузить `.env` без внешних зависимостей."""
    path = Path(dotenv_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cleaned = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), cleaned)


def make_engine_from_env() -> Engine:
    """Создать SQLAlchemy engine из `.env` или PG-переменных."""
    load_dotenv_local()

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return create_engine(database_url)

    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT", "5432")
    dbname = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")

    if all([host, dbname, user, password]):
        return create_engine(
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
        )

    raise RuntimeError(
        "Database connection is missing. "
        "Set DATABASE_URL or PG* variables."
    )


def parse_relation_name(relation_name: str) -> tuple[str, str]:
    """Разделить relation на schema и table и проверить идентификаторы."""
    parts = relation_name.split(".", 1)
    if len(parts) == 2:
        schema_name, table_name = parts
    else:
        schema_name, table_name = "public", relation_name

    if not IDENTIFIER_PATTERN.match(schema_name):
        raise ValueError(f"Invalid schema name: {schema_name}")
    if not IDENTIFIER_PATTERN.match(table_name):
        raise ValueError(f"Invalid table name: {table_name}")

    return schema_name, table_name


def relation_exists(engine: Engine, relation_name: str) -> bool:
    """Проверить, что relation существует в БД."""
    schema_name, table_name = parse_relation_name(relation_name)
    query = text(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = :schema_name
              AND table_name = :table_name
        );
        """
    )
    with engine.connect() as conn:
        return bool(
            conn.execute(
                query,
                {
                    "schema_name": schema_name,
                    "table_name": table_name,
                },
            ).scalar_one()
        )


def relation_columns(engine: Engine, relation_name: str) -> tuple[str, ...]:
    """Получить список колонок relation."""
    schema_name, table_name = parse_relation_name(relation_name)
    query = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema_name
          AND table_name = :table_name
        ORDER BY ordinal_position;
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {
                "schema_name": schema_name,
                "table_name": table_name,
            },
        ).fetchall()
    return tuple(str(row.column_name) for row in rows)


def missing_columns(
    available_columns: Sequence[str],
    required_columns: Iterable[str],
) -> tuple[str, ...]:
    """Вернуть список отсутствующих колонок."""
    available = set(available_columns)
    return tuple(
        column for column in required_columns if column not in available
    )


def collect_dataset_summary(
    engine: Engine,
    relation_name: str,
) -> DatasetSummary:
    """Собрать базовую статистику по входному набору."""
    schema_name, table_name = parse_relation_name(relation_name)
    query = f"""
    SELECT
        COUNT(*) AS row_count,
        COUNT(*) FILTER (
            WHERE source_id IS NULL
        ) AS n_source_id_null,
        COUNT(*) FILTER (
            WHERE ra IS NULL OR dec IS NULL
        ) AS n_coords_null,
        COUNT(*) FILTER (
            WHERE teff_gspphot IS NULL
        ) AS n_teff_null,
        COUNT(*) FILTER (
            WHERE logg_gspphot IS NULL
        ) AS n_logg_null,
        COUNT(*) FILTER (
            WHERE radius_gspphot IS NULL
        ) AS n_radius_null,
        COUNT(*) FILTER (
            WHERE mh_gspphot IS NULL
        ) AS n_mh_null,
        COUNT(*) FILTER (
            WHERE parallax IS NULL
        ) AS n_parallax_null,
        COUNT(*) FILTER (
            WHERE parallax_over_error IS NULL
        ) AS n_plx_err_null,
        COUNT(*) FILTER (
            WHERE ruwe IS NULL
        ) AS n_ruwe_null,
        COUNT(*) - COUNT(DISTINCT source_id) AS n_duplicate_source_ids,
        MIN(teff_gspphot) AS min_teff,
        MIN(radius_gspphot) AS min_radius,
        MIN(ruwe) AS min_ruwe
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
    """Проверить входной набор и вернуть verdict."""
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
            validated_at=datetime.now(timezone.utc),
        )

    columns = relation_columns(engine, relation_name)
    missing_required = missing_columns(columns, REQUIRED_COLUMNS)
    missing_optional = missing_columns(columns, OPTIONAL_COLUMNS)

    summary = collect_dataset_summary(engine, relation_name)
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
        validated_at=datetime.now(timezone.utc),
    )


def ensure_registry_table(engine: Engine) -> None:
    """Создать registry-таблицу для входных датасетов."""
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
    """Собрать короткий текст notes для registry-таблицы."""
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
    """Записать verdict входного датасета в registry-таблицу."""
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

    payload: dict[str, Any] = {
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

    with engine.begin() as conn:
        conn.execute(query, payload)


def print_validation_result(result: DatasetValidationResult) -> None:
    """Печатает короткий отчёт по проверке входного набора."""
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
    """Разобрать аргументы CLI."""
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
    """CLI-точка входа для валидации и регистрации датасета."""
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
