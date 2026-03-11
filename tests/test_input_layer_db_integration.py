"""DB-backed интеграционные тесты для `input_layer` и registry-таблицы."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

import input_layer


@pytest.fixture()
def cleanup_registry_rows(postgres_test_engine: Engine) -> Iterator[list[str]]:
    """Удалить registry-записи, созданные интеграционным тестом."""
    relation_names: list[str] = []
    yield relation_names
    if not relation_names:
        return

    with postgres_test_engine.begin() as conn:
        exists = conn.execute(
            text("SELECT to_regclass(:registry_table)"),
            {"registry_table": input_layer.REGISTRY_TABLE},
        ).scalar()
        if exists is None:
            return
        for relation_name in relation_names:
            conn.execute(
                text(
                    f"""
                    DELETE FROM {input_layer.REGISTRY_TABLE}
                    WHERE relation_name = :relation_name
                    """
                ),
                {"relation_name": relation_name},
            )


@pytest.mark.db_integration
def test_input_layer_registers_ready_dataset_result(
    postgres_test_engine: Engine,
    temp_pg_schema: str,
    cleanup_registry_rows: list[str],
) -> None:
    """Валидный relation должен попадать в registry со статусом `READY`."""
    relation_name = f"{temp_pg_schema}.gaia_input_ready"
    cleanup_registry_rows.append(relation_name)

    df = pd.DataFrame(
        [
            {
                "source_id": 1,
                "ra": 10.0,
                "dec": -5.0,
                "teff_gspphot": 4700.0,
                "logg_gspphot": 4.6,
                "radius_gspphot": 0.8,
                "mh_gspphot": 0.1,
                "parallax": 12.0,
                "parallax_over_error": 18.0,
                "ruwe": 1.0,
            },
            {
                "source_id": 1,
                "ra": 10.1,
                "dec": -4.9,
                "teff_gspphot": 4695.0,
                "logg_gspphot": 4.6,
                "radius_gspphot": 0.79,
                "mh_gspphot": 0.1,
                "parallax": 13.0,
                "parallax_over_error": 24.0,
                "ruwe": 0.95,
            },
            {
                "source_id": 2,
                "ra": 20.0,
                "dec": 1.0,
                "teff_gspphot": 5600.0,
                "logg_gspphot": 4.3,
                "radius_gspphot": 1.0,
                "mh_gspphot": -0.1,
                "parallax": 8.0,
                "parallax_over_error": 12.0,
                "ruwe": 1.1,
            },
        ]
    )
    df.to_sql(
        name="gaia_input_ready",
        schema=temp_pg_schema,
        con=postgres_test_engine,
        if_exists="replace",
        index=False,
        method="multi",
    )

    result = input_layer.validate_dataset(
        engine=postgres_test_engine,
        relation_name=relation_name,
        source_name="QA integration ready sample",
        mark_ready=True,
    )
    input_layer.register_dataset_result(postgres_test_engine, result)

    assert result.status == input_layer.DatasetStatus.READY
    assert result.summary.row_count == 3
    assert result.summary.n_duplicate_source_ids == 1
    assert result.missing_required_columns == ()
    assert result.missing_optional_columns == input_layer.OPTIONAL_COLUMNS
    assert result.errors == ()
    assert result.warnings == (
        "source_id contains duplicates; "
        "orchestrator will select one row deterministically.",
        "Missing optional columns: "
        + ", ".join(input_layer.OPTIONAL_COLUMNS),
    )

    registry_row = pd.read_sql(
        text(
            f"""
            SELECT relation_name, source_name, status, row_count,
                   n_duplicate_source_ids, notes
            FROM {input_layer.REGISTRY_TABLE}
            WHERE relation_name = :relation_name
            """
        ),
        postgres_test_engine,
        params={"relation_name": relation_name},
    ).iloc[0]

    assert registry_row["relation_name"] == relation_name
    assert registry_row["source_name"] == "QA integration ready sample"
    assert registry_row["status"] == input_layer.DatasetStatus.READY.value
    assert int(registry_row["row_count"]) == 3
    assert int(registry_row["n_duplicate_source_ids"]) == 1
    notes = str(registry_row["notes"])
    assert "source_id contains duplicates" in notes
    assert "Missing optional columns" in notes


@pytest.mark.db_integration
def test_input_layer_registers_failed_result_for_missing_required_columns(
    postgres_test_engine: Engine,
    temp_pg_schema: str,
    cleanup_registry_rows: list[str],
) -> None:
    """Relation с неполной схемой должен давать `FAILED`, а не SQL-падение."""
    relation_name = f"{temp_pg_schema}.gaia_input_missing_required"
    cleanup_registry_rows.append(relation_name)

    df = pd.DataFrame(
        [
            {
                "source_id": 10,
                "ra": 42.0,
                "dec": 3.0,
                "teff_gspphot": 4800.0,
                "logg_gspphot": 4.5,
                "radius_gspphot": 0.82,
                "parallax": 14.0,
                "parallax_over_error": 19.0,
            }
        ]
    )
    df.to_sql(
        name="gaia_input_missing_required",
        schema=temp_pg_schema,
        con=postgres_test_engine,
        if_exists="replace",
        index=False,
        method="multi",
    )

    result = input_layer.validate_dataset(
        engine=postgres_test_engine,
        relation_name=relation_name,
        source_name="QA integration broken schema",
        mark_ready=True,
    )
    input_layer.register_dataset_result(postgres_test_engine, result)

    assert result.status == input_layer.DatasetStatus.FAILED
    assert result.summary.row_count == 1
    assert result.summary.n_mh_null == 0
    assert result.summary.n_ruwe_null == 0
    assert result.missing_required_columns == ("mh_gspphot", "ruwe")
    assert result.errors == (
        "Missing required columns: mh_gspphot, ruwe",
    )
    assert result.warnings == (
        "Missing optional columns: "
        + ", ".join(input_layer.OPTIONAL_COLUMNS),
    )

    registry_row = pd.read_sql(
        text(
            f"""
            SELECT relation_name, status, row_count, notes
            FROM {input_layer.REGISTRY_TABLE}
            WHERE relation_name = :relation_name
            """
        ),
        postgres_test_engine,
        params={"relation_name": relation_name},
    ).iloc[0]

    assert registry_row["relation_name"] == relation_name
    assert registry_row["status"] == input_layer.DatasetStatus.FAILED.value
    assert int(registry_row["row_count"]) == 1
    notes = str(registry_row["notes"])
    assert "Missing required columns: mh_gspphot, ruwe" in notes
    assert "Missing optional columns" in notes
