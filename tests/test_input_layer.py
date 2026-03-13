"""Unit-тесты для branch-логики `input_layer` без БД."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy.engine import Engine

import input_layer


def make_summary(**overrides: Any) -> input_layer.DatasetSummary:
    """Собрать минимальную сводку датасета с возможностью точечных override."""
    payload: dict[str, Any] = {
        "relation_name": "lab.synthetic_input",
        "row_count": 10,
        "n_source_id_null": 0,
        "n_coords_null": 0,
        "n_teff_null": 0,
        "n_logg_null": 0,
        "n_radius_null": 0,
        "n_mh_null": 0,
        "n_parallax_null": 0,
        "n_plx_err_null": 0,
        "n_ruwe_null": 0,
        "n_duplicate_source_ids": 0,
        "min_teff": 4200.0,
        "min_radius": 0.7,
        "min_ruwe": 0.9,
    }
    payload.update(overrides)
    return input_layer.DatasetSummary(**payload)


def test_missing_columns_preserves_requested_order() -> None:
    """Helper должен возвращать отсутствующие колонки в порядке контракта."""
    available_columns = ("ruwe", "source_id", "ra")
    required_columns = ("source_id", "dec", "ruwe", "teff_gspphot")

    result = input_layer.missing_columns(
        available_columns=available_columns,
        required_columns=required_columns,
    )

    assert result == ("dec", "teff_gspphot")


def test_validate_dataset_returns_failed_for_missing_relation(
    monkeypatch: Any,
) -> None:
    """Несуществующий relation должен давать `FAILED` без похода глубже в SQL."""
    monkeypatch.setattr(input_layer, "relation_exists", lambda *_args: False)

    result = input_layer.validate_dataset(
        engine=cast(Engine, object()),
        relation_name="lab.missing_input",
        source_name="QA missing relation",
    )

    assert result.status == input_layer.DatasetStatus.FAILED
    assert result.summary.row_count == 0
    assert result.missing_required_columns == input_layer.REQUIRED_COLUMNS
    assert result.missing_optional_columns == input_layer.OPTIONAL_COLUMNS
    assert result.errors == ("Relation does not exist: lab.missing_input",)
    assert result.warnings == ()


def test_validate_dataset_marks_validated_without_ready_flag(
    monkeypatch: Any,
) -> None:
    """`mark_ready=False` должен приводить к `VALIDATED`, если ошибок нет."""
    all_columns = (
        *input_layer.REQUIRED_COLUMNS,
        *input_layer.OPTIONAL_COLUMNS,
    )

    monkeypatch.setattr(input_layer, "relation_exists", lambda *_args: True)
    monkeypatch.setattr(input_layer, "relation_columns", lambda *_args: all_columns)
    monkeypatch.setattr(
        input_layer,
        "collect_dataset_summary",
        lambda *_args, **_kwargs: make_summary(),
    )

    result = input_layer.validate_dataset(
        engine=cast(Engine, object()),
        relation_name="lab.validated_only_input",
        source_name="QA validated-only sample",
        mark_ready=False,
    )

    assert result.status == input_layer.DatasetStatus.VALIDATED
    assert result.errors == ()
    assert result.warnings == ()
    assert result.missing_required_columns == ()
    assert result.missing_optional_columns == ()


def test_validate_dataset_collects_errors_and_warnings_from_summary(
    monkeypatch: Any,
) -> None:
    """Branch-логика должна собирать и ошибки, и warnings из агрегатной сводки."""
    monkeypatch.setattr(input_layer, "relation_exists", lambda *_args: True)
    monkeypatch.setattr(
        input_layer,
        "relation_columns",
        lambda *_args: input_layer.REQUIRED_COLUMNS,
    )
    monkeypatch.setattr(
        input_layer,
        "collect_dataset_summary",
        lambda *_args, **_kwargs: make_summary(
            row_count=2,
            n_ruwe_null=1,
            n_duplicate_source_ids=1,
            min_radius=0.0,
        ),
    )

    result = input_layer.validate_dataset(
        engine=cast(Engine, object()),
        relation_name="lab.problematic_input",
        source_name="QA problematic sample",
    )

    assert result.status == input_layer.DatasetStatus.FAILED
    assert result.errors == (
        "ruwe contains NULL values.",
        "radius_gspphot contains non-positive values.",
    )
    assert result.warnings == (
        "source_id contains duplicates; "
        "orchestrator will select one row deterministically.",
        "Missing optional columns: " + ", ".join(input_layer.OPTIONAL_COLUMNS),
    )


def test_build_registry_notes_combines_errors_and_warnings() -> None:
    """Registry notes должен собирать errors и warnings в читаемую сводку."""
    result = input_layer.DatasetValidationResult(
        relation_name="lab.notes_input",
        source_name="QA notes sample",
        status=input_layer.DatasetStatus.FAILED,
        summary=make_summary(),
        missing_required_columns=("ruwe",),
        missing_optional_columns=("bp_rp",),
        errors=("Missing required columns: ruwe",),
        warnings=("Missing optional columns: bp_rp",),
        validated_at=datetime.now(UTC),
    )

    notes = input_layer.build_registry_notes(result)

    assert notes == (
        "errors: Missing required columns: ruwe\n"
        "warnings: Missing optional columns: bp_rp"
    )
