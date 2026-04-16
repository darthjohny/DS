# Тестовый файл `test_bmk_catalog_parser.py` домена `ingestion`.
#
# Этот файл проверяет только:
# - проверку логики домена: ингест, parser-слой и нормализацию внешних меток;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `ingestion` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest
from astropy.table import Table

from exohost.ingestion.bmk import (
    B_MK_CATALOG_NAME,
    B_MK_FILTERED_COLUMNS,
    B_MK_RAW_COLUMNS,
    B_MK_REJECTED_COLUMNS,
    BmkCatalogSource,
    BmkExportPaths,
    BmkImportSummary,
    BmkPrimaryFilterSummary,
    BmkTransformBundle,
    build_bmk_import_summary,
    build_bmk_primary_filter_frames,
    build_bmk_primary_filter_summary,
    build_bmk_raw_frame,
    build_bmk_transform_bundle,
    read_bmk_catalog,
    write_bmk_csv_bundle,
    write_bmk_filtered_csv,
    write_bmk_raw_csv,
    write_bmk_rejected_csv,
)


def test_read_bmk_catalog_uses_cds_reader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # На границе чтения важно убедиться, что мы действительно используем CDS-reader
    # с правильными путями, а не подменяем формат или README без явного сигнала.
    captured: dict[str, object] = {}
    expected_table = Table(rows=[("G2V",)], names=["SpType"])
    source = BmkCatalogSource(
        readme_path=tmp_path / "ReadMe",
        data_path=tmp_path / "mktypes.dat",
    )
    source.readme_path.write_text("ReadMe", encoding="utf-8")
    source.data_path.write_text("mktypes", encoding="utf-8")

    def fake_read(*args: object, **kwargs: object) -> Table:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return expected_table

    monkeypatch.setattr("exohost.ingestion.bmk.reader.Table.read", fake_read)

    result = read_bmk_catalog(source)

    assert result is expected_table
    assert captured["args"] == (source.data_path,)
    assert captured["kwargs"] == {
        "readme": str(source.readme_path),
        "format": "ascii.cds",
    }


def test_build_bmk_raw_frame_normalizes_expected_columns() -> None:
    # Raw-слой должен дать канонический набор колонок и координаты в градусах,
    # чтобы downstream-фильтры и CSV-экспорт не зависели от формы исходной таблицы.
    table = Table(
        rows=[
            (
                "2012ApJS..203...21A",
                "SDSS J000000.67+004315.5",
                0,
                0,
                0.67,
                "+",
                0,
                43,
                15.5,
                15.9,
                "K5V",
                "note",
            ),
        ],
        names=["Bibcode", "Name", "RAh", "RAm", "RAs", "DE-", "DEd", "DEm", "DEs", "Mag", "SpType", "Remarks"],
    )

    raw_frame = build_bmk_raw_frame(table)

    assert list(raw_frame.columns) == list(B_MK_RAW_COLUMNS)
    assert raw_frame.loc[0, "external_catalog_name"] == B_MK_CATALOG_NAME
    assert raw_frame.loc[0, "external_object_id"] == "SDSS J000000.67+004315.5"
    assert raw_frame.loc[0, "raw_sptype"] == "K5V"
    assert raw_frame.loc[0, "raw_source_bibcode"] == "2012ApJS..203...21A"
    assert raw_frame.loc[0, "raw_notes"] == "note"
    ra_deg = raw_frame.loc[0, "ra_deg"]
    dec_deg = raw_frame.loc[0, "dec_deg"]

    assert isinstance(ra_deg, float)
    assert isinstance(dec_deg, float)
    assert ra_deg == pytest.approx(0.0027916667, rel=1e-7)
    assert dec_deg == pytest.approx(0.7209722222, rel=1e-7)


def test_build_bmk_import_summary_counts_expected_rows() -> None:
    # Import summary нужен для быстрого контроля качества сырого каталога:
    # сколько строк вообще пригодно к дальнейшему разбору и crossmatch.
    table = Table(
        rows=[
            ("2012ApJS..203...21A", "Star A", 0, 0, 0.67, "+", 0, 43, 15.5, 15.9, "K5V", "note"),
            ("2012ApJS..203...21B", "Star B", 1, 2, 3.40, "-", 3, 4, 5.6, 12.2, "WD", ""),
            ("2012ApJS..203...21C", "Star C", None, 2, 3.40, "-", 3, 4, 5.6, 12.2, "G2V", ""),
            ("2012ApJS..203...21D", "Star D", 5, 6, 7.80, "+", 7, 8, 9.0, 12.2, "", ""),
        ],
        names=["Bibcode", "Name", "RAh", "RAm", "RAs", "DE-", "DEd", "DEm", "DEs", "Mag", "SpType", "Remarks"],
    )
    raw_frame = build_bmk_raw_frame(table)

    summary = build_bmk_import_summary(table, raw_frame)

    assert summary == BmkImportSummary(
        total_rows=4,
        rows_with_coordinates=3,
        rows_with_raw_sptype=3,
        rows_with_supported_spectral_prefix=2,
        exported_rows=2,
    )


def test_build_bmk_transform_bundle_builds_layers_and_summaries_in_one_call() -> None:
    # Здесь страхуем основной orchestration helper ingestion-слоя:
    # он должен за один вызов собрать raw/filtered/rejected кадры и обе сводки.
    table = Table(
        rows=[
            ("2012ApJS..203...21A", "Star A", 0, 0, 0.67, "+", 0, 43, 15.5, 15.9, "K5V", "note"),
            ("2012ApJS..203...21B", "Star B", 1, 2, 3.40, "-", 3, 4, 5.6, 12.2, "F0", ""),
            ("2012ApJS..203...21C", "Star C", 2, 2, 3.40, "-", 3, 4, 5.6, 12.2, "DA", ""),
            ("2012ApJS..203...21D", "Star D", None, 2, 3.40, "-", 3, 4, 5.6, 12.2, "G2V", ""),
            ("2012ApJS..203...21E", "Star E", 5, 6, 7.80, "+", 7, 8, 9.0, 12.2, "", ""),
        ],
        names=["Bibcode", "Name", "RAh", "RAm", "RAs", "DE-", "DEd", "DEm", "DEs", "Mag", "SpType", "Remarks"],
    )

    transform_bundle = build_bmk_transform_bundle(table)

    assert isinstance(transform_bundle, BmkTransformBundle)
    assert transform_bundle.import_summary == BmkImportSummary(
        total_rows=5,
        rows_with_coordinates=4,
        rows_with_raw_sptype=4,
        rows_with_supported_spectral_prefix=3,
        exported_rows=3,
    )
    assert transform_bundle.primary_filter_summary == BmkPrimaryFilterSummary(
        total_rows=5,
        filtered_rows=2,
        rejected_rows=3,
        rows_ready_for_gaia_crossmatch=2,
        rejected_missing_coordinates=1,
        rejected_missing_raw_sptype=1,
        rejected_unsupported_spectral_prefix=1,
    )
    assert transform_bundle.raw_frame["external_object_id"].tolist() == [
        "Star A",
        "Star B",
        "Star C",
    ]
    assert transform_bundle.filtered_frame["external_object_id"].tolist() == ["Star A", "Star B"]
    assert transform_bundle.rejected_frame["external_object_id"].tolist() == [
        "Star C",
        "Star D",
        "Star E",
    ]


def test_write_bmk_raw_csv_writes_expected_columns(tmp_path: Path) -> None:
    # Экспорт raw CSV должен сохранять порядок колонок и пустые ячейки так,
    # чтобы файл был стабилен для локального аудита и дальнейшей загрузки.
    raw_frame = pd.DataFrame(
        [
            {
                "external_row_id": 0,
                "external_catalog_name": "bmk",
                "external_object_id": "Star A",
                "ra_deg": 10.5,
                "dec_deg": -5.25,
                "raw_sptype": "G2V",
                "raw_magnitude": None,
                "raw_source_bibcode": "2014yCat....1.2023S",
                "raw_notes": None,
            },
        ],
        columns=B_MK_RAW_COLUMNS,
    )
    output_path = tmp_path / "exports" / "bmk_external_raw.csv"

    written_path = write_bmk_raw_csv(raw_frame, output_path)

    assert written_path == output_path
    with output_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)

    assert reader.fieldnames == list(B_MK_RAW_COLUMNS)
    assert rows == [
        {
            "external_row_id": "0",
            "external_catalog_name": "bmk",
            "external_object_id": "Star A",
            "ra_deg": "10.5",
            "dec_deg": "-5.25",
            "raw_sptype": "G2V",
            "raw_magnitude": "",
            "raw_source_bibcode": "2014yCat....1.2023S",
            "raw_notes": "",
        },
    ]


def test_write_bmk_filtered_csv_writes_empty_cells_for_missing_numeric_values(
    tmp_path: Path,
) -> None:
    # Для filtered CSV отдельно страхуем сериализацию пропусков: числовые пустые
    # значения должны уходить как пустые ячейки, а не как `nan` или `None`.
    filtered_frame = pd.DataFrame(
        [
            {
                "external_row_id": 0,
                "external_catalog_name": "bmk",
                "external_object_id": "Star A",
                "ra_deg": 10.5,
                "dec_deg": -5.25,
                "raw_sptype": "F0",
                "raw_magnitude": pd.NA,
                "raw_source_bibcode": "2014yCat....1.2023S",
                "raw_notes": None,
                "spectral_prefix": "F",
                "spectral_class": "F",
                "spectral_subclass": pd.NA,
                "luminosity_class": pd.NA,
                "parse_status": "partial",
                "parse_note": "missing_integer_subclass",
                "has_supported_prefix": True,
                "has_coordinates": True,
                "has_raw_sptype": True,
                "ready_for_gaia_crossmatch": True,
            },
        ],
        columns=B_MK_FILTERED_COLUMNS,
    )
    output_path = tmp_path / "exports" / "bmk_external_filtered.csv"

    written_path = write_bmk_filtered_csv(filtered_frame, output_path)

    assert written_path == output_path
    with output_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)

    assert reader.fieldnames == list(B_MK_FILTERED_COLUMNS)
    assert rows == [
        {
            "external_row_id": "0",
            "external_catalog_name": "bmk",
            "external_object_id": "Star A",
            "ra_deg": "10.5",
            "dec_deg": "-5.25",
            "raw_sptype": "F0",
            "raw_magnitude": "",
            "raw_source_bibcode": "2014yCat....1.2023S",
            "raw_notes": "",
            "spectral_prefix": "F",
            "spectral_class": "F",
            "spectral_subclass": "",
            "luminosity_class": "",
            "parse_status": "partial",
            "parse_note": "missing_integer_subclass",
            "has_supported_prefix": "True",
            "has_coordinates": "True",
            "has_raw_sptype": "True",
            "ready_for_gaia_crossmatch": "True",
        },
    ]


def test_write_bmk_filtered_csv_serializes_nullable_integer_without_decimal_suffix(
    tmp_path: Path,
) -> None:
    filtered_frame = pd.DataFrame(
        [
            {
                "external_row_id": 1,
                "external_catalog_name": "bmk",
                "external_object_id": "Star B",
                "ra_deg": 11.5,
                "dec_deg": -4.25,
                "raw_sptype": "G0V",
                "raw_magnitude": 12.3,
                "raw_source_bibcode": "2014yCat....1.2023S",
                "raw_notes": None,
                "spectral_prefix": "G",
                "spectral_class": "G",
                "spectral_subclass": 0.0,
                "luminosity_class": "V",
                "parse_status": "parsed",
                "parse_note": None,
                "has_supported_prefix": True,
                "has_coordinates": True,
                "has_raw_sptype": True,
                "ready_for_gaia_crossmatch": True,
            },
        ],
        columns=B_MK_FILTERED_COLUMNS,
    )
    output_path = tmp_path / "exports" / "bmk_external_filtered.csv"

    write_bmk_filtered_csv(filtered_frame, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)

    assert rows[0]["spectral_subclass"] == "0"


def test_write_bmk_rejected_csv_writes_expected_columns(tmp_path: Path) -> None:
    rejected_frame = pd.DataFrame(
        [
            {
                "external_row_id": 3,
                "external_catalog_name": "bmk",
                "external_object_id": "Star D",
                "ra_deg": None,
                "dec_deg": None,
                "raw_sptype": "G2V",
                "raw_magnitude": 12.2,
                "raw_source_bibcode": "2012ApJS..203...21D",
                "raw_notes": None,
                "spectral_prefix": None,
                "reject_reason": "missing_coordinates",
            },
        ],
        columns=B_MK_REJECTED_COLUMNS,
    )
    output_path = tmp_path / "exports" / "bmk_external_rejected.csv"

    written_path = write_bmk_rejected_csv(rejected_frame, output_path)

    assert written_path == output_path
    with output_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)

    assert reader.fieldnames == list(B_MK_REJECTED_COLUMNS)
    assert rows == [
        {
            "external_row_id": "3",
            "external_catalog_name": "bmk",
            "external_object_id": "Star D",
            "ra_deg": "",
            "dec_deg": "",
            "raw_sptype": "G2V",
            "raw_magnitude": "12.2",
            "raw_source_bibcode": "2012ApJS..203...21D",
            "raw_notes": "",
            "spectral_prefix": "",
            "reject_reason": "missing_coordinates",
        },
    ]


def test_write_bmk_csv_bundle_writes_three_staging_files(tmp_path: Path) -> None:
    raw_frame = pd.DataFrame(
        [
            {
                "external_row_id": 0,
                "external_catalog_name": "bmk",
                "external_object_id": "Star A",
                "ra_deg": 10.5,
                "dec_deg": -5.25,
                "raw_sptype": "G2V",
                "raw_magnitude": None,
                "raw_source_bibcode": "2014yCat....1.2023S",
                "raw_notes": None,
            },
        ],
        columns=B_MK_RAW_COLUMNS,
    )
    filtered_frame = pd.DataFrame(
        [
            {
                "external_row_id": 0,
                "external_catalog_name": "bmk",
                "external_object_id": "Star A",
                "ra_deg": 10.5,
                "dec_deg": -5.25,
                "raw_sptype": "G2V",
                "raw_magnitude": None,
                "raw_source_bibcode": "2014yCat....1.2023S",
                "raw_notes": None,
                "spectral_prefix": "G",
                "spectral_class": "G",
                "spectral_subclass": 2,
                "luminosity_class": "V",
                "parse_status": "parsed",
                "parse_note": None,
                "has_supported_prefix": True,
                "has_coordinates": True,
                "has_raw_sptype": True,
                "ready_for_gaia_crossmatch": True,
            },
        ],
        columns=B_MK_FILTERED_COLUMNS,
    )
    rejected_frame = pd.DataFrame(columns=B_MK_REJECTED_COLUMNS)

    export_paths = write_bmk_csv_bundle(
        raw_frame,
        filtered_frame,
        rejected_frame,
        output_dir=tmp_path / "bundle",
    )

    assert export_paths == BmkExportPaths(
        raw_csv_path=tmp_path / "bundle" / "bmk_external_raw.csv",
        filtered_csv_path=tmp_path / "bundle" / "bmk_external_filtered.csv",
        rejected_csv_path=tmp_path / "bundle" / "bmk_external_rejected.csv",
    )
    assert export_paths.raw_csv_path.exists()
    assert export_paths.filtered_csv_path.exists()
    assert export_paths.rejected_csv_path.exists()


def test_build_bmk_primary_filter_frames_splits_supported_and_rejected_rows() -> None:
    table = Table(
        rows=[
            ("2012ApJS..203...21A", "Star A", 0, 0, 0.67, "+", 0, 43, 15.5, 15.9, "K5V", "note"),
            ("2012ApJS..203...21B", "Star B", 1, 2, 3.40, "-", 3, 4, 5.6, 12.2, "F0", ""),
            ("2012ApJS..203...21C", "Star C", 2, 2, 3.40, "-", 3, 4, 5.6, 12.2, "DA", ""),
            ("2012ApJS..203...21D", "Star D", None, 2, 3.40, "-", 3, 4, 5.6, 12.2, "G2V", ""),
            ("2012ApJS..203...21E", "Star E", 5, 6, 7.80, "+", 7, 8, 9.0, 12.2, "", ""),
        ],
        names=["Bibcode", "Name", "RAh", "RAm", "RAs", "DE-", "DEd", "DEm", "DEs", "Mag", "SpType", "Remarks"],
    )

    filtered_frame, rejected_frame = build_bmk_primary_filter_frames(table)

    assert list(filtered_frame.columns) == list(B_MK_FILTERED_COLUMNS)
    assert list(rejected_frame.columns) == list(B_MK_REJECTED_COLUMNS)

    assert filtered_frame["external_object_id"].tolist() == ["Star A", "Star B"]
    assert filtered_frame["spectral_prefix"].tolist() == ["K", "F"]
    assert filtered_frame["parse_status"].tolist() == ["parsed", "partial"]
    assert filtered_frame["ready_for_gaia_crossmatch"].tolist() == [True, True]

    assert rejected_frame["external_object_id"].tolist() == ["Star C", "Star D", "Star E"]
    assert rejected_frame["reject_reason"].tolist() == [
        "unsupported_spectral_prefix",
        "missing_coordinates",
        "missing_raw_sptype",
    ]


def test_build_bmk_primary_filter_summary_counts_rejection_reasons() -> None:
    table = Table(
        rows=[
            ("2012ApJS..203...21A", "Star A", 0, 0, 0.67, "+", 0, 43, 15.5, 15.9, "K5V", "note"),
            ("2012ApJS..203...21B", "Star B", 1, 2, 3.40, "-", 3, 4, 5.6, 12.2, "DA", ""),
            ("2012ApJS..203...21C", "Star C", None, 2, 3.40, "-", 3, 4, 5.6, 12.2, "G2V", ""),
            ("2012ApJS..203...21D", "Star D", 5, 6, 7.80, "+", 7, 8, 9.0, 12.2, "", ""),
        ],
        names=["Bibcode", "Name", "RAh", "RAm", "RAs", "DE-", "DEd", "DEm", "DEs", "Mag", "SpType", "Remarks"],
    )

    filtered_frame, rejected_frame = build_bmk_primary_filter_frames(table)
    summary = build_bmk_primary_filter_summary(table, filtered_frame, rejected_frame)

    assert summary == BmkPrimaryFilterSummary(
        total_rows=4,
        filtered_rows=1,
        rejected_rows=3,
        rows_ready_for_gaia_crossmatch=1,
        rejected_missing_coordinates=1,
        rejected_missing_raw_sptype=1,
        rejected_unsupported_spectral_prefix=1,
    )
