# Тестовый файл `test_bmk_ingestion_pipeline.py` домена `ingestion`.
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

from astropy.table import Table

from exohost.ingestion.bmk import (
    BmkExportBundle,
    BmkImportSummary,
    BmkPrimaryFilterSummary,
    build_bmk_export_bundle_from_table,
)


def test_build_bmk_export_bundle_from_table_writes_csv_and_summaries(
    tmp_path: Path,
) -> None:
    table = Table(
        rows=[
            ("2012ApJS..203...21A", "Star A", 0, 0, 0.67, "+", 0, 43, 15.5, 15.9, "K5V", "note"),
            ("2012ApJS..203...21B", "Star B", 1, 2, 3.40, "-", 3, 4, 5.6, 12.2, "DA", ""),
            ("2012ApJS..203...21C", "Star C", None, 2, 3.40, "-", 3, 4, 5.6, 12.2, "G2V", ""),
            ("2012ApJS..203...21D", "Star D", 5, 6, 7.80, "+", 7, 8, 9.0, 12.2, "", ""),
        ],
        names=["Bibcode", "Name", "RAh", "RAm", "RAs", "DE-", "DEd", "DEm", "DEs", "Mag", "SpType", "Remarks"],
    )

    export_bundle = build_bmk_export_bundle_from_table(
        table,
        output_dir=tmp_path / "ingestion",
    )

    assert export_bundle == BmkExportBundle(
        source=None,
        export_paths=export_bundle.export_paths,
        import_summary=BmkImportSummary(
            total_rows=4,
            rows_with_coordinates=3,
            rows_with_raw_sptype=3,
            rows_with_supported_spectral_prefix=2,
            exported_rows=2,
        ),
        primary_filter_summary=BmkPrimaryFilterSummary(
            total_rows=4,
            filtered_rows=1,
            rejected_rows=3,
            rows_ready_for_gaia_crossmatch=1,
            rejected_missing_coordinates=1,
            rejected_missing_raw_sptype=1,
            rejected_unsupported_spectral_prefix=1,
        ),
    )

    for csv_path in (
        export_bundle.export_paths.raw_csv_path,
        export_bundle.export_paths.filtered_csv_path,
        export_bundle.export_paths.rejected_csv_path,
    ):
        assert csv_path.exists()

    with export_bundle.export_paths.filtered_csv_path.open("r", encoding="utf-8", newline="") as input_file:
        filtered_rows = list(csv.DictReader(input_file))
    with export_bundle.export_paths.rejected_csv_path.open("r", encoding="utf-8", newline="") as input_file:
        rejected_rows = list(csv.DictReader(input_file))

    assert len(filtered_rows) == 1
    assert filtered_rows[0]["external_object_id"] == "Star A"
    assert len(rejected_rows) == 3
