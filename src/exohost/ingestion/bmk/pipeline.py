# Файл `pipeline.py` слоя `ingestion`.
#
# Этот файл отвечает только за:
# - разбор и нормализацию внешних B/mk-меток;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `ingestion` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from pathlib import Path

from astropy.table import Table

from exohost.ingestion.bmk.contracts import (
    BmkCatalogSource,
    BmkExportBundle,
)
from exohost.ingestion.bmk.export import write_bmk_csv_bundle
from exohost.ingestion.bmk.filtering import (
    build_bmk_transform_bundle,
)
from exohost.ingestion.bmk.reader import read_bmk_catalog


def build_bmk_export_bundle_from_table(
    table: Table,
    *,
    output_dir: Path,
) -> BmkExportBundle:
    # Строим все parser/export артефакты из уже прочитанной CDS-таблицы.
    transform_bundle = build_bmk_transform_bundle(table)
    export_paths = write_bmk_csv_bundle(
        transform_bundle.raw_frame,
        transform_bundle.filtered_frame,
        transform_bundle.rejected_frame,
        output_dir=output_dir,
    )
    return BmkExportBundle(
        source=None,
        export_paths=export_paths,
        import_summary=transform_bundle.import_summary,
        primary_filter_summary=transform_bundle.primary_filter_summary,
    )


def build_bmk_export_bundle(
    source: BmkCatalogSource,
    *,
    output_dir: Path,
) -> BmkExportBundle:
    # Читаем внешний B/mk источник и сохраняем три staging CSV.
    table = read_bmk_catalog(source)
    export_bundle = build_bmk_export_bundle_from_table(
        table,
        output_dir=output_dir,
    )
    return BmkExportBundle(
        source=source,
        export_paths=export_bundle.export_paths,
        import_summary=export_bundle.import_summary,
        primary_filter_summary=export_bundle.primary_filter_summary,
    )
