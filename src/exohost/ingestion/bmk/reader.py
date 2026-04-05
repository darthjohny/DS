# Файл `reader.py` слоя `ingestion`.
#
# Этот файл отвечает только за:
# - разбор и нормализацию внешних B/mk-меток;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `ingestion` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from typing import cast

from astropy.table import Table

from exohost.ingestion.bmk.contracts import BmkCatalogSource


def read_bmk_catalog(source: BmkCatalogSource) -> Table:
    # Читаем mktypes.dat по официальному ReadMe через CDS-ридер Astropy.
    if not source.readme_path.exists():
        raise FileNotFoundError(f"B/mk ReadMe was not found: {source.readme_path}")
    if not source.data_path.exists():
        raise FileNotFoundError(f"B/mk data file was not found: {source.data_path}")

    return cast(
        Table,
        Table.read(
            source.data_path,
            readme=str(source.readme_path),
            format="ascii.cds",
        ),
    )
