# Тестовый файл `test_db_bmk_labeled_export.py` домена `db`.
#
# Этот файл проверяет только:
# - проверку логики домена: relation-layer, materialization и SQL-helper;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `db` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pandas as pd
from sqlalchemy.engine import Engine

from exohost.db.bmk_labeled import (
    B_MK_EXTERNAL_LABELED_COLUMNS,
    B_MK_EXTERNAL_LABELED_CSV_FILENAME,
    B_MK_EXTERNAL_LABELED_SOURCE_CROSSMATCH_RELATION_NAME,
    B_MK_EXTERNAL_LABELED_SOURCE_FILTERED_RELATION_NAME,
    BmkExternalLabeledExportSummary,
    export_bmk_external_labeled_csv,
)


def test_export_bmk_external_labeled_csv_writes_parsed_rows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_chunk = pd.DataFrame(
        [
            {
                "xmatch_batch_id": "xmatch_bmk_gaia_dr3__2026_03_26",
                "source_id": 1001,
                "external_row_id": 7,
                "xmatch_separation_arcsec": 0.41,
                "external_catalog_name": "bmk",
                "external_object_id": "Star A",
                "raw_sptype": "G2VFE-1",
                "source_conflict_count": 3,
            },
            {
                "xmatch_batch_id": "xmatch_bmk_gaia_dr3__2026_03_26",
                "source_id": 1002,
                "external_row_id": 8,
                "xmatch_separation_arcsec": 0.52,
                "external_catalog_name": "bmk",
                "external_object_id": "Star B",
                "raw_sptype": "OB-",
                "source_conflict_count": 1,
            },
        ]
    )
    captured: dict[str, object] = {}

    def fake_read_sql(
        query: str,
        engine: object,
        chunksize: int,
    ) -> Iterator[pd.DataFrame]:
        captured["query"] = query
        captured["engine"] = engine
        captured["chunksize"] = chunksize
        yield source_chunk

    monkeypatch.setattr("exohost.db.bmk_labeled_export.pd.read_sql", fake_read_sql)

    output_csv_path = tmp_path / "labeled" / B_MK_EXTERNAL_LABELED_CSV_FILENAME
    fake_engine = object()
    export_summary = export_bmk_external_labeled_csv(
        cast(Engine, fake_engine),
        output_csv_path=output_csv_path,
        xmatch_batch_id="xmatch_bmk_gaia_dr3__2026_03_26",
        chunksize=10,
    )

    assert export_summary == BmkExternalLabeledExportSummary(
        filtered_relation_name=B_MK_EXTERNAL_LABELED_SOURCE_FILTERED_RELATION_NAME,
        crossmatch_relation_name=B_MK_EXTERNAL_LABELED_SOURCE_CROSSMATCH_RELATION_NAME,
        output_csv_path=output_csv_path,
        xmatch_batch_id="xmatch_bmk_gaia_dr3__2026_03_26",
        rows_exported=2,
    )
    assert captured["engine"] is fake_engine
    assert captured["chunksize"] == 10
    assert 'c."xmatch_selected" IS TRUE' in str(captured["query"])
    assert output_csv_path.exists()
    assert output_csv_path.read_text(encoding="utf-8").splitlines() == [
        ",".join(B_MK_EXTERNAL_LABELED_COLUMNS),
        "xmatch_bmk_gaia_dr3__2026_03_26,1001,7,bmk,Star A,G2VFE-1,G,2,V,FE-1,parsed,,0.41,True,3",
        "xmatch_bmk_gaia_dr3__2026_03_26,1002,8,bmk,Star B,OB-,OB,,,OB-,partial,ambiguous_ob_boundary_label,0.52,False,1",
    ]
