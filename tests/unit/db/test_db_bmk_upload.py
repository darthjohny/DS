# Тестовый файл `test_db_bmk_upload.py` домена `db`.
#
# Этот файл проверяет только:
# - проверку логики домена: relation-layer, materialization и SQL-helper;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `db` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
from sqlalchemy.engine import Engine

from exohost.db.bmk_ingestion import B_MK_FILTERED_RELATION_NAME
from exohost.db.bmk_upload import (
    B_MK_GAIA_UPLOAD_COLUMNS,
    B_MK_GAIA_UPLOAD_CSV_FILENAME,
    B_MK_GAIA_UPLOAD_SOURCE_RELATION_NAME,
    BmkGaiaUploadExportSummary,
    build_bmk_gaia_upload_query,
    export_bmk_gaia_upload_csv,
)


def test_bmk_gaia_upload_contract_uses_filtered_relation_by_default() -> None:
    assert B_MK_GAIA_UPLOAD_SOURCE_RELATION_NAME == B_MK_FILTERED_RELATION_NAME
    assert B_MK_GAIA_UPLOAD_COLUMNS == (
        "external_row_id",
        "external_catalog_name",
        "external_object_id",
        "ra_deg",
        "dec_deg",
        "raw_sptype",
    )


def test_build_bmk_gaia_upload_query_uses_minimal_columns_and_ready_filter() -> None:
    query = build_bmk_gaia_upload_query()

    assert 'FROM "lab"."gaia_mk_external_filtered"' in query
    assert '"external_row_id"' in query
    assert '"external_catalog_name"' in query
    assert '"external_object_id"' in query
    assert '"ra_deg"' in query
    assert '"dec_deg"' in query
    assert '"raw_sptype"' in query
    assert '"ready_for_gaia_crossmatch" IS TRUE' in query
    assert 'ORDER BY "external_row_id" ASC' in query


def test_build_bmk_gaia_upload_query_adds_limit_when_requested() -> None:
    query = build_bmk_gaia_upload_query(limit=100)

    assert "LIMIT 100" in query


@dataclass
class FakeEngine:
    # Минимальный engine-placeholder для read_sql monkeypatch теста.
    disposed: bool = False

    def dispose(self) -> None:
        self.disposed = True


def test_export_bmk_gaia_upload_csv_writes_csv_and_returns_summary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    frame = pd.DataFrame(
        [
            {
                "external_row_id": 0,
                "external_catalog_name": "bmk",
                "external_object_id": "Star A",
                "ra_deg": 10.5,
                "dec_deg": -5.25,
                "raw_sptype": "G2V",
            },
        ],
        columns=B_MK_GAIA_UPLOAD_COLUMNS,
    )

    def fake_read_sql(query: str, engine: FakeEngine) -> pd.DataFrame:
        captured["query"] = query
        captured["engine"] = engine
        return frame

    monkeypatch.setattr("exohost.db.bmk_upload.pd.read_sql", fake_read_sql)

    output_csv_path = tmp_path / "gaia_upload" / B_MK_GAIA_UPLOAD_CSV_FILENAME
    fake_engine = FakeEngine()
    export_summary = export_bmk_gaia_upload_csv(
        cast(Engine, fake_engine),
        output_csv_path=output_csv_path,
    )

    assert export_summary == BmkGaiaUploadExportSummary(
        relation_name=B_MK_GAIA_UPLOAD_SOURCE_RELATION_NAME,
        output_csv_path=output_csv_path,
        rows_exported=1,
    )
    assert captured["engine"] is fake_engine
    assert 'FROM "lab"."gaia_mk_external_filtered"' in str(captured["query"])
    assert output_csv_path.exists()
    assert output_csv_path.read_text(encoding="utf-8").splitlines() == [
        "external_row_id,external_catalog_name,external_object_id,ra_deg,dec_deg,raw_sptype",
        "0,bmk,Star A,10.5,-5.25,G2V",
    ]
