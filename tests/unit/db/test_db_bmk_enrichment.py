# Тестовый файл `test_db_bmk_enrichment.py` домена `db`.
#
# Этот файл проверяет только:
# - проверку логики домена: relation-layer, materialization и SQL-helper;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `db` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from sqlalchemy.engine import Engine

from exohost.db.bmk_enrichment import (
    B_MK_GAIA_ENRICHMENT_COLUMNS,
    B_MK_GAIA_ENRICHMENT_MANIFEST_FILENAME,
    B_MK_GAIA_ENRICHMENT_QUERY_TEMPLATE_FILENAME,
    B_MK_GAIA_ENRICHMENT_SOURCE_RELATION_NAME,
    BmkGaiaEnrichmentExportSummary,
    build_bmk_gaia_enrichment_source_query,
    build_gaia_radius_flame_query_template,
    export_bmk_gaia_enrichment_batches,
)


def test_bmk_gaia_enrichment_contract_uses_external_labeled_relation() -> None:
    assert B_MK_GAIA_ENRICHMENT_SOURCE_RELATION_NAME == "lab.gaia_mk_external_labeled"
    assert B_MK_GAIA_ENRICHMENT_COLUMNS == ("source_id",)


def test_build_bmk_gaia_enrichment_source_query_filters_conflict_free_by_default() -> None:
    query = build_bmk_gaia_enrichment_source_query(
        xmatch_batch_id="xmatch_bmk_gaia_dr3__2026_03_26",
    )

    assert 'FROM "lab"."gaia_mk_external_labeled"' in query
    assert '"xmatch_batch_id" =' in query
    assert '"has_source_conflict" IS FALSE' in query
    assert 'ORDER BY "source_id" ASC' in query


def test_build_bmk_gaia_enrichment_source_query_can_include_conflicts() -> None:
    query = build_bmk_gaia_enrichment_source_query(
        xmatch_batch_id="xmatch_bmk_gaia_dr3__2026_03_26",
        only_conflict_free=False,
        limit=10,
    )

    assert '"has_source_conflict" IS FALSE' not in query
    assert "LIMIT 10" in query


def test_build_gaia_radius_flame_query_template_uses_official_field_name() -> None:
    query = build_gaia_radius_flame_query_template(
        user_table_name="user_test.batch_0001",
    )

    assert "user_test.batch_0001" in query
    assert "gaiadr3.astrophysical_parameters" in query
    assert "ap.radius_flame" in query


def test_export_bmk_gaia_enrichment_batches_writes_batches_manifest_and_template(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    source_frame = pd.DataFrame(
        [
            {"source_id": 1001},
            {"source_id": 1002},
            {"source_id": 1003},
            {"source_id": 1004},
            {"source_id": 1005},
        ],
        columns=B_MK_GAIA_ENRICHMENT_COLUMNS,
    )

    monkeypatch.setattr(
        "exohost.db.bmk_enrichment.relation_columns",
        lambda _engine, _relation_name: (
            "xmatch_batch_id",
            "source_id",
            "has_source_conflict",
        ),
    )

    def fake_read_sql(query: str, engine: object) -> pd.DataFrame:
        captured["query"] = query
        captured["engine"] = engine
        return source_frame

    monkeypatch.setattr("exohost.db.bmk_enrichment.pd.read_sql", fake_read_sql)

    output_dir = tmp_path / "gaia_enrichment"
    fake_engine = object()
    summary = export_bmk_gaia_enrichment_batches(
        cast(Engine, fake_engine),
        output_dir=output_dir,
        xmatch_batch_id="xmatch_bmk_gaia_dr3__2026_03_26",
        batch_size=2,
    )

    assert summary == BmkGaiaEnrichmentExportSummary(
        relation_name=B_MK_GAIA_ENRICHMENT_SOURCE_RELATION_NAME,
        output_dir=output_dir,
        manifest_path=output_dir / B_MK_GAIA_ENRICHMENT_MANIFEST_FILENAME,
        query_template_path=output_dir / B_MK_GAIA_ENRICHMENT_QUERY_TEMPLATE_FILENAME,
        xmatch_batch_id="xmatch_bmk_gaia_dr3__2026_03_26",
        only_conflict_free=True,
        total_rows_exported=5,
        total_batches=3,
        batch_size=2,
    )
    assert captured["engine"] is fake_engine
    assert '"has_source_conflict" IS FALSE' in str(captured["query"])
    assert (output_dir / "bmk_gaia_enrichment_batch_0001.csv").read_text(
        encoding="utf-8"
    ).splitlines() == [
        "source_id",
        "1001",
        "1002",
    ]
    assert (output_dir / "bmk_gaia_enrichment_batch_0003.csv").read_text(
        encoding="utf-8"
    ).splitlines() == [
        "source_id",
        "1005",
    ]
    manifest_lines = (output_dir / B_MK_GAIA_ENRICHMENT_MANIFEST_FILENAME).read_text(
        encoding="utf-8"
    ).splitlines()
    assert manifest_lines[0] == "batch_index,csv_filename,rows_exported,source_id_min,source_id_max"
    assert "bmk_gaia_enrichment_batch_0001.csv" in manifest_lines[1]
    assert "bmk_gaia_enrichment_batch_0003.csv" in manifest_lines[3]
    query_template = (
        output_dir / B_MK_GAIA_ENRICHMENT_QUERY_TEMPLATE_FILENAME
    ).read_text(encoding="utf-8")
    assert "ap.radius_flame" in query_template


def test_export_bmk_gaia_enrichment_batches_rejects_nonpositive_batch_size(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "exohost.db.bmk_enrichment.relation_columns",
        lambda _engine, _relation_name: (
            "xmatch_batch_id",
            "source_id",
            "has_source_conflict",
        ),
    )
    monkeypatch.setattr(
        "exohost.db.bmk_enrichment.pd.read_sql",
        lambda _query, _engine: pd.DataFrame(columns=B_MK_GAIA_ENRICHMENT_COLUMNS),
    )

    with pytest.raises(ValueError, match="batch_size must be positive"):
        export_bmk_gaia_enrichment_batches(
            cast(Engine, object()),
            output_dir=tmp_path,
            xmatch_batch_id="xmatch_bmk_gaia_dr3__2026_03_26",
            batch_size=0,
        )
