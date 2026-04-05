# Тестовый файл `test_db_bmk_labeled_load.py` домена `db`.
#
# Этот файл проверяет только:
# - проверку логики домена: relation-layer, materialization и SQL-helper;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `db` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from dataclasses import dataclass, field
from io import TextIOBase
from pathlib import Path
from types import TracebackType
from typing import cast

from sqlalchemy.engine import Engine

from exohost.db.bmk_labeled import (
    B_MK_EXTERNAL_LABELED_COLUMNS,
    B_MK_EXTERNAL_LABELED_RELATION_NAME,
    B_MK_EXTERNAL_LABELED_SOURCE_CROSSMATCH_RELATION_NAME,
    B_MK_EXTERNAL_LABELED_SOURCE_FILTERED_RELATION_NAME,
    BmkExternalLabeledExportSummary,
    BmkExternalLabeledLoadSummary,
    materialize_bmk_external_labeled_relation,
)


@dataclass
class FakeCursor:
    # Минимальный DBAPI cursor для проверки DDL/COPY/count без реальной БД.
    counts: list[tuple[int]] = field(
        default_factory=lambda: [(2,), (2,), (2,), (1,), (1,), (1,), (0,), (0,), (1,)]
    )
    executed: list[str] = field(default_factory=list)
    copied: list[tuple[str, str]] = field(default_factory=list)
    closed: bool = False

    def execute(self, sql: str) -> None:
        self.executed.append(sql)

    def copy_expert(self, sql: str, file_obj: TextIOBase) -> None:
        self.copied.append((sql, file_obj.read()))

    def fetchone(self) -> tuple[int]:
        return self.counts.pop(0)

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeConnection:
    # Минимальный raw DBAPI connection для проверки transaction-границы.
    cursor_obj: FakeCursor = field(default_factory=FakeCursor)
    committed: bool = False
    rolled_back: bool = False
    closed: bool = False

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeEngine:
    # Минимальный SQLAlchemy-like engine с raw_connection().
    connection: FakeConnection = field(default_factory=FakeConnection)

    def raw_connection(self) -> FakeConnection:
        return self.connection


def test_materialize_bmk_external_labeled_relation_copies_csv_and_counts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    engine = FakeEngine()

    monkeypatch.setattr(
        "exohost.db.bmk_labeled_validation.relation_columns",
        lambda _engine, relation_name: (
            ("external_row_id", "external_catalog_name", "external_object_id", "raw_sptype")
            if relation_name == B_MK_EXTERNAL_LABELED_SOURCE_FILTERED_RELATION_NAME
            else (
                "xmatch_batch_id",
                "source_id",
                "external_row_id",
                "xmatch_separation_arcsec",
                "xmatch_selected",
            )
        ),
    )

    def fake_export_bmk_external_labeled_csv(
        _engine: object,
        *,
        output_csv_path: Path,
        xmatch_batch_id: str,
        filtered_relation_name: str,
        crossmatch_relation_name: str,
        chunksize: int,
        limit: int | None,
    ) -> BmkExternalLabeledExportSummary:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        output_csv_path.write_text(
            "\n".join(
                [
                    ",".join(B_MK_EXTERNAL_LABELED_COLUMNS),
                    "xmatch_bmk_gaia_dr3__2026_03_26,1001,7,bmk,Star A,G2V,G,2,V,,parsed,,0.41,False,1",
                    "xmatch_bmk_gaia_dr3__2026_03_26,1002,8,bmk,Star B,B9.5V,B,,V,,partial,fractional_subclass_requires_separate_policy,0.52,True,2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return BmkExternalLabeledExportSummary(
            filtered_relation_name=filtered_relation_name,
            crossmatch_relation_name=crossmatch_relation_name,
            output_csv_path=output_csv_path,
            xmatch_batch_id=xmatch_batch_id,
            rows_exported=2,
        )

    monkeypatch.setattr(
        "exohost.db.bmk_labeled.export_bmk_external_labeled_csv",
        fake_export_bmk_external_labeled_csv,
    )

    @dataclass
    class FakeTemporaryDirectory:
        path: Path

        def __enter__(self) -> str:
            self.path.mkdir(parents=True, exist_ok=True)
            return str(self.path)

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            return None

    monkeypatch.setattr(
        "exohost.db.bmk_labeled.TemporaryDirectory",
        lambda prefix: FakeTemporaryDirectory(tmp_path / "tempdir"),
    )

    load_summary = materialize_bmk_external_labeled_relation(
        cast(Engine, engine),
        xmatch_batch_id="xmatch_bmk_gaia_dr3__2026_03_26",
    )

    assert load_summary == BmkExternalLabeledLoadSummary(
        filtered_relation_name=B_MK_EXTERNAL_LABELED_SOURCE_FILTERED_RELATION_NAME,
        crossmatch_relation_name=B_MK_EXTERNAL_LABELED_SOURCE_CROSSMATCH_RELATION_NAME,
        target_relation_name=B_MK_EXTERNAL_LABELED_RELATION_NAME,
        xmatch_batch_id="xmatch_bmk_gaia_dr3__2026_03_26",
        rows_loaded=2,
        distinct_external_rows=2,
        distinct_source_ids=2,
        duplicate_source_ids=1,
        parsed_rows=1,
        partial_rows=1,
        unsupported_rows=0,
        empty_rows=0,
        rows_without_luminosity_class=1,
    )
    assert engine.connection.committed is True
    assert engine.connection.rolled_back is False
    assert engine.connection.closed is True
    assert engine.connection.cursor_obj.closed is True
    assert len(engine.connection.cursor_obj.copied) == 1
    assert any("CREATE TABLE IF NOT EXISTS" in sql for sql in engine.connection.cursor_obj.executed)
    assert any("DELETE FROM" in sql for sql in engine.connection.cursor_obj.executed)
    assert any('COUNT(DISTINCT "source_id")' in sql for sql in engine.connection.cursor_obj.executed)


def test_materialize_bmk_external_labeled_relation_requires_source_columns(
    monkeypatch,
) -> None:
    engine = FakeEngine()

    monkeypatch.setattr(
        "exohost.db.bmk_labeled_validation.relation_columns",
        lambda _engine, relation_name: (
            ("external_row_id", "external_catalog_name", "raw_sptype")
            if relation_name == B_MK_EXTERNAL_LABELED_SOURCE_FILTERED_RELATION_NAME
            else (
                "xmatch_batch_id",
                "source_id",
                "external_row_id",
                "xmatch_separation_arcsec",
                "xmatch_selected",
            )
        ),
    )

    try:
        materialize_bmk_external_labeled_relation(
            cast(Engine, engine),
            xmatch_batch_id="xmatch_bmk_gaia_dr3__2026_03_26",
        )
    except ValueError as exc:
        assert "Missing required filtered columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing filtered columns")
