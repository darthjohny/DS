# Тестовый файл `test_cli_ingest.py` домена `cli`.
#
# Этот файл проверяет только:
# - проверку логики домена: CLI-команды и их orchestration-сценарии;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `cli` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from exohost.cli.main import main
from exohost.db.bmk_ingestion import BmkDatabaseLoadSummary
from exohost.ingestion.bmk import (
    BmkCatalogSource,
    BmkExportBundle,
    BmkExportPaths,
    BmkImportSummary,
    BmkPrimaryFilterSummary,
)


@dataclass
class FakeEngine:
    # Минимальный engine для проверки dispose() в CLI ingest.
    disposed: bool = False

    def dispose(self) -> None:
        self.disposed = True


def build_export_bundle(tmp_path: Path) -> BmkExportBundle:
    export_dir = tmp_path / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_paths = BmkExportPaths(
        raw_csv_path=export_dir / "bmk_external_raw.csv",
        filtered_csv_path=export_dir / "bmk_external_filtered.csv",
        rejected_csv_path=export_dir / "bmk_external_rejected.csv",
    )
    for csv_path in (
        export_paths.raw_csv_path,
        export_paths.filtered_csv_path,
        export_paths.rejected_csv_path,
    ):
        csv_path.write_text("header\n", encoding="utf-8")

    return BmkExportBundle(
        source=None,
        export_paths=export_paths,
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


def test_cli_ingest_runs_export_and_db_load(monkeypatch, tmp_path: Path) -> None:
    readme_path = tmp_path / "ReadMe"
    data_path = tmp_path / "mktypes.dat"
    readme_path.write_text("ReadMe", encoding="utf-8")
    data_path.write_text("mktypes", encoding="utf-8")

    captured: dict[str, object] = {}
    fake_engine = FakeEngine()

    def fake_build_bmk_export_bundle(
        source: BmkCatalogSource,
        *,
        output_dir: Path,
    ) -> BmkExportBundle:
        captured["source"] = source
        captured["output_dir"] = output_dir
        return build_export_bundle(tmp_path)

    def fake_make_write_engine(**kwargs: object) -> FakeEngine:
        captured["engine_kwargs"] = kwargs
        return fake_engine

    def fake_load_bmk_exports_into_db(
        engine: FakeEngine,
        export_paths: BmkExportPaths,
    ) -> BmkDatabaseLoadSummary:
        captured["engine"] = engine
        captured["export_paths"] = export_paths
        return BmkDatabaseLoadSummary(
            raw_relation_name="lab.gaia_mk_external_raw",
            filtered_relation_name="lab.gaia_mk_external_filtered",
            rejected_relation_name="lab.gaia_mk_external_rejected",
            raw_rows_loaded=2,
            filtered_rows_loaded=1,
            rejected_rows_loaded=3,
        )

    monkeypatch.setattr(
        "exohost.cli.ingest.command.build_bmk_export_bundle",
        fake_build_bmk_export_bundle,
    )
    monkeypatch.setattr(
        "exohost.cli.ingest.command.make_write_engine",
        fake_make_write_engine,
    )
    monkeypatch.setattr(
        "exohost.cli.ingest.command.load_bmk_exports_into_db",
        fake_load_bmk_exports_into_db,
    )

    exit_code = main(
        [
            "ingest",
            "--readme-path",
            str(readme_path),
            "--data-path",
            str(data_path),
            "--output-dir",
            str(tmp_path / "artifacts"),
        ]
    )

    assert exit_code == 0
    source = cast(BmkCatalogSource, captured["source"])
    assert captured["engine"] is fake_engine
    assert fake_engine.disposed is True
    assert str(source.readme_path) == str(readme_path)
    assert str(source.data_path) == str(data_path)
    assert str(captured["output_dir"]).startswith(str(tmp_path / "artifacts"))


def test_cli_ingest_supports_skip_db_load(monkeypatch, tmp_path: Path) -> None:
    readme_path = tmp_path / "ReadMe"
    data_path = tmp_path / "mktypes.dat"
    readme_path.write_text("ReadMe", encoding="utf-8")
    data_path.write_text("mktypes", encoding="utf-8")

    monkeypatch.setattr(
        "exohost.cli.ingest.command.build_bmk_export_bundle",
        lambda source, *, output_dir: build_export_bundle(tmp_path),
    )

    exit_code = main(
        [
            "ingest",
            "--readme-path",
            str(readme_path),
            "--data-path",
            str(data_path),
            "--skip-db-load",
        ]
    )

    assert exit_code == 0
