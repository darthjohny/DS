# Тестовый файл `test_cli_prepare_enrichment.py` домена `cli`.
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

from exohost.cli.main import main
from exohost.db.bmk_enrichment import BmkGaiaEnrichmentExportSummary


@dataclass
class FakeEngine:
    # Минимальный engine для проверки dispose() в CLI prepare-enrichment.
    disposed: bool = False

    def dispose(self) -> None:
        self.disposed = True


def test_cli_prepare_enrichment_exports_local_batch_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    fake_engine = FakeEngine()

    def fake_make_read_only_engine(**kwargs: object) -> FakeEngine:
        captured["engine_kwargs"] = kwargs
        return fake_engine

    def fake_export_bmk_gaia_enrichment_batches(
        engine: FakeEngine,
        *,
        output_dir: Path,
        relation_name: str,
        xmatch_batch_id: str,
        batch_size: int,
        only_conflict_free: bool,
        limit: int | None,
    ) -> BmkGaiaEnrichmentExportSummary:
        captured["engine"] = engine
        captured["output_dir"] = output_dir
        captured["relation_name"] = relation_name
        captured["xmatch_batch_id"] = xmatch_batch_id
        captured["batch_size"] = batch_size
        captured["only_conflict_free"] = only_conflict_free
        captured["limit"] = limit
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.csv"
        query_template_path = output_dir / "query.sql"
        manifest_path.write_text("header\n", encoding="utf-8")
        query_template_path.write_text("SELECT 1;\n", encoding="utf-8")
        return BmkGaiaEnrichmentExportSummary(
            relation_name=relation_name,
            output_dir=output_dir,
            manifest_path=manifest_path,
            query_template_path=query_template_path,
            xmatch_batch_id=xmatch_batch_id,
            only_conflict_free=only_conflict_free,
            total_rows_exported=100,
            total_batches=2,
            batch_size=batch_size,
        )

    monkeypatch.setattr(
        "exohost.cli.prepare_enrichment.command.make_read_only_engine",
        fake_make_read_only_engine,
    )
    monkeypatch.setattr(
        "exohost.cli.prepare_enrichment.command.export_bmk_gaia_enrichment_batches",
        fake_export_bmk_gaia_enrichment_batches,
    )

    exit_code = main(
        [
            "prepare-enrichment",
            "--xmatch-batch-id",
            "xmatch_bmk_gaia_dr3__2026_03_26",
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--batch-size",
            "25",
            "--limit",
            "100",
        ]
    )

    assert exit_code == 0
    assert captured["engine"] is fake_engine
    assert fake_engine.disposed is True
    assert captured["relation_name"] == "lab.gaia_mk_external_labeled"
    assert captured["xmatch_batch_id"] == "xmatch_bmk_gaia_dr3__2026_03_26"
    assert captured["batch_size"] == 25
    assert captured["only_conflict_free"] is True
    assert captured["limit"] == 100
    assert str(captured["output_dir"]).startswith(str(tmp_path / "artifacts"))
