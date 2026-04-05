# Тестовый файл `test_cli_prepare_upload.py` домена `cli`.
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
from exohost.db.bmk_upload import BmkGaiaUploadExportSummary


@dataclass
class FakeEngine:
    # Минимальный engine для проверки dispose() в CLI prepare-upload.
    disposed: bool = False

    def dispose(self) -> None:
        self.disposed = True


def test_cli_prepare_upload_exports_local_gaia_upload_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    fake_engine = FakeEngine()

    def fake_make_read_only_engine(**kwargs: object) -> FakeEngine:
        captured["engine_kwargs"] = kwargs
        return fake_engine

    def fake_export_bmk_gaia_upload_csv(
        engine: FakeEngine,
        *,
        output_csv_path: Path,
        relation_name: str,
        limit: int | None,
    ) -> BmkGaiaUploadExportSummary:
        captured["engine"] = engine
        captured["output_csv_path"] = output_csv_path
        captured["relation_name"] = relation_name
        captured["limit"] = limit
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        output_csv_path.write_text("header\n", encoding="utf-8")
        return BmkGaiaUploadExportSummary(
            relation_name=relation_name,
            output_csv_path=output_csv_path,
            rows_exported=100,
        )

    monkeypatch.setattr(
        "exohost.cli.prepare_upload.command.make_read_only_engine",
        fake_make_read_only_engine,
    )
    monkeypatch.setattr(
        "exohost.cli.prepare_upload.command.export_bmk_gaia_upload_csv",
        fake_export_bmk_gaia_upload_csv,
    )

    exit_code = main(
        [
            "prepare-upload",
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--limit",
            "100",
        ]
    )

    assert exit_code == 0
    assert captured["engine"] is fake_engine
    assert fake_engine.disposed is True
    assert captured["relation_name"] == "lab.gaia_mk_external_filtered"
    assert captured["limit"] == 100
    assert str(captured["output_csv_path"]).startswith(str(tmp_path / "artifacts"))
