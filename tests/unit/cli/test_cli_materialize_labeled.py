# Тестовый файл `test_cli_materialize_labeled.py` домена `cli`.
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

from sqlalchemy.engine import Engine

from exohost.cli.main import main
from exohost.db.bmk_labeled import BmkExternalLabeledLoadSummary


@dataclass
class FakeEngine:
    # Минимальный engine для проверки dispose() в CLI materialize-labeled.
    disposed: bool = False

    def dispose(self) -> None:
        self.disposed = True


def test_cli_materialize_labeled_runs_db_materialization(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    fake_engine = FakeEngine()

    def fake_make_write_engine(**kwargs: object) -> FakeEngine:
        captured["engine_kwargs"] = kwargs
        return fake_engine

    def fake_materialize_bmk_external_labeled_relation(
        engine: Engine,
        *,
        xmatch_batch_id: str,
        filtered_relation_name: str,
        crossmatch_relation_name: str,
        target_relation_name: str,
        chunksize: int,
        limit: int | None,
    ) -> BmkExternalLabeledLoadSummary:
        captured["engine"] = engine
        captured["xmatch_batch_id"] = xmatch_batch_id
        captured["filtered_relation_name"] = filtered_relation_name
        captured["crossmatch_relation_name"] = crossmatch_relation_name
        captured["target_relation_name"] = target_relation_name
        captured["chunksize"] = chunksize
        captured["limit"] = limit
        return BmkExternalLabeledLoadSummary(
            filtered_relation_name=filtered_relation_name,
            crossmatch_relation_name=crossmatch_relation_name,
            target_relation_name=target_relation_name,
            xmatch_batch_id=xmatch_batch_id,
            rows_loaded=809832,
            distinct_external_rows=809832,
            distinct_source_ids=564820,
            duplicate_source_ids=162594,
            parsed_rows=284887,
            partial_rows=524945,
            unsupported_rows=0,
            empty_rows=0,
            rows_without_luminosity_class=123456,
        )

    monkeypatch.setattr(
        "exohost.cli.materialize_labeled.command.make_write_engine",
        fake_make_write_engine,
    )
    monkeypatch.setattr(
        "exohost.cli.materialize_labeled.command.materialize_bmk_external_labeled_relation",
        fake_materialize_bmk_external_labeled_relation,
    )

    exit_code = main(
        [
            "materialize-labeled",
            "--xmatch-batch-id",
            "xmatch_bmk_gaia_dr3__2026_03_26",
            "--chunksize",
            "1000",
            "--limit",
            "10",
        ]
    )

    assert exit_code == 0
    assert captured["engine"] is fake_engine
    assert fake_engine.disposed is True
    assert captured["filtered_relation_name"] == "lab.gaia_mk_external_filtered"
    assert captured["crossmatch_relation_name"] == "lab.gaia_mk_external_crossmatch"
    assert captured["target_relation_name"] == "lab.gaia_mk_external_labeled"
    assert captured["xmatch_batch_id"] == "xmatch_bmk_gaia_dr3__2026_03_26"
    assert captured["chunksize"] == 1000
    assert captured["limit"] == 10
