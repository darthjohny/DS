# Тестовый файл `test_cli_materialize_crossmatch.py` домена `cli`.
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
from exohost.db.bmk_crossmatch import BmkCrossmatchMaterializationSummary


@dataclass
class FakeEngine:
    # Минимальный engine для проверки dispose() в CLI materialize-crossmatch.
    disposed: bool = False

    def dispose(self) -> None:
        self.disposed = True


def test_cli_materialize_crossmatch_runs_db_materialization(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    fake_engine = FakeEngine()

    def fake_make_write_engine(**kwargs: object) -> FakeEngine:
        captured["engine_kwargs"] = kwargs
        return fake_engine

    def fake_materialize_bmk_crossmatch_relation(
        engine: Engine,
        *,
        source_relation_name: str,
        target_relation_name: str,
        xmatch_batch_id: str,
    ) -> BmkCrossmatchMaterializationSummary:
        captured["engine"] = engine
        captured["source_relation_name"] = source_relation_name
        captured["target_relation_name"] = target_relation_name
        captured["xmatch_batch_id"] = xmatch_batch_id
        return BmkCrossmatchMaterializationSummary(
            source_relation_name=source_relation_name,
            target_relation_name=target_relation_name,
            xmatch_batch_id=xmatch_batch_id,
            rows_loaded=824038,
            distinct_external_rows=824038,
            selected_rows=824038,
            multi_match_external_rows=13905,
        )

    monkeypatch.setattr(
        "exohost.cli.materialize_crossmatch.command.make_write_engine",
        fake_make_write_engine,
    )
    monkeypatch.setattr(
        "exohost.cli.materialize_crossmatch.command.materialize_bmk_crossmatch_relation",
        fake_materialize_bmk_crossmatch_relation,
    )

    exit_code = main(
        [
            "materialize-crossmatch",
            "--xmatch-batch-id",
            "xmatch_bmk_gaia_dr3__2026_03_26",
        ]
    )

    assert exit_code == 0
    assert captured["engine"] is fake_engine
    assert fake_engine.disposed is True
    assert captured["source_relation_name"] == "public.raw_landing_table"
    assert captured["target_relation_name"] == "lab.gaia_mk_external_crossmatch"
    assert captured["xmatch_batch_id"] == "xmatch_bmk_gaia_dr3__2026_03_26"
