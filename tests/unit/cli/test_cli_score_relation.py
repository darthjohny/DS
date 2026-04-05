# Тестовый файл `test_cli_score_relation.py` домена `cli`.
#
# Этот файл проверяет только:
# - проверку логики домена: CLI-команды и их orchestration-сценарии;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `cli` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from exohost.cli.main import main
from exohost.reporting.model_artifacts import save_model_artifacts
from exohost.training.train_runner import TrainRunResult


class DummyEngine:
    # Минимальный read-only engine для проверки dispose в CLI score.

    def dispose(self) -> None:
        # В тесте достаточно самого факта наличия dispose.
        return None


def test_cli_score_applies_saved_model_to_relation_source(
    monkeypatch,
    tmp_path: Path,
    small_spectral_class_train_result: TrainRunResult,
) -> None:
    # Проверяем DB-backed scoring без реального подключения к БД.
    train_result = small_spectral_class_train_result
    model_paths = save_model_artifacts(train_result, output_dir=tmp_path / "models")
    output_dir = tmp_path / "scoring_artifacts"

    monkeypatch.setattr(
        "exohost.cli.score.command.make_read_only_engine",
        lambda **kwargs: DummyEngine(),
    )
    monkeypatch.setattr(
        "exohost.cli.score.command.load_model_scoring_dataset",
        lambda engine, **kwargs: pd.DataFrame(
            [
                {
                    "source_id": "201",
                    "teff_gspphot": 5810.0,
                    "logg_gspphot": 4.4,
                    "radius_gspphot": 1.0,
                    "parallax": 15.1,
                    "parallax_over_error": 18.2,
                    "ruwe": 1.01,
                    "bp_rp": 0.74,
                    "mh_gspphot": 0.08,
                }
            ]
        ),
    )

    exit_code = main(
        [
            "score",
            "--relation-name",
            "lab.v_gaia_random_stars",
            "--model-run-dir",
            str(model_paths.run_dir),
            "--output-dir",
            str(output_dir),
            "--limit",
            "1",
        ]
    )

    assert exit_code == 0
    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "scored.csv").exists()
    assert (run_dirs[0] / "metadata.json").exists()


def test_cli_score_builds_ranking_from_relation_source(
    monkeypatch,
    tmp_path: Path,
    small_host_field_train_result: TrainRunResult,
) -> None:
    # Проверяем комбинированный DB-backed host scoring + ranking.
    train_result = small_host_field_train_result
    model_paths = save_model_artifacts(train_result, output_dir=tmp_path / "models")
    scoring_output_dir = tmp_path / "scoring_artifacts"
    ranking_output_dir = tmp_path / "ranking_artifacts"

    monkeypatch.setattr(
        "exohost.cli.score.command.make_read_only_engine",
        lambda **kwargs: DummyEngine(),
    )
    monkeypatch.setattr(
        "exohost.cli.score.command.load_model_scoring_dataset",
        lambda engine, **kwargs: pd.DataFrame(
            [
                {
                    "source_id": "401",
                    "spec_class": "G",
                    "evolution_stage": "dwarf",
                    "teff_gspphot": 5815.0,
                    "logg_gspphot": 4.4,
                    "radius_gspphot": 1.0,
                    "parallax": 15.1,
                    "phot_g_mean_mag": 10.9,
                    "parallax_over_error": 18.2,
                    "ruwe": 1.01,
                    "bp_rp": 0.74,
                    "mh_gspphot": 0.08,
                    "validation_factor": 0.93,
                }
            ]
        ),
    )

    exit_code = main(
        [
            "score",
            "--relation-name",
            "lab.v_gaia_random_stars",
            "--model-run-dir",
            str(model_paths.run_dir),
            "--output-dir",
            str(scoring_output_dir),
            "--ranking-output-dir",
            str(ranking_output_dir),
            "--with-ranking",
            "--limit",
            "1",
        ]
    )

    assert exit_code == 0

    scoring_run_dirs = list(scoring_output_dir.iterdir())
    assert len(scoring_run_dirs) == 1
    assert (scoring_run_dirs[0] / "scored.csv").exists()

    ranking_run_dirs = list(ranking_output_dir.iterdir())
    assert len(ranking_run_dirs) == 1
    saved_ranking = pd.read_csv(ranking_run_dirs[0] / "ranking.csv")
    assert "priority_score" in saved_ranking.columns
