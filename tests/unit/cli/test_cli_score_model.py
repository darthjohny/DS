# Тестовый файл `test_cli_score_model.py` домена `cli`.
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


def test_cli_score_applies_saved_model_artifact(
    tmp_path: Path,
    small_spectral_class_train_result: TrainRunResult,
) -> None:
    # Проверяем end-to-end model-scoring через сохраненный model artifact.
    train_result = small_spectral_class_train_result
    model_paths = save_model_artifacts(train_result, output_dir=tmp_path / "models")

    input_csv_path = tmp_path / "candidates.csv"
    output_dir = tmp_path / "scoring_artifacts"
    pd.DataFrame(
        [
            {
                "source_id": "101",
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
    ).to_csv(input_csv_path, index=False)

    exit_code = main(
        [
            "score",
            "--input-csv",
            str(input_csv_path),
            "--model-run-dir",
            str(model_paths.run_dir),
            "--output-dir",
            str(output_dir),
            "--preview-rows",
            "1",
        ]
    )

    assert exit_code == 0
    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "scored.csv").exists()
    assert (run_dirs[0] / "metadata.json").exists()


def test_cli_score_builds_ranking_after_host_model_scoring(
    tmp_path: Path,
    small_host_field_train_result: TrainRunResult,
) -> None:
    # После host model-scoring должен уметь строиться ranking по scored output.
    train_result = small_host_field_train_result
    model_paths = save_model_artifacts(train_result, output_dir=tmp_path / "models")

    input_csv_path = tmp_path / "host_candidates.csv"
    scoring_output_dir = tmp_path / "scoring_artifacts"
    ranking_output_dir = tmp_path / "ranking_artifacts"
    pd.DataFrame(
        [
            {
                "source_id": "301",
                "spec_class": "G",
                "evolution_stage": "dwarf",
                "teff_gspphot": 5830.0,
                "logg_gspphot": 4.4,
                "radius_gspphot": 1.0,
                "parallax": 16.0,
                "phot_g_mean_mag": 10.8,
                "parallax_over_error": 19.5,
                "ruwe": 1.01,
                "bp_rp": 0.75,
                "mh_gspphot": 0.06,
                "validation_factor": 0.94,
            }
        ]
    ).to_csv(input_csv_path, index=False)

    exit_code = main(
        [
            "score",
            "--input-csv",
            str(input_csv_path),
            "--model-run-dir",
            str(model_paths.run_dir),
            "--output-dir",
            str(scoring_output_dir),
            "--ranking-output-dir",
            str(ranking_output_dir),
            "--with-ranking",
            "--preview-rows",
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
    assert saved_ranking.loc[0, "priority_label"] in {"low", "medium", "high"}
