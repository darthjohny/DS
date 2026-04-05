# Тестовый файл `test_cli_prioritize.py` домена `cli`.
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
import pytest

from exohost.cli.main import main
from exohost.reporting.model_artifacts import save_model_artifacts
from exohost.training.train_runner import TrainRunResult


class DummyEngine:
    # Минимальный read-only engine для DB-backed prioritize-тестов.

    def dispose(self) -> None:
        # В тесте достаточно наличия dispose.
        return None


def test_cli_prioritize_builds_scoring_and_ranking_from_csv(
    tmp_path: Path,
    small_spectral_class_train_result: TrainRunResult,
    small_host_field_train_result: TrainRunResult,
) -> None:
    # Сквозной prioritize должен собрать scoring и ranking из одного CSV.
    router_model_paths = save_model_artifacts(
        small_spectral_class_train_result,
        output_dir=tmp_path / "models",
    )
    host_model_paths = save_model_artifacts(
        small_host_field_train_result,
        output_dir=tmp_path / "models",
    )

    input_csv_path = tmp_path / "candidate_batch.csv"
    scoring_output_dir = tmp_path / "scoring_artifacts"
    ranking_output_dir = tmp_path / "ranking_artifacts"
    pd.DataFrame(
        [
            {
                "source_id": "501",
                "teff_gspphot": 5820.0,
                "logg_gspphot": 4.4,
                "radius_gspphot": 1.0,
                "parallax": 16.0,
                "phot_g_mean_mag": 10.8,
                "parallax_over_error": 19.0,
                "ruwe": 1.01,
                "bp_rp": 0.75,
                "mh_gspphot": 0.08,
                "validation_factor": 0.93,
            },
            {
                "source_id": "502",
                "teff_gspphot": 9000.0,
                "logg_gspphot": 4.2,
                "radius_gspphot": 2.1,
                "parallax": 8.0,
                "phot_g_mean_mag": 12.5,
                "parallax_over_error": 9.5,
                "ruwe": 1.15,
                "bp_rp": 0.10,
                "mh_gspphot": -0.05,
                "validation_factor": 0.70,
            },
        ]
    ).to_csv(input_csv_path, index=False)

    exit_code = main(
        [
            "prioritize",
            "--input-csv",
            str(input_csv_path),
            "--router-model-run-dir",
            str(router_model_paths.run_dir),
            "--host-model-run-dir",
            str(host_model_paths.run_dir),
            "--output-dir",
            str(scoring_output_dir),
            "--ranking-output-dir",
            str(ranking_output_dir),
            "--preview-rows",
            "2",
        ]
    )

    assert exit_code == 0

    scoring_run_dirs = list(scoring_output_dir.iterdir())
    assert len(scoring_run_dirs) == 1
    scored_df = pd.read_csv(scoring_run_dirs[0] / "scored.csv")
    assert "predicted_spec_class" in scored_df.columns
    assert "predicted_host_label" in scored_df.columns
    assert "host_similarity_score" in scored_df.columns
    assert "spec_class" in scored_df.columns

    ranking_run_dirs = list(ranking_output_dir.iterdir())
    assert len(ranking_run_dirs) == 1
    ranking_df = pd.read_csv(ranking_run_dirs[0] / "ranking.csv")
    assert "priority_score" in ranking_df.columns
    assert "priority_label" in ranking_df.columns
    assert "priority_reason" in ranking_df.columns


def test_cli_prioritize_builds_scoring_and_ranking_from_relation(
    monkeypatch,
    tmp_path: Path,
    small_spectral_class_train_result: TrainRunResult,
    small_host_field_train_result: TrainRunResult,
) -> None:
    # Сквозной prioritize должен уметь работать и на DB-backed relation.
    router_model_paths = save_model_artifacts(
        small_spectral_class_train_result,
        output_dir=tmp_path / "models",
    )
    host_model_paths = save_model_artifacts(
        small_host_field_train_result,
        output_dir=tmp_path / "models",
    )
    scoring_output_dir = tmp_path / "scoring_artifacts"
    ranking_output_dir = tmp_path / "ranking_artifacts"

    monkeypatch.setattr(
        "exohost.cli.prioritize.command.make_read_only_engine",
        lambda **kwargs: DummyEngine(),
    )
    monkeypatch.setattr(
        "exohost.cli.prioritize.command.load_model_scoring_dataset",
        lambda engine, **kwargs: pd.DataFrame(
            [
                {
                    "source_id": "601",
                    "teff_gspphot": 5810.0,
                    "logg_gspphot": 4.4,
                    "radius_gspphot": 1.0,
                    "parallax": 15.1,
                    "phot_g_mean_mag": 10.9,
                    "parallax_over_error": 18.2,
                    "ruwe": 1.01,
                    "bp_rp": 0.74,
                    "mh_gspphot": 0.08,
                    "validation_factor": 0.92,
                }
            ]
        ),
    )

    exit_code = main(
        [
            "prioritize",
            "--relation-name",
            "lab.v_gaia_random_stars",
            "--router-model-run-dir",
            str(router_model_paths.run_dir),
            "--host-model-run-dir",
            str(host_model_paths.run_dir),
            "--output-dir",
            str(scoring_output_dir),
            "--ranking-output-dir",
            str(ranking_output_dir),
            "--limit",
            "1",
        ]
    )

    assert exit_code == 0
    assert len(list(scoring_output_dir.iterdir())) == 1
    assert len(list(ranking_output_dir.iterdir())) == 1


def test_cli_prioritize_rejects_wrong_host_artifact(
    tmp_path: Path,
    small_spectral_class_train_result: TrainRunResult,
) -> None:
    # Host-слой нельзя подменять artifact'ом от другой target-задачи.
    router_model_paths = save_model_artifacts(
        small_spectral_class_train_result,
        output_dir=tmp_path / "models",
    )

    input_csv_path = tmp_path / "candidate_batch.csv"
    pd.DataFrame(
        [
            {
                "source_id": "701",
                "teff_gspphot": 5820.0,
                "logg_gspphot": 4.4,
                "radius_gspphot": 1.0,
                "parallax": 16.0,
                "phot_g_mean_mag": 10.8,
                "parallax_over_error": 19.0,
                "ruwe": 1.01,
                "bp_rp": 0.75,
                "mh_gspphot": 0.08,
                "validation_factor": 0.93,
            }
        ]
    ).to_csv(input_csv_path, index=False)

    with pytest.raises(ValueError, match="host_label"):
        main(
            [
                "prioritize",
                "--input-csv",
                str(input_csv_path),
                "--router-model-run-dir",
                str(router_model_paths.run_dir),
                "--host-model-run-dir",
                str(router_model_paths.run_dir),
            ]
        )
