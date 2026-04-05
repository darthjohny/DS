# Тестовый файл `test_cli_score.py` домена `cli`.
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


def test_cli_score_builds_ranking_artifacts_from_csv(tmp_path: Path) -> None:
    # Проверяем минимальный end-to-end прогон score-команды на CSV.
    input_csv_path = tmp_path / "candidates.csv"
    output_dir = tmp_path / "artifacts"
    pd.DataFrame(
        [
            {
                "source_id": "1",
                "spec_class": "G",
                "evolution_stage": "dwarf",
                "host_similarity_score": 0.90,
                "parallax": 15.0,
                "phot_g_mean_mag": 11.0,
                "parallax_over_error": 18.0,
                "ruwe": 1.02,
                "validation_factor": 0.95,
            }
        ]
    ).to_csv(input_csv_path, index=False)

    exit_code = main(
        [
            "score",
            "--input-csv",
            str(input_csv_path),
            "--output-dir",
            str(output_dir),
            "--preview-rows",
            "1",
        ]
    )

    assert exit_code == 0
    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "ranking.csv").exists()
    assert (run_dirs[0] / "metadata.json").exists()


def test_cli_score_rejects_with_ranking_without_model_run_dir(tmp_path: Path) -> None:
    # Комбинированный ranking после model-scoring нельзя включать без model artifact.
    input_csv_path = tmp_path / "candidates.csv"
    pd.DataFrame(
        [
            {
                "source_id": "1",
                "spec_class": "G",
                "host_similarity_score": 0.8,
            }
        ]
    ).to_csv(input_csv_path, index=False)

    with pytest.raises(ValueError, match="requires --model-run-dir"):
        main(
            [
                "score",
                "--input-csv",
                str(input_csv_path),
                "--with-ranking",
            ]
        )
