# Integration-тест roundtrip контура train -> model artifact -> score.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from exohost.cli.main import main
from exohost.training.train_runner import TrainRunResult


def test_train_score_roundtrip_creates_scored_artifacts(
    monkeypatch,
    tmp_path: Path,
    small_spectral_class_train_result: TrainRunResult,
    small_model_scoring_frame: pd.DataFrame,
) -> None:
    # Проверяем сквозной CLI-путь:
    # train сохраняет model artifact, а score потом применяет его к новым данным.
    monkeypatch.setattr(
        "exohost.cli.train.command.run_router_training_from_env",
        lambda **kwargs: small_spectral_class_train_result,
    )

    model_output_dir = tmp_path / "models"
    score_output_dir = tmp_path / "scoring"
    input_csv_path = tmp_path / "candidates.csv"
    small_model_scoring_frame.to_csv(input_csv_path, index=False)

    train_exit_code = main(
        [
            "train",
            "--task",
            "spectral_class_classification",
            "--model",
            "hist_gradient_boosting",
            "--output-dir",
            str(model_output_dir),
        ]
    )
    assert train_exit_code == 0

    model_run_dirs = list(model_output_dir.iterdir())
    assert len(model_run_dirs) == 1

    score_exit_code = main(
        [
            "score",
            "--input-csv",
            str(input_csv_path),
            "--model-run-dir",
            str(model_run_dirs[0]),
            "--output-dir",
            str(score_output_dir),
        ]
    )
    assert score_exit_code == 0

    scoring_run_dirs = list(score_output_dir.iterdir())
    assert len(scoring_run_dirs) == 1
    scored_df = pd.read_csv(scoring_run_dirs[0] / "scored.csv")
    assert "predicted_spec_class" in scored_df.columns
