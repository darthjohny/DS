# Тестовый файл `test_cli_train.py` домена `cli`.
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

from exohost.cli.main import main
from exohost.training.train_runner import TrainRunResult


def test_cli_train_saves_model_artifacts(
    monkeypatch,
    tmp_path: Path,
    small_spectral_class_train_result: TrainRunResult,
) -> None:
    # Проверяем минимальный end-to-end прогон train-команды без реальной БД.
    monkeypatch.setattr(
        "exohost.cli.train.command.run_router_training_from_env",
        lambda **kwargs: small_spectral_class_train_result,
    )

    output_dir = tmp_path / "artifacts"
    exit_code = main(
        [
            "train",
            "--task",
            "spectral_class_classification",
            "--model",
            "hist_gradient_boosting",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "model.joblib").exists()
    assert (run_dirs[0] / "metadata.json").exists()


def test_cli_train_dispatches_host_training(
    monkeypatch,
    tmp_path: Path,
    small_host_field_train_result: TrainRunResult,
) -> None:
    # Проверяем, что host-задача уходит в host training-контур.
    monkeypatch.setattr(
        "exohost.cli.train.command.run_host_training_from_env",
        lambda **kwargs: small_host_field_train_result,
    )

    output_dir = tmp_path / "artifacts"
    exit_code = main(
        [
            "train",
            "--task",
            "host_field_classification",
            "--model",
            "hist_gradient_boosting",
            "--host-limit",
            "100",
            "--router-limit",
            "200",
            "--field-to-host-ratio",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "model.joblib").exists()


def test_cli_train_dispatches_hierarchical_training(
    monkeypatch,
    tmp_path: Path,
    small_spectral_class_train_result: TrainRunResult,
) -> None:
    monkeypatch.setattr(
        "exohost.cli.train.command.run_hierarchical_training_from_env",
        lambda **kwargs: small_spectral_class_train_result,
    )

    output_dir = tmp_path / "artifacts"
    exit_code = main(
        [
            "train",
            "--task",
            "gaia_id_coarse_classification",
            "--model",
            "hist_gradient_boosting",
            "--limit",
            "1000",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "model.joblib").exists()


def test_cli_train_dispatches_refinement_family_training(
    monkeypatch,
    tmp_path: Path,
    small_spectral_class_train_result: TrainRunResult,
) -> None:
    monkeypatch.setattr(
        "exohost.cli.train.command.run_refinement_family_training_from_env",
        lambda **kwargs: small_spectral_class_train_result,
    )

    output_dir = tmp_path / "artifacts"
    exit_code = main(
        [
            "train",
            "--task",
            "gaia_mk_refinement_g_classification",
            "--model",
            "hist_gradient_boosting",
            "--limit",
            "1000",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "model.joblib").exists()
