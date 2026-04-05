# Тестовый файл `test_cli_public_tasks.py` домена `cli`.
#
# Этот файл проверяет только:
# - проверку логики домена: CLI-команды и их orchestration-сценарии;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `cli` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import pytest

from exohost.cli.main import build_parser


def test_cli_hides_subclass_task_until_source_is_ready() -> None:
    # Подклассную задачу не публикуем в CLI, пока под нее не готов training source.
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "benchmark",
                "--task",
                "spectral_subclass_classification",
            ]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "train",
                "--task",
                "spectral_subclass_classification",
            ]
        )


def test_cli_exposes_second_wave_refinement_family_tasks() -> None:
    parser = build_parser()

    benchmark_args = parser.parse_args(
        [
            "benchmark",
            "--task",
            "gaia_mk_refinement_g_classification",
        ]
    )
    train_args = parser.parse_args(
        [
            "train",
            "--task",
            "gaia_mk_refinement_g_classification",
            "--model",
            "hist_gradient_boosting",
        ]
    )

    assert benchmark_args.task == "gaia_mk_refinement_g_classification"
    assert train_args.task == "gaia_mk_refinement_g_classification"
