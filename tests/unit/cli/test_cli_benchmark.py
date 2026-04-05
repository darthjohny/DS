# Тестовый файл `test_cli_benchmark.py` домена `cli`.
#
# Этот файл проверяет только:
# - проверку логики домена: CLI-команды и их orchestration-сценарии;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `cli` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from exohost.cli.main import main
from exohost.evaluation.split import DatasetSplit
from exohost.training.benchmark_runner import BenchmarkRunResult


def build_benchmark_result(task_name: str) -> BenchmarkRunResult:
    # Небольшой synthetic benchmark-результат для CLI benchmark.
    split = DatasetSplit(
        full_df=pd.DataFrame({"source_id": [1, 2, 3]}),
        train_df=pd.DataFrame({"source_id": [1, 2]}),
        test_df=pd.DataFrame({"source_id": [3]}),
    )
    metrics_df = pd.DataFrame(
        [
            {
                "model_name": "hist_gradient_boosting",
                "split_name": "test",
                "accuracy": 0.91,
            }
        ]
    )
    cv_summary_df = pd.DataFrame(
        [
            {
                "model_name": "hist_gradient_boosting",
                "cv_folds": 10,
                "mean_accuracy": 0.88,
            }
        ]
    )
    target_distribution_df = pd.DataFrame(
        [
            {
                "split_name": "full",
                "target_label": "G",
                "n_rows": 3,
                "share": 1.0,
            }
        ]
    )
    return BenchmarkRunResult(
        task_name=task_name,
        split=split,
        metrics_df=metrics_df,
        cv_summary_df=cv_summary_df,
        target_distribution_df=target_distribution_df,
    )


def test_cli_benchmark_runs_router_task_and_saves_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # CLI benchmark должен запускать router-вариант и сохранять артефакты.
    captured: dict[str, object] = {}

    def fake_run_router_benchmark_from_env(**kwargs: object) -> BenchmarkRunResult:
        captured.update(kwargs)
        return build_benchmark_result("spectral_class_classification")

    monkeypatch.setattr(
        "exohost.cli.benchmark.command.run_router_benchmark_from_env",
        fake_run_router_benchmark_from_env,
    )

    output_dir = tmp_path / "benchmark_artifacts"
    exit_code = main(
        [
            "benchmark",
            "--task",
            "spectral_class_classification",
            "--models",
            "mlp_classifier,gmm_classifier",
            "--limit",
            "50",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert captured["task_name"] == "spectral_class_classification"
    assert captured["selected_model_names"] == ("mlp_classifier", "gmm_classifier")
    assert captured["limit"] == 50

    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1
    metadata = json.loads((run_dirs[0] / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["context"]["selected_models"] == ["mlp_classifier", "gmm_classifier"]
    assert (run_dirs[0] / "metrics.csv").exists()


def test_cli_benchmark_runs_host_task_dispatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # CLI benchmark должен корректно переключаться на host benchmark-контур.
    captured: dict[str, object] = {}

    def fake_run_host_benchmark_from_env(**kwargs: object) -> BenchmarkRunResult:
        captured.update(kwargs)
        return build_benchmark_result("host_field_classification")

    monkeypatch.setattr(
        "exohost.cli.benchmark.command.run_host_benchmark_from_env",
        fake_run_host_benchmark_from_env,
    )

    output_dir = tmp_path / "benchmark_artifacts"
    exit_code = main(
        [
            "benchmark",
            "--task",
            "host_field_classification",
            "--host-limit",
            "120",
            "--router-limit",
            "240",
            "--field-to-host-ratio",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert captured["task_name"] == "host_field_classification"
    assert captured["host_limit"] == 120
    assert captured["router_limit"] == 240
    assert captured["field_to_host_ratio"] == 2


def test_cli_benchmark_runs_hierarchical_task_dispatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_hierarchical_benchmark_from_env(**kwargs: object) -> BenchmarkRunResult:
        captured.update(kwargs)
        return build_benchmark_result("gaia_id_ood_classification")

    monkeypatch.setattr(
        "exohost.cli.benchmark.command.run_hierarchical_benchmark_from_env",
        fake_run_hierarchical_benchmark_from_env,
    )

    output_dir = tmp_path / "benchmark_artifacts"
    exit_code = main(
        [
            "benchmark",
            "--task",
            "gaia_id_ood_classification",
            "--limit",
            "500",
            "--models",
            "hist_gradient_boosting",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert captured["task_name"] == "gaia_id_ood_classification"
    assert captured["limit"] == 500
    assert captured["selected_model_names"] == ("hist_gradient_boosting",)


def test_cli_benchmark_runs_refinement_family_task_dispatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_refinement_family_benchmark_from_env(
        **kwargs: object,
    ) -> BenchmarkRunResult:
        captured.update(kwargs)
        return build_benchmark_result("gaia_mk_refinement_g_classification")

    monkeypatch.setattr(
        "exohost.cli.benchmark.command.run_refinement_family_benchmark_from_env",
        fake_run_refinement_family_benchmark_from_env,
    )

    output_dir = tmp_path / "benchmark_artifacts"
    exit_code = main(
        [
            "benchmark",
            "--task",
            "gaia_mk_refinement_g_classification",
            "--limit",
            "250",
            "--models",
            "hist_gradient_boosting",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert captured["task_name"] == "gaia_mk_refinement_g_classification"
    assert captured["limit"] == 250
    assert captured["selected_model_names"] == ("hist_gradient_boosting",)
