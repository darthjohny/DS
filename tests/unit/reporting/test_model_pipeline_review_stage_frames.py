# Тестовый файл `test_model_pipeline_review_stage_frames.py` домена `reporting`.
#
# Этот файл проверяет только:
# - проверку логики домена: review-helper, сводки и notebook display-слой;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `reporting` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from pathlib import Path

from exohost.reporting.model_pipeline_review import (
    build_pipeline_stage_overview_frame,
    build_split_metrics_frame,
    build_stage_metric_long_frame,
    build_target_distribution_frame,
    load_benchmark_review_bundle,
)

from .model_pipeline_review_testkit import write_benchmark_run


def test_build_pipeline_stage_overview_frame_combines_multiple_runs(tmp_path: Path) -> None:
    coarse_run_dir = tmp_path / "coarse_run"
    ood_run_dir = tmp_path / "ood_run"
    write_benchmark_run(
        coarse_run_dir,
        task_name="gaia_id_coarse_classification",
        accuracy=0.95,
        macro_f1=0.94,
        balanced_accuracy=0.93,
    )
    write_benchmark_run(
        ood_run_dir,
        task_name="gaia_id_ood_classification",
        accuracy=0.98,
        macro_f1=0.97,
        balanced_accuracy=0.96,
    )

    overview_df = build_pipeline_stage_overview_frame(
        {
            "coarse": coarse_run_dir,
            "ood": ood_run_dir,
        }
    )

    assert overview_df["stage_name"].tolist() == ["coarse", "ood"]
    assert overview_df["task_name"].tolist() == [
        "gaia_id_coarse_classification",
        "gaia_id_ood_classification",
    ]

    metric_long_df = build_stage_metric_long_frame(overview_df)
    assert set(metric_long_df["stage_name"]) == {"coarse", "ood"}
    assert "test_accuracy" in set(metric_long_df["metric_name"])


def test_build_split_and_target_frames_return_sorted_review_tables(tmp_path: Path) -> None:
    run_dir = tmp_path / "coarse_run"
    write_benchmark_run(
        run_dir,
        task_name="gaia_id_coarse_classification",
        accuracy=0.95,
        macro_f1=0.94,
        balanced_accuracy=0.93,
    )

    bundle = load_benchmark_review_bundle(run_dir)
    split_metrics_df = build_split_metrics_frame(bundle)
    target_distribution_df = build_target_distribution_frame(bundle, split_name="full")

    assert split_metrics_df["split_name"].tolist() == ["train", "test"]
    assert target_distribution_df["target_label"].tolist() == ["A", "B"]
