# Тестовый файл `test_model_pipeline_review_bundle.py` домена `reporting`.
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
    build_benchmark_summary_frame,
    load_benchmark_review_bundle,
)

from .model_pipeline_review_testkit import (
    require_float_scalar,
    require_int_scalar,
    write_benchmark_run,
)


def test_load_benchmark_review_bundle_reads_all_review_tables(tmp_path: Path) -> None:
    run_dir = tmp_path / "coarse_run"
    write_benchmark_run(
        run_dir,
        task_name="gaia_id_coarse_classification",
        accuracy=0.95,
        macro_f1=0.94,
        balanced_accuracy=0.93,
    )

    bundle = load_benchmark_review_bundle(run_dir)

    assert bundle.run_dir == run_dir
    assert list(bundle.metrics_df["split_name"]) == ["test", "train"]
    assert bundle.metadata["task_name"] == "gaia_id_coarse_classification"
    assert require_int_scalar(bundle.target_distribution_df.shape[0]) == 3


def test_build_benchmark_summary_frame_extracts_top_level_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "coarse_run"
    write_benchmark_run(
        run_dir,
        task_name="gaia_id_coarse_classification",
        accuracy=0.95,
        macro_f1=0.94,
        balanced_accuracy=0.93,
    )

    bundle = load_benchmark_review_bundle(run_dir)
    summary_df = build_benchmark_summary_frame(bundle)

    assert list(summary_df["task_name"]) == ["gaia_id_coarse_classification"]
    assert require_float_scalar(summary_df.loc[0, "test_accuracy"]) == 0.95
    assert require_float_scalar(summary_df.loc[0, "cv_mean_macro_f1"]) == 0.93
