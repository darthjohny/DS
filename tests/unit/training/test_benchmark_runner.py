# Тестовый файл `test_benchmark_runner.py` домена `training`.
#
# Этот файл проверяет только:
# - проверку логики домена: обучающие orchestration-сценарии и benchmark-runner;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `training` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import pandas as pd

from exohost.evaluation.protocol import (
    STAGE_CLASSIFICATION_TASK,
    BenchmarkProtocol,
    CrossValidationConfig,
    SplitConfig,
)
from exohost.models.gmm_classifier import GMMClassifier
from exohost.models.hgb_classifier import HGBClassifier
from exohost.models.mlp_classifier import MLPClassifier
from exohost.models.protocol import ModelSpec
from exohost.training.benchmark_runner import run_benchmark


def build_router_benchmark_frame() -> pd.DataFrame:
    # Synthetic dataset с достаточным числом строк для split и CV.
    rows: list[dict[str, object]] = []
    base_rows = [
        ("G", "G2", "dwarf", 5700.0, 4.40, 1.00, 11.0, 15.0, 1.01, 0.82, 0.0),
        ("G", "G3", "dwarf", 5600.0, 4.35, 0.97, 10.0, 14.0, 1.02, 0.84, 0.1),
        ("K", "K4", "dwarf", 4700.0, 4.55, 0.82, 9.0, 13.0, 1.03, 1.08, -0.1),
        ("K", "K5", "dwarf", 4550.0, 4.60, 0.79, 8.5, 12.0, 1.01, 1.12, -0.2),
        ("G", "G1", "evolved", 5400.0, 3.60, 2.10, 6.0, 9.0, 1.05, 0.95, 0.0),
        ("G", "G0", "evolved", 5300.0, 3.50, 2.30, 5.5, 8.0, 1.04, 0.98, 0.1),
        ("K", "K2", "evolved", 4400.0, 3.30, 3.10, 4.8, 7.0, 1.06, 1.20, -0.1),
        ("K", "K3", "evolved", 4300.0, 3.20, 3.35, 4.5, 6.5, 1.07, 1.24, -0.2),
    ]
    for repetition in range(3):
        for index, row in enumerate(base_rows, start=1):
            spec_class, spec_subclass, stage, teff, logg, radius, parallax, poe, ruwe, bp_rp, mh = row
            source_id = repetition * 100 + index
            rows.append(
                {
                    "source_id": source_id,
                    "spec_class": spec_class,
                    "spec_subclass": spec_subclass,
                    "evolution_stage": stage,
                    "teff_gspphot": teff + repetition,
                    "logg_gspphot": logg,
                    "radius_gspphot": radius,
                    "parallax": parallax,
                    "parallax_over_error": poe,
                    "ruwe": ruwe,
                    "bp_rp": bp_rp,
                    "mh_gspphot": mh,
                }
            )
    return pd.DataFrame(rows)


def build_small_dataset_model_specs(
    feature_columns: tuple[str, ...],
) -> tuple[ModelSpec, ...]:
    # Для маленького synthetic датасета sklearn рекомендует lbfgs для MLP
    # вместо adam, чтобы модель сходилась устойчивее.
    return (
        ModelSpec(
            model_name="gmm_classifier",
            estimator=GMMClassifier(
                feature_columns=feature_columns,
                n_components=2,
                covariance_type="diag",
                reg_covar=1e-5,
                max_iter=200,
                random_state=42,
                scale_numeric=True,
                model_name="gmm_classifier",
            ),
        ),
        ModelSpec(
            model_name="hist_gradient_boosting",
            estimator=HGBClassifier(
                feature_columns=feature_columns,
                learning_rate=0.1,
                max_iter=200,
                max_leaf_nodes=31,
                min_samples_leaf=10,
                random_state=42,
                model_name="hist_gradient_boosting",
            ),
        ),
        ModelSpec(
            model_name="mlp_classifier",
            estimator=MLPClassifier(
                feature_columns=feature_columns,
                hidden_layer_sizes=(32, 16),
                solver="lbfgs",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=300,
                max_fun=15000,
                random_state=42,
                model_name="mlp_classifier",
            ),
        ),
    )


def test_run_benchmark_returns_metrics_and_cv_tables() -> None:
    # Проверяем полный benchmark-прогон на synthetic router dataset.
    benchmark_df = build_router_benchmark_frame()
    protocol = BenchmarkProtocol(
        split=SplitConfig(test_size=0.5, random_state=42),
        cv=CrossValidationConfig(n_splits=3, shuffle=True, random_state=42),
    )

    result = run_benchmark(
        benchmark_df,
        task=STAGE_CLASSIFICATION_TASK,
        model_specs=build_small_dataset_model_specs(STAGE_CLASSIFICATION_TASK.feature_columns),
        protocol=protocol,
    )

    assert result.task_name == "stage_classification"
    assert result.metrics_df.shape[0] == 6
    assert set(result.metrics_df["split_name"].tolist()) == {"train", "test"}
    assert result.cv_summary_df.shape[0] == 3
    assert "total_seconds" in result.cv_summary_df.columns
    assert set(result.target_distribution_df["split_name"].tolist()) == {
        "full",
        "train",
        "test",
    }
