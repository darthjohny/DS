# Тестовый файл `model_pipeline_review_testkit.py` домена `reporting`.
#
# Этот файл проверяет только:
# - проверку логики домена: review-helper, сводки и notebook display-слой;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `reporting` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import json
from numbers import Integral, Real
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator

from exohost.training.train_runner import TrainRunResult


class DummyEstimator(BaseEstimator):
    pass


def require_int_scalar(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Expected integer-like scalar, got {type(value).__name__}.")
    return int(value)


def require_float_scalar(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"Expected real scalar, got {type(value).__name__}.")
    return float(value)


def write_benchmark_run(
    run_dir: Path,
    *,
    task_name: str,
    accuracy: float,
    macro_f1: float,
    balanced_accuracy: float,
) -> None:
    # Сохраняем минимальный набор CSV/JSON артефактов, который ожидает review-helper.
    # Здесь важна не полнота эксперимента, а стабильная схема файлов и основных полей.
    run_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(
        [
            {
                "model_name": "hist_gradient_boosting",
                "split_name": "test",
                "n_rows": 100,
                "n_classes": 3,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "macro_precision": 0.8,
                "macro_recall": balanced_accuracy,
                "macro_f1": macro_f1,
                "roc_auc_ovr": 0.91,
            },
            {
                "model_name": "hist_gradient_boosting",
                "split_name": "train",
                "n_rows": 200,
                "n_classes": 3,
                "accuracy": 0.99,
                "balanced_accuracy": 0.99,
                "macro_precision": 0.99,
                "macro_recall": 0.99,
                "macro_f1": 0.99,
                "roc_auc_ovr": 1.0,
            },
        ]
    ).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "model_name": "hist_gradient_boosting",
                "cv_folds": 10,
                "mean_accuracy": accuracy - 0.01,
                "mean_balanced_accuracy": balanced_accuracy - 0.01,
                "mean_macro_f1": macro_f1 - 0.01,
                "fit_seconds": 1.5,
                "cv_seconds": 12.0,
                "total_seconds": 13.5,
            }
        ]
    ).to_csv(run_dir / "cv_summary.csv", index=False)
    pd.DataFrame(
        [
            {"split_name": "full", "target_label": "A", "n_rows": 60, "share": 0.6},
            {"split_name": "full", "target_label": "B", "n_rows": 40, "share": 0.4},
            {"split_name": "test", "target_label": "A", "n_rows": 30, "share": 0.6},
        ]
    ).to_csv(run_dir / "target_distribution.csv", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "task_name": task_name,
                "created_at_utc": "2026-03-28T19:00:00+00:00",
                "n_rows_full": 100,
                "n_rows_train": 70,
                "n_rows_test": 30,
            }
        ),
        encoding="utf-8",
    )


def build_train_result() -> TrainRunResult:
    # TrainRunResult нужен для artifact review и должен напоминать реальный run,
    # но оставаться маленьким и полностью контролируемым внутри теста.
    label_distribution_df = pd.DataFrame(
        [
            {"target_label": "A", "n_rows": 60, "share": 0.6},
            {"target_label": "B", "n_rows": 40, "share": 0.4},
        ]
    )
    return TrainRunResult(
        task_name="gaia_id_coarse_classification",
        target_column="spec_class",
        model_name="hist_gradient_boosting",
        estimator=DummyEstimator(),
        label_distribution_df=label_distribution_df,
        n_rows=100,
        feature_columns=("teff_gspphot", "bp_rp"),
        class_labels=("A", "B"),
    )
