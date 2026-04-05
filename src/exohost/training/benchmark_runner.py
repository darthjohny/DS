# Файл `benchmark_runner.py` слоя `training`.
#
# Этот файл отвечает только за:
# - оркестрацию обучения и benchmark-прогонов;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `training` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate

from exohost.evaluation.metrics import (
    ClassificationMetricsRecord,
    build_classification_metrics,
)
from exohost.evaluation.protocol import (
    DEFAULT_BENCHMARK_PROTOCOL,
    BenchmarkProtocol,
    ClassificationTask,
)
from exohost.evaluation.split import DatasetSplit, build_cv_splitter, split_dataset
from exohost.models.protocol import ClassifierModel, ModelSpec

CV_SCORING: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
)


def build_prediction_series(
    values: npt.NDArray[np.object_],
    *,
    index: pd.Index,
) -> pd.Series:
    # Приводим предсказания модели к явному строковому Series для метрик.
    return pd.Series(values.astype(str).tolist(), index=index, dtype="string")


@dataclass(frozen=True, slots=True)
class CrossValidationSummary:
    # Сводка CV-метрик одной модели.
    model_name: str
    cv_folds: int
    mean_accuracy: float
    mean_balanced_accuracy: float
    mean_macro_f1: float
    fit_seconds: float
    cv_seconds: float
    total_seconds: float


@dataclass(slots=True)
class BenchmarkRunResult:
    # Полный результат benchmark-прогона по одной задаче.
    task_name: str
    split: DatasetSplit
    metrics_df: pd.DataFrame
    cv_summary_df: pd.DataFrame
    target_distribution_df: pd.DataFrame


def build_probability_frame(
    model: ClassifierModel,
    X: pd.DataFrame,
) -> pd.DataFrame | None:
    # Преобразуем вероятности модели в DataFrame с именами классов.
    predict_proba = getattr(model, "predict_proba", None)
    classes = getattr(model, "classes_", None)
    if predict_proba is None or classes is None:
        return None

    probability_matrix = predict_proba(X)
    return pd.DataFrame(probability_matrix, columns=[str(name) for name in classes])


def run_single_model(
    model_spec: ModelSpec,
    split: DatasetSplit,
    *,
    task: ClassificationTask,
    protocol: BenchmarkProtocol,
) -> tuple[list[dict[str, object]], CrossValidationSummary]:
    # Обучаем одну модель, считаем train/test метрики и CV summary.
    estimator = cast(ClassifierModel, clone(model_spec.estimator))

    train_X = split.train_df.loc[:, list(task.feature_columns)]
    train_y = split.train_df.loc[:, task.target_column].astype(str)
    test_X = split.test_df.loc[:, list(task.feature_columns)]
    test_y = split.test_df.loc[:, task.target_column].astype(str)

    fit_started_at = perf_counter()
    estimator.fit(train_X, train_y)
    fit_seconds = perf_counter() - fit_started_at

    train_prediction = build_prediction_series(
        estimator.predict(train_X),
        index=train_y.index,
    )
    test_prediction = build_prediction_series(
        estimator.predict(test_X),
        index=test_y.index,
    )
    train_probability = build_probability_frame(estimator, train_X)
    test_probability = build_probability_frame(estimator, test_X)

    train_metrics = build_classification_metrics(
        train_y,
        train_prediction,
        split_name="train",
        y_proba=train_probability,
    )
    test_metrics = build_classification_metrics(
        test_y,
        test_prediction,
        split_name="test",
        y_proba=test_probability,
    )

    cv_started_at = perf_counter()
    cv_result = cross_validate(
        clone(model_spec.estimator),
        train_X,
        train_y,
        cv=build_cv_splitter(protocol.cv),
        scoring=list(CV_SCORING),
        n_jobs=1,
    )
    cv_seconds = perf_counter() - cv_started_at
    cv_summary = CrossValidationSummary(
        model_name=model_spec.model_name,
        cv_folds=protocol.cv.n_splits,
        mean_accuracy=float(cv_result["test_accuracy"].mean()),
        mean_balanced_accuracy=float(cv_result["test_balanced_accuracy"].mean()),
        mean_macro_f1=float(cv_result["test_f1_macro"].mean()),
        fit_seconds=fit_seconds,
        cv_seconds=cv_seconds,
        total_seconds=fit_seconds + cv_seconds,
    )

    return [
        build_metrics_row(model_spec.model_name, train_metrics),
        build_metrics_row(model_spec.model_name, test_metrics),
    ], cv_summary


def build_metrics_row(
    model_name: str,
    metrics: ClassificationMetricsRecord,
) -> dict[str, object]:
    # Преобразуем dataclass метрик в плоскую таблицу benchmark-результатов.
    return {
        "model_name": model_name,
        "split_name": metrics.split_name,
        "n_rows": metrics.n_rows,
        "n_classes": metrics.n_classes,
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "macro_f1": metrics.macro_f1,
        "roc_auc_ovr": metrics.roc_auc_ovr,
    }


def build_cv_summary_row(summary: CrossValidationSummary) -> dict[str, object]:
    # Преобразуем CV summary в табличный вид.
    return {
        "model_name": summary.model_name,
        "cv_folds": summary.cv_folds,
        "mean_accuracy": summary.mean_accuracy,
        "mean_balanced_accuracy": summary.mean_balanced_accuracy,
        "mean_macro_f1": summary.mean_macro_f1,
        "fit_seconds": summary.fit_seconds,
        "cv_seconds": summary.cv_seconds,
        "total_seconds": summary.total_seconds,
    }


def build_target_distribution_rows(
    y_values: pd.Series,
    *,
    split_name: str,
) -> list[dict[str, object]]:
    # Собираем распределение целевой метки для одного split.
    value_counts = y_values.astype(str).value_counts(dropna=False)
    total_rows = int(value_counts.sum())
    rows: list[dict[str, object]] = []

    for target_label, row_count in value_counts.items():
        rows.append(
            {
                "split_name": split_name,
                "target_label": str(target_label),
                "n_rows": int(row_count),
                "share": float(row_count / total_rows),
            }
        )

    return rows


def build_target_distribution_frame(
    split: DatasetSplit,
    *,
    target_column: str,
) -> pd.DataFrame:
    # Собираем распределение таргета по full/train/test частям.
    rows: list[dict[str, object]] = []
    split_frames: tuple[tuple[str, pd.DataFrame], ...] = (
        ("full", split.full_df),
        ("train", split.train_df),
        ("test", split.test_df),
    )

    for split_name, frame in split_frames:
        rows.extend(
            build_target_distribution_rows(
                frame.loc[:, target_column].astype(str),
                split_name=split_name,
            )
        )

    return pd.DataFrame.from_records(rows).sort_values(
        ["split_name", "target_label"],
        ignore_index=True,
    )


def run_benchmark(
    df: pd.DataFrame,
    *,
    task: ClassificationTask,
    model_specs: tuple[ModelSpec, ...],
    protocol: BenchmarkProtocol = DEFAULT_BENCHMARK_PROTOCOL,
) -> BenchmarkRunResult:
    # Выполняем benchmark-прогон по одной classification-задаче.
    split = split_dataset(
        df,
        split_config=protocol.split,
        stratify_columns=task.stratify_columns,
    )

    metrics_rows: list[dict[str, object]] = []
    cv_rows: list[dict[str, object]] = []

    for model_spec in model_specs:
        model_metrics_rows, cv_summary = run_single_model(
            model_spec,
            split,
            task=task,
            protocol=protocol,
        )
        metrics_rows.extend(model_metrics_rows)
        cv_rows.append(build_cv_summary_row(cv_summary))

    metrics_df = pd.DataFrame.from_records(metrics_rows).sort_values(
        ["model_name", "split_name"],
        ignore_index=True,
    )
    cv_summary_df = pd.DataFrame.from_records(cv_rows).sort_values(
        "model_name",
        ignore_index=True,
    )
    target_distribution_df = build_target_distribution_frame(
        split,
        target_column=task.target_column,
    )
    return BenchmarkRunResult(
        task_name=task.name,
        split=split,
        metrics_df=metrics_df,
        cv_summary_df=cv_summary_df,
        target_distribution_df=target_distribution_df,
    )
