# Тестовый файл `test_evaluation_metrics.py` домена `evaluation`.
#
# Этот файл проверяет только:
# - проверку логики домена: метрики, split-логику и benchmark contracts;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `evaluation` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import math

import pandas as pd

from exohost.evaluation.metrics import (
    build_classification_metrics,
    format_metric_value,
    metrics_record_to_frame,
)


def test_build_classification_metrics_with_probabilities() -> None:
    # Проверяем сборку базовых multiclass-метрик.
    y_true = pd.Series(["G", "G", "K", "K"])
    y_pred = pd.Series(["G", "K", "K", "K"])
    y_proba = pd.DataFrame(
        [
            [0.8, 0.2],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.1, 0.9],
        ],
        columns=["G", "K"],
    )

    record = build_classification_metrics(
        y_true,
        y_pred,
        split_name="test",
        y_proba=y_proba,
    )

    assert record.split_name == "test"
    assert record.n_rows == 4
    assert record.n_classes == 2
    assert record.macro_f1 >= 0.0
    assert math.isnan(record.roc_auc_ovr) is False


def test_build_classification_metrics_without_probabilities_returns_nan_auc() -> None:
    # Если вероятностей нет, AUC оставляем как nan.
    y_true = pd.Series(["G", "K"])
    y_pred = pd.Series(["G", "K"])

    record = build_classification_metrics(
        y_true,
        y_pred,
        split_name="train",
    )

    assert math.isnan(record.roc_auc_ovr)


def test_metrics_record_to_frame_builds_single_row_frame() -> None:
    # Проверяем простой tabular output для reporting-слоя.
    y_true = pd.Series(["G", "K"])
    y_pred = pd.Series(["G", "K"])
    record = build_classification_metrics(y_true, y_pred, split_name="train")
    frame = metrics_record_to_frame(record)

    assert frame.shape == (1, 9)


def test_format_metric_value_formats_scalars() -> None:
    # Проверяем компактное форматирование метрик для CLI и markdown.
    assert format_metric_value(0.12345) == "0.1235"
    assert format_metric_value(None) == "-"
