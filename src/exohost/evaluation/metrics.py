# Файл `metrics.py` слоя `evaluation`.
#
# Этот файл отвечает только за:
# - метрики, split-логику и benchmark-task contracts;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `evaluation` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, cast

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass(frozen=True, slots=True)
class ClassificationMetricsRecord:
    # Сводка метрик для одного прогона модели на одном split.
    split_name: str
    n_rows: int
    n_classes: int
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    roc_auc_ovr: float


def safe_roc_auc_ovr(
    y_true: pd.Series,
    y_proba: pd.DataFrame | None,
) -> float:
    # Считаем multiclass ROC-AUC OVR только если вероятности реально доступны.
    if y_proba is None:
        return float("nan")
    if y_true.nunique(dropna=False) < 2:
        return float("nan")

    try:
        probability_frame = y_proba.copy()
        probability_frame.columns = probability_frame.columns.astype(str)
        label_series = y_true.astype(str)

        if probability_frame.shape[1] == 2:
            positive_label = str(probability_frame.columns[1])
            binary_target = (label_series == positive_label).astype(int)
            return float(roc_auc_score(binary_target, probability_frame[positive_label]))

        return float(
            roc_auc_score(
                label_series,
                probability_frame,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return float("nan")


def build_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    split_name: str,
    y_proba: pd.DataFrame | None = None,
) -> ClassificationMetricsRecord:
    # Собираем компактный и воспроизводимый набор benchmark-метрик.
    zero_division_fallback = cast(Any, 0)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=zero_division_fallback,
    )

    return ClassificationMetricsRecord(
        split_name=split_name,
        n_rows=int(y_true.shape[0]),
        n_classes=int(y_true.nunique()),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_precision=float(precision),
        macro_recall=float(recall),
        macro_f1=float(f1),
        roc_auc_ovr=safe_roc_auc_ovr(y_true, y_proba),
    )


def metrics_record_to_frame(record: ClassificationMetricsRecord) -> pd.DataFrame:
    # Преобразуем запись метрик в однотабличный вид для reporting-слоя.
    return pd.DataFrame.from_records([asdict(record)])


def format_metric_value(value: Any) -> str:
    # Преобразуем скаляр метрики в компактный текст для markdown и CLI.
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)
