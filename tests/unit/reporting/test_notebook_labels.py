# Тестовый файл `test_notebook_labels.py` домена `reporting`.
#
# Этот файл проверяет только:
# - проверку логики домена: review-helper, сводки и notebook display-слой;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `reporting` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.reporting.notebook_labels import (
    BOOLEAN_LABELS,
    FINAL_REFINEMENT_STATE_LABELS,
    HOST_CALIBRATION_GROUP_LABELS,
    PIPELINE_RUNTIME_LABELS,
    PIPELINE_STAGE_METRIC_LABELS,
    PRIORITY_COMPONENT_LABELS,
    PRIORITY_LABELS,
    QUALITY_STATE_LABELS,
)


def test_priority_labels_are_stable() -> None:
    assert PRIORITY_LABELS == {
        "high": "Высокий",
        "medium": "Средний",
        "low": "Низкий",
    }


def test_quality_state_labels_are_stable() -> None:
    assert QUALITY_STATE_LABELS["pass"] == "Допуск"
    assert QUALITY_STATE_LABELS["unknown"] == "Проверить"
    assert QUALITY_STATE_LABELS["reject"] == "Отклонено"


def test_pipeline_metric_labels_cover_key_metrics() -> None:
    assert PIPELINE_STAGE_METRIC_LABELS["test_macro_f1"] == "Macro F1 на тесте"
    assert PIPELINE_STAGE_METRIC_LABELS["cv_mean_balanced_accuracy"] == (
        "Средняя сбалансированная точность CV"
    )
    assert PIPELINE_STAGE_METRIC_LABELS["roc_auc_ovr"] == "ROC AUC OvR"


def test_runtime_and_boolean_labels_are_stable() -> None:
    assert PIPELINE_RUNTIME_LABELS["fit_seconds"] == "Обучение"
    assert PIPELINE_RUNTIME_LABELS["total_seconds"] == "Всего"
    assert BOOLEAN_LABELS[True] == "Да"
    assert BOOLEAN_LABELS[False] == "Нет"


def test_notebook_labels_cover_refinement_host_and_priority_metrics() -> None:
    assert FINAL_REFINEMENT_STATE_LABELS["accepted"] == "Уточнение принято"
    assert HOST_CALIBRATION_GROUP_LABELS["positive_rate"] == "Доля host-класса"
    assert PRIORITY_COMPONENT_LABELS["host_similarity_score"] == "Сходство с host-профилем"
