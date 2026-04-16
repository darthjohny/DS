# Файл `notebook_labels.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

BOOLEAN_LABELS: dict[bool, str] = {
    True: "Да",
    False: "Нет",
}

# Базовые короткие словари для состояний и итоговых категорий.
PRIORITY_LABELS: dict[str, str] = {
    "high": "Высокий",
    "medium": "Средний",
    "low": "Низкий",
}

QUALITY_STATE_LABELS: dict[str, str] = {
    "pass": "Допуск",
    "unknown": "Проверить",
    "reject": "Отклонено",
}

OOD_STATE_LABELS: dict[str, str] = {
    "in_domain": "Внутри домена",
    "candidate_ood": "Пограничный OOD",
    "ood": "OOD",
}

FINAL_DOMAIN_STATE_LABELS: dict[str, str] = {
    "id": "ID",
    "unknown": "Проверить",
    "ood": "OOD",
}

FINAL_REFINEMENT_STATE_LABELS: dict[str, str] = {
    "accepted": "Уточнение принято",
    "not_attempted": "Уточнение не запускалось",
    "rejected_to_unknown": "Уточнение увело в проверку",
}

PIPELINE_RUNTIME_LABELS: dict[str, str] = {
    "fit_seconds": "Обучение",
    "cv_seconds": "Кросс-валидация",
    "total_seconds": "Всего",
}

# Подписи метрик вынесены в один модуль, чтобы notebook не разъезжались
# по терминологии и не дублировали почти одинаковые строки вручную.
PIPELINE_STAGE_METRIC_LABELS: dict[str, str] = {
    "test_accuracy": "Точность на тесте (accuracy)",
    "test_balanced_accuracy": "Сбалансированная точность на тесте",
    "test_macro_f1": "Macro F1 на тесте",
    "test_roc_auc_ovr": "ROC AUC OvR на тесте",
    "cv_mean_accuracy": "Средняя точность CV (accuracy)",
    "cv_mean_balanced_accuracy": "Средняя сбалансированная точность CV",
    "cv_mean_macro_f1": "Средний Macro F1 CV",
    "accuracy": "Точность (accuracy)",
    "balanced_accuracy": "Сбалансированная точность",
    "macro_precision": "Macro precision",
    "macro_recall": "Macro recall",
    "macro_f1": "Macro F1",
    "roc_auc_ovr": "ROC AUC OvR",
}

SCORING_ALIGNMENT_METRIC_LABELS: dict[str, str] = {
    "top_n": "Размер верхнего среза",
    "target_class_share": "Доля целевых классов",
    "low_priority_class_share": "Доля низкоприоритетных классов",
    "dwarf_share": "Доля карликов",
    "evolved_share": "Доля эволюционировавших объектов",
    "high_priority_share": "Доля высокого приоритета",
    "medium_priority_share": "Доля среднего приоритета",
    "low_priority_share": "Доля низкого приоритета",
    "mean_host_similarity_score": "Средний host_similarity_score",
    "mean_observability_score": "Средний observability_score",
    "mean_observability_evidence_count": "Среднее число сигналов observability",
    "mean_priority_score": "Средний priority_score",
}

HOST_CALIBRATION_METRIC_LABELS: dict[str, str] = {
    "positive_rate": "Доля положительного класса",
    "mean_predicted_probability": "Средняя предсказанная вероятность",
    "brier_score": "Brier score",
    "log_loss": "Log loss",
    "roc_auc": "ROC AUC",
}

HOST_CALIBRATION_GROUP_LABELS: dict[str, str] = {
    "positive_rate": "Доля host-класса",
    "mean_host_similarity_score": "Средний host_similarity_score",
    "median_host_similarity_score": "Медианный host_similarity_score",
}

QUALITY_POLICY_LABELS: dict[str, str] = {
    "baseline": "Базовый вариант",
    "relaxed": "Смягченный вариант",
    "strict": "Строгий вариант",
}

QUALITY_POLICY_METRIC_LABELS: dict[str, str] = {
    "share_pass": "Доля допуска",
    "share_unknown": "Доля проверки",
    "share_reject": "Доля отклонения",
}

# Эти словари используются в исследовательских notebook, где важно показать
# человеку не кодовое имя группы, а физически понятную подпись на русском языке.
DOMAIN_NAME_LABELS: dict[str, str] = {
    "train_time": "Обучающий домен",
    "downstream_pass": "Рабочий домен",
}

REFERENCE_GROUP_LABELS: dict[str, str] = {
    "reference_b": "Чистый эталон `B`",
    "reference_evolved_b": "Эталон эволюционировавших `B`",
    "reference_o": "Чистый эталон `O`",
    "reference_evolved_o": "Эталон эволюционировавших `O`",
    "secure_o_tail": "Текущий надежный хвост `O`",
}

PRIORITY_THRESHOLD_VARIANT_LABELS: dict[str, str] = {
    "baseline": "Базовый вариант",
    "strict_high_080": "Строгий high ≥ 0.80",
    "strict_high_medium_085_055": "Строгие high/medium ≥ 0.85 / 0.55",
}

PRIORITY_THRESHOLD_SHARE_LABELS: dict[str, str] = {
    "share_high": "Доля высокого приоритета",
    "share_medium": "Доля среднего приоритета",
    "share_low": "Доля низкого приоритета",
}

PIPELINE_FINDING_LABELS: dict[str, str] = {
    "best_stage_by_macro_f1": "Лучший этап по Macro F1",
    "worst_stage_by_macro_f1": "Самый слабый этап по Macro F1",
    "slowest_stage_by_total_seconds": "Самый медленный этап",
}

PRIORITY_COMPONENT_LABELS: dict[str, str] = {
    "priority_score": "Итоговый priority_score",
    "host_similarity_score": "Сходство с host-профилем",
    "observability_score": "Оценка наблюдаемости",
    "class_priority_score": "Приоритет спектрального класса",
    "brightness_score": "Сигнал яркости",
    "distance_score": "Сигнал расстояния",
    "astrometry_score": "Астрометрический сигнал",
    "mean_priority_score": "Средний priority_score",
    "median_priority_score": "Медианный priority_score",
    "max_priority_score": "Максимальный priority_score",
    "min_priority_score": "Минимальный priority_score",
}

NUMERIC_SIGNAL_LABELS: dict[str, str] = {
    "host_similarity_score": "Сходство с host-профилем",
    "ruwe": "RUWE",
    "parallax_over_error": "SNR параллакса",
}


__all__ = [
    "BOOLEAN_LABELS",
    "DOMAIN_NAME_LABELS",
    "FINAL_DOMAIN_STATE_LABELS",
    "FINAL_REFINEMENT_STATE_LABELS",
    "HOST_CALIBRATION_GROUP_LABELS",
    "HOST_CALIBRATION_METRIC_LABELS",
    "NUMERIC_SIGNAL_LABELS",
    "OOD_STATE_LABELS",
    "PIPELINE_FINDING_LABELS",
    "PIPELINE_RUNTIME_LABELS",
    "PIPELINE_STAGE_METRIC_LABELS",
    "PRIORITY_LABELS",
    "PRIORITY_COMPONENT_LABELS",
    "PRIORITY_THRESHOLD_SHARE_LABELS",
    "PRIORITY_THRESHOLD_VARIANT_LABELS",
    "QUALITY_POLICY_LABELS",
    "QUALITY_POLICY_METRIC_LABELS",
    "QUALITY_STATE_LABELS",
    "REFERENCE_GROUP_LABELS",
    "SCORING_ALIGNMENT_METRIC_LABELS",
]
