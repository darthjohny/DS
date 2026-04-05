# Тестовый файл `test_posthoc_threshold_tuning.py` домена `posthoc`.
#
# Этот файл проверяет только:
# - проверку логики домена: post-hoc routing, gate и final decision policy;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `posthoc` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import pytest
from sklearn.linear_model import LogisticRegression

from exohost.posthoc.threshold_tuning import (
    ThresholdTuningConfig,
    build_tuned_threshold_classifier,
)


def test_threshold_tuning_config_rejects_invalid_threshold_count() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        ThresholdTuningConfig(thresholds=1)


def test_build_tuned_threshold_classifier_uses_config_values() -> None:
    estimator = LogisticRegression()
    tuner = build_tuned_threshold_classifier(
        estimator,
        config=ThresholdTuningConfig(
            scoring="balanced_accuracy",
            cv=3,
            response_method="predict_proba",
            thresholds=25,
        ),
    )

    assert tuner.scoring == "balanced_accuracy"
    assert tuner.cv == 3
    assert tuner.response_method == "predict_proba"
