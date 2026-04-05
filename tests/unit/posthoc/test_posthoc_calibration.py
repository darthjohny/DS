# Тестовый файл `test_posthoc_calibration.py` домена `posthoc`.
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

from exohost.posthoc.calibration import (
    CalibrationConfig,
    build_calibrated_classifier,
)


def test_calibration_config_rejects_invalid_cv() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        CalibrationConfig(cv=1)


def test_build_calibrated_classifier_uses_config_values() -> None:
    estimator = LogisticRegression()
    calibrator = build_calibrated_classifier(
        estimator,
        config=CalibrationConfig(method="sigmoid", cv=3, ensemble=False),
    )
    calibrator_params = calibrator.get_params(deep=False)

    assert calibrator_params["method"] == "sigmoid"
    assert calibrator_params["cv"] == 3
    assert calibrator_params["ensemble"] is False
