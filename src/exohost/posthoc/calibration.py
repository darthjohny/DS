# Файл `calibration.py` слоя `posthoc`.
#
# Этот файл отвечает только за:
# - post-hoc scoring, routing и final decision policy;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `posthoc` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV

CalibrationMethod = Literal["sigmoid", "isotonic"]
CalibrationEnsemble = bool | Literal["auto"]


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    # Конфиг post-hoc probability calibration по official scikit-learn semantics.
    method: CalibrationMethod = "sigmoid"
    cv: int = 5
    ensemble: CalibrationEnsemble = "auto"

    def __post_init__(self) -> None:
        if self.cv < 2:
            raise ValueError("CalibrationConfig.cv must be at least 2.")


DEFAULT_CALIBRATION_CONFIG = CalibrationConfig()


def build_calibrated_classifier(
    estimator: BaseEstimator,
    *,
    config: CalibrationConfig = DEFAULT_CALIBRATION_CONFIG,
) -> CalibratedClassifierCV:
    # Строим отдельный calibration wrapper и не смешиваем его с base estimator.
    calibrated_classifier = CalibratedClassifierCV(
        estimator=estimator,
        method=config.method,
        cv=config.cv,
    )
    # В runtime sklearn 1.8 принимает bool | "auto" через estimator params;
    # локальные IDE stubs могут сужать constructor signature.
    calibrated_classifier.set_params(ensemble=config.ensemble)
    return calibrated_classifier
