# Файл `threshold_tuning.py` слоя `posthoc`.
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
from sklearn.model_selection import TunedThresholdClassifierCV

ThresholdScoring = Literal["balanced_accuracy"]
ThresholdResponseMethod = Literal["auto", "predict_proba", "decision_function"]


@dataclass(frozen=True, slots=True)
class ThresholdTuningConfig:
    # Конфиг official threshold-tuning stage для binary gate.
    scoring: ThresholdScoring = "balanced_accuracy"
    cv: int = 5
    response_method: ThresholdResponseMethod = "auto"
    thresholds: int = 100

    def __post_init__(self) -> None:
        if self.cv < 2:
            raise ValueError("ThresholdTuningConfig.cv must be at least 2.")
        if self.thresholds < 2:
            raise ValueError("ThresholdTuningConfig.thresholds must be at least 2.")


DEFAULT_THRESHOLD_TUNING_CONFIG = ThresholdTuningConfig()


def build_tuned_threshold_classifier(
    estimator: BaseEstimator,
    *,
    config: ThresholdTuningConfig = DEFAULT_THRESHOLD_TUNING_CONFIG,
) -> TunedThresholdClassifierCV:
    # Строим post-hoc threshold tuner отдельно от calibration и base fit.
    return TunedThresholdClassifierCV(
        estimator=estimator,
        scoring=config.scoring,
        cv=config.cv,
        response_method=config.response_method,
        thresholds=config.thresholds,
    )
