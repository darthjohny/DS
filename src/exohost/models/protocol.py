# Файл `protocol.py` слоя `models`.
#
# Этот файл отвечает только за:
# - обертки моделей и inference-протоколы;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `models` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator


class ClassifierModel(Protocol):
    # Минимальный интерфейс модели для benchmark runner.
    model_name: str
    classes_: npt.NDArray[np.object_]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ClassifierModel:
        ...

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.object_]:
        ...

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        ...


@dataclass(frozen=True, slots=True)
class ModelSpec:
    # Описание одной модели, которую можно запускать в benchmark-контуре.
    model_name: str
    estimator: BaseEstimator
