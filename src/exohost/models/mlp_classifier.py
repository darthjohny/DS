# Файл `mlp_classifier.py` слоя `models`.
#
# Этот файл отвечает только за:
# - обертки моделей и inference-протоколы;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `models` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import (
    MLPClassifier as SklearnMLPClassifier,
)
from sklearn.pipeline import Pipeline

from exohost.features.preprocessing import (
    NumericPreprocessingConfig,
    build_numeric_preprocessor,
)

type MlpSolver = Literal["lbfgs", "sgd", "adam"]


class MLPClassifier(ClassifierMixin, BaseEstimator):
    # Небольшая MLP без лишней гибкости для первой волны benchmark.

    def __init__(
        self,
        feature_columns: tuple[str, ...],
        *,
        hidden_layer_sizes: tuple[int, ...] = (32, 16),
        solver: MlpSolver = "adam",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 300,
        max_fun: int = 15000,
        random_state: int = 42,
        model_name: str = "mlp_classifier",
    ) -> None:
        self.feature_columns = feature_columns
        self.hidden_layer_sizes = hidden_layer_sizes
        self.solver: MlpSolver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.max_fun = max_fun
        self.random_state = random_state
        self.model_name = model_name

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MLPClassifier:
        # Обучаем StandardScaler + MLPClassifier как единый pipeline.
        preprocessor = build_numeric_preprocessor(
            self.feature_columns,
            config=NumericPreprocessingConfig(scale_numeric=True),
        )
        classifier = SklearnMLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            max_fun=self.max_fun,
            random_state=self.random_state,
            early_stopping=False,
        )
        self._pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )
        self._pipeline.fit(X, y.astype(str))
        self.classes_ = np.asarray(classifier.classes_, dtype=object)
        return self

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.object_]:
        # Возвращаем label-предсказания обученного pipeline.
        return np.asarray(self._pipeline.predict(X), dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        # Возвращаем вероятности классов в порядке self.classes_.
        probabilities = self._pipeline.predict_proba(X)
        return np.asarray(probabilities, dtype=float)
