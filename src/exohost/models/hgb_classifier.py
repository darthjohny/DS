# Файл `hgb_classifier.py` слоя `models`.
#
# Этот файл отвечает только за:
# - обертки моделей и inference-протоколы;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `models` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from exohost.features.preprocessing import (
    NumericPreprocessingConfig,
    build_numeric_preprocessor,
)


class HGBClassifier(ClassifierMixin, BaseEstimator):
    # Компактный нелинейный baseline для табличных признаков.

    def __init__(
        self,
        feature_columns: tuple[str, ...],
        *,
        learning_rate: float = 0.1,
        max_iter: int = 200,
        max_depth: int | None = None,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        random_state: int = 42,
        model_name: str = "hist_gradient_boosting",
    ) -> None:
        self.feature_columns = feature_columns
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model_name = model_name

    def fit(self, X: pd.DataFrame, y: pd.Series) -> HGBClassifier:
        # Обучаем pipeline с имputation, без лишней стандартизации.
        preprocessor = build_numeric_preprocessor(
            self.feature_columns,
            config=NumericPreprocessingConfig(scale_numeric=False),
        )
        classifier = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
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
        # Возвращаем label-предсказания из sklearn pipeline.
        return np.asarray(self._pipeline.predict(X), dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        # Возвращаем вероятности классов в порядке self.classes_.
        probabilities = self._pipeline.predict_proba(X)
        return np.asarray(probabilities, dtype=float)
