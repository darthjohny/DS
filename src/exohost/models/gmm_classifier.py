# Файл `gmm_classifier.py` слоя `models`.
#
# Этот файл отвечает только за:
# - обертки моделей и inference-протоколы;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `models` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture

from exohost.features.preprocessing import (
    NumericPreprocessingConfig,
    build_numeric_preprocessor,
)

CovarianceType = Literal["full", "tied", "diag", "spherical"]


class GMMClassifier(ClassifierMixin, BaseEstimator):
    # Один GaussianMixture на каждый класс.
    # Итоговая вероятность считается по score_samples и class priors.

    def __init__(
        self,
        feature_columns: tuple[str, ...],
        *,
        n_components: int = 2,
        covariance_type: CovarianceType = "diag",
        reg_covar: float = 1e-6,
        max_iter: int = 200,
        random_state: int = 42,
        scale_numeric: bool = True,
        model_name: str = "gmm_classifier",
    ) -> None:
        self.feature_columns = feature_columns
        self.n_components = n_components
        self.covariance_type: CovarianceType = covariance_type
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_state = random_state
        self.scale_numeric = scale_numeric
        self.model_name = model_name

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GMMClassifier:
        # Обучаем отдельную смесь на каждом классе.
        self._preprocessor = build_numeric_preprocessor(
            self.feature_columns,
            config=NumericPreprocessingConfig(scale_numeric=self.scale_numeric),
        )
        transformed = self._preprocessor.fit_transform(X)
        transformed_array = np.asarray(transformed, dtype=float)
        label_series = y.astype(str).reset_index(drop=True)
        self.classes_ = np.array(sorted(label_series.unique().tolist()), dtype=object)

        models_by_class: dict[str, GaussianMixture] = {}
        class_log_prior_by_label: dict[str, float] = {}
        total_rows = int(label_series.shape[0])

        for class_label in self.classes_:
            class_mask = label_series == str(class_label)
            class_rows = transformed_array[class_mask.to_numpy()]
            n_rows = int(class_rows.shape[0])
            if n_rows == 0:
                raise ValueError(f"GMM received no rows for class {class_label}.")

            effective_components = max(1, min(int(self.n_components), n_rows))
            estimator = GaussianMixture(
                n_components=effective_components,
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            estimator.fit(class_rows)
            models_by_class[str(class_label)] = estimator
            class_log_prior_by_label[str(class_label)] = math.log(n_rows / total_rows)

        self._models_by_class = models_by_class
        self._class_log_prior_by_label = class_log_prior_by_label
        return self

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.object_]:
        # Выбираем класс с максимальной posterior-like вероятностью.
        probability_matrix = self.predict_proba(X)
        predicted_indices = np.argmax(probability_matrix, axis=1)
        return self.classes_[predicted_indices]

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        # Преобразуем log-likelihood по классам в нормированные вероятности.
        transformed = self._preprocessor.transform(X)
        transformed_array = np.asarray(transformed, dtype=float)
        log_probability_columns: list[np.ndarray] = []

        for class_label in self.classes_:
            label = str(class_label)
            estimator = self._models_by_class[label]
            log_score = estimator.score_samples(transformed_array)
            log_score = log_score + self._class_log_prior_by_label[label]
            log_probability_columns.append(log_score)

        log_probability_matrix = np.column_stack(log_probability_columns)
        row_max = np.max(log_probability_matrix, axis=1, keepdims=True)
        stabilized = np.exp(log_probability_matrix - row_max)
        normalization = stabilized.sum(axis=1, keepdims=True)
        return stabilized / normalization
