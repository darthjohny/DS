# Файл `probability_summary.py` слоя `posthoc`.
#
# Этот файл отвечает только за:
# - post-hoc scoring, routing и final decision policy;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `posthoc` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

import numpy as np
import pandas as pd


def build_probability_summary_frame(
    probability_frame: pd.DataFrame,
    *,
    prediction_column_name: str,
    confidence_column_name: str,
    margin_column_name: str,
) -> pd.DataFrame:
    # Собираем predicted label, max probability и margin top-2 классов.
    if probability_frame.empty:
        # Пустой probability-кадр не должен ломать downstream-слой.
        # Возвращаем ту же схему, которую затем ожидают scoring и routing helper.
        return pd.DataFrame(
            {
                prediction_column_name: pd.Series(dtype="string"),
                confidence_column_name: pd.Series(dtype="float64"),
                margin_column_name: pd.Series(dtype="float64"),
            },
            index=probability_frame.index,
        )

    probability_matrix = probability_frame.to_numpy(dtype=float)
    # Метку берем через `idxmax`, а числовые summary считаем уже по матрице,
    # чтобы не тащить лишние циклы по строкам pandas.
    predicted_labels = probability_frame.idxmax(axis=1).astype("string")
    confidence_series = pd.Series(
        probability_matrix.max(axis=1),
        index=probability_frame.index,
        dtype="float64",
    )
    margin_series = pd.Series(
        _compute_probability_margin(probability_matrix),
        index=probability_frame.index,
        dtype="float64",
    )
    return pd.DataFrame(
        {
            prediction_column_name: predicted_labels,
            confidence_column_name: confidence_series,
            margin_column_name: margin_series,
        },
        index=probability_frame.index,
    )


def _compute_probability_margin(probability_matrix: np.ndarray) -> np.ndarray:
    if probability_matrix.ndim != 2:
        raise ValueError("Probability matrix must be 2-dimensional.")
    if probability_matrix.shape[1] == 0:
        raise ValueError("Probability matrix must contain at least one class column.")

    # Margin нужен как простой индикатор уверенности: разница между лучшим
    # и вторым по вероятности классом лучше показывает пограничные случаи.
    sorted_probabilities = np.sort(probability_matrix, axis=1)
    max_probability = sorted_probabilities[:, -1]
    if probability_matrix.shape[1] == 1:
        second_probability = np.zeros(probability_matrix.shape[0], dtype=float)
    else:
        second_probability = sorted_probabilities[:, -2]
    return max_probability - second_probability
