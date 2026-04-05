# Тестовый файл `test_gmm_classifier.py` домена `models`.
#
# Этот файл проверяет только:
# - проверку логики домена: обертки моделей и inference-контракты;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `models` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import numpy as np
import pandas as pd

from exohost.models.gmm_classifier import GMMClassifier


def build_multiclass_frame() -> tuple[pd.DataFrame, pd.Series]:
    # Небольшой synthetic dataset для проверки multiclass поведения.
    frame = pd.DataFrame(
        {
            "f1": [-2.0, -1.8, -2.2, 0.0, 0.2, -0.1, 2.0, 2.2, 1.8],
            "f2": [-1.9, -2.1, -1.7, 0.2, 0.0, -0.2, 2.1, 1.9, 2.2],
        }
    )
    labels = pd.Series(
        ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        dtype="string",
    )
    return frame, labels


def test_gmm_classifier_fits_and_predicts_probabilities() -> None:
    # Проверяем базовый fit/predict/predict_proba контракт.
    frame, labels = build_multiclass_frame()
    model = GMMClassifier(
        feature_columns=("f1", "f2"),
        n_components=1,
        random_state=42,
        model_name="gmm_classifier",
    )

    model.fit(frame, labels)
    probabilities = model.predict_proba(frame)
    predictions = model.predict(frame)

    assert probabilities.shape == (9, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(predictions.tolist()) <= {"A", "B", "C"}
