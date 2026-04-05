# Тестовый файл `test_model_scoring.py` домена `evaluation`.
#
# Этот файл проверяет только:
# - проверку логики домена: метрики, split-логику и benchmark contracts;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `evaluation` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from numbers import Real

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from exohost.evaluation.protocol import SPECTRAL_CLASS_CLASSIFICATION_TASK
from exohost.models.inference import score_with_model
from exohost.training.train_runner import TrainRunResult


def get_float_cell(df: pd.DataFrame, row_index: int, column_name: str) -> float:
    # Читаем numeric-ячейку через явную runtime-проверку типа.
    value = df.loc[row_index, column_name]
    if isinstance(value, Real):
        return float(value)
    raise AssertionError(f"Expected numeric value in column {column_name}.")


def get_str_cell(df: pd.DataFrame, row_index: int, column_name: str) -> str:
    # Читаем строковую ячейку через явную runtime-проверку типа.
    value = df.loc[row_index, column_name]
    if isinstance(value, str):
        return value
    raise AssertionError(f"Expected string value in column {column_name}.")


class DummyHostClassifier:
    # Минимальная модель с host/field вероятностями для проверки host-like колонки.

    def __init__(self) -> None:
        self.model_name = "dummy_host_classifier"
        self.classes_ = np.asarray(["field", "host"], dtype=object)

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.object_]:
        # Всегда возвращаем host, чтобы не усложнять synthetic-тест.
        return np.asarray(["host"] * int(X.shape[0]), dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        # Возвращаем фиксированную уверенность по каждому объекту.
        row = np.asarray([[0.2, 0.8]], dtype=float)
        return np.repeat(row, repeats=int(X.shape[0]), axis=0)


def test_score_with_model_adds_predictions_and_probabilities(
    small_model_scoring_frame: pd.DataFrame,
    small_spectral_class_train_result: TrainRunResult,
) -> None:
    # Проверяем, что scoring добавляет предсказания, confidence и probability-колонки.
    train_result = small_spectral_class_train_result

    scoring_result = score_with_model(
        small_model_scoring_frame,
        estimator=train_result.estimator,
        task_name=train_result.task_name,
        target_column=train_result.target_column,
        feature_columns=train_result.feature_columns,
        model_name=train_result.model_name,
    )

    assert scoring_result.task_name == "spectral_class_classification"
    assert scoring_result.n_rows == 2
    assert get_str_cell(scoring_result.scored_df, 0, "predicted_spec_class") in {"G", "K"}
    assert get_float_cell(scoring_result.scored_df, 0, "predicted_spec_class_confidence") >= 0.0
    assert "probability__G" in scoring_result.scored_df.columns
    assert "probability__K" in scoring_result.scored_df.columns


def test_score_with_model_rejects_missing_feature_columns(
    small_model_scoring_frame: pd.DataFrame,
    small_spectral_class_train_result: TrainRunResult,
) -> None:
    # Скоринг должен падать сразу, если во входном frame нет нужного признака.
    train_result = small_spectral_class_train_result
    broken_frame = small_model_scoring_frame.drop(columns="mh_gspphot")

    with pytest.raises(ValueError, match="missing required feature columns"):
        score_with_model(
            broken_frame,
            estimator=train_result.estimator,
            task_name=train_result.task_name,
            target_column=train_result.target_column,
            feature_columns=train_result.feature_columns,
            model_name=train_result.model_name,
        )


def test_score_with_model_maps_host_probability_to_host_similarity_score() -> None:
    # Для host-модели автоматически пробрасываем вероятность host в ranking-совместимую колонку.
    scoring_result = score_with_model(
        pd.DataFrame(
            [
                {
                    "source_id": "10",
                    "teff_gspphot": 5820.0,
                    "logg_gspphot": 4.4,
                    "radius_gspphot": 1.0,
                    "parallax": 14.5,
                    "parallax_over_error": 19.0,
                    "ruwe": 1.01,
                    "bp_rp": 0.76,
                    "mh_gspphot": 0.05,
                }
            ]
        ),
        estimator=DummyHostClassifier(),
        task_name="host_field_classification",
        target_column="host_label",
        feature_columns=SPECTRAL_CLASS_CLASSIFICATION_TASK.feature_columns,
        model_name="dummy_host_classifier",
    )

    assert get_str_cell(scoring_result.scored_df, 0, "predicted_host_label") == "host"
    assert get_float_cell(scoring_result.scored_df, 0, "host_similarity_score") == 0.8
