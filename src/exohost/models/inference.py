# Файл `inference.py` слоя `models`.
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
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from exohost.models.protocol import ClassifierModel


@dataclass(slots=True)
class ModelScoringResult:
    # Полный результат применения одной модели к входному DataFrame.
    task_name: str
    target_column: str
    model_name: str
    n_rows: int
    scored_df: pd.DataFrame


def build_prediction_series(
    values: npt.NDArray[np.object_],
    *,
    index: pd.Index,
) -> pd.Series:
    # Приводим предсказания к явному строковому Series.
    return pd.Series(values.astype(str).tolist(), index=index, dtype="string")


def require_feature_columns(
    df: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
) -> None:
    # Проверяем, что входной frame содержит все признаки модели.
    missing_columns = [name for name in feature_columns if name not in df.columns]
    if missing_columns:
        missing_sql = ", ".join(missing_columns)
        raise ValueError(f"Scoring frame is missing required feature columns: {missing_sql}")


def build_probability_frame(
    model: ClassifierModel,
    X: pd.DataFrame,
) -> pd.DataFrame | None:
    # Преобразуем predict_proba-выход в именованный DataFrame.
    predict_proba = getattr(model, "predict_proba", None)
    classes = getattr(model, "classes_", None)
    if predict_proba is None or classes is None:
        return None

    probability_matrix = predict_proba(X)
    return pd.DataFrame(
        probability_matrix,
        columns=[str(name) for name in classes],
        index=X.index,
    )


def build_scored_frame(
    df: pd.DataFrame,
    *,
    feature_frame: pd.DataFrame,
    model: ClassifierModel,
    target_column: str,
    host_score_column: str,
) -> pd.DataFrame:
    # Добавляем к входному frame предсказания и вероятности модели.
    prediction_column = f"predicted_{target_column}"
    confidence_column = f"{prediction_column}_confidence"

    scored_df = df.copy()
    prediction_series = build_prediction_series(
        model.predict(feature_frame),
        index=df.index,
    )
    scored_df[prediction_column] = prediction_series

    probability_frame = build_probability_frame(model, feature_frame)
    if probability_frame is None:
        scored_df[confidence_column] = float("nan")
        return scored_df

    probability_columns = {
        column_name: f"probability__{column_name}"
        for column_name in probability_frame.columns.astype(str).tolist()
    }
    renamed_probability_frame = probability_frame.rename(columns=probability_columns)
    scored_df = scored_df.join(renamed_probability_frame)
    scored_df[confidence_column] = probability_frame.max(axis=1).astype(float)

    if target_column == "host_label" and "host" in probability_frame.columns:
        scored_df[host_score_column] = probability_frame["host"].astype(float)

    return scored_df


def score_with_model(
    df: pd.DataFrame,
    *,
    estimator: object,
    task_name: str,
    target_column: str,
    feature_columns: tuple[str, ...],
    model_name: str,
    host_score_column: str = "host_similarity_score",
) -> ModelScoringResult:
    # Применяем сохраненную модель к новому DataFrame.
    require_feature_columns(df, feature_columns=feature_columns)
    model = cast(ClassifierModel, estimator)
    feature_frame = df.loc[:, list(feature_columns)].copy()
    scored_df = build_scored_frame(
        df,
        feature_frame=feature_frame,
        model=model,
        target_column=target_column,
        host_score_column=host_score_column,
    )

    return ModelScoringResult(
        task_name=task_name,
        target_column=target_column,
        model_name=model_name,
        n_rows=int(feature_frame.shape[0]),
        scored_df=scored_df,
    )
