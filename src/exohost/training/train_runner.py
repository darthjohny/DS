# Файл `train_runner.py` слоя `training`.
#
# Этот файл отвечает только за:
# - оркестрацию обучения и benchmark-прогонов;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `training` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd
from sklearn.base import BaseEstimator, clone

from exohost.evaluation.protocol import ClassificationTask
from exohost.models.protocol import ClassifierModel, ModelSpec


@dataclass(slots=True)
class TrainRunResult:
    # Полный результат обучения одной модели на одной задаче.
    task_name: str
    model_name: str
    target_column: str
    feature_columns: tuple[str, ...]
    n_rows: int
    class_labels: tuple[str, ...]
    estimator: BaseEstimator
    label_distribution_df: pd.DataFrame


def build_label_distribution_frame(
    df: pd.DataFrame,
    *,
    target_column: str,
) -> pd.DataFrame:
    # Собираем распределение таргета на полном train frame.
    label_counts = df.loc[:, target_column].astype(str).value_counts(dropna=False)
    total_rows = int(label_counts.sum())
    rows: list[dict[str, object]] = []

    for target_label, row_count in label_counts.items():
        rows.append(
            {
                "target_label": str(target_label),
                "n_rows": int(row_count),
                "share": float(row_count / total_rows),
            }
        )

    return pd.DataFrame.from_records(rows).sort_values(
        "target_label",
        ignore_index=True,
    )


def run_training(
    df: pd.DataFrame,
    *,
    task: ClassificationTask,
    model_spec: ModelSpec,
) -> TrainRunResult:
    # Обучаем одну выбранную модель на полном task-ready frame.
    estimator = cast(ClassifierModel, clone(model_spec.estimator))
    train_X = df.loc[:, list(task.feature_columns)]
    train_y = df.loc[:, task.target_column].astype(str)

    fitted_estimator = estimator.fit(train_X, train_y)
    class_labels = tuple(str(label) for label in fitted_estimator.classes_.tolist())
    label_distribution_df = build_label_distribution_frame(
        df,
        target_column=task.target_column,
    )

    return TrainRunResult(
        task_name=task.name,
        model_name=model_spec.model_name,
        target_column=task.target_column,
        feature_columns=task.feature_columns,
        n_rows=int(df.shape[0]),
        class_labels=class_labels,
        estimator=cast(BaseEstimator, fitted_estimator),
        label_distribution_df=label_distribution_df,
    )
