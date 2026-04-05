# Файл `split.py` слоя `evaluation`.
#
# Этот файл отвечает только за:
# - метрики, split-логику и benchmark-task contracts;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `evaluation` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)

from exohost.evaluation.protocol import CrossValidationConfig, SplitConfig


@dataclass(slots=True)
class DatasetSplit:
    # Полный датасет и его train/test части.
    full_df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def build_stratify_labels(
    df: pd.DataFrame,
    stratify_columns: tuple[str, ...],
) -> pd.Series:
    # Собираем строковые метки стратификации по заданным колонкам.
    if not stratify_columns:
        raise ValueError("stratify_columns must not be empty.")

    missing_columns = [name for name in stratify_columns if name not in df.columns]
    if missing_columns:
        missing_columns_sql = ", ".join(missing_columns)
        raise ValueError(f"Missing stratify columns: {missing_columns_sql}")

    label_frame = df.loc[:, list(stratify_columns)].astype(str)
    return label_frame.agg("|".join, axis=1)


def validate_split_inputs(
    df: pd.DataFrame,
    *,
    split_config: SplitConfig,
    stratify_columns: tuple[str, ...],
) -> pd.Series:
    # Проверяем, что данных хватает для стратифицированного split.
    if df.empty:
        raise ValueError("Cannot split an empty dataset.")

    stratify_labels = build_stratify_labels(df, stratify_columns)
    label_counts = stratify_labels.value_counts()
    broken_labels = [
        str(label)
        for label, row_count in label_counts.items()
        if int(row_count) < 2
    ]
    if broken_labels:
        broken_labels_sql = ", ".join(broken_labels[:5])
        raise ValueError(
            "Each stratify label must contain at least two rows. "
            f"Broken labels: {broken_labels_sql}"
        )

    n_rows = int(df.shape[0])
    n_labels = int(len(label_counts))
    n_test_rows = math.ceil(n_rows * split_config.test_size)
    if n_test_rows < n_labels:
        raise ValueError(
            "Test split is too small for the chosen stratify labels. "
            f"Need at least {n_labels} rows in test, got {n_test_rows}."
        )

    return stratify_labels


def split_dataset(
    df: pd.DataFrame,
    *,
    split_config: SplitConfig,
    stratify_columns: tuple[str, ...],
) -> DatasetSplit:
    # Выполняем детерминированный train/test split без leakage по индексам.
    prepared = df.reset_index(drop=True).copy()
    stratify_labels = validate_split_inputs(
        prepared,
        split_config=split_config,
        stratify_columns=stratify_columns,
    )

    train_index, test_index = train_test_split(
        prepared.index.to_numpy(),
        test_size=split_config.test_size,
        random_state=split_config.random_state,
        shuffle=True,
        stratify=stratify_labels.to_numpy(),
    )

    train_df = prepared.loc[train_index].reset_index(drop=True)
    test_df = prepared.loc[test_index].reset_index(drop=True)
    return DatasetSplit(
        full_df=prepared,
        train_df=train_df,
        test_df=test_df,
    )


def build_cv_splitter(cv_config: CrossValidationConfig) -> StratifiedKFold:
    # Возвращаем единый splitter для benchmark-контра и model search.
    return StratifiedKFold(
        n_splits=cv_config.n_splits,
        shuffle=cv_config.shuffle,
        random_state=cv_config.random_state,
    )
