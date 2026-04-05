# Файл `host_training_review.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

import pandas as pd

from exohost.datasets.load_host_training_dataset import load_host_training_dataset
from exohost.db.engine import make_read_only_engine
from exohost.features.training_frame import prepare_host_training_frame


def load_host_training_review_frame(
    *,
    limit: int | None = None,
    dotenv_path: str = ".env",
) -> pd.DataFrame:
    # Загружаем и нормализуем current host source для observability-notebook.
    engine = make_read_only_engine(dotenv_path=dotenv_path)
    raw_df = load_host_training_dataset(engine, limit=limit)
    return prepare_host_training_frame(raw_df)


def build_host_training_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Компактная one-row summary по current host source.
    source_id = _require_series_column(df, "source_id")
    return pd.DataFrame(
        [
            {
                "n_rows": int(df.shape[0]),
                "n_unique_source_id": int(source_id.astype(str).nunique(dropna=False)),
                "n_unique_hostname": _count_unique_string_values(df, "hostname"),
                "n_spec_subclass_rows": _count_non_missing_values(df, "spec_subclass"),
                "n_supported_classes": _count_unique_string_values(df, "spec_class"),
                "n_supported_stages": _count_unique_string_values(df, "evolution_stage"),
            }
        ]
    )


def build_host_class_distribution_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Распределение spec_class в host source.
    return _build_distribution_frame(df, column_name="spec_class", label_name="spec_class")


def build_host_stage_distribution_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Распределение evolution_stage в host source.
    return _build_distribution_frame(
        df,
        column_name="evolution_stage",
        label_name="evolution_stage",
    )


def build_host_class_stage_crosstab_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Crosstab по host source для быстрого обзора class/stage balance.
    spec_class = _require_series_column(df, "spec_class")
    evolution_stage = _require_series_column(df, "evolution_stage")
    return pd.crosstab(
        spec_class,
        evolution_stage,
        dropna=False,
    )


def _build_distribution_frame(
    df: pd.DataFrame,
    *,
    column_name: str,
    label_name: str,
) -> pd.DataFrame:
    if column_name not in df.columns:
        return pd.DataFrame(columns=[label_name, "n_rows", "share"])

    counts = df.loc[:, column_name].astype(str).value_counts(dropna=False)
    total_rows = int(counts.sum())
    rows = [
        {
            label_name: str(label_value),
            "n_rows": int(n_rows),
            "share": float(n_rows / total_rows),
        }
        for label_value, n_rows in counts.items()
    ]
    return pd.DataFrame.from_records(rows)


def _count_unique_string_values(df: pd.DataFrame, column_name: str) -> int:
    if column_name not in df.columns:
        return 0
    column = _require_series_column(df, column_name)
    return int(column.dropna().astype(str).nunique(dropna=False))


def _count_non_missing_values(df: pd.DataFrame, column_name: str) -> int:
    if column_name not in df.columns:
        return 0
    column = _require_series_column(df, column_name)
    return int(column.notna().sum())


def _require_series_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    column = df.loc[:, column_name]
    if not isinstance(column, pd.Series):
        raise TypeError(f"{column_name} must resolve to a pandas Series.")
    return column
