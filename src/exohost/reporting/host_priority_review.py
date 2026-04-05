# Файл `host_priority_review.py` слоя `reporting`.
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

from exohost.contracts.host_priority_feature_contracts import (
    HOST_PRIORITY_ALL_FEATURES,
    HOST_PRIORITY_CANONICAL_RADIUS_COLUMN,
    HOST_PRIORITY_CONTEXT_FEATURES,
    HOST_PRIORITY_CORE_FEATURES,
    HOST_PRIORITY_OBSERVABILITY_FEATURES,
)
from exohost.reporting.host_training_review import load_host_training_review_frame


def load_host_priority_review_frame(
    *,
    limit: int | None = None,
    dotenv_path: str = ".env",
) -> pd.DataFrame:
    # Загружаем current host source для review относительно clean priority contract.
    return load_host_training_review_frame(
        limit=limit,
        dotenv_path=dotenv_path,
    )


def build_host_priority_contract_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    # One-row summary по readiness current host source к новому clean contract.
    source_id = _require_series_column(df, "source_id")
    clean_core_ready_mask = _build_clean_core_ready_mask(df)
    canonical_radius_present = HOST_PRIORITY_CANONICAL_RADIUS_COLUMN in df.columns

    return pd.DataFrame(
        [
            {
                "n_rows": int(df.shape[0]),
                "n_unique_source_id": int(source_id.astype(str).nunique(dropna=False)),
                "n_supported_classes": _count_unique_string_values(df, "spec_class"),
                "n_supported_stages": _count_unique_string_values(df, "evolution_stage"),
                "n_core_features": int(len(HOST_PRIORITY_CORE_FEATURES)),
                "n_core_features_present": int(
                    sum(1 for name in HOST_PRIORITY_CORE_FEATURES if name in df.columns)
                ),
                "has_canonical_radius_column": bool(canonical_radius_present),
                "n_rows_with_canonical_radius": _count_non_missing_values(
                    df,
                    HOST_PRIORITY_CANONICAL_RADIUS_COLUMN,
                ),
                "share_rows_with_canonical_radius": _compute_non_missing_share(
                    df,
                    HOST_PRIORITY_CANONICAL_RADIUS_COLUMN,
                ),
                "n_rows_clean_core_ready": int(clean_core_ready_mask.sum()),
                "share_rows_clean_core_ready": float(clean_core_ready_mask.mean())
                if int(df.shape[0]) > 0
                else 0.0,
            }
        ]
    )


def build_host_priority_feature_coverage_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Подробная coverage-таблица по clean host/priority feature contract.
    rows: list[dict[str, object]] = []

    for feature_name in HOST_PRIORITY_ALL_FEATURES:
        rows.append(
            {
                "feature_name": feature_name,
                "feature_group": _resolve_feature_group(feature_name),
                "is_required": bool(feature_name in HOST_PRIORITY_CORE_FEATURES),
                "column_present": bool(feature_name in df.columns),
                "n_non_missing": _count_non_missing_values(df, feature_name),
                "share_non_missing": _compute_non_missing_share(df, feature_name),
            }
        )

    return pd.DataFrame.from_records(rows)


def build_host_priority_missing_core_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Top missing-core breakdown по current host source.
    rows: list[dict[str, object]] = []
    total_rows = int(df.shape[0])

    for feature_name in HOST_PRIORITY_CORE_FEATURES:
        if feature_name not in df.columns:
            rows.append(
                {
                    "feature_name": feature_name,
                    "n_missing_rows": total_rows,
                    "share_missing_rows": 1.0 if total_rows > 0 else 0.0,
                    "column_present": False,
                }
            )
            continue

        series = _require_series_column(df, feature_name)
        n_missing_rows = int(series.isna().sum())
        rows.append(
            {
                "feature_name": feature_name,
                "n_missing_rows": n_missing_rows,
                "share_missing_rows": float(n_missing_rows / total_rows)
                if total_rows > 0
                else 0.0,
                "column_present": True,
            }
        )

    return pd.DataFrame.from_records(rows).sort_values(
        ["n_missing_rows", "feature_name"],
        ascending=[False, True],
        kind="mergesort",
        ignore_index=True,
    )


def _build_clean_core_ready_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    mask = pd.Series(True, index=df.index, dtype=bool)
    for feature_name in HOST_PRIORITY_CORE_FEATURES:
        if feature_name not in df.columns:
            return pd.Series(False, index=df.index, dtype=bool)
        mask &= _require_series_column(df, feature_name).notna()
    return mask


def _resolve_feature_group(feature_name: str) -> str:
    if feature_name in HOST_PRIORITY_CORE_FEATURES:
        return "core"
    if feature_name in HOST_PRIORITY_OBSERVABILITY_FEATURES:
        return "observability"
    if feature_name in HOST_PRIORITY_CONTEXT_FEATURES:
        return "context"
    return "optional_physical"


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


def _compute_non_missing_share(df: pd.DataFrame, column_name: str) -> float:
    if int(df.shape[0]) == 0 or column_name not in df.columns:
        return 0.0
    column = _require_series_column(df, column_name)
    return float(column.notna().mean())


def _require_series_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    column = df.loc[:, column_name]
    if not isinstance(column, pd.Series):
        raise TypeError(f"{column_name} must resolve to a pandas Series.")
    return column
