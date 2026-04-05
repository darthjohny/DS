# Файл `model_pipeline_review_summary.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

from collections.abc import Hashable, Mapping
from pathlib import Path

import pandas as pd

from exohost.reporting.model_pipeline_review_bundle import load_benchmark_review_bundle
from exohost.reporting.model_pipeline_review_contracts import BenchmarkReviewBundle


def build_benchmark_summary_frame(bundle: BenchmarkReviewBundle) -> pd.DataFrame:
    # Собираем компактный stage-level summary по test и CV метрикам.
    best_test_row = _select_best_metrics_row(bundle.metrics_df)
    best_cv_row = _select_best_cv_row(bundle.cv_summary_df)
    best_test_payload = _normalize_series_record(best_test_row) if best_test_row is not None else None
    best_cv_payload = _normalize_series_record(best_cv_row) if best_cv_row is not None else None

    row: dict[str, object] = {
        "run_dir": bundle.run_dir.name,
        "task_name": bundle.metadata.get("task_name", "unknown"),
        "created_at_utc": bundle.metadata.get("created_at_utc", "unknown"),
        "n_rows_full": bundle.metadata.get(
            "n_rows_full",
            int(bundle.target_distribution_df.shape[0]),
        ),
        "n_rows_train": bundle.metadata.get("n_rows_train", pd.NA),
        "n_rows_test": bundle.metadata.get("n_rows_test", pd.NA),
        "best_test_model": pd.NA,
        "test_accuracy": pd.NA,
        "test_balanced_accuracy": pd.NA,
        "test_macro_f1": pd.NA,
        "test_roc_auc_ovr": pd.NA,
        "best_cv_model": pd.NA,
        "cv_folds": pd.NA,
        "cv_mean_accuracy": pd.NA,
        "cv_mean_balanced_accuracy": pd.NA,
        "cv_mean_macro_f1": pd.NA,
        "fit_seconds": pd.NA,
        "cv_seconds": pd.NA,
        "total_seconds": pd.NA,
    }

    if best_test_payload is not None:
        test_roc_auc_ovr = _to_optional_float(best_test_payload.get("roc_auc_ovr"))
        row.update(
            {
                "best_test_model": str(best_test_payload["model_name"]),
                "test_accuracy": _require_float(best_test_payload["accuracy"]),
                "test_balanced_accuracy": _require_float(
                    best_test_payload["balanced_accuracy"]
                ),
                "test_macro_f1": _require_float(best_test_payload["macro_f1"]),
                "test_roc_auc_ovr": test_roc_auc_ovr if test_roc_auc_ovr is not None else pd.NA,
            }
        )

    if best_cv_payload is not None:
        row.update(
            {
                "best_cv_model": str(best_cv_payload["model_name"]),
                "cv_folds": _require_int(best_cv_payload["cv_folds"]),
                "cv_mean_accuracy": _require_float(best_cv_payload["mean_accuracy"]),
                "cv_mean_balanced_accuracy": _require_float(
                    best_cv_payload["mean_balanced_accuracy"]
                ),
                "cv_mean_macro_f1": _require_float(best_cv_payload["mean_macro_f1"]),
                "fit_seconds": _require_float(best_cv_payload["fit_seconds"]),
                "cv_seconds": _require_float(best_cv_payload["cv_seconds"]),
                "total_seconds": _require_float(best_cv_payload["total_seconds"]),
            }
        )

    return pd.DataFrame([row])


def build_pipeline_stage_overview_frame(
    stage_run_dirs: Mapping[str, str | Path],
) -> pd.DataFrame:
    # Строим единую таблицу по нескольким pipeline stages.
    rows: list[dict[str, object]] = []
    for stage_name, run_dir in stage_run_dirs.items():
        bundle = load_benchmark_review_bundle(run_dir)
        stage_row = _normalize_series_record(build_benchmark_summary_frame(bundle).iloc[0])
        stage_row["stage_name"] = stage_name
        rows.append(stage_row)

    overview_df = pd.DataFrame.from_records(rows)
    if overview_df.empty:
        return overview_df

    return overview_df.loc[
        :,
        [
            "stage_name",
            "task_name",
            "run_dir",
            "created_at_utc",
            "n_rows_full",
            "n_rows_train",
            "n_rows_test",
            "best_test_model",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_macro_f1",
            "test_roc_auc_ovr",
            "best_cv_model",
            "cv_folds",
            "cv_mean_accuracy",
            "cv_mean_balanced_accuracy",
            "cv_mean_macro_f1",
            "fit_seconds",
            "cv_seconds",
            "total_seconds",
        ],
    ].copy()


def build_stage_metric_long_frame(
    overview_df: pd.DataFrame,
    *,
    metric_columns: tuple[str, ...] = (
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "cv_mean_accuracy",
        "cv_mean_balanced_accuracy",
        "cv_mean_macro_f1",
    ),
) -> pd.DataFrame:
    # Преобразуем overview в tidy long-format для barplot/lineplot.
    available_metric_columns = [
        column_name for column_name in metric_columns if column_name in overview_df.columns
    ]
    if overview_df.empty or not available_metric_columns:
        return pd.DataFrame(columns=["stage_name", "metric_name", "metric_value"])

    metric_long_df = overview_df.loc[:, ["stage_name", *available_metric_columns]].melt(
        id_vars="stage_name",
        var_name="metric_name",
        value_name="metric_value",
    )
    return metric_long_df.dropna(subset=["metric_value"]).reset_index(drop=True)


def _select_best_metrics_row(metrics_df: pd.DataFrame) -> pd.Series | None:
    if metrics_df.empty:
        return None
    test_metrics_df = metrics_df.loc[metrics_df["split_name"] == "test"].copy()
    if test_metrics_df.empty:
        return None
    return test_metrics_df.sort_values(
        ["accuracy", "macro_f1", "balanced_accuracy"],
        ascending=[False, False, False],
        kind="mergesort",
        ignore_index=True,
    ).iloc[0]


def _select_best_cv_row(cv_summary_df: pd.DataFrame) -> pd.Series | None:
    if cv_summary_df.empty:
        return None
    return cv_summary_df.sort_values(
        ["mean_macro_f1", "mean_accuracy", "mean_balanced_accuracy"],
        ascending=[False, False, False],
        kind="mergesort",
        ignore_index=True,
    ).iloc[0]


def _normalize_series_record(row: pd.Series) -> dict[str, object]:
    return _normalize_record_mapping(row.to_dict())


def _normalize_record_mapping(row: Mapping[Hashable, object]) -> dict[str, object]:
    return {str(key): value for key, value in row.items()}


def _to_optional_float(value: object) -> float | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value)
    return None


def _require_float(value: object) -> float:
    optional_value = _to_optional_float(value)
    if optional_value is None:
        raise ValueError("Expected numeric scalar in benchmark review payload.")
    return optional_value


def _require_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if pd.isna(value):
            raise ValueError("Expected integer scalar in benchmark review payload.")
        return int(value)
    if isinstance(value, str):
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("Expected integer scalar in benchmark review payload.")
        return int(normalized_value)
    raise ValueError("Expected integer scalar in benchmark review payload.")


__all__ = [
    "build_benchmark_summary_frame",
    "build_pipeline_stage_overview_frame",
    "build_stage_metric_long_frame",
]
