# Файл `quality_gate_review.py` слоя `reporting`.
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

from exohost.contracts.dataset_contracts import DatasetContract
from exohost.contracts.quality_gate_dataset_contracts import (
    GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT,
    GAIA_MK_UNKNOWN_REVIEW_AUDIT_CONTRACT,
    QUALITY_GATE_SIGNAL_COLUMNS,
)
from exohost.datasets.load_quality_gate_audit_dataset import (
    load_quality_gate_audit_dataset,
)
from exohost.db.engine import make_read_only_engine


def load_quality_gate_review_frame(
    *,
    contract: DatasetContract = GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT,
    limit: int | None = None,
    dotenv_path: str = ".env",
) -> pd.DataFrame:
    # Загружаем один quality-gate audit relation для calibration-review.
    engine = make_read_only_engine(dotenv_path=dotenv_path)
    try:
        return load_quality_gate_audit_dataset(
            engine,
            contract=contract,
            limit=limit,
        )
    finally:
        engine.dispose()


def load_unknown_review_frame(
    *,
    limit: int | None = None,
    dotenv_path: str = ".env",
) -> pd.DataFrame:
    # Удобный wrapper для unknown/review relation без дублирования notebook-кода.
    return load_quality_gate_review_frame(
        contract=GAIA_MK_UNKNOWN_REVIEW_AUDIT_CONTRACT,
        limit=limit,
        dotenv_path=dotenv_path,
    )


def build_quality_gate_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Компактная one-row summary по quality-gate review source.
    source_id = _require_series_column(df, "source_id")
    quality_state = _require_series_column(df, "quality_state")
    ood_state = _require_series_column(df, "ood_state")
    return pd.DataFrame(
        [
            {
                "n_rows": int(df.shape[0]),
                "n_unique_source_id": int(source_id.astype(str).nunique(dropna=False)),
                "n_pass_rows": int((quality_state == "pass").sum()),
                "n_unknown_rows": int((quality_state == "unknown").sum()),
                "n_reject_rows": int((quality_state == "reject").sum()),
                "n_in_domain_rows": int((ood_state == "in_domain").sum()),
                "n_candidate_ood_rows": int((ood_state == "candidate_ood").sum()),
                "n_ood_rows": int((ood_state == "ood").sum()),
            }
        ]
    )


def build_quality_state_distribution_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Распределение quality_state для gate calibration.
    return _build_distribution_frame(
        df,
        column_name="quality_state",
        label_name="quality_state",
    )


def build_ood_state_distribution_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Распределение ood_state для gate calibration.
    return _build_distribution_frame(
        df,
        column_name="ood_state",
        label_name="ood_state",
    )


def build_quality_reason_distribution_frame(
    df: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    # Top quality reasons для review/reject analysis.
    distribution_df = _build_distribution_frame(
        df,
        column_name="quality_reason",
        label_name="quality_reason",
    )
    return distribution_df.head(top_n).copy()


def build_review_bucket_distribution_frame(
    df: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    # Top review buckets для calibration-study.
    distribution_df = _build_distribution_frame(
        df,
        column_name="review_bucket",
        label_name="review_bucket",
    )
    return distribution_df.head(top_n).copy()


def build_quality_gate_signal_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Сводка по отдельным gate signals и их вкладу в pass/unknown/reject.
    rows: list[dict[str, object]] = []
    total_rows = int(df.shape[0])

    for signal_name in QUALITY_GATE_SIGNAL_COLUMNS:
        if signal_name not in df.columns:
            continue

        signal_series = _require_series_column(df, signal_name).fillna(False).astype(bool)
        signal_df = df.loc[signal_series].copy()
        quality_state = (
            _require_series_column(signal_df, "quality_state")
            if not signal_df.empty
            else pd.Series(dtype="object")
        )
        ood_state = (
            _require_series_column(signal_df, "ood_state")
            if not signal_df.empty
            else pd.Series(dtype="object")
        )
        n_rows_true = int(signal_df.shape[0])

        rows.append(
            {
                "signal_name": signal_name,
                "n_rows_true": n_rows_true,
                "share_true": float(n_rows_true / total_rows) if total_rows > 0 else 0.0,
                "n_pass_true": int((quality_state == "pass").sum()),
                "n_unknown_true": int((quality_state == "unknown").sum()),
                "n_reject_true": int((quality_state == "reject").sum()),
                "n_in_domain_true": int((ood_state == "in_domain").sum()),
                "n_candidate_ood_true": int((ood_state == "candidate_ood").sum()),
                "n_ood_true": int((ood_state == "ood").sum()),
            }
        )

    return pd.DataFrame.from_records(rows)


def build_quality_review_crosstab_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Crosstab помогает понять, как quality_state соотносится с review buckets.
    quality_state = _require_series_column(df, "quality_state")
    review_bucket = _require_series_column(df, "review_bucket")
    return pd.crosstab(quality_state, review_bucket, dropna=False)


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


def _require_series_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    column = df.loc[:, column_name]
    if not isinstance(column, pd.Series):
        raise TypeError(f"{column_name} must resolve to a pandas Series.")
    return column
