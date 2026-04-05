# Файл `priority_input.py` слоя `posthoc`.
#
# Этот файл отвечает только за:
# - post-hoc scoring, routing и final decision policy;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `posthoc` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

import pandas as pd

from exohost.ranking.priority_score import (
    DEFAULT_HOST_SCORE_COLUMN,
    RANKING_OPTIONAL_COLUMNS,
)

_FINAL_DECISION_REQUIRED_COLUMNS: tuple[str, ...] = (
    "source_id",
    "final_domain_state",
    "final_quality_state",
    "final_coarse_class",
)


def build_priority_input_frame(
    base_df: pd.DataFrame,
    *,
    final_decision_df: pd.DataFrame,
    host_score_column: str = DEFAULT_HOST_SCORE_COLUMN,
) -> pd.DataFrame:
    # Строим ranking-ready input только для final in-domain rows.
    _require_unique_source_id(base_df, frame_name="priority base frame")
    _require_unique_source_id(
        final_decision_df,
        frame_name="final decision frame",
    )
    _require_columns(
        base_df,
        ("source_id", host_score_column),
        frame_name="priority base frame",
    )
    _require_columns(
        final_decision_df,
        _FINAL_DECISION_REQUIRED_COLUMNS,
        frame_name="final decision frame",
    )

    eligible_decision_df = _select_priority_eligible_rows(final_decision_df)
    if eligible_decision_df.empty:
        return _build_empty_priority_input_frame(host_score_column=host_score_column)

    available_base_columns = [
        "source_id",
        host_score_column,
        *[
            column_name
            for column_name in RANKING_OPTIONAL_COLUMNS
            if column_name in base_df.columns
        ],
    ]
    merged_df = eligible_decision_df.merge(
        base_df.loc[:, available_base_columns].copy(),
        on="source_id",
        how="left",
        validate="one_to_one",
    )
    merged_df["spec_class"] = (
        merged_df.loc[:, "final_coarse_class"].astype(str).str.strip().str.upper()
    )
    result_columns = [
        "source_id",
        "spec_class",
        host_score_column,
        *[
            column_name
            for column_name in RANKING_OPTIONAL_COLUMNS
            if column_name in merged_df.columns
        ],
    ]
    return merged_df.loc[:, result_columns].copy()


def _select_priority_eligible_rows(final_decision_df: pd.DataFrame) -> pd.DataFrame:
    eligible_mask = (
        final_decision_df["final_domain_state"].astype(str).str.lower().eq("id")
        & final_decision_df["final_quality_state"].astype(str).str.lower().eq("pass")
        & final_decision_df["final_coarse_class"].notna()
    )
    return (
        final_decision_df.loc[eligible_mask, ["source_id", "final_coarse_class"]]
        .reset_index(drop=True)
        .copy()
    )


def _build_empty_priority_input_frame(
    *,
    host_score_column: str,
) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {
        "source_id": pd.Series(dtype="object"),
        "spec_class": pd.Series(dtype="string"),
        host_score_column: pd.Series(dtype="float64"),
    }
    for column_name in RANKING_OPTIONAL_COLUMNS:
        columns[column_name] = pd.Series(dtype="float64")
    return pd.DataFrame(columns)


def _require_columns(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    frame_name: str,
) -> None:
    missing_columns = [
        column_name for column_name in columns if column_name not in df.columns
    ]
    if missing_columns:
        missing_columns_sql = ", ".join(missing_columns)
        raise ValueError(f"{frame_name} is missing required columns: {missing_columns_sql}")


def _require_unique_source_id(df: pd.DataFrame, *, frame_name: str) -> None:
    _require_columns(df, ("source_id",), frame_name=frame_name)
    duplicate_mask = df["source_id"].astype(str).duplicated(keep=False)
    if bool(duplicate_mask.to_numpy().any()):
        raise ValueError(f"{frame_name} contains duplicate source_id values.")
