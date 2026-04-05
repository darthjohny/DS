# Файл `priority_integration.py` слоя `posthoc`.
#
# Этот файл отвечает только за:
# - post-hoc scoring, routing и final decision policy;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `posthoc` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

from exohost.posthoc.priority_input import build_priority_input_frame
from exohost.ranking.priority_score import (
    DEFAULT_HOST_SCORE_COLUMN,
    DEFAULT_PRIORITY_THRESHOLDS,
    DEFAULT_RANKING_WEIGHTS,
    PriorityThresholds,
    RankingWeights,
    build_priority_ranking_frame,
)

_PRIORITY_OUTPUT_COLUMNS: tuple[str, ...] = (
    "class_priority_score",
    "host_similarity_score",
    "brightness_score",
    "distance_score",
    "astrometry_score",
    "observability_score",
    "brightness_available",
    "distance_available",
    "astrometry_available",
    "observability_evidence_count",
    "priority_score",
    "priority_label",
    "priority_reason",
)


@dataclass(frozen=True, slots=True)
class PriorityIntegrationConfig:
    # Конфиг узкого integration-layer поверх explainable ranking.
    host_score_column: str = DEFAULT_HOST_SCORE_COLUMN
    weights: RankingWeights = DEFAULT_RANKING_WEIGHTS
    thresholds: PriorityThresholds = DEFAULT_PRIORITY_THRESHOLDS


@dataclass(frozen=True, slots=True)
class PriorityIntegrationResult:
    # Полный результат priority integration после final routing.
    priority_input_df: pd.DataFrame
    priority_ranking_df: pd.DataFrame
    final_decision_df: pd.DataFrame


DEFAULT_PRIORITY_INTEGRATION_CONFIG = PriorityIntegrationConfig()


def build_priority_integration_result(
    base_df: pd.DataFrame,
    *,
    final_decision_df: pd.DataFrame,
    config: PriorityIntegrationConfig = DEFAULT_PRIORITY_INTEGRATION_CONFIG,
) -> PriorityIntegrationResult:
    # Считаем explainable priority только для final in-domain rows.
    priority_input_df = build_priority_input_frame(
        base_df,
        final_decision_df=final_decision_df,
        host_score_column=config.host_score_column,
    )
    if priority_input_df.empty:
        return PriorityIntegrationResult(
            priority_input_df=priority_input_df,
            priority_ranking_df=_build_empty_priority_ranking_frame(),
            final_decision_df=_merge_priority_into_final_decision(
                final_decision_df,
                priority_ranking_df=_build_empty_priority_ranking_frame(),
            ),
        )

    priority_ranking_df = build_priority_ranking_frame(
        priority_input_df,
        host_score_column=config.host_score_column,
        weights=config.weights,
        thresholds=config.thresholds,
    )
    return PriorityIntegrationResult(
        priority_input_df=priority_input_df,
        priority_ranking_df=priority_ranking_df,
        final_decision_df=_merge_priority_into_final_decision(
            final_decision_df,
            priority_ranking_df=priority_ranking_df,
        ),
    )


def _merge_priority_into_final_decision(
    final_decision_df: pd.DataFrame,
    *,
    priority_ranking_df: pd.DataFrame,
) -> pd.DataFrame:
    result = final_decision_df.copy()
    result["priority_state"] = pd.NA
    result["_priority_source_id"] = result["source_id"].astype(str)
    ranking_merge_df = priority_ranking_df.copy()
    ranking_merge_df["_priority_source_id"] = ranking_merge_df["source_id"].astype(str)
    available_merge_columns = [
        "_priority_source_id",
        *[
            column_name
            for column_name in _PRIORITY_OUTPUT_COLUMNS
            if column_name in ranking_merge_df.columns
        ],
    ]
    result = result.merge(
        ranking_merge_df.loc[:, available_merge_columns].copy(),
        on="_priority_source_id",
        how="left",
        validate="one_to_one",
    )
    result = result.drop(columns="_priority_source_id")
    result["priority_state"] = _build_priority_state_series(result)
    return result


def _build_empty_priority_ranking_frame() -> pd.DataFrame:
    columns: dict[str, pd.Series] = {"source_id": pd.Series(dtype="object")}
    for column_name in _PRIORITY_OUTPUT_COLUMNS:
        if column_name.endswith("_available"):
            columns[column_name] = pd.Series(dtype="boolean")
        elif column_name == "observability_evidence_count":
            columns[column_name] = pd.Series(dtype="int64")
        elif column_name in ("priority_label", "priority_reason"):
            columns[column_name] = pd.Series(dtype="string")
        else:
            columns[column_name] = pd.Series(dtype="float64")
    return pd.DataFrame(columns)


def _build_priority_state_series(df: pd.DataFrame) -> pd.Series:
    # Каноническое состояние priority повторяет priority_label, но хранится в string dtype.
    if "priority_label" not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    priority_label = df["priority_label"]
    if not isinstance(priority_label, pd.Series):
        raise TypeError("priority_label column must resolve to a pandas Series.")
    priority_state = priority_label.astype("string")
    if not isinstance(priority_state, pd.Series):
        raise TypeError("priority_state conversion must produce a pandas Series.")
    return cast(pd.Series, priority_state)
