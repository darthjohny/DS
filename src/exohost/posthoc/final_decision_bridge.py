# Файл `final_decision_bridge.py` слоя `posthoc`.
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

import pandas as pd

from exohost.posthoc.final_decision import (
    FinalDecisionPolicy,
    build_final_decision_frame,
)


@dataclass(frozen=True, slots=True)
class FinalDecisionBridgeResult:
    # Полный результат merge stage outputs и final routing.
    input_df: pd.DataFrame
    final_decision_df: pd.DataFrame


def build_final_decision_bridge_result(
    base_df: pd.DataFrame,
    *,
    ood_scored_df: pd.DataFrame,
    coarse_scored_df: pd.DataFrame,
    policy: FinalDecisionPolicy,
    refinement_scored_df: pd.DataFrame | None = None,
) -> FinalDecisionBridgeResult:
    # Собираем final decision input frame из реальных stage outputs.
    input_df = build_final_decision_input_frame(
        base_df,
        ood_scored_df=ood_scored_df,
        coarse_scored_df=coarse_scored_df,
        refinement_scored_df=refinement_scored_df,
    )
    final_decision_df = build_final_decision_frame(input_df, policy=policy)
    return FinalDecisionBridgeResult(
        input_df=input_df,
        final_decision_df=final_decision_df,
    )


def build_final_decision_input_frame(
    base_df: pd.DataFrame,
    *,
    ood_scored_df: pd.DataFrame,
    coarse_scored_df: pd.DataFrame,
    refinement_scored_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    # Мержим stage outputs в единый routing-ready frame по source_id.
    _require_unique_source_id(base_df, frame_name="final decision base frame")
    _require_columns(base_df, ("source_id", "quality_state"), frame_name="final decision base frame")
    _require_unique_source_id(ood_scored_df, frame_name="ood scored frame")
    _require_columns(
        ood_scored_df,
        ("source_id", "ood_decision"),
        frame_name="ood scored frame",
    )
    _require_unique_source_id(coarse_scored_df, frame_name="coarse scored frame")
    _require_columns(
        coarse_scored_df,
        (
            "source_id",
            "coarse_predicted_label",
            "coarse_probability_max",
            "coarse_probability_margin",
        ),
        frame_name="coarse scored frame",
    )

    result = base_df.copy()
    result = result.merge(ood_scored_df, on="source_id", how="left", validate="one_to_one")
    result = result.merge(
        coarse_scored_df,
        on="source_id",
        how="left",
        validate="one_to_one",
    )

    if refinement_scored_df is None:
        result["refinement_predicted_label"] = pd.NA
        result["refinement_probability_max"] = pd.NA
        result["refinement_probability_margin"] = pd.NA
        result["refinement_family_name"] = pd.NA
        return result

    _require_unique_source_id(refinement_scored_df, frame_name="refinement scored frame")
    _require_columns(
        refinement_scored_df,
        (
            "source_id",
            "refinement_predicted_label",
            "refinement_probability_max",
        ),
        frame_name="refinement scored frame",
    )
    return result.merge(
        refinement_scored_df,
        on="source_id",
        how="left",
        validate="one_to_one",
    )


def _require_columns(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    frame_name: str,
) -> None:
    missing_columns = [column_name for column_name in columns if column_name not in df.columns]
    if missing_columns:
        missing_columns_sql = ", ".join(missing_columns)
        raise ValueError(f"{frame_name} is missing required columns: {missing_columns_sql}")


def _require_unique_source_id(df: pd.DataFrame, *, frame_name: str) -> None:
    _require_columns(df, ("source_id",), frame_name=frame_name)
    duplicate_mask = df["source_id"].astype(str).duplicated(keep=False)
    if bool(duplicate_mask.to_numpy().any()):
        raise ValueError(f"{frame_name} contains duplicate source_id values.")
