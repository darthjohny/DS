# Файл `final_decision_runner.py` слоя `posthoc`.
#
# Этот файл отвечает только за:
# - post-hoc scoring, routing и final decision policy;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `posthoc` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from exohost.posthoc.coarse_scoring import build_coarse_scored_frame
from exohost.posthoc.final_decision import FinalDecisionPolicy
from exohost.posthoc.final_decision_bridge import (
    FinalDecisionBridgeResult,
    build_final_decision_bridge_result,
)
from exohost.posthoc.id_ood_gate import (
    ID_OOD_IN_DOMAIN_STATE,
    IdOodThresholdPolicy,
    build_id_ood_gate_scored_frame,
)
from exohost.posthoc.refinement_family_scoring import (
    build_refinement_family_scored_frame,
)
from exohost.posthoc.refinement_handoff import decide_refinement_handoff


@dataclass(frozen=True, slots=True)
class FinalDecisionRunnerConfig:
    # Все model-side зависимости final decision pipeline в одном типизированном config.
    ood_estimator: object
    ood_feature_columns: tuple[str, ...]
    ood_threshold_policy: IdOodThresholdPolicy
    coarse_estimator: object
    coarse_feature_columns: tuple[str, ...]
    final_decision_policy: FinalDecisionPolicy
    refinement_estimators_by_family: Mapping[str, object] | None = None
    refinement_feature_columns: tuple[str, ...] | None = None
    coarse_model_name: str | None = None
    refinement_model_names_by_family: Mapping[str, str] | None = None


def run_final_decision_pipeline(
    base_df: pd.DataFrame,
    *,
    config: FinalDecisionRunnerConfig,
) -> FinalDecisionBridgeResult:
    # Собираем stage outputs и final routing без смешивания с CLI или persistence.
    ood_scored_df = build_id_ood_gate_scored_frame(
        base_df,
        estimator=config.ood_estimator,
        feature_columns=config.ood_feature_columns,
        policy=config.ood_threshold_policy,
    )
    ood_scored_df = _select_ood_stage_columns(ood_scored_df)
    in_domain_base_df = _select_in_domain_rows(base_df, ood_scored_df=ood_scored_df)
    coarse_scored_df = build_coarse_scored_frame(
        in_domain_base_df,
        estimator=config.coarse_estimator,
        feature_columns=config.coarse_feature_columns,
        model_name=config.coarse_model_name,
    )
    refinement_scored_df = _build_refinement_scored_frame(
        base_df,
        coarse_scored_df=coarse_scored_df,
        config=config,
    )
    return build_final_decision_bridge_result(
        base_df,
        ood_scored_df=ood_scored_df,
        coarse_scored_df=coarse_scored_df,
        refinement_scored_df=refinement_scored_df,
        policy=config.final_decision_policy,
    )


def _select_in_domain_rows(
    base_df: pd.DataFrame,
    *,
    ood_scored_df: pd.DataFrame,
) -> pd.DataFrame:
    in_domain_source_ids = (
        ood_scored_df.loc[
            ood_scored_df["ood_decision"] == ID_OOD_IN_DOMAIN_STATE,
            "source_id",
        ]
        .astype(str)
        .tolist()
    )
    return (
        base_df.loc[base_df["source_id"].astype(str).isin(in_domain_source_ids)]
        .reset_index(drop=True)
        .copy()
    )


def _select_ood_stage_columns(df: pd.DataFrame) -> pd.DataFrame:
    stage_columns = (
        "source_id",
        "ood_decision",
        "ood_probability",
        "id_probability",
        "ood_threshold",
        "candidate_ood_threshold",
        "ood_threshold_name",
        "ood_threshold_metric",
        "ood_threshold_fit_scope",
        "ood_threshold_policy_version",
        "predicted_domain_target",
    )
    available_stage_columns = [column_name for column_name in stage_columns if column_name in df.columns]
    return df.loc[:, available_stage_columns].copy()


def _build_refinement_scored_frame(
    base_df: pd.DataFrame,
    *,
    coarse_scored_df: pd.DataFrame,
    config: FinalDecisionRunnerConfig,
) -> pd.DataFrame | None:
    if config.refinement_estimators_by_family is None:
        return None
    if config.refinement_feature_columns is None:
        raise ValueError(
            "Final decision runner requires refinement_feature_columns when "
            "refinement_estimators_by_family is provided."
        )

    refinement_frames: list[pd.DataFrame] = []
    for family_name, family_estimator in config.refinement_estimators_by_family.items():
        family_input_df = _select_family_refinement_rows(
            base_df,
            coarse_scored_df=coarse_scored_df,
            family_name=family_name,
            policy=config.final_decision_policy,
        )
        if family_input_df.empty:
            continue

        refinement_frames.append(
            build_refinement_family_scored_frame(
                family_input_df,
                estimator=family_estimator,
                feature_columns=config.refinement_feature_columns,
                family_name=family_name,
                model_name=_resolve_refinement_model_name(
                    family_name=family_name,
                    model_names_by_family=config.refinement_model_names_by_family,
                ),
            )
        )

    if not refinement_frames:
        return None
    return pd.concat(refinement_frames, ignore_index=True)


def _select_family_refinement_rows(
    base_df: pd.DataFrame,
    *,
    coarse_scored_df: pd.DataFrame,
    family_name: str,
    policy: FinalDecisionPolicy,
) -> pd.DataFrame:
    family_rows = coarse_scored_df.loc[
        coarse_scored_df["coarse_predicted_label"].astype(str).str.upper() == family_name,
        :,
    ].copy()
    if family_rows.empty:
        return base_df.iloc[0:0].copy()

    refinement_source_ids: list[str] = []
    for row in family_rows.to_dict(orient="records"):
        handoff_decision = decide_refinement_handoff(
            final_domain_state="id",
            coarse_class=str(row.get("coarse_predicted_label", "")),
            coarse_probability_max=_to_optional_float(row.get("coarse_probability_max")),
            coarse_probability_margin=_to_optional_float(row.get("coarse_probability_margin")),
            policy=policy.refinement_handoff_policy,
        )
        if handoff_decision.should_attempt_refinement:
            refinement_source_ids.append(str(row["source_id"]))

    return (
        base_df.loc[base_df["source_id"].astype(str).isin(refinement_source_ids)]
        .reset_index(drop=True)
        .copy()
    )


def _resolve_refinement_model_name(
    *,
    family_name: str,
    model_names_by_family: Mapping[str, str] | None,
) -> str | None:
    if model_names_by_family is None:
        return None
    return model_names_by_family.get(family_name)


def _to_optional_float(value: object) -> float | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return None if pd.isna(value) else value
    if isinstance(value, str):
        normalized_value = value.strip()
        if not normalized_value:
            return None
        try:
            return float(normalized_value)
        except ValueError:
            return None
    return None
