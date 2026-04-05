# Файл `final_decision_artifact_runner.py` слоя `posthoc`.
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

from exohost.models.inference import score_with_model
from exohost.posthoc.decision_model_bundle import (
    FinalDecisionModelBundle,
    build_final_decision_runner_config_from_bundle,
)
from exohost.posthoc.final_decision import FinalDecisionPolicy
from exohost.posthoc.final_decision_runner import run_final_decision_pipeline
from exohost.posthoc.priority_integration import (
    DEFAULT_PRIORITY_INTEGRATION_CONFIG,
    PriorityIntegrationConfig,
    build_priority_integration_result,
)


@dataclass(frozen=True, slots=True)
class FinalDecisionArtifactRunResult:
    # Полный результат higher-level decision pipeline поверх saved artifacts.
    decision_input_df: pd.DataFrame
    final_decision_df: pd.DataFrame
    priority_input_df: pd.DataFrame
    priority_ranking_df: pd.DataFrame


def run_final_decision_with_artifacts(
    base_df: pd.DataFrame,
    *,
    bundle: FinalDecisionModelBundle,
    final_decision_policy: FinalDecisionPolicy,
    priority_config: PriorityIntegrationConfig = DEFAULT_PRIORITY_INTEGRATION_CONFIG,
) -> FinalDecisionArtifactRunResult:
    # Собираем final decision и optional priority integration из saved artifacts.
    runner_config = build_final_decision_runner_config_from_bundle(
        bundle,
        final_decision_policy=final_decision_policy,
    )
    decision_result = run_final_decision_pipeline(base_df, config=runner_config)

    if bundle.host_artifact is None:
        return FinalDecisionArtifactRunResult(
            decision_input_df=decision_result.input_df,
            final_decision_df=decision_result.final_decision_df,
            priority_input_df=_build_empty_priority_input_frame(),
            priority_ranking_df=_build_empty_priority_ranking_frame(),
        )

    host_scoring_result = score_with_model(
        base_df,
        estimator=bundle.host_artifact.estimator,
        task_name=bundle.host_artifact.task_name,
        target_column=bundle.host_artifact.target_column,
        feature_columns=bundle.host_artifact.feature_columns,
        model_name=bundle.host_artifact.model_name,
        host_score_column=priority_config.host_score_column,
    )
    priority_result = build_priority_integration_result(
        host_scoring_result.scored_df,
        final_decision_df=decision_result.final_decision_df,
        config=priority_config,
    )
    return FinalDecisionArtifactRunResult(
        decision_input_df=decision_result.input_df,
        final_decision_df=priority_result.final_decision_df,
        priority_input_df=priority_result.priority_input_df,
        priority_ranking_df=priority_result.priority_ranking_df,
    )


def _build_empty_priority_input_frame() -> pd.DataFrame:
    return pd.DataFrame({"source_id": pd.Series(dtype="object")})


def _build_empty_priority_ranking_frame() -> pd.DataFrame:
    return pd.DataFrame({"source_id": pd.Series(dtype="object")})
