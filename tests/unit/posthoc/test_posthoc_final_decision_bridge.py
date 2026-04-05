# Тестовый файл `test_posthoc_final_decision_bridge.py` домена `posthoc`.
#
# Этот файл проверяет только:
# - проверку логики домена: post-hoc routing, gate и final decision policy;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `posthoc` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import pandas as pd
import pytest

from exohost.posthoc.final_decision import FINAL_DOMAIN_ID, FinalDecisionPolicy
from exohost.posthoc.final_decision_bridge import (
    build_final_decision_bridge_result,
    build_final_decision_input_frame,
)


def build_base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2],
            "quality_state": ["pass", "pass"],
            "priority_state": ["high", "low"],
        }
    )


def build_ood_scored_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2],
            "ood_decision": ["in_domain", "in_domain"],
            "ood_probability": [0.02, 0.03],
        }
    )


def build_coarse_scored_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2],
            "coarse_predicted_label": ["G", "O"],
            "coarse_probability_max": [0.92, 0.87],
            "coarse_probability_margin": [0.24, 0.31],
        }
    )


def build_refinement_scored_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1],
            "refinement_predicted_label": ["G4"],
            "refinement_probability_max": [0.81],
            "refinement_probability_margin": [0.62],
            "refinement_family_name": ["G"],
        }
    )


def test_build_final_decision_input_frame_merges_stage_outputs() -> None:
    result = build_final_decision_input_frame(
        build_base_frame(),
        ood_scored_df=build_ood_scored_frame(),
        coarse_scored_df=build_coarse_scored_frame(),
        refinement_scored_df=build_refinement_scored_frame(),
    )

    assert "ood_decision" in result.columns
    assert "coarse_predicted_label" in result.columns
    assert "refinement_predicted_label" in result.columns
    assert result.loc[0, "refinement_predicted_label"] == "G4"
    assert pd.isna(result.loc[1, "refinement_predicted_label"])


def test_build_final_decision_bridge_result_produces_routed_frame() -> None:
    result = build_final_decision_bridge_result(
        build_base_frame(),
        ood_scored_df=build_ood_scored_frame(),
        coarse_scored_df=build_coarse_scored_frame(),
        refinement_scored_df=build_refinement_scored_frame(),
        policy=FinalDecisionPolicy(decision_policy_version="final_decision_v1"),
    )

    assert result.final_decision_df.loc[0, "final_domain_state"] == FINAL_DOMAIN_ID
    assert result.final_decision_df.loc[0, "final_refinement_label"] == "G4"
    assert result.final_decision_df.loc[1, "final_decision_reason"] == "coarse_class_not_enabled"


def test_build_final_decision_input_frame_rejects_duplicate_source_ids() -> None:
    duplicate_ood_frame = pd.DataFrame(
        {
            "source_id": [1, 1],
            "ood_decision": ["in_domain", "ood"],
        }
    )

    with pytest.raises(ValueError, match="duplicate source_id"):
        build_final_decision_input_frame(
            build_base_frame(),
            ood_scored_df=duplicate_ood_frame,
            coarse_scored_df=build_coarse_scored_frame(),
        )
