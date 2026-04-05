# Тестовый файл `test_posthoc_final_decision.py` домена `posthoc`.
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

from exohost.posthoc.candidate_ood_policy import (
    CANDIDATE_OOD_MAP_TO_UNKNOWN_DISPOSITION,
    CandidateOodPolicy,
)
from exohost.posthoc.final_decision import (
    FINAL_DOMAIN_CANDIDATE_OOD,
    FINAL_DOMAIN_ID,
    FINAL_DOMAIN_OOD,
    FINAL_DOMAIN_UNKNOWN,
    FINAL_REFINEMENT_ACCEPTED,
    FINAL_REFINEMENT_NOT_ATTEMPTED,
    FINAL_REFINEMENT_REJECTED_TO_UNKNOWN,
    FinalDecisionPolicy,
    build_final_decision_frame,
)
from exohost.posthoc.refinement_handoff import RefinementHandoffPolicy


def build_input_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": 1,
                "quality_state": "reject",
                "ood_decision": "in_domain",
                "coarse_predicted_label": "G",
                "coarse_probability_max": 0.90,
                "coarse_probability_margin": 0.35,
                "refinement_predicted_label": "G2",
                "refinement_probability_max": 0.82,
                "priority_state": "high",
            },
            {
                "source_id": 2,
                "quality_state": "pass",
                "ood_decision": "ood",
                "coarse_predicted_label": "K",
                "coarse_probability_max": 0.88,
                "coarse_probability_margin": 0.22,
                "refinement_predicted_label": "K3",
                "refinement_probability_max": 0.79,
                "priority_state": "medium",
            },
            {
                "source_id": 3,
                "quality_state": "pass",
                "ood_decision": "candidate_ood",
                "coarse_predicted_label": "M",
                "coarse_probability_max": 0.77,
                "coarse_probability_margin": 0.11,
                "refinement_predicted_label": "M2",
                "refinement_probability_max": 0.73,
                "priority_state": "medium",
            },
            {
                "source_id": 4,
                "quality_state": "pass",
                "ood_decision": "in_domain",
                "coarse_predicted_label": "O",
                "coarse_probability_max": 0.91,
                "coarse_probability_margin": 0.44,
                "refinement_predicted_label": None,
                "refinement_probability_max": None,
                "priority_state": "low",
            },
            {
                "source_id": 5,
                "quality_state": "pass",
                "ood_decision": "in_domain",
                "coarse_predicted_label": "G",
                "coarse_probability_max": 0.93,
                "coarse_probability_margin": 0.27,
                "refinement_predicted_label": "G4",
                "refinement_probability_max": 0.81,
                "priority_state": "high",
            },
            {
                "source_id": 6,
                "quality_state": "pass",
                "ood_decision": "in_domain",
                "coarse_predicted_label": "K",
                "coarse_probability_max": 0.89,
                "coarse_probability_margin": 0.18,
                "refinement_predicted_label": "K2",
                "refinement_probability_max": 0.45,
                "priority_state": "high",
            },
        ]
    )


def test_build_final_decision_frame_routes_quality_ood_and_refinement_cases() -> None:
    policy = FinalDecisionPolicy(
        decision_policy_version="final_decision_v1",
        min_refinement_confidence=0.6,
    )

    result = build_final_decision_frame(build_input_frame(), policy=policy)

    assert result["final_domain_state"].tolist() == [
        FINAL_DOMAIN_UNKNOWN,
        FINAL_DOMAIN_OOD,
        FINAL_DOMAIN_CANDIDATE_OOD,
        FINAL_DOMAIN_ID,
        FINAL_DOMAIN_ID,
        FINAL_DOMAIN_UNKNOWN,
    ]
    assert result["final_refinement_state"].tolist() == [
        FINAL_REFINEMENT_NOT_ATTEMPTED,
        FINAL_REFINEMENT_NOT_ATTEMPTED,
        FINAL_REFINEMENT_NOT_ATTEMPTED,
        FINAL_REFINEMENT_NOT_ATTEMPTED,
        FINAL_REFINEMENT_ACCEPTED,
        FINAL_REFINEMENT_REJECTED_TO_UNKNOWN,
    ]
    assert pd.isna(result.loc[0, "final_refinement_label"])
    assert pd.isna(result.loc[1, "final_refinement_label"])
    assert pd.isna(result.loc[2, "final_refinement_label"])
    assert pd.isna(result.loc[3, "final_refinement_label"])
    assert result.loc[4, "final_refinement_label"] == "G4"
    assert pd.isna(result.loc[5, "final_refinement_label"])
    assert result["final_decision_reason"].tolist() == [
        "quality_reject",
        "hard_ood",
        "candidate_ood",
        "coarse_class_not_enabled",
        "refinement_accepted",
        "refinement_confidence_below_threshold",
    ]
    assert pd.isna(result.loc[0, "priority_state"])
    assert pd.isna(result.loc[1, "priority_state"])
    assert pd.isna(result.loc[2, "priority_state"])
    assert pd.isna(result.loc[3, "priority_state"])
    assert pd.isna(result.loc[4, "priority_state"])
    assert pd.isna(result.loc[5, "priority_state"])


def test_build_final_decision_frame_respects_probability_handoff_thresholds() -> None:
    policy = FinalDecisionPolicy(
        decision_policy_version="final_decision_v1",
        refinement_handoff_policy=RefinementHandoffPolicy(min_coarse_probability=0.95),
    )

    frame = pd.DataFrame(
        [
            {
                "source_id": 10,
                "quality_state": "pass",
                "ood_decision": "in_domain",
                "coarse_predicted_label": "G",
                "coarse_probability_max": 0.91,
                "coarse_probability_margin": 0.21,
                "refinement_predicted_label": "G3",
                "refinement_probability_max": 0.88,
                "priority_state": "medium",
            }
        ]
    )

    result = build_final_decision_frame(frame, policy=policy)

    assert result.loc[0, "final_domain_state"] == FINAL_DOMAIN_ID
    assert result.loc[0, "final_refinement_state"] == FINAL_REFINEMENT_NOT_ATTEMPTED
    assert result.loc[0, "final_decision_reason"] == "coarse_probability_below_threshold"


def test_build_final_decision_frame_requires_contract_columns() -> None:
    frame = pd.DataFrame([{"source_id": 1}])

    try:
        build_final_decision_frame(
            frame,
            policy=FinalDecisionPolicy(decision_policy_version="final_decision_v1"),
        )
    except ValueError as exc:
        assert "missing required columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing final decision columns.")


def test_build_final_decision_frame_can_map_candidate_ood_to_unknown() -> None:
    policy = FinalDecisionPolicy(
        decision_policy_version="final_decision_v1",
        candidate_ood_policy=CandidateOodPolicy(
            disposition=CANDIDATE_OOD_MAP_TO_UNKNOWN_DISPOSITION
        ),
    )

    result = build_final_decision_frame(build_input_frame(), policy=policy)

    assert result.loc[2, "final_domain_state"] == FINAL_DOMAIN_UNKNOWN
    assert result.loc[2, "final_decision_reason"] == "candidate_ood_mapped_to_unknown"
