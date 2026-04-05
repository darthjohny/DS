# Файл `final_decision.py` слоя `posthoc`.
#
# Этот файл отвечает только за:
# - post-hoc scoring, routing и final decision policy;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `posthoc` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Final

import pandas as pd

from exohost.posthoc.candidate_ood_policy import (
    DEFAULT_CANDIDATE_OOD_POLICY,
    CandidateOodPolicy,
    decide_candidate_ood_disposition,
)
from exohost.posthoc.id_ood_gate import (
    ID_OOD_CANDIDATE_OOD_STATE,
    ID_OOD_IN_DOMAIN_STATE,
    ID_OOD_OOD_STATE,
)
from exohost.posthoc.refinement_handoff import (
    DEFAULT_REFINEMENT_HANDOFF_POLICY,
    RefinementHandoffPolicy,
    decide_refinement_handoff,
)

FINAL_DOMAIN_ID: Final[str] = "id"
FINAL_DOMAIN_CANDIDATE_OOD: Final[str] = "candidate_ood"
FINAL_DOMAIN_OOD: Final[str] = "ood"
FINAL_DOMAIN_UNKNOWN: Final[str] = "unknown"

FINAL_REFINEMENT_NOT_ATTEMPTED: Final[str] = "not_attempted"
FINAL_REFINEMENT_ACCEPTED: Final[str] = "accepted"
FINAL_REFINEMENT_REJECTED_TO_UNKNOWN: Final[str] = "rejected_to_unknown"


@dataclass(frozen=True, slots=True)
class FinalDecisionPolicy:
    # Версионированный policy contract для final routing layer.
    decision_policy_version: str
    refinement_handoff_policy: RefinementHandoffPolicy = DEFAULT_REFINEMENT_HANDOFF_POLICY
    candidate_ood_policy: CandidateOodPolicy = DEFAULT_CANDIDATE_OOD_POLICY
    min_refinement_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.min_refinement_confidence is None:
            return
        if not 0.0 <= self.min_refinement_confidence <= 1.0:
            raise ValueError(
                "FinalDecisionPolicy.min_refinement_confidence must be between 0 and 1."
            )


def build_final_decision_frame(
    df: pd.DataFrame,
    *,
    policy: FinalDecisionPolicy,
) -> pd.DataFrame:
    # Преобразуем upstream stage outputs в final decision contract.
    _require_final_decision_columns(df)

    rows = [
        _build_final_decision_row(_normalize_record_mapping(row), policy=policy)
        for row in df.to_dict(orient="records")
    ]
    return pd.DataFrame.from_records(rows, index=df.index)


def _build_final_decision_row(
    row: Mapping[str, object],
    *,
    policy: FinalDecisionPolicy,
) -> dict[str, object]:
    source_id = row["source_id"]
    quality_state = _normalize_text(row["quality_state"])
    ood_decision = _normalize_text(row["ood_decision"])
    coarse_class = _normalize_optional_text(row.get("coarse_predicted_label"))
    coarse_probability = _to_optional_float(row.get("coarse_probability_max"))
    coarse_margin = _to_optional_float(row.get("coarse_probability_margin"))
    refinement_label = _normalize_optional_text(row.get("refinement_predicted_label"))
    refinement_confidence = _to_optional_float(row.get("refinement_probability_max"))

    result = {
        "source_id": source_id,
        "final_domain_state": FINAL_DOMAIN_UNKNOWN,
        "final_quality_state": quality_state,
        "final_coarse_class": pd.NA,
        "final_coarse_confidence": pd.NA,
        "final_refinement_label": pd.NA,
        "final_refinement_state": FINAL_REFINEMENT_NOT_ATTEMPTED,
        "final_refinement_confidence": pd.NA,
        "final_decision_reason": "quality_unknown",
        "final_decision_policy_version": policy.decision_policy_version,
        "priority_state": pd.NA,
    }

    if quality_state == "reject":
        result["final_decision_reason"] = "quality_reject"
        return result

    if quality_state == "unknown":
        result["final_decision_reason"] = "quality_unknown"
        return result

    if ood_decision == ID_OOD_OOD_STATE:
        result["final_domain_state"] = FINAL_DOMAIN_OOD
        result["final_decision_reason"] = "hard_ood"
        return result

    if ood_decision == ID_OOD_CANDIDATE_OOD_STATE:
        candidate_ood_decision = decide_candidate_ood_disposition(
            policy=policy.candidate_ood_policy
        )
        result["final_domain_state"] = candidate_ood_decision.final_domain_state
        result["final_decision_reason"] = candidate_ood_decision.reason
        return result

    if ood_decision != ID_OOD_IN_DOMAIN_STATE:
        result["final_decision_reason"] = "unsupported_ood_state"
        return result

    result["final_domain_state"] = FINAL_DOMAIN_ID
    result["final_coarse_class"] = coarse_class if coarse_class is not None else pd.NA
    result["final_coarse_confidence"] = (
        coarse_probability if coarse_probability is not None else pd.NA
    )

    handoff_decision = decide_refinement_handoff(
        final_domain_state=FINAL_DOMAIN_ID,
        coarse_class=coarse_class,
        coarse_probability_max=coarse_probability,
        coarse_probability_margin=coarse_margin,
        policy=policy.refinement_handoff_policy,
    )
    if not handoff_decision.should_attempt_refinement:
        result["final_decision_reason"] = handoff_decision.reason
        return result

    if refinement_label is None:
        result["final_domain_state"] = FINAL_DOMAIN_UNKNOWN
        result["final_refinement_state"] = FINAL_REFINEMENT_REJECTED_TO_UNKNOWN
        result["final_decision_reason"] = "refinement_label_missing"
        return result

    if (
        policy.min_refinement_confidence is not None
        and refinement_confidence is not None
        and refinement_confidence < policy.min_refinement_confidence
    ):
        result["final_domain_state"] = FINAL_DOMAIN_UNKNOWN
        result["final_refinement_state"] = FINAL_REFINEMENT_REJECTED_TO_UNKNOWN
        result["final_decision_reason"] = "refinement_confidence_below_threshold"
        return result

    result["final_refinement_label"] = refinement_label
    result["final_refinement_state"] = FINAL_REFINEMENT_ACCEPTED
    result["final_refinement_confidence"] = (
        refinement_confidence if refinement_confidence is not None else pd.NA
    )
    result["final_decision_reason"] = "refinement_accepted"
    return result


def _normalize_record_mapping(row: Mapping[Hashable, object]) -> dict[str, object]:
    return {str(key): value for key, value in row.items()}


def _require_final_decision_columns(df: pd.DataFrame) -> None:
    required_columns = (
        "source_id",
        "quality_state",
        "ood_decision",
        "coarse_predicted_label",
        "coarse_probability_max",
        "coarse_probability_margin",
        "refinement_predicted_label",
        "refinement_probability_max",
    )
    missing_columns = [
        column_name for column_name in required_columns if column_name not in df.columns
    ]
    if missing_columns:
        missing_columns_sql = ", ".join(missing_columns)
        raise ValueError(
            "Final decision frame is missing required columns: "
            f"{missing_columns_sql}"
        )


def _normalize_text(value: object) -> str:
    return str(value).strip().lower()


def _normalize_optional_text(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    normalized_value = str(value).strip()
    return normalized_value if normalized_value else None


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
