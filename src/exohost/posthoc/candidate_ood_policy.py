# Файл `candidate_ood_policy.py` слоя `posthoc`.
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
from typing import Literal

CandidateOodDisposition = Literal["keep_candidate_ood", "map_to_unknown"]

CANDIDATE_OOD_KEEP_DISPOSITION: CandidateOodDisposition = "keep_candidate_ood"
CANDIDATE_OOD_MAP_TO_UNKNOWN_DISPOSITION: CandidateOodDisposition = "map_to_unknown"


@dataclass(frozen=True, slots=True)
class CandidateOodPolicy:
    # Версионированный project policy для secondary routing candidate_ood.
    disposition: CandidateOodDisposition = CANDIDATE_OOD_KEEP_DISPOSITION
    policy_version: str = "candidate_ood_v1"


DEFAULT_CANDIDATE_OOD_POLICY = CandidateOodPolicy()


@dataclass(frozen=True, slots=True)
class CandidateOodDecision:
    # Явный результат secondary routing для candidate_ood.
    final_domain_state: Literal["candidate_ood", "unknown"]
    reason: str


def decide_candidate_ood_disposition(
    *,
    policy: CandidateOodPolicy = DEFAULT_CANDIDATE_OOD_POLICY,
) -> CandidateOodDecision:
    # Candidate OOD не схлопываем с clean ID; допускаем только два contract-state.
    if policy.disposition == CANDIDATE_OOD_KEEP_DISPOSITION:
        return CandidateOodDecision(
            final_domain_state="candidate_ood",
            reason="candidate_ood",
        )
    return CandidateOodDecision(
        final_domain_state="unknown",
        reason="candidate_ood_mapped_to_unknown",
    )
