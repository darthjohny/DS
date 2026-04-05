# Тестовый файл `test_posthoc_candidate_ood_policy.py` домена `posthoc`.
#
# Этот файл проверяет только:
# - проверку логики домена: post-hoc routing, gate и final decision policy;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `posthoc` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.posthoc.candidate_ood_policy import (
    CANDIDATE_OOD_KEEP_DISPOSITION,
    CANDIDATE_OOD_MAP_TO_UNKNOWN_DISPOSITION,
    CandidateOodPolicy,
    decide_candidate_ood_disposition,
)


def test_decide_candidate_ood_disposition_keeps_candidate_state_by_default() -> None:
    decision = decide_candidate_ood_disposition(
        policy=CandidateOodPolicy(disposition=CANDIDATE_OOD_KEEP_DISPOSITION)
    )

    assert decision.final_domain_state == "candidate_ood"
    assert decision.reason == "candidate_ood"


def test_decide_candidate_ood_disposition_can_map_to_unknown() -> None:
    decision = decide_candidate_ood_disposition(
        policy=CandidateOodPolicy(disposition=CANDIDATE_OOD_MAP_TO_UNKNOWN_DISPOSITION)
    )

    assert decision.final_domain_state == "unknown"
    assert decision.reason == "candidate_ood_mapped_to_unknown"
