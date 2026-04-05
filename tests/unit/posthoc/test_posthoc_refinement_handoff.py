# Тестовый файл `test_posthoc_refinement_handoff.py` домена `posthoc`.
#
# Этот файл проверяет только:
# - проверку логики домена: post-hoc routing, gate и final decision policy;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `posthoc` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import pytest

from exohost.posthoc.refinement_handoff import (
    REFINEMENT_HANDOFF_ALLOWED,
    REFINEMENT_HANDOFF_CLASS_BLOCKED,
    REFINEMENT_HANDOFF_DOMAIN_BLOCKED,
    REFINEMENT_HANDOFF_MARGIN_BLOCKED,
    REFINEMENT_HANDOFF_PROBABILITY_BLOCKED,
    RefinementHandoffPolicy,
    decide_refinement_handoff,
)


def test_refinement_handoff_rejects_non_id_domain() -> None:
    decision = decide_refinement_handoff(
        final_domain_state="candidate_ood",
        coarse_class="G",
    )

    assert decision.should_attempt_refinement is False
    assert decision.reason == REFINEMENT_HANDOFF_DOMAIN_BLOCKED


def test_refinement_handoff_rejects_disabled_class() -> None:
    decision = decide_refinement_handoff(
        final_domain_state="id",
        coarse_class="O",
    )

    assert decision.should_attempt_refinement is False
    assert decision.reason == REFINEMENT_HANDOFF_CLASS_BLOCKED


def test_refinement_handoff_rejects_low_probability() -> None:
    decision = decide_refinement_handoff(
        final_domain_state="id",
        coarse_class="G",
        coarse_probability_max=0.54,
        policy=RefinementHandoffPolicy(min_coarse_probability=0.6),
    )

    assert decision.should_attempt_refinement is False
    assert decision.reason == REFINEMENT_HANDOFF_PROBABILITY_BLOCKED


def test_refinement_handoff_rejects_low_margin() -> None:
    decision = decide_refinement_handoff(
        final_domain_state="id",
        coarse_class="G",
        coarse_probability_margin=0.08,
        policy=RefinementHandoffPolicy(min_coarse_margin=0.1),
    )

    assert decision.should_attempt_refinement is False
    assert decision.reason == REFINEMENT_HANDOFF_MARGIN_BLOCKED


def test_refinement_handoff_accepts_enabled_class() -> None:
    decision = decide_refinement_handoff(
        final_domain_state="id",
        coarse_class="G",
        coarse_probability_max=0.78,
        coarse_probability_margin=0.21,
        policy=RefinementHandoffPolicy(
            min_coarse_probability=0.6,
            min_coarse_margin=0.1,
        ),
    )

    assert decision.should_attempt_refinement is True
    assert decision.reason == REFINEMENT_HANDOFF_ALLOWED


def test_refinement_handoff_policy_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        RefinementHandoffPolicy(min_coarse_probability=1.1)
