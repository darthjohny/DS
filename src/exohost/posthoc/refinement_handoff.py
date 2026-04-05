# Файл `refinement_handoff.py` слоя `posthoc`.
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
from typing import Final

REFINEMENT_HANDOFF_ALLOWED: Final[str] = "refinement_allowed"
REFINEMENT_HANDOFF_DOMAIN_BLOCKED: Final[str] = "domain_not_id"
REFINEMENT_HANDOFF_CLASS_BLOCKED: Final[str] = "coarse_class_not_enabled"
REFINEMENT_HANDOFF_PROBABILITY_BLOCKED: Final[str] = (
    "coarse_probability_below_threshold"
)
REFINEMENT_HANDOFF_MARGIN_BLOCKED: Final[str] = "coarse_margin_below_threshold"

DEFAULT_REFINEMENT_ENABLED_CLASSES: Final[frozenset[str]] = frozenset(
    {"A", "B", "F", "G", "K", "M"}
)


@dataclass(frozen=True, slots=True)
class RefinementHandoffPolicy:
    # Явная policy-обертка для second-wave handoff.
    enabled_classes: frozenset[str] = DEFAULT_REFINEMENT_ENABLED_CLASSES
    min_coarse_probability: float | None = None
    min_coarse_margin: float | None = None

    def __post_init__(self) -> None:
        if self.min_coarse_probability is not None and not 0.0 <= self.min_coarse_probability <= 1.0:
            raise ValueError(
                "RefinementHandoffPolicy.min_coarse_probability must be between 0 and 1."
            )
        if self.min_coarse_margin is not None and not 0.0 <= self.min_coarse_margin <= 1.0:
            raise ValueError(
                "RefinementHandoffPolicy.min_coarse_margin must be between 0 and 1."
            )


DEFAULT_REFINEMENT_HANDOFF_POLICY = RefinementHandoffPolicy()


@dataclass(frozen=True, slots=True)
class RefinementHandoffDecision:
    # Результат handoff без смешивания с final routing.
    should_attempt_refinement: bool
    reason: str


def decide_refinement_handoff(
    *,
    final_domain_state: str,
    coarse_class: str | None,
    coarse_probability_max: float | None = None,
    coarse_probability_margin: float | None = None,
    policy: RefinementHandoffPolicy = DEFAULT_REFINEMENT_HANDOFF_POLICY,
) -> RefinementHandoffDecision:
    # Refinement разрешаем только для clean ID и coarse classes из enable-list.
    if final_domain_state != "id":
        return RefinementHandoffDecision(
            should_attempt_refinement=False,
            reason=REFINEMENT_HANDOFF_DOMAIN_BLOCKED,
        )

    normalized_class = _normalize_coarse_class(coarse_class)
    if normalized_class is None or normalized_class not in policy.enabled_classes:
        return RefinementHandoffDecision(
            should_attempt_refinement=False,
            reason=REFINEMENT_HANDOFF_CLASS_BLOCKED,
        )

    if (
        policy.min_coarse_probability is not None
        and coarse_probability_max is not None
        and coarse_probability_max < policy.min_coarse_probability
    ):
        return RefinementHandoffDecision(
            should_attempt_refinement=False,
            reason=REFINEMENT_HANDOFF_PROBABILITY_BLOCKED,
        )

    if (
        policy.min_coarse_margin is not None
        and coarse_probability_margin is not None
        and coarse_probability_margin < policy.min_coarse_margin
    ):
        return RefinementHandoffDecision(
            should_attempt_refinement=False,
            reason=REFINEMENT_HANDOFF_MARGIN_BLOCKED,
        )

    return RefinementHandoffDecision(
        should_attempt_refinement=True,
        reason=REFINEMENT_HANDOFF_ALLOWED,
    )


def _normalize_coarse_class(coarse_class: str | None) -> str | None:
    if coarse_class is None:
        return None
    normalized_class = coarse_class.strip().upper()
    return normalized_class or None
