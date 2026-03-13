"""Reject-option и open-set policy для Gaussian router.

Модуль инкапсулирует правила, по которым результат router scoring может
быть переведён в `UNKNOWN`, не раздувая `score.py` условными ветками и
не захардкоживая пороги внутри основного scoring-контура.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from router_model.artifacts import (
    DISABLED_OOD_POLICY_VERSION,
    RouterScoreResult,
)
from router_model.labels import (
    UNKNOWN_EVOLUTION_STAGE,
    UNKNOWN_ROUTER_LABEL,
    UNKNOWN_SPEC_CLASS,
)


@dataclass(frozen=True)
class RouterOODPolicy:
    """Пороговая policy для reject-option поверх raw router scoring."""

    allow_unknown: bool
    policy_version: str
    min_router_log_posterior: float | None
    min_posterior_margin: float | None
    min_router_similarity: float | None


@dataclass(frozen=True)
class RouterOODDecision:
    """Итог решения open-set слоя для одной строки."""

    should_reject: bool
    reject_reason: str | None


def _optional_float(value: Any) -> float | None:
    """Преобразовать значение в `float | None` для OOD metadata."""
    if value is None:
        return None
    numeric = float(value)
    if math.isnan(numeric):
        return None
    return numeric


def load_ood_policy(meta: Mapping[str, Any]) -> RouterOODPolicy:
    """Прочитать OOD policy из metadata router artifact.

    Для старых artifact, где поля OOD ещё не добавлены, функция
    возвращает безопасный режим с `allow_unknown=False`.
    """
    return RouterOODPolicy(
        allow_unknown=bool(meta.get("allow_unknown", False)),
        policy_version=str(
            meta.get("ood_policy_version", DISABLED_OOD_POLICY_VERSION)
        ),
        min_router_log_posterior=_optional_float(
            meta.get("min_router_log_posterior")
        ),
        min_posterior_margin=_optional_float(meta.get("min_posterior_margin")),
        min_router_similarity=_optional_float(meta.get("min_router_similarity")),
    )


def build_unknown_router_score(
    model_version: str,
    diagnostics: Mapping[str, Any] | None = None,
) -> RouterScoreResult:
    """Собрать canonical router result для `UNKNOWN`.

    Если в `diagnostics` уже есть численные поля raw scoring, функция
    сохраняет их, а не затирает, чтобы QA и EDA могли анализировать
    rejected rows.
    """
    diagnostics = diagnostics or {}
    return {
        "predicted_spec_class": UNKNOWN_SPEC_CLASS,
        "predicted_evolution_stage": UNKNOWN_EVOLUTION_STAGE,
        "router_label": UNKNOWN_ROUTER_LABEL,
        "d_mahal_router": float(
            diagnostics.get("d_mahal_router", float("nan"))
        ),
        "router_similarity": float(
            diagnostics.get("router_similarity", 0.0)
        ),
        "router_log_likelihood": float(
            diagnostics.get("router_log_likelihood", float("nan"))
        ),
        "router_log_posterior": float(
            diagnostics.get("router_log_posterior", float("nan"))
        ),
        "second_best_label": str(
            diagnostics.get("second_best_label", UNKNOWN_ROUTER_LABEL)
        ),
        "margin": float(diagnostics.get("margin", float("nan"))),
        "posterior_margin": float(
            diagnostics.get("posterior_margin", float("nan"))
        ),
        "model_version": model_version,
    }


def _is_below_threshold(value: Any, threshold: float | None) -> bool:
    """Проверить, что значение не проходит нижний порог уверенности."""
    if threshold is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return True
    return not math.isfinite(numeric) or numeric < threshold


def evaluate_router_ood(
    result: RouterScoreResult,
    meta: Mapping[str, Any],
) -> RouterOODDecision:
    """Принять open-set решение для уже посчитанного router result."""
    policy = load_ood_policy(meta)
    if not policy.allow_unknown:
        return RouterOODDecision(should_reject=False, reject_reason=None)

    if _is_below_threshold(
        result["router_log_posterior"],
        policy.min_router_log_posterior,
    ):
        return RouterOODDecision(
            should_reject=True,
            reject_reason="LOW_POSTERIOR",
        )

    margin_rule_enabled = (
        policy.min_posterior_margin is not None
        and policy.min_router_similarity is not None
    )
    if margin_rule_enabled and (
        _is_below_threshold(
            result["posterior_margin"],
            policy.min_posterior_margin,
        )
        and _is_below_threshold(
            result["router_similarity"],
            policy.min_router_similarity,
        )
    ):
        return RouterOODDecision(
            should_reject=True,
            reject_reason="LOW_MARGIN_AND_SIMILARITY",
        )

    return RouterOODDecision(should_reject=False, reject_reason=None)


def apply_ood_policy(
    result: RouterScoreResult,
    meta: Mapping[str, Any],
) -> RouterScoreResult:
    """Применить OOD policy к raw router result."""
    decision = evaluate_router_ood(result=result, meta=meta)
    if not decision.should_reject:
        return result
    return build_unknown_router_score(
        model_version=result["model_version"],
        diagnostics=result,
    )


__all__ = [
    "RouterOODDecision",
    "RouterOODPolicy",
    "apply_ood_policy",
    "build_unknown_router_score",
    "evaluate_router_ood",
    "load_ood_policy",
]
