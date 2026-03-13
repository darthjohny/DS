"""Точечные тесты standalone OOD-policy для router."""

from __future__ import annotations

import math

from router_model.ood import (
    build_unknown_router_score,
    evaluate_router_ood,
    load_ood_policy,
)


def test_load_ood_policy_returns_disabled_defaults_for_legacy_meta() -> None:
    """Legacy artifact без OOD metadata должен читаться как disabled-policy."""
    policy = load_ood_policy({})

    assert policy.allow_unknown is False
    assert policy.policy_version == "disabled"
    assert policy.min_router_log_posterior is None
    assert policy.min_posterior_margin is None
    assert policy.min_router_similarity is None


def test_evaluate_router_ood_rejects_low_posterior_when_unknown_enabled() -> None:
    """OOD policy должна отвергать результат по явному posterior-порогу."""
    decision = evaluate_router_ood(
        result={
            "predicted_spec_class": "K",
            "predicted_evolution_stage": "dwarf",
            "router_label": "K_dwarf",
            "d_mahal_router": 0.2,
            "router_similarity": 0.8,
            "router_log_likelihood": -1.8,
            "router_log_posterior": -6.5,
            "second_best_label": "M_dwarf",
            "margin": 0.2,
            "posterior_margin": 0.3,
            "model_version": "gaussian_router_v1",
        },
        meta={
            "allow_unknown": True,
            "ood_policy_version": "posterior_reject_v1",
            "min_router_log_posterior": -6.0,
        },
    )

    assert decision.should_reject is True
    assert decision.reject_reason == "LOW_POSTERIOR"


def test_evaluate_router_ood_requires_both_margin_and_similarity_for_joint_rule() -> None:
    """Joint rule должна срабатывать только когда оба сигнала ниже порога."""
    accepted = evaluate_router_ood(
        result={
            "predicted_spec_class": "K",
            "predicted_evolution_stage": "dwarf",
            "router_label": "K_dwarf",
            "d_mahal_router": 0.3,
            "router_similarity": 0.55,
            "router_log_likelihood": -1.2,
            "router_log_posterior": -2.0,
            "second_best_label": "M_dwarf",
            "margin": 0.2,
            "posterior_margin": 0.1,
            "model_version": "gaussian_router_v1",
        },
        meta={
            "allow_unknown": True,
            "ood_policy_version": "posterior_reject_v1",
            "min_router_log_posterior": -6.0,
            "min_posterior_margin": 0.2,
            "min_router_similarity": 0.4,
        },
    )
    rejected = evaluate_router_ood(
        result={
            "predicted_spec_class": "K",
            "predicted_evolution_stage": "dwarf",
            "router_label": "K_dwarf",
            "d_mahal_router": 0.3,
            "router_similarity": 0.25,
            "router_log_likelihood": -1.2,
            "router_log_posterior": -2.0,
            "second_best_label": "M_dwarf",
            "margin": 0.2,
            "posterior_margin": 0.1,
            "model_version": "gaussian_router_v1",
        },
        meta={
            "allow_unknown": True,
            "ood_policy_version": "posterior_reject_v1",
            "min_router_log_posterior": -6.0,
            "min_posterior_margin": 0.2,
            "min_router_similarity": 0.4,
        },
    )

    assert accepted.should_reject is False
    assert accepted.reject_reason is None
    assert rejected.should_reject is True
    assert rejected.reject_reason == "LOW_MARGIN_AND_SIMILARITY"


def test_build_unknown_router_score_preserves_existing_diagnostics() -> None:
    """Canonical UNKNOWN result не должен затирать уже посчитанные diagnostics."""
    result = build_unknown_router_score(
        model_version="gaussian_router_v1",
        diagnostics={
            "d_mahal_router": 1.7,
            "router_similarity": 0.33,
            "router_log_likelihood": -3.2,
            "router_log_posterior": -4.4,
            "second_best_label": "G_dwarf",
            "margin": 0.05,
            "posterior_margin": 0.07,
        },
    )

    assert result["predicted_spec_class"] == "UNKNOWN"
    assert result["predicted_evolution_stage"] == "unknown"
    assert result["router_label"] == "UNKNOWN"
    assert result["d_mahal_router"] == 1.7
    assert result["router_similarity"] == 0.33
    assert result["router_log_likelihood"] == -3.2
    assert result["router_log_posterior"] == -4.4
    assert result["second_best_label"] == "G_dwarf"
    assert result["margin"] == 0.05
    assert result["posterior_margin"] == 0.07
    assert result["model_version"] == "gaussian_router_v1"


def test_build_unknown_router_score_uses_nan_defaults_for_missing_diagnostics() -> None:
    """Без diagnostics canonical UNKNOWN result должен использовать safe fallback values."""
    result = build_unknown_router_score(model_version="gaussian_router_v1")

    assert result["router_similarity"] == 0.0
    assert result["second_best_label"] == "UNKNOWN"
    assert math.isnan(result["d_mahal_router"])
    assert math.isnan(result["router_log_likelihood"])
    assert math.isnan(result["router_log_posterior"])
    assert math.isnan(result["margin"])
    assert math.isnan(result["posterior_margin"])
