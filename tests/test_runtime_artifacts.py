"""Регрессии на production JSON-артефакты в каталоге data/."""

from __future__ import annotations

from pathlib import Path

from gaussian_router import load_router_model
from model_gaussian import load_model, validate_host_model_artifact

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROUTER_ARTIFACT = PROJECT_ROOT / "data" / "router_gaussian_params.json"
HOST_ARTIFACT = PROJECT_ROOT / "data" / "model_gaussian_params.json"


def test_router_artifact_matches_posterior_contract() -> None:
    """Router artifact должен быть пересобран в posterior-aware формате."""
    model = load_router_model(str(ROUTER_ARTIFACT))

    assert model["meta"]["score_mode"] == "gaussian_log_posterior_v1"
    assert model["meta"]["prior_mode"] == "uniform"
    for params in model["classes"].values():
        assert "effective_cov" in params
        assert "log_det_cov" in params


def test_host_artifact_matches_contrastive_contract() -> None:
    """Host artifact должен соответствовать новому host-vs-field runtime contract."""
    model = load_model(str(HOST_ARTIFACT))

    validate_host_model_artifact(model)
    assert model["meta"]["model_version"] == "gaussian_host_field_v1"
    assert model["meta"]["score_mode"] == "host_vs_field_log_lr_v1"
