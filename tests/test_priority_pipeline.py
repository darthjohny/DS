"""Минимальный end-to-end тест боевого pipeline."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
from sqlalchemy.engine import Engine

import priority_pipeline.pipeline as pipeline
from priority_pipeline.branching import split_router_branches
from priority_pipeline.frame_contract import ensure_decision_columns


def test_split_router_branches_separates_unknown_from_known_low() -> None:
    """Branching должен разводить host, low_known и unknown отдельно."""
    df_router = pd.DataFrame(
        [
            {
                "source_id": 1,
                "predicted_spec_class": "K",
                "predicted_evolution_stage": "dwarf",
                "router_label": "K_dwarf",
            },
            {
                "source_id": 2,
                "predicted_spec_class": "A",
                "predicted_evolution_stage": "dwarf",
                "router_label": "A_dwarf",
            },
            {
                "source_id": 3,
                "predicted_spec_class": "UNKNOWN",
                "predicted_evolution_stage": "unknown",
                "router_label": "UNKNOWN",
            },
        ]
    )

    branches = split_router_branches(df_router)

    assert branches.host_df["source_id"].tolist() == [1]
    assert branches.low_known_df["source_id"].tolist() == [2]
    assert branches.unknown_df["source_id"].tolist() == [3]


def test_ensure_decision_columns_adds_neutral_defaults() -> None:
    """Frame contract должен добавлять мягкие decision defaults без перезаписи данных."""
    df = pd.DataFrame(
        [
            {
                "source_id": 1,
                "predicted_spec_class": "K",
                "bp_rp": 1.2,
                "validation_factor": 0.9,
            }
        ]
    )

    normalized = ensure_decision_columns(df)

    assert normalized["source_id"].tolist() == [1]
    assert normalized["predicted_spec_class"].tolist() == ["K"]
    assert normalized["bp_rp"].tolist() == [1.2]
    assert normalized["validation_factor"].tolist() == [0.9]
    assert normalized["parallax"].isna().all()
    assert normalized["parallax_over_error"].isna().all()
    assert normalized["ruwe"].isna().all()
    assert normalized["mh_gspphot"].isna().all()


def test_run_pipeline_mini_batch_end_to_end(
    monkeypatch: Any,
) -> None:
    """Pipeline должен пройти mini-batch и корректно собрать обе ветки."""
    df_input = pd.DataFrame(
        [
            {
                "source_id": 1,
                "ra": 10.0,
                "dec": -5.0,
                "teff_gspphot": 4600.0,
                "logg_gspphot": 4.7,
                "radius_gspphot": 0.80,
                "ruwe": 1.0,
                "parallax_over_error": 20.0,
                "parallax": 15.0,
                "bp_rp": 1.3,
                "mh_gspphot": 0.1,
                "validation_factor": 1.0,
            },
            {
                "source_id": 2,
                "ra": 20.0,
                "dec": 1.0,
                "teff_gspphot": 8200.0,
                "logg_gspphot": 4.1,
                "radius_gspphot": 1.90,
                "ruwe": 1.2,
                "parallax_over_error": 12.0,
                "parallax": 9.0,
                "bp_rp": 0.2,
                "mh_gspphot": -0.1,
                "validation_factor": 1.0,
            },
            {
                "source_id": 3,
                "ra": 30.0,
                "dec": 4.0,
                "teff_gspphot": 4700.0,
                "logg_gspphot": 3.2,
                "radius_gspphot": 4.10,
                "ruwe": 1.1,
                "parallax_over_error": 14.0,
                "parallax": 11.0,
                "bp_rp": 1.1,
                "mh_gspphot": 0.0,
                "validation_factor": 1.0,
            },
            {
                "source_id": 4,
                "ra": 40.0,
                "dec": 7.0,
                "teff_gspphot": None,
                "logg_gspphot": None,
                "radius_gspphot": None,
                "ruwe": 1.4,
                "parallax_over_error": None,
                "parallax": None,
                "bp_rp": None,
                "mh_gspphot": None,
                "validation_factor": 1.0,
            },
        ]
    )

    def fake_load_input_candidates(
        engine: object,
        source_name: str,
        limit: int | None = None,
    ) -> pd.DataFrame:
        assert source_name == "qa.synthetic_batch"
        assert limit == 4
        return df_input.copy()

    def fake_load_models(
        router_model_path: object,
        host_model_path: object,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            {"meta": {"model_version": "gaussian_router_v1"}},
            {
                "meta": {
                    "model_version": "gaussian_host_field_v1",
                    "score_mode": "host_vs_field_log_lr_v1",
                    "shrink_alpha": 0.15,
                    "use_m_subclasses": True,
                }
            },
        )

    def fake_run_router(
        df: pd.DataFrame,
        router_model: dict[str, Any],
    ) -> pd.DataFrame:
        result = df.copy()
        result["predicted_spec_class"] = ["K", "A", "K", "UNKNOWN"]
        result["predicted_evolution_stage"] = [
            "dwarf",
            "dwarf",
            "evolved",
            "unknown",
        ]
        result["router_label"] = [
            "K_dwarf",
            "A_dwarf",
            "K_evolved",
            "UNKNOWN",
        ]
        result["second_best_label"] = [
            "M_dwarf",
            "F_dwarf",
            "G_evolved",
            "UNKNOWN",
        ]
        result["d_mahal_router"] = [0.2, 0.8, 0.7, float("nan")]
        result["router_similarity"] = [0.95, 0.35, 0.40, 0.0]
        result["router_log_likelihood"] = [-0.10, -1.50, -1.00, float("nan")]
        result["router_log_posterior"] = [-0.10, -1.50, -1.00, float("nan")]
        result["margin"] = [0.40, 0.10, 0.20, float("nan")]
        result["posterior_margin"] = [0.60, 0.20, 0.30, float("nan")]
        result["router_model_version"] = "gaussian_router_v1"
        return result

    def fake_run_host_similarity(
        df_host: pd.DataFrame,
        host_model: dict[str, Any],
    ) -> pd.DataFrame:
        assert df_host["source_id"].tolist() == [1]
        result = df_host.copy()
        result["gauss_label"] = "K"
        result["host_log_likelihood"] = -0.50
        result["field_log_likelihood"] = -1.00
        result["host_log_lr"] = 0.50
        result["host_posterior"] = 0.80
        result["class_prior"] = 0.95
        result["quality_factor"] = 0.95
        result["metallicity_factor"] = 1.00
        result["color_factor"] = 1.00
        result["validation_factor"] = 1.00
        result["d_mahal"] = None
        result["similarity"] = None
        result["final_score"] = 0.72
        result["priority_tier"] = "HIGH"
        result["reason_code"] = "HOST_SCORING"
        result["host_model_version"] = (
            "gaussian_host_field_v1_"
            "host_vs_field_log_lr_v1_"
            "msub_True_shrink_0.15"
        )
        return result

    monkeypatch.setattr(
        pipeline,
        "load_input_candidates",
        fake_load_input_candidates,
    )
    monkeypatch.setattr(pipeline, "load_models", fake_load_models)
    monkeypatch.setattr(pipeline, "run_router", fake_run_router)
    monkeypatch.setattr(
        pipeline,
        "run_host_similarity",
        fake_run_host_similarity,
    )
    monkeypatch.setattr(pipeline, "make_run_id", lambda: "run_test_1")

    result = pipeline.run_pipeline(
        engine=cast(Engine, object()),
        input_source="qa.synthetic_batch",
        limit=4,
        persist=False,
    )

    assert result.run_id == "run_test_1"
    assert len(result.router_results) == 4
    assert len(result.priority_results) == 4
    assert result.router_results["run_id"].tolist() == ["run_test_1"] * 4
    assert result.priority_results["run_id"].tolist() == ["run_test_1"] * 4
    assert result.priority_results["source_id"].tolist() == [1, 3, 2, 4]
    assert result.priority_results["priority_tier"].tolist() == [
        "HIGH",
        "LOW",
        "LOW",
        "LOW",
    ]
    assert result.priority_results["reason_code"].tolist() == [
        "HOST_SCORING",
        "EVOLVED_STAR",
        "HOT_STAR",
        "ROUTER_UNKNOWN",
    ]
    assert result.priority_results["final_score"].tolist() == [
        0.72,
        0.0,
        0.0,
        0.0,
    ]
    assert result.priority_results["host_posterior"].notna().sum() == 1
