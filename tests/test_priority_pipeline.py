"""Минимальный end-to-end тест боевого pipeline."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
from sqlalchemy.engine import Engine

import priority_pipeline.pipeline as pipeline


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
        ]
    )

    def fake_load_input_candidates(
        engine: object,
        source_name: str,
        limit: int | None = None,
    ) -> pd.DataFrame:
        assert source_name == "qa.synthetic_batch"
        assert limit == 3
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
        result["predicted_spec_class"] = ["K", "A", "K"]
        result["predicted_evolution_stage"] = ["dwarf", "dwarf", "evolved"]
        result["router_label"] = ["K_dwarf", "A_dwarf", "K_evolved"]
        result["second_best_label"] = ["M_dwarf", "F_dwarf", "G_evolved"]
        result["d_mahal_router"] = [0.2, 0.8, 0.7]
        result["router_similarity"] = [0.95, 0.35, 0.40]
        result["router_log_likelihood"] = [-0.10, -1.50, -1.00]
        result["router_log_posterior"] = [-0.10, -1.50, -1.00]
        result["margin"] = [0.40, 0.10, 0.20]
        result["posterior_margin"] = [0.60, 0.20, 0.30]
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
        limit=3,
        persist=False,
    )

    assert result.run_id == "run_test_1"
    assert len(result.router_results) == 3
    assert len(result.priority_results) == 3
    assert result.router_results["run_id"].tolist() == ["run_test_1"] * 3
    assert result.priority_results["run_id"].tolist() == ["run_test_1"] * 3
    assert result.priority_results["source_id"].tolist() == [1, 3, 2]
    assert result.priority_results["priority_tier"].tolist() == [
        "HIGH",
        "LOW",
        "LOW",
    ]
    assert result.priority_results["reason_code"].tolist() == [
        "HOST_SCORING",
        "EVOLVED_STAR",
        "HOT_STAR",
    ]
    assert result.priority_results["final_score"].tolist() == [0.72, 0.0, 0.0]
    assert result.priority_results["host_posterior"].notna().sum() == 1
