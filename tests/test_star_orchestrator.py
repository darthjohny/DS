"""Тесты для orchestration и decision layer."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import star_orchestrator as orchestrator


def test_split_branches_only_mkgf_dwarfs_go_to_host() -> None:
    """В host-ветку должны попадать только MKGF dwarf."""
    df_router = pd.DataFrame(
        [
            {
                "source_id": 1,
                "predicted_spec_class": "M",
                "predicted_evolution_stage": "dwarf",
            },
            {
                "source_id": 2,
                "predicted_spec_class": "A",
                "predicted_evolution_stage": "dwarf",
            },
            {
                "source_id": 3,
                "predicted_spec_class": "K",
                "predicted_evolution_stage": "evolved",
            },
        ]
    )

    df_host, df_low = orchestrator.split_branches(df_router)

    assert list(df_host["source_id"]) == [1]
    assert list(df_low["source_id"]) == [2, 3]


def test_build_low_priority_stub_sets_reason_codes() -> None:
    """A/B/O и evolved должны получать корректные заглушки."""
    df_low = pd.DataFrame(
        [
            {
                "source_id": 1,
                "predicted_spec_class": "A",
                "predicted_evolution_stage": "dwarf",
                "router_label": "A_dwarf",
                "d_mahal_router": 0.7,
                "router_similarity": 0.6,
                "validation_factor": 1.0,
            },
            {
                "source_id": 2,
                "predicted_spec_class": "K",
                "predicted_evolution_stage": "evolved",
                "router_label": "K_evolved",
                "d_mahal_router": 0.9,
                "router_similarity": 0.5,
                "validation_factor": 1.0,
            },
        ]
    )

    stub = orchestrator.build_low_priority_stub(df_low)

    assert stub["priority_tier"].tolist() == ["LOW", "LOW"]
    assert stub["reason_code"].tolist() == ["HOT_STAR", "EVOLVED_STAR"]
    assert stub["final_score"].tolist() == [0.0, 0.0]
    assert stub["gauss_label"].isna().all()
    assert stub["host_posterior"].isna().all()
    assert stub["host_log_lr"].isna().all()
    assert stub["similarity"].isna().all()


def test_run_host_similarity_applies_decision_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host-ветка должна применять все множители к host_posterior."""

    def fake_score_host_df(
        model: dict[str, Any],
        df: pd.DataFrame,
        spec_class_col: str = "spec_class",
    ) -> pd.DataFrame:
        result = df.copy()
        result["gauss_label"] = result[spec_class_col]
        result["host_log_likelihood"] = -1.0
        result["field_log_likelihood"] = -2.0
        result["host_log_lr"] = 1.0
        result["host_posterior"] = 0.5
        return result

    monkeypatch.setattr(orchestrator, "score_host_df", fake_score_host_df)

    df_host = pd.DataFrame(
        [
            {
                "source_id": 1,
                "predicted_spec_class": "M",
                "predicted_evolution_stage": "dwarf",
                "router_label": "M_dwarf",
                "d_mahal_router": 0.3,
                "router_similarity": 0.9,
                "ruwe": 1.0,
                "parallax_over_error": 20.0,
                "parallax": 15.0,
                "mh_gspphot": 0.1,
                "bp_rp": 1.2,
                "validation_factor": 0.9,
                "teff_gspphot": 3500.0,
                "logg_gspphot": 4.8,
                "radius_gspphot": 0.45,
            }
        ]
    )
    host_model: dict[str, Any] = {
        "meta": {
            "model_version": "gaussian_host_field_v1",
            "score_mode": "host_vs_field_log_lr_v1",
            "shrink_alpha": 0.15,
            "use_m_subclasses": True,
        }
    }

    scored = orchestrator.run_host_similarity(
        df_host=df_host,
        host_model=host_model,
    )
    expected = orchestrator.clip_unit_interval(
        0.5
        * orchestrator.class_prior("M")
        * orchestrator.quality_factor(1.0, 20.0, 15.0)
        * orchestrator.metallicity_factor(0.1)
        * orchestrator.color_factor(1.2)
        * orchestrator.normalized_validation_factor(0.9)
    )
    final_score_value: Any = scored.at[0, "final_score"]
    priority_tier_value: Any = scored.at[0, "priority_tier"]
    reason_code_value: Any = scored.at[0, "reason_code"]
    host_log_lr_value: Any = scored.at[0, "host_log_lr"]
    host_posterior_value: Any = scored.at[0, "host_posterior"]
    similarity_value: Any = scored.at[0, "similarity"]

    assert math.isclose(
        float(final_score_value),
        expected,
        rel_tol=1e-9,
        abs_tol=1e-12,
    )
    assert str(priority_tier_value) == orchestrator.priority_tier_from_score(
        expected
    )
    assert str(reason_code_value) == "HOST_SCORING"
    assert math.isclose(float(host_log_lr_value), 1.0, rel_tol=1e-9)
    assert math.isclose(float(host_posterior_value), 0.5, rel_tol=1e-9)
    assert pd.isna(similarity_value)


def test_load_models_rejects_legacy_host_artifact(tmp_path: Path) -> None:
    """load_models должен явно отклонять устаревший legacy host artifact."""
    router_model = {
        "global_mu": [0.0, 0.0, 0.0],
        "global_sigma": [1.0, 1.0, 1.0],
        "features": ["teff_gspphot", "logg_gspphot", "radius_gspphot"],
        "classes": {
            "K_dwarf": {
                "n": 3,
                "spec_class": "K",
                "evolution_stage": "dwarf",
                "mu": [0.0, 0.0, 0.0],
                "cov": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "effective_cov": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "inv_cov": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "log_det_cov": 0.0,
            }
        },
        "meta": {
            "model_version": "gaussian_router_v1",
            "source_view": "lab.v_gaia_router_training",
            "shrink_alpha": 0.15,
            "min_class_size": 3,
            "score_mode": "gaussian_log_posterior_v1",
            "prior_mode": "uniform",
        },
    }
    legacy_host_model = {
        "global_mu": [0.0, 0.0, 0.0],
        "global_sigma": [1.0, 1.0, 1.0],
        "features": ["teff_gspphot", "logg_gspphot", "radius_gspphot"],
        "classes": {
            "K": {
                "n": 3,
                "mu": [0.0, 0.0, 0.0],
                "cov": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "inv_cov": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            }
        },
        "meta": {
            "logg_dwarf_min": 4.0,
            "use_m_subclasses": True,
            "shrink_alpha": 0.15,
        },
    }

    router_path: Path = tmp_path / "router.json"
    host_path: Path = tmp_path / "host.json"
    router_path.write_text(json.dumps(router_model), encoding="utf-8")
    host_path.write_text(json.dumps(legacy_host_model), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy"):
        orchestrator.load_models(
            router_model_path=router_path,
            host_model_path=host_path,
        )


def test_build_persist_payload_drops_unknown_db_columns() -> None:
    """Persist payload должен отбрасывать поля, которых нет в текущей таблице."""
    df = pd.DataFrame(
        [
            {
                "run_id": "run_1",
                "source_id": 1,
                "predicted_spec_class": "K",
                "predicted_evolution_stage": "dwarf",
                "router_label": "K_dwarf",
                "final_score": 0.8,
                "priority_tier": "HIGH",
                "reason_code": "HOST_SCORING",
                "host_posterior": 0.9,
            }
        ]
    )

    payload = orchestrator.build_persist_payload(
        df=df,
        ordered_columns=orchestrator.PRIORITY_RESULTS_COLUMNS,
        available_columns=[
            "run_id",
            "source_id",
            "predicted_spec_class",
            "predicted_evolution_stage",
            "router_label",
            "final_score",
            "priority_tier",
            "reason_code",
        ],
        required_columns=orchestrator.PRIORITY_REQUIRED_DB_COLUMNS,
        table_name="lab.gaia_priority_results",
    )

    assert list(payload.columns) == [
        "run_id",
        "source_id",
        "predicted_spec_class",
        "predicted_evolution_stage",
        "router_label",
        "final_score",
        "priority_tier",
        "reason_code",
    ]


def test_build_persist_payload_requires_minimum_db_schema() -> None:
    """Persist payload должен явно падать, если в таблице нет ключевых колонок."""
    df = pd.DataFrame(
        [
            {
                "run_id": "run_1",
                "source_id": 1,
                "predicted_spec_class": "K",
                "predicted_evolution_stage": "dwarf",
                "router_label": "K_dwarf",
                "final_score": 0.8,
                "priority_tier": "HIGH",
                "reason_code": "HOST_SCORING",
            }
        ]
    )

    with pytest.raises(RuntimeError, match="missing required columns"):
        orchestrator.build_persist_payload(
            df=df,
            ordered_columns=orchestrator.PRIORITY_RESULTS_COLUMNS,
            available_columns=[
                "run_id",
                "source_id",
                "predicted_spec_class",
            ],
            required_columns=orchestrator.PRIORITY_REQUIRED_DB_COLUMNS,
            table_name="lab.gaia_priority_results",
        )


def test_build_persist_payload_keeps_full_router_schema_when_available() -> None:
    """При полной схеме router persist должен писать весь текущий контракт."""
    row: dict[str, object | None] = {
        column: None for column in orchestrator.ROUTER_RESULTS_COLUMNS
    }
    row.update(
        {
            "run_id": "run_1",
            "source_id": 1,
            "ra": 10.0,
            "dec": -5.0,
            "predicted_spec_class": "K",
            "predicted_evolution_stage": "dwarf",
            "router_label": "K_dwarf",
            "router_model_version": "router_v2",
        }
    )
    df = pd.DataFrame([row])

    payload = orchestrator.build_persist_payload(
        df=df,
        ordered_columns=orchestrator.ROUTER_RESULTS_COLUMNS,
        available_columns=list(orchestrator.ROUTER_RESULTS_COLUMNS),
        required_columns=orchestrator.ROUTER_REQUIRED_DB_COLUMNS,
        table_name="lab.gaia_router_results",
    )

    assert list(payload.columns) == list(orchestrator.ROUTER_RESULTS_COLUMNS)


def test_build_persist_payload_keeps_full_priority_schema_when_available() -> None:
    """При полной схеме priority persist должен писать весь текущий контракт."""
    row: dict[str, object | None] = {
        column: None for column in orchestrator.PRIORITY_RESULTS_COLUMNS
    }
    row.update(
        {
            "run_id": "run_1",
            "source_id": 1,
            "ra": 10.0,
            "dec": -5.0,
            "predicted_spec_class": "K",
            "predicted_evolution_stage": "dwarf",
            "router_label": "K_dwarf",
            "final_score": 0.8,
            "priority_tier": "HIGH",
            "reason_code": "HOST_SCORING",
            "router_model_version": "router_v2",
            "host_model_version": "host_v2",
        }
    )
    df = pd.DataFrame([row])

    payload = orchestrator.build_persist_payload(
        df=df,
        ordered_columns=orchestrator.PRIORITY_RESULTS_COLUMNS,
        available_columns=list(orchestrator.PRIORITY_RESULTS_COLUMNS),
        required_columns=orchestrator.PRIORITY_REQUIRED_DB_COLUMNS,
        table_name="lab.gaia_priority_results",
    )

    assert list(payload.columns) == list(orchestrator.PRIORITY_RESULTS_COLUMNS)
