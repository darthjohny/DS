"""Тесты для orchestration и decision layer."""

from __future__ import annotations

import math
from typing import Any, Dict

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
    assert stub["similarity"].isna().all()


def test_run_host_similarity_applies_decision_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host-ветка должна применять все множители к similarity."""

    def fake_score_host_df(
        model: Dict[str, Any],
        df: pd.DataFrame,
        spec_class_col: str = "spec_class",
    ) -> pd.DataFrame:
        result = df.copy()
        result["gauss_label"] = result[spec_class_col]
        result["d_mahal"] = 0.5
        result["similarity"] = 0.5
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
    host_model: Dict[str, Any] = {
        "meta": {
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
