"""Тесты для офлайн-калибратора decision layer."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
import pandas as pd
import pytest

import decision_calibration.runtime as calibration_runtime
from decision_calibration import (
    CalibrationConfig,
    apply_calibration_config,
    build_low_priority_preview,
    build_unknown_preview,
    distance_factor,
    distance_pc_from_parallax,
    quality_factor,
    run_base_scoring,
)


def scalar_to_float(value: object) -> float:
    """Преобразовать pandas-скаляр во `float` с явной runtime-проверкой."""
    if isinstance(value, (Real, np.integer, np.floating)) and not isinstance(
        value,
        bool,
    ):
        return float(value)
    raise TypeError(f"Value is not float-compatible: {value!r}")


def test_apply_calibration_config_uses_host_posterior() -> None:
    """Калибратор должен строить final_score от host_posterior, не от similarity."""
    df_scored = pd.DataFrame(
        [
            {
                "source_id": 1,
                "predicted_spec_class": "M",
                "predicted_evolution_stage": "dwarf",
                "gauss_label": "M_EARLY",
                "host_log_likelihood": -1.0,
                "field_log_likelihood": -2.0,
                "host_log_lr": 1.0,
                "host_posterior": 0.80,
                "similarity": 0.01,
                "parallax": 20.0,
                "parallax_over_error": 25.0,
                "ruwe": 1.0,
                "mh_gspphot": 0.0,
            }
        ]
    )

    scored = apply_calibration_config(
        df_scored=df_scored,
        config=CalibrationConfig(),
        host_model_version_value="gaussian_host_field_v1",
    )

    expected = 0.80 * 1.02
    assert math.isclose(
        scalar_to_float(scored.at[0, "final_score"]),
        expected,
        rel_tol=1e-9,
        abs_tol=1e-12,
    )
    assert math.isclose(
        scalar_to_float(scored.at[0, "host_log_lr"]),
        1.0,
        rel_tol=1e-9,
    )
    assert math.isclose(
        scalar_to_float(scored.at[0, "host_posterior"]),
        0.80,
        rel_tol=1e-9,
    )
    assert pd.isna(scored.at[0, "similarity"])


def test_build_low_priority_preview_adds_contrastive_host_columns() -> None:
    """Low-priority preview должен нести новые host diagnostic поля как NULL."""
    df_low = pd.DataFrame(
        [
            {
                "source_id": 2,
                "predicted_spec_class": "A",
                "predicted_evolution_stage": "dwarf",
                "router_label": "A_dwarf",
            }
        ]
    )

    preview = build_low_priority_preview(df_low)

    assert preview["priority_tier"].tolist() == ["LOW"]
    assert preview["host_log_likelihood"].isna().all()
    assert preview["field_log_likelihood"].isna().all()
    assert preview["host_log_lr"].isna().all()
    assert preview["host_posterior"].isna().all()


def test_build_unknown_preview_sets_router_unknown_reason() -> None:
    """Unknown-preview должен сохранять отдельный reason-code и LOW tier."""
    df_unknown = pd.DataFrame(
        [
            {
                "source_id": 9,
                "predicted_spec_class": "UNKNOWN",
                "predicted_evolution_stage": "unknown",
                "router_label": "UNKNOWN",
            }
        ]
    )

    preview = build_unknown_preview(df_unknown)

    assert preview["priority_tier"].tolist() == ["LOW"]
    assert preview["reason_code"].tolist() == ["ROUTER_UNKNOWN"]
    assert preview["final_score"].tolist() == [0.0]
    assert preview["host_posterior"].isna().all()


def test_run_base_scoring_keeps_legacy_low_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combined low-frame должен сохранять порядок non-host строк router output."""
    df_input = pd.DataFrame({"source_id": [1, 2, 3, 4]})

    def fake_run_router(df: pd.DataFrame, router_model: object) -> pd.DataFrame:
        result = df.copy()
        result["predicted_spec_class"] = ["K", "A", "UNKNOWN", "G"]
        result["predicted_evolution_stage"] = [
            "dwarf",
            "dwarf",
            "unknown",
            "evolved",
        ]
        result["router_label"] = [
            "K_dwarf",
            "A_dwarf",
            "UNKNOWN",
            "G_evolved",
        ]
        return result

    def fake_score_host_df(
        model: dict[str, object],
        df: pd.DataFrame,
        spec_class_col: str = "spec_class",
    ) -> pd.DataFrame:
        return df.copy()

    monkeypatch.setattr(calibration_runtime, "run_router", fake_run_router)
    monkeypatch.setattr(
        calibration_runtime,
        "score_host_df",
        fake_score_host_df,
    )

    result = run_base_scoring(
        df_input=df_input,
        router_model={"meta": {"model_version": "router_test"}},
        host_model={"meta": {"model_version": "host_test"}},
    )

    assert result.host_input_df["source_id"].tolist() == [1]
    assert result.low_known_input_df["source_id"].tolist() == [2, 4]
    assert result.unknown_input_df["source_id"].tolist() == [3]
    assert result.low_input_df["source_id"].tolist() == [2, 3, 4]


@pytest.mark.parametrize("parallax", [None, float("nan"), 0.0, -2.5])
def test_distance_pc_from_parallax_rejects_invalid_values(
    parallax: object,
) -> None:
    """Неположительный или отсутствующий параллакс не должен давать расстояние."""
    assert distance_pc_from_parallax(parallax) is None


@pytest.mark.parametrize("parallax", [None, float("nan"), 0.0, -2.5])
def test_distance_factor_uses_invalid_factor_for_invalid_parallax(
    parallax: object,
) -> None:
    """Distance factor должен идти в invalid branch на плохом параллаксе."""
    config = CalibrationConfig()

    assert math.isclose(
        distance_factor(parallax, config),
        config.distance.invalid_factor,
        rel_tol=1e-9,
        abs_tol=1e-12,
    )


def test_quality_factor_handles_missing_ruwe_and_poor_precision() -> None:
    """Quality factor должен корректно умножать missing/pоor branches."""
    config = CalibrationConfig()

    value = quality_factor(
        ruwe_value=None,
        parallax_over_error=2.0,
        config=config,
    )

    expected = (
        config.quality.ruwe.missing_factor
        * config.quality.parallax_precision.poor_factor
    )
    assert math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12)
