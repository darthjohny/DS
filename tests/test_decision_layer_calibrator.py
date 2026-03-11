"""Тесты для офлайн-калибратора decision layer."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
import pandas as pd

from decision_layer_calibrator import (
    CalibrationConfig,
    apply_calibration_config,
    build_low_priority_preview,
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
