# Тестовый файл `test_posthoc_refinement_family_scoring.py` домена `posthoc`.
#
# Этот файл проверяет только:
# - проверку логики домена: post-hoc routing, gate и final decision policy;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `posthoc` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from exohost.posthoc.refinement_family_scoring import (
    build_refinement_family_scored_frame,
)


class DummyGFamilyEstimator:
    classes_ = np.asarray(["G2", "G4"], dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.asarray(
            [
                [0.7, 0.3],
                [0.15, 0.85],
            ],
            dtype=float,
        )


class DummyInvalidFamilyEstimator:
    classes_ = np.asarray(["F2", "F5"], dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.asarray([[0.9, 0.1]], dtype=float)


class DummyDigitOnlyGFamilyEstimator:
    classes_ = np.asarray(["2", "4"], dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.asarray(
            [
                [0.7, 0.3],
                [0.15, 0.85],
            ],
            dtype=float,
        )


def test_build_refinement_family_scored_frame_returns_family_scored_output() -> None:
    frame = pd.DataFrame(
        {
            "source_id": [1, 2],
            "spectral_class": ["G", "G"],
            "teff_gspphot": [5700.0, 5400.0],
        }
    )

    result = build_refinement_family_scored_frame(
        frame,
        estimator=DummyGFamilyEstimator(),
        feature_columns=("teff_gspphot",),
        family_name="G",
        model_name="hist_gradient_boosting",
    )

    assert result["refinement_predicted_label"].tolist() == ["G2", "G4"]
    assert result["refinement_probability_max"].tolist() == [0.7, 0.85]
    assert result["refinement_probability_margin"].tolist() == pytest.approx([0.4, 0.7])
    assert result["refinement_family_name"].tolist() == ["G", "G"]
    assert result["refinement_model_name"].tolist() == [
        "hist_gradient_boosting",
        "hist_gradient_boosting",
    ]


def test_build_refinement_family_scored_frame_normalizes_digit_only_labels() -> None:
    frame = pd.DataFrame(
        {
            "source_id": [1, 2],
            "spectral_class": ["G", "G"],
            "teff_gspphot": [5700.0, 5400.0],
        }
    )

    result = build_refinement_family_scored_frame(
        frame,
        estimator=DummyDigitOnlyGFamilyEstimator(),
        feature_columns=("teff_gspphot",),
        family_name="G",
    )

    assert result["refinement_predicted_label"].tolist() == ["G2", "G4"]


def test_build_refinement_family_scored_frame_rejects_mixed_input_family() -> None:
    frame = pd.DataFrame(
        {
            "source_id": [1, 2],
            "spectral_class": ["G", "K"],
            "teff_gspphot": [5700.0, 5000.0],
        }
    )

    with pytest.raises(ValueError, match="outside requested family"):
        build_refinement_family_scored_frame(
            frame,
            estimator=DummyGFamilyEstimator(),
            feature_columns=("teff_gspphot",),
            family_name="G",
        )


def test_build_refinement_family_scored_frame_rejects_invalid_predicted_family() -> None:
    frame = pd.DataFrame(
        {
            "source_id": [1],
            "spectral_class": ["G"],
            "teff_gspphot": [5700.0],
        }
    )

    with pytest.raises(ValueError, match="outside requested family"):
        build_refinement_family_scored_frame(
            frame,
            estimator=DummyInvalidFamilyEstimator(),
            feature_columns=("teff_gspphot",),
            family_name="G",
        )
