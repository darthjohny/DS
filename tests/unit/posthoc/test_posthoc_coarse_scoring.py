# Тестовый файл `test_posthoc_coarse_scoring.py` домена `posthoc`.
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

from exohost.posthoc.coarse_scoring import build_coarse_scored_frame


class DummyCoarseEstimator:
    classes_ = np.asarray(["G", "K"], dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.asarray(
            [
                [0.8, 0.2],
                [0.35, 0.65],
            ],
            dtype=float,
        )


def test_build_coarse_scored_frame_returns_expected_columns() -> None:
    frame = pd.DataFrame(
        {
            "source_id": [1, 2],
            "teff_gspphot": [5800.0, 4700.0],
        }
    )

    result = build_coarse_scored_frame(
        frame,
        estimator=DummyCoarseEstimator(),
        feature_columns=("teff_gspphot",),
        model_name="hist_gradient_boosting",
    )

    assert result["coarse_predicted_label"].tolist() == ["G", "K"]
    assert result["coarse_probability_max"].tolist() == [0.8, 0.65]
    assert result["coarse_probability_margin"].tolist() == pytest.approx([0.6, 0.3])
    assert result["coarse_model_name"].tolist() == [
        "hist_gradient_boosting",
        "hist_gradient_boosting",
    ]
