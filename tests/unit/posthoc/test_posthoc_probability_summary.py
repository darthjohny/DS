# Тестовый файл `test_posthoc_probability_summary.py` домена `posthoc`.
#
# Этот файл проверяет только:
# - проверку логики домена: post-hoc routing, gate и final decision policy;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `posthoc` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import pandas as pd
import pytest

from exohost.posthoc.probability_summary import build_probability_summary_frame


def test_build_probability_summary_frame_builds_prediction_confidence_and_margin() -> None:
    probability_frame = pd.DataFrame(
        {
            "G": [0.7, 0.3],
            "K": [0.2, 0.6],
            "M": [0.1, 0.1],
        }
    )

    result = build_probability_summary_frame(
        probability_frame,
        prediction_column_name="predicted_label",
        confidence_column_name="probability_max",
        margin_column_name="probability_margin",
    )

    assert result["predicted_label"].tolist() == ["G", "K"]
    assert result["probability_max"].tolist() == [0.7, 0.6]
    assert result["probability_margin"].tolist() == pytest.approx([0.5, 0.3])


def test_build_probability_summary_frame_supports_empty_frames() -> None:
    probability_frame = pd.DataFrame(columns=["A", "F"])

    result = build_probability_summary_frame(
        probability_frame,
        prediction_column_name="predicted_label",
        confidence_column_name="probability_max",
        margin_column_name="probability_margin",
    )

    assert list(result.columns) == [
        "predicted_label",
        "probability_max",
        "probability_margin",
    ]
    assert result.empty is True
