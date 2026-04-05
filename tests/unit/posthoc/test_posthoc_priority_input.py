# Тестовый файл `test_posthoc_priority_input.py` домена `posthoc`.
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

from exohost.posthoc.priority_input import build_priority_input_frame


def build_base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2, 3, 4],
            "host_similarity_score": [0.91, 0.52, 0.30, 0.98],
            "evolution_stage": ["dwarf", "dwarf", "evolved", "dwarf"],
            "parallax": [16.0, 10.0, 7.0, 20.0],
            "phot_g_mean_mag": [10.8, 12.2, 13.5, 9.5],
            "parallax_over_error": [18.0, 12.0, 9.0, 25.0],
            "ruwe": [1.02, 1.10, 1.18, 1.01],
        }
    )


def build_final_decision_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2, 3, 4],
            "final_domain_state": ["id", "candidate_ood", "unknown", "id"],
            "final_quality_state": ["pass", "pass", "pass", "pass"],
            "final_coarse_class": ["G", "K", pd.NA, "A"],
        }
    )


def test_build_priority_input_frame_selects_only_priority_eligible_rows() -> None:
    result = build_priority_input_frame(
        build_base_frame(),
        final_decision_df=build_final_decision_frame(),
    )

    assert result["source_id"].astype(int).tolist() == [1, 4]
    assert result["spec_class"].astype(str).tolist() == ["G", "A"]
    assert result["host_similarity_score"].astype(float).tolist() == [0.91, 0.98]
    assert "evolution_stage" in result.columns
    assert "parallax" in result.columns
    assert "ruwe" in result.columns


def test_build_priority_input_frame_requires_host_score_column() -> None:
    base_df = build_base_frame().drop(columns="host_similarity_score")

    with pytest.raises(ValueError, match="host_similarity_score"):
        build_priority_input_frame(
            base_df,
            final_decision_df=build_final_decision_frame(),
        )
