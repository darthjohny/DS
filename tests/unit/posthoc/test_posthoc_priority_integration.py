# Тестовый файл `test_posthoc_priority_integration.py` домена `posthoc`.
#
# Этот файл проверяет только:
# - проверку логики домена: post-hoc routing, gate и final decision policy;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `posthoc` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from numbers import Real

import pandas as pd

from exohost.posthoc.priority_integration import build_priority_integration_result


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
            "validation_factor": [0.93, 0.70, 0.55, 1.00],
        }
    )


def build_final_decision_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2, 3, 4],
            "final_domain_state": ["id", "candidate_ood", "unknown", "id"],
            "final_quality_state": ["pass", "pass", "reject", "pass"],
            "final_coarse_class": ["G", "K", pd.NA, "A"],
            "priority_state": ["stale_high", "stale_medium", "stale_low", "stale_high"],
        }
    )


def test_build_priority_integration_result_merges_priority_only_for_eligible_rows() -> None:
    result = build_priority_integration_result(
        build_base_frame(),
        final_decision_df=build_final_decision_frame(),
    )

    assert result.priority_input_df["source_id"].astype(int).tolist() == [1, 4]
    assert result.priority_ranking_df["source_id"].astype(str).tolist() == ["1", "4"]

    integrated_df = result.final_decision_df
    assert integrated_df["source_id"].astype(int).tolist() == [1, 2, 3, 4]
    assert integrated_df.loc[0, "priority_state"] == "high"
    assert integrated_df.loc[3, "priority_state"] == "low"
    assert pd.isna(integrated_df.loc[1, "priority_state"])
    assert pd.isna(integrated_df.loc[2, "priority_state"])
    assert pd.isna(integrated_df.loc[1, "priority_score"])
    assert pd.isna(integrated_df.loc[2, "priority_score"])
    assert integrated_df.loc[0, "priority_label"] == "high"
    assert integrated_df.loc[3, "priority_label"] == "low"

    high_score = integrated_df.loc[0, "priority_score"]
    low_score = integrated_df.loc[3, "priority_score"]
    assert isinstance(high_score, Real)
    assert isinstance(low_score, Real)
    assert float(high_score) > float(low_score)


def test_build_priority_integration_result_overwrites_stale_priority_state() -> None:
    result = build_priority_integration_result(
        build_base_frame(),
        final_decision_df=build_final_decision_frame(),
    )

    assert result.final_decision_df.loc[0, "priority_state"] != "stale_high"
    assert pd.isna(result.final_decision_df.loc[1, "priority_state"])
