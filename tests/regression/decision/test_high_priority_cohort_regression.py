# Регресс-тест верхней приоритетной группы в decision-слое.
#
# Этот файл отвечает только за:
# - проверку того, что high-priority cohort остается непустой и host-like;
# - защиту review-helper слоя от тихого дрейфа базовых cohort-сводок.
#
# Следующий слой:
# - summary-регрессии reporting-слоя;
# - active review-helper для final decision в `src/exohost/reporting`.

from __future__ import annotations

from pathlib import Path

from exohost.reporting.final_decision_review import (
    build_high_priority_candidate_physics_frame,
    build_high_priority_coarse_class_frame,
    build_high_priority_component_summary_frame,
    build_high_priority_summary_frame,
    load_final_decision_review_bundle,
)
from tests.regression.assertions import (
    assert_required_columns,
    require_float_scalar,
    require_int_scalar,
)
from tests.regression.decision.decide_roundtrip_testkit import (
    execute_decide_roundtrip_regression,
    prepare_decide_roundtrip_regression_context,
)


def test_high_priority_cohort_stays_non_empty_and_host_like_on_frozen_bundle(
    tmp_path: Path,
) -> None:
    context = prepare_decide_roundtrip_regression_context(tmp_path)
    run_dir = execute_decide_roundtrip_regression(context)
    bundle = load_final_decision_review_bundle(run_dir)

    summary_df = build_high_priority_summary_frame(bundle)
    coarse_df = build_high_priority_coarse_class_frame(bundle)
    component_df = build_high_priority_component_summary_frame(bundle)
    physics_df = build_high_priority_candidate_physics_frame(bundle)

    assert_required_columns(
        summary_df,
        required_columns=(
            "n_rows",
            "share_priority_ranking",
            "share_final_id",
            "median_priority_score",
            "median_host_similarity_score",
        ),
    )
    assert_required_columns(
        coarse_df,
        required_columns=("final_coarse_class", "n_rows", "share"),
    )
    assert_required_columns(
        component_df,
        required_columns=("metric_name", "p25_value", "median_value", "p75_value"),
    )
    assert_required_columns(
        physics_df,
        required_columns=(
            "source_id",
            "final_coarse_class",
            "priority_score",
            "host_similarity_score",
            "priority_reason",
        ),
    )

    assert require_int_scalar(summary_df.loc[0, "n_rows"]) == 1
    assert require_float_scalar(summary_df.loc[0, "share_priority_ranking"]) == 1.0
    assert require_float_scalar(summary_df.loc[0, "share_final_id"]) == 1.0
    assert require_float_scalar(summary_df.loc[0, "median_priority_score"]) > 0.85
    assert require_float_scalar(summary_df.loc[0, "median_host_similarity_score"]) > 0.9

    assert coarse_df.loc[0, "final_coarse_class"] == "G"
    assert require_int_scalar(coarse_df.loc[0, "n_rows"]) == 1
    assert {"priority_score", "host_similarity_score", "observability_score"} <= set(
        component_df["metric_name"].astype(str)
    )

    assert physics_df["source_id"].astype(int).tolist() == [501]
    assert physics_df.loc[0, "final_coarse_class"] == "G"
    assert "host-like" in str(physics_df.loc[0, "priority_reason"]).lower()
