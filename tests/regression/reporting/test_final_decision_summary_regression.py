# Регресс-тест итоговых review summary для final decision.
#
# Этот файл отвечает только за:
# - проверку стабильности summary- и distribution-таблиц review-слоя;
# - защиту notebook/docs от тихого дрейфа причин, групп и базовых статусных сводок.
#
# Следующий слой:
# - интеграция regression-контура в общую документацию и QA-политику;
# - active review-helper `final_decision_review` в `src/exohost/reporting`.

from __future__ import annotations

from pathlib import Path

from exohost.reporting.final_decision_review import (
    build_decision_reason_frame,
    build_domain_distribution_frame,
    build_domain_quality_crosstab_frame,
    build_final_decision_summary_frame,
    build_host_priority_status_frame,
    build_priority_distribution_frame,
    build_priority_reason_frame,
    build_quality_distribution_frame,
    build_quality_reason_frame,
    build_refinement_distribution_frame,
    build_review_bucket_frame,
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


def test_final_decision_review_summary_frames_keep_expected_groups_and_reasons(
    tmp_path: Path,
) -> None:
    context = prepare_decide_roundtrip_regression_context(tmp_path)
    run_dir = execute_decide_roundtrip_regression(context)
    bundle = load_final_decision_review_bundle(run_dir)

    summary_df = build_final_decision_summary_frame(bundle)
    domain_df = build_domain_distribution_frame(bundle)
    quality_df = build_quality_distribution_frame(bundle)
    refinement_df = build_refinement_distribution_frame(bundle)
    decision_reason_df = build_decision_reason_frame(bundle)
    quality_reason_df = build_quality_reason_frame(bundle)
    review_bucket_df = build_review_bucket_frame(bundle)
    crosstab_df = build_domain_quality_crosstab_frame(bundle)
    priority_df = build_priority_distribution_frame(bundle)
    priority_reason_df = build_priority_reason_frame(bundle)
    priority_status_df = build_host_priority_status_frame(bundle)

    assert_required_columns(
        summary_df,
        required_columns=(
            "pipeline_name",
            "n_rows_input",
            "n_rows_final_decision",
            "n_rows_priority_input",
            "n_rows_priority_ranking",
            "n_unique_source_id",
        ),
    )
    assert_required_columns(
        domain_df,
        required_columns=("final_domain_state", "n_rows", "share"),
    )
    assert_required_columns(
        decision_reason_df,
        required_columns=("final_decision_reason", "n_rows", "share"),
    )
    assert_required_columns(
        priority_status_df,
        required_columns=(
            "host_signal_available",
            "priority_input_rows",
            "priority_output_rows",
            "status_note",
        ),
    )

    assert summary_df.loc[0, "pipeline_name"] == "hierarchical_final_decision"
    assert require_int_scalar(summary_df.loc[0, "n_rows_input"]) == 2
    assert require_int_scalar(summary_df.loc[0, "n_rows_final_decision"]) == 2
    assert require_int_scalar(summary_df.loc[0, "n_rows_priority_input"]) == 1
    assert require_int_scalar(summary_df.loc[0, "n_rows_priority_ranking"]) == 1
    assert require_int_scalar(summary_df.loc[0, "n_unique_source_id"]) == 2

    assert set(domain_df["final_domain_state"].astype(str)) == {"id", "ood"}
    assert set(quality_df["final_quality_state"].astype(str)) == {"pass"}
    assert set(refinement_df["final_refinement_state"].astype(str)) == {"not_attempted"}
    assert set(decision_reason_df["final_decision_reason"].astype(str)) == {
        "coarse_probability_below_threshold",
        "hard_ood",
    }
    assert set(quality_reason_df["quality_reason"].astype(str)) == {"pass"}
    assert set(review_bucket_df["review_bucket"].astype(str)) == {"pass"}
    assert set(priority_df["priority_label"].astype(str)) == {"high"}
    assert "host-like" in " ".join(priority_reason_df["priority_reason"].astype(str)).lower()

    assert require_int_scalar(crosstab_df.loc["id", "pass"]) == 1
    assert require_int_scalar(crosstab_df.loc["ood", "pass"]) == 1
    assert bool(priority_status_df.loc[0, "host_signal_available"]) is True
    assert require_int_scalar(priority_status_df.loc[0, "priority_input_rows"]) == 1
    assert require_int_scalar(priority_status_df.loc[0, "priority_output_rows"]) == 1
    assert "host-сигнал" in str(priority_status_df.loc[0, "status_note"]).lower()

    id_share = domain_df.loc[
        domain_df["final_domain_state"].astype(str) == "id",
        "share",
    ].iloc[0]
    high_priority_share = priority_df.loc[
        priority_df["priority_label"].astype(str) == "high",
        "share",
    ].iloc[0]

    assert require_float_scalar(id_share) == 0.5
    assert require_float_scalar(high_priority_share) == 1.0
