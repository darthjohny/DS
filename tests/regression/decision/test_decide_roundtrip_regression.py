# Регресс-тест малого сквозного `decide`-прогона.
#
# Этот файл отвечает только за:
# - проверку полного artifact bundle на frozen input;
# - защиту `decide`-контура от тихого дрейфа metadata, колонок и допустимых состояний.
#
# Следующий слой:
# - schema- и cohort-регрессии decision-слоя;
# - active orchestration `decide` в `src/exohost/cli` и `src/exohost/posthoc`.

from __future__ import annotations

from pathlib import Path

from exohost.reporting.final_decision_artifacts import load_final_decision_artifacts
from tests.regression.assertions import assert_required_columns, require_int_scalar
from tests.regression.decision.decide_roundtrip_testkit import (
    execute_decide_roundtrip_regression,
    prepare_decide_roundtrip_regression_context,
)

_ALLOWED_FINAL_DOMAIN_STATES = {"id", "candidate_ood", "ood", "unknown"}
_ALLOWED_PRIORITY_LABELS = {"high", "medium", "low"}


def test_decide_roundtrip_builds_complete_bundle_on_frozen_input(
    tmp_path: Path,
) -> None:
    context = prepare_decide_roundtrip_regression_context(tmp_path)
    run_dir = execute_decide_roundtrip_regression(context)

    assert (run_dir / "decision_input.csv").exists()
    assert (run_dir / "final_decision.csv").exists()
    assert (run_dir / "priority_input.csv").exists()
    assert (run_dir / "priority_ranking.csv").exists()
    assert (run_dir / "metadata.json").exists()

    bundle = load_final_decision_artifacts(run_dir)

    assert_required_columns(
        bundle.decision_input_df,
        required_columns=(
            "source_id",
            "quality_state",
            "quality_reason",
            "review_bucket",
            "ood_state",
            "ood_reason",
        ),
    )
    assert_required_columns(
        bundle.final_decision_df,
        required_columns=(
            "source_id",
            "final_domain_state",
            "final_quality_state",
            "final_coarse_class",
            "final_decision_reason",
        ),
    )
    assert_required_columns(
        bundle.priority_input_df,
        required_columns=("source_id",),
    )
    assert_required_columns(
        bundle.priority_ranking_df,
        required_columns=(
            "source_id",
            "priority_score",
            "priority_label",
            "priority_reason",
        ),
    )

    assert require_int_scalar(bundle.metadata["n_rows_input"]) == 2
    assert require_int_scalar(bundle.metadata["n_rows_final_decision"]) == 2
    assert require_int_scalar(bundle.metadata["n_rows_priority_input"]) == 1
    assert require_int_scalar(bundle.metadata["n_rows_priority_ranking"]) == 1

    metadata_context = bundle.metadata["context"]
    assert isinstance(metadata_context, dict)
    assert metadata_context["quality_gate_policy_name"] == "decide_tuned_quality_gate"
    assert metadata_context["quality_require_flame_for_pass"] is False
    assert metadata_context["min_coarse_probability"] == 0.99
    assert metadata_context["priority_high_min"] == 0.85
    assert metadata_context["priority_medium_min"] == 0.55

    final_source_ids = bundle.final_decision_df["source_id"].astype(int).tolist()
    priority_source_ids = bundle.priority_ranking_df["source_id"].astype(int).tolist()

    assert final_source_ids == [501, 502]
    assert priority_source_ids == [501]
    assert set(bundle.final_decision_df["final_domain_state"].astype(str)) <= _ALLOWED_FINAL_DOMAIN_STATES
    assert set(bundle.priority_ranking_df["priority_label"].astype(str)) <= _ALLOWED_PRIORITY_LABELS
