# Регресс-тесты policy-слоя `priority`.
#
# Этот файл отвечает только за:
# - проверку системных инвариантов tuned priority thresholds;
# - защиту ranking-контура от тихого изменения допуска и распределения labels.
#
# Следующий слой:
# - остальные regression-тесты `posthoc`-слоя;
# - active priority integration и ranking-логика в `src/exohost`.

from __future__ import annotations

from exohost.posthoc.priority_integration import (
    PriorityIntegrationConfig,
    build_priority_integration_result,
)
from exohost.ranking.priority_score import PriorityThresholds
from tests.regression.assertions import assert_required_columns, require_int_scalar
from tests.regression.conftest import (
    PRIORITY_BASE_SMALL_FIXTURE_PATH,
    PRIORITY_FINAL_DECISION_SMALL_FIXTURE_PATH,
)
from tests.regression.fixture_loaders import load_regression_csv_fixture


def test_priority_tuned_thresholds_shrink_high_zone_without_breaking_labels() -> None:
    base_df = load_regression_csv_fixture(PRIORITY_BASE_SMALL_FIXTURE_PATH)
    final_decision_df = load_regression_csv_fixture(PRIORITY_FINAL_DECISION_SMALL_FIXTURE_PATH)

    baseline_result = build_priority_integration_result(
        base_df,
        final_decision_df=final_decision_df,
    )
    tuned_result = build_priority_integration_result(
        base_df,
        final_decision_df=final_decision_df,
        config=PriorityIntegrationConfig(
            thresholds=PriorityThresholds(high_min=0.85, medium_min=0.55),
        ),
    )

    baseline_priority_df = baseline_result.priority_ranking_df
    tuned_priority_df = tuned_result.priority_ranking_df

    assert_required_columns(
        tuned_priority_df,
        required_columns=("source_id", "priority_score", "priority_label", "priority_reason"),
    )

    baseline_high_count = require_int_scalar(
        baseline_priority_df["priority_label"].eq("high").sum()
    )
    tuned_high_count = require_int_scalar(tuned_priority_df["priority_label"].eq("high").sum())
    tuned_medium_count = require_int_scalar(
        tuned_priority_df["priority_label"].eq("medium").sum()
    )

    assert baseline_high_count == 1
    assert tuned_high_count == 0
    assert tuned_medium_count == 1
    assert tuned_priority_df["priority_label"].tolist() == ["medium", "low"]


def test_priority_tuned_policy_keeps_only_eligible_rows_in_ranking_contour() -> None:
    base_df = load_regression_csv_fixture(PRIORITY_BASE_SMALL_FIXTURE_PATH)
    final_decision_df = load_regression_csv_fixture(PRIORITY_FINAL_DECISION_SMALL_FIXTURE_PATH)

    result = build_priority_integration_result(
        base_df,
        final_decision_df=final_decision_df,
        config=PriorityIntegrationConfig(
            thresholds=PriorityThresholds(high_min=0.85, medium_min=0.55),
        ),
    )

    priority_input_df = result.priority_input_df
    integrated_df = result.final_decision_df

    assert priority_input_df["source_id"].astype(int).tolist() == [1, 4]
    assert result.priority_ranking_df["source_id"].astype(int).tolist() == [1, 4]

    eligible_state_df = integrated_df.loc[
        integrated_df["source_id"].astype(int).isin([1, 4]),
        ["source_id", "priority_state", "priority_label"],
    ].copy()
    blocked_state_df = integrated_df.loc[
        integrated_df["source_id"].astype(int).isin([2, 3]),
        ["source_id", "priority_state", "priority_label"],
    ].copy()

    assert eligible_state_df["priority_state"].astype(str).tolist() == ["medium", "low"]
    assert eligible_state_df["priority_label"].astype(str).tolist() == ["medium", "low"]
    assert blocked_state_df["priority_state"].isna().all()
    assert blocked_state_df["priority_label"].isna().all()
