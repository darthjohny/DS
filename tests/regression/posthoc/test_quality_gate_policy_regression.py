# Регресс-тесты policy-слоя `quality_gate`.
#
# Этот файл отвечает только за:
# - проверку базовых инвариантов `quality_gate` на frozen fixture;
# - защиту от тихого размывания `reject` и смешения `review`-правил.
#
# Следующий слой:
# - остальные regression-тесты `posthoc`-слоя;
# - active posthoc-логика quality gate tuning в `src/exohost/posthoc`.

from __future__ import annotations

import pandas as pd

from exohost.contracts.quality_gate_rule_roles import QUALITY_GATE_RULE_SPECS
from exohost.posthoc.quality_gate_tuning import (
    QualityGateTuningConfig,
    apply_quality_gate_tuning,
)
from tests.regression.assertions import assert_required_columns, assert_small_frame_equal
from tests.regression.conftest import QUALITY_GATE_SMALL_FIXTURE_PATH
from tests.regression.fixture_loaders import load_regression_csv_fixture


def test_quality_gate_baseline_policy_keeps_frozen_expected_states() -> None:
    review_df = load_regression_csv_fixture(QUALITY_GATE_SMALL_FIXTURE_PATH)

    result_df = apply_quality_gate_tuning(review_df)
    assert_required_columns(
        result_df,
        required_columns=("source_id", "quality_state", "quality_reason", "review_bucket"),
    )

    expected_df = pd.DataFrame(
        {
            "source_id": [1, 2, 3, 4],
            "quality_state": ["pass", "unknown", "unknown", "reject"],
            "quality_reason": [
                "pass",
                "review_missing_radius_flame",
                "review_high_ruwe",
                "reject_missing_core_features",
            ],
            "review_bucket": [
                "pass",
                "review_missing_radius_flame",
                "review_high_ruwe",
                "reject_missing_core_features",
            ],
        }
    )

    assert_small_frame_equal(
        result_df.loc[:, expected_df.columns],
        expected_df,
    )


def test_quality_gate_tuned_policy_changes_only_allowed_review_rows() -> None:
    review_df = load_regression_csv_fixture(QUALITY_GATE_SMALL_FIXTURE_PATH)

    baseline_df = apply_quality_gate_tuning(review_df)
    tuned_df = apply_quality_gate_tuning(
        review_df,
        config=QualityGateTuningConfig(
            policy_name="no_flame",
            ruwe_unknown_threshold=1.4,
            parallax_snr_unknown_threshold=5.0,
            require_flame_for_pass=False,
        ),
    )

    changed_mask = baseline_df["quality_state"].astype("string").ne(
        tuned_df["quality_state"].astype("string")
    )
    changed_source_ids = tuned_df.loc[changed_mask, "source_id"].astype(int).tolist()

    assert changed_source_ids == [2]
    assert tuned_df.loc[tuned_df["source_id"] == 2, "quality_state"].iloc[0] == "pass"
    assert tuned_df.loc[tuned_df["source_id"] == 3, "quality_state"].iloc[0] == "unknown"
    assert tuned_df.loc[tuned_df["source_id"] == 4, "quality_state"].iloc[0] == "reject"
    assert (
        tuned_df.loc[tuned_df["source_id"] == 4, "quality_reason"].iloc[0]
        == "reject_missing_core_features"
    )


def test_quality_gate_reason_labels_follow_rule_roles_contract() -> None:
    review_df = load_regression_csv_fixture(QUALITY_GATE_SMALL_FIXTURE_PATH)

    result_df = apply_quality_gate_tuning(review_df)

    reject_reason_labels = {
        reason_label
        for spec in QUALITY_GATE_RULE_SPECS
        if spec.role == "reject"
        for reason_label in spec.live_reason_labels
    }
    review_reason_labels = {
        reason_label
        for spec in QUALITY_GATE_RULE_SPECS
        if spec.role == "review"
        for reason_label in spec.live_reason_labels
    }

    unknown_reason_labels = set(
        result_df.loc[result_df["quality_state"] == "unknown", "quality_reason"].astype(str)
    )
    reject_reason_values = set(
        result_df.loc[result_df["quality_state"] == "reject", "quality_reason"].astype(str)
    )

    assert unknown_reason_labels <= review_reason_labels
    assert reject_reason_values <= reject_reason_labels
    assert (
        result_df.loc[:, "quality_reason"].astype("string")
        == result_df.loc[:, "review_bucket"].astype("string")
    ).all()
