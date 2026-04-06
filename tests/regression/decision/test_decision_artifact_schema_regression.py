# Регресс-тест schema и metadata decision artifacts.
#
# Этот файл отвечает только за:
# - проверку контрактов `metadata.json`, `final_decision.csv` и `priority_ranking.csv`;
# - защиту малого `decide`-bundle от тихого дрейфа обязательных полей и допустимых значений.
#
# Следующий слой:
# - cohort- и summary-регрессии decision-слоя;
# - active artifact contracts в `src/exohost/reporting`.

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from exohost.cli.main import main
from exohost.reporting.final_decision_artifacts import load_final_decision_artifacts
from tests.regression.assertions import assert_required_columns, require_int_scalar
from tests.regression.conftest import DECISION_ARTIFACT_SCHEMA_FIXTURE_PATH
from tests.regression.decision.decide_roundtrip_testkit import (
    build_decide_roundtrip_cli_argv,
    prepare_decide_roundtrip_regression_context,
    resolve_single_decide_run_dir,
)
from tests.regression.fixture_loaders import load_regression_json_fixture


def test_decision_artifact_bundle_matches_frozen_schema_contract(
    tmp_path: Path,
) -> None:
    schema = load_regression_json_fixture(DECISION_ARTIFACT_SCHEMA_FIXTURE_PATH)
    context = prepare_decide_roundtrip_regression_context(tmp_path)

    exit_code = main(build_decide_roundtrip_cli_argv(context))
    assert exit_code == 0

    bundle = load_final_decision_artifacts(resolve_single_decide_run_dir(context.output_dir))

    assert_required_columns(
        bundle.decision_input_df,
        required_columns=_require_string_list(
            schema,
            field_name="decision_input_required_columns",
        ),
    )
    assert_required_columns(
        bundle.final_decision_df,
        required_columns=_require_string_list(
            schema,
            field_name="final_decision_required_columns",
        ),
    )
    assert_required_columns(
        bundle.priority_input_df,
        required_columns=_require_string_list(
            schema,
            field_name="priority_input_required_columns",
        ),
    )
    assert_required_columns(
        bundle.priority_ranking_df,
        required_columns=_require_string_list(
            schema,
            field_name="priority_ranking_required_columns",
        ),
    )

    assert set(bundle.metadata) >= set(
        _require_string_list(schema, field_name="metadata_required_keys")
    )
    metadata_context = _require_metadata_context(bundle.metadata.get("context"))
    assert set(metadata_context) >= set(
        _require_string_list(schema, field_name="metadata_context_required_keys")
    )

    assert bundle.metadata["pipeline_name"] == "hierarchical_final_decision"
    assert require_int_scalar(bundle.metadata["n_rows_input"]) == int(bundle.decision_input_df.shape[0])
    assert require_int_scalar(bundle.metadata["n_rows_final_decision"]) == int(
        bundle.final_decision_df.shape[0]
    )
    assert require_int_scalar(bundle.metadata["n_rows_priority_input"]) == int(
        bundle.priority_input_df.shape[0]
    )
    assert require_int_scalar(bundle.metadata["n_rows_priority_ranking"]) == int(
        bundle.priority_ranking_df.shape[0]
    )

    assert bundle.metadata["decision_input_columns"] == bundle.decision_input_df.columns.astype(str).tolist()
    assert bundle.metadata["final_decision_columns"] == bundle.final_decision_df.columns.astype(str).tolist()
    assert bundle.metadata["priority_input_columns"] == bundle.priority_input_df.columns.astype(str).tolist()
    assert bundle.metadata["priority_ranking_columns"] == bundle.priority_ranking_df.columns.astype(str).tolist()

    assert set(bundle.final_decision_df["final_domain_state"].astype(str)) <= set(
        _require_string_list(schema, field_name="allowed_final_domain_states")
    )
    assert set(bundle.final_decision_df["final_refinement_state"].astype(str)) <= set(
        _require_string_list(schema, field_name="allowed_final_refinement_states")
    )
    assert set(bundle.priority_ranking_df["priority_label"].astype(str)) <= set(
        _require_string_list(schema, field_name="allowed_priority_labels")
    )


def _require_string_list(
    payload: dict[str, Any] | Any,
    *,
    field_name: str,
) -> list[str]:
    raw_value = cast(dict[str, Any], payload).get(field_name)
    if not isinstance(raw_value, list) or not raw_value:
        raise AssertionError(f"Regression schema field '{field_name}' must be a non-empty list.")

    normalized_values: list[str] = []
    for raw_item in raw_value:
        if not isinstance(raw_item, str) or not raw_item.strip():
            raise AssertionError(
                f"Regression schema field '{field_name}' must contain non-empty strings."
            )
        normalized_values.append(raw_item)
    return normalized_values


def _require_metadata_context(raw_value: object) -> dict[str, Any]:
    if not isinstance(raw_value, dict):
        raise AssertionError("Decision artifact metadata must contain object-valued context.")
    return raw_value
