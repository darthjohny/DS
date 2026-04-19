# Тестовый файл `test_ui_contracts.py` домена `ui`.
#
# Этот файл проверяет только:
# - стабильность интерфейсных контрактов Streamlit;
# - согласованность UI-слоя с текущими final-decision artifacts и внешним CSV.
#
# Следующий слой:
# - loader-, state- и page-модули интерфейса;
# - unit-тесты соседних helper-модулей UI-пакета.

from __future__ import annotations

from datetime import UTC, datetime

from exohost.reporting.final_decision_artifacts import build_final_decision_artifact_paths
from exohost.ui.contracts import (
    UI_BENCHMARK_STAGE_CONTRACTS,
    UI_EXTERNAL_CSV_CONTRACT,
    UI_FINAL_DECISION_RUN_CONTRACT,
    UI_SESSION_STATE_CONTRACT,
    build_ui_artifact_table_map,
)


def test_ui_run_contract_matches_current_artifact_filenames() -> None:
    paths = build_final_decision_artifact_paths(
        output_dir="artifacts/decisions",
        pipeline_name=UI_FINAL_DECISION_RUN_CONTRACT.pipeline_name,
        now=datetime(2026, 4, 16, tzinfo=UTC),
    )

    expected_filenames = {
        paths.decision_input_csv_path.name,
        paths.final_decision_csv_path.name,
        paths.priority_input_csv_path.name,
        paths.priority_ranking_csv_path.name,
        paths.metadata_json_path.name,
    }

    assert set(UI_FINAL_DECISION_RUN_CONTRACT.required_filenames) == expected_filenames


def test_ui_run_contract_keeps_unique_table_names_and_filenames() -> None:
    table_contracts = UI_FINAL_DECISION_RUN_CONTRACT.table_contracts

    table_names = [contract.table_name for contract in table_contracts]
    filenames = [contract.filename for contract in table_contracts]

    assert len(table_names) == len(set(table_names))
    assert len(filenames) == len(set(filenames))
    assert all(contract.required_columns for contract in table_contracts)


def test_ui_table_map_returns_all_declared_contracts() -> None:
    table_map = build_ui_artifact_table_map(UI_FINAL_DECISION_RUN_CONTRACT)

    assert set(table_map) == {
        "decision_input",
        "final_decision",
        "priority_input",
        "priority_ranking",
    }
    assert table_map["priority_ranking"].filename == "priority_ranking.csv"


def test_ui_external_csv_contract_keeps_current_minimal_input_columns() -> None:
    required_columns = set(UI_EXTERNAL_CSV_CONTRACT.required_columns)

    assert "source_id" in required_columns
    assert "quality_state" in required_columns
    assert "radius_flame" in required_columns
    assert "lum_flame" in required_columns
    assert UI_EXTERNAL_CSV_CONTRACT.quality_state_default == "pass"


def test_ui_session_state_contract_keeps_unique_state_keys() -> None:
    state_keys = (
        UI_SESSION_STATE_CONTRACT.selected_run_dir_key,
        UI_SESSION_STATE_CONTRACT.selected_source_id_key,
        UI_SESSION_STATE_CONTRACT.uploaded_csv_path_key,
        UI_SESSION_STATE_CONTRACT.generated_run_dir_key,
        UI_SESSION_STATE_CONTRACT.run_load_error_key,
        UI_SESSION_STATE_CONTRACT.csv_validation_error_key,
    )

    assert UI_SESSION_STATE_CONTRACT.default_page == "home"
    assert len(state_keys) == len(set(state_keys))


def test_ui_benchmark_stage_contracts_keep_expected_stage_keys() -> None:
    stage_keys = [contract.stage_key for contract in UI_BENCHMARK_STAGE_CONTRACTS]
    task_prefixes = [contract.task_name_prefix for contract in UI_BENCHMARK_STAGE_CONTRACTS]

    assert stage_keys == ["id_ood", "coarse", "host", "refinement"]
    assert len(task_prefixes) == len(set(task_prefixes))


def test_ui_run_contract_requires_refinement_model_run_dirs_for_csv_rerun() -> None:
    assert "refinement_model_run_dirs" in UI_FINAL_DECISION_RUN_CONTRACT.required_metadata_context_keys
