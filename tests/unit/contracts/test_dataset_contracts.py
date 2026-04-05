# Тестовый файл `test_dataset_contracts.py` домена `contracts`.
#
# Этот файл проверяет только:
# - проверку логики домена: контракты датасетов, колонок и policy-слоев;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `contracts` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.contracts.dataset_contracts import (
    ROUTER_TRAINING_CONTRACT,
    DatasetContract,
    missing_contract_columns,
    select_contract_columns,
)


def test_missing_contract_columns_returns_only_missing_required_fields() -> None:
    # Проверяем, что optional-поля не попадают в список обязательных пропусков.
    available_columns = set(ROUTER_TRAINING_CONTRACT.required_columns[:-1])
    missing_columns = missing_contract_columns(
        ROUTER_TRAINING_CONTRACT,
        available_columns,
    )

    assert missing_columns == ("evolution_stage",)


def test_select_contract_columns_appends_available_optional_columns() -> None:
    # Проверяем порядок обязательных и дополнительных колонок.
    available_columns = set(ROUTER_TRAINING_CONTRACT.required_columns) | {
        "ruwe",
        "bp_rp",
    }

    selected_columns = select_contract_columns(
        ROUTER_TRAINING_CONTRACT,
        available_columns,
    )

    assert selected_columns[: len(ROUTER_TRAINING_CONTRACT.required_columns)] == (
        ROUTER_TRAINING_CONTRACT.required_columns
    )
    assert selected_columns[-2:] == ("ruwe", "bp_rp")


def test_select_contract_columns_deduplicates_repeated_optional_fields() -> None:
    # Повторяющиеся optional-колонки не должны дублироваться в итоговом SELECT.
    contract = DatasetContract(
        relation_name="lab.example",
        required_columns=("source_id",),
        optional_columns=("ruwe", "ruwe", "bp_rp"),
    )

    selected_columns = select_contract_columns(
        contract,
        {"source_id", "ruwe", "bp_rp"},
    )

    assert selected_columns == ("source_id", "ruwe", "bp_rp")
