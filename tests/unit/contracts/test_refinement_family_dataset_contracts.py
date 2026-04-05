# Тестовый файл `test_refinement_family_dataset_contracts.py` домена `contracts`.
#
# Этот файл проверяет только:
# - проверку логики домена: контракты датасетов, колонок и policy-слоев;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `contracts` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import pytest

from exohost.contracts.refinement_family_dataset_contracts import (
    REFINEMENT_ENABLED_SPECTRAL_CLASSES,
    build_gaia_mk_refinement_family_training_contract,
    build_refinement_family_view_name,
    validate_refinement_family_class,
)


def test_validate_refinement_family_class_accepts_supported_value() -> None:
    assert validate_refinement_family_class(" g ") == "G"


def test_validate_refinement_family_class_rejects_unsupported_value() -> None:
    with pytest.raises(ValueError, match="Unsupported refinement family class"):
        validate_refinement_family_class("O")


def test_build_refinement_family_view_name_uses_lab_prefix() -> None:
    assert build_refinement_family_view_name("K") == "lab.v_gaia_mk_refinement_training_k"


def test_build_gaia_mk_refinement_family_training_contract_uses_family_view() -> None:
    contract = build_gaia_mk_refinement_family_training_contract("M")

    assert contract.relation_name == "lab.v_gaia_mk_refinement_training_m"
    assert "full_subclass_label" in contract.required_columns
    assert "evolstage_flame" in contract.required_columns
    assert "luminosity_class" in contract.optional_columns


def test_refinement_enabled_spectral_classes_are_ordered() -> None:
    assert REFINEMENT_ENABLED_SPECTRAL_CLASSES == ("A", "B", "F", "G", "K", "M")
