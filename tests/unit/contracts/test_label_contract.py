# Тестовый файл `test_label_contract.py` домена `contracts`.
#
# Этот файл проверяет только:
# - проверку логики домена: контракты датасетов, колонок и policy-слоев;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `contracts` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.contracts.label_contract import (
    is_supported_evolution_stage,
    is_supported_spectral_class,
    is_valid_spectral_subclass,
    normalize_evolution_stage,
    normalize_spectral_subclass,
)


def test_supported_spectral_class_is_recognized() -> None:
    # Проверяем поддержку целевого спектрального класса.
    assert is_supported_spectral_class("G") is True


def test_supported_evolution_stage_is_recognized() -> None:
    # Проверяем поддержку базовой стадии эволюции.
    assert is_supported_evolution_stage("dwarf") is True
    assert is_supported_evolution_stage("subgiant") is True


def test_spectral_subclass_is_normalized_before_validation() -> None:
    # Подкласс должен проходить через простую нормализацию.
    assert normalize_spectral_subclass(" g2 ") == "G2"
    assert is_valid_spectral_subclass(" g2 ") is True


def test_invalid_spectral_subclass_is_rejected() -> None:
    # Значения вроде G10 не входят в зафиксированный контракт первой волны.
    assert is_valid_spectral_subclass("G10") is False


def test_evolution_stage_is_normalized_to_canonical_schema() -> None:
    # Внешние источники giant/subgiant должны ложиться в схему dwarf/evolved.
    assert normalize_evolution_stage(" giant ") == "evolved"
    assert normalize_evolution_stage("subgiant") == "evolved"
