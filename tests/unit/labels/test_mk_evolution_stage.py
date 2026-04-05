# Тестовый файл `test_mk_evolution_stage.py` домена `labels`.
#
# Этот файл проверяет только:
# - проверку логики домена: семантику меток и label-helper;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `labels` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import pytest

from exohost.labels.mk_evolution_stage import (
    has_supported_evolution_stage_mapping,
    map_luminosity_class_to_evolution_stage,
    normalize_luminosity_class_code,
)


@pytest.mark.parametrize(
    ("raw_value", "expected_value"),
    [
        (" V ", "V"),
        ("iv", "IV"),
        ("Ia+", "IA+"),
    ],
)
def test_normalize_luminosity_class_code_trims_and_uppercases(
    raw_value: str,
    expected_value: str,
) -> None:
    assert normalize_luminosity_class_code(raw_value) == expected_value


@pytest.mark.parametrize(
    ("luminosity_class", "expected_stage"),
    [
        ("V", "dwarf"),
        ("vi", "dwarf"),
        ("VII", "dwarf"),
        ("IV", "evolved"),
        ("iii", "evolved"),
        ("Ia+", "evolved"),
        ("Iab", "evolved"),
        ("0", "evolved"),
        (None, None),
        ("", None),
        ("?", None),
    ],
)
def test_map_luminosity_class_to_evolution_stage_returns_expected_stage(
    luminosity_class: str | None,
    expected_stage: str | None,
) -> None:
    assert map_luminosity_class_to_evolution_stage(luminosity_class) == expected_stage


def test_has_supported_evolution_stage_mapping_matches_mapping_result() -> None:
    assert has_supported_evolution_stage_mapping("V") is True
    assert has_supported_evolution_stage_mapping("Iab") is True
    assert has_supported_evolution_stage_mapping(None) is False
    assert has_supported_evolution_stage_mapping("pec") is False
