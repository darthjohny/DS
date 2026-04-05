# Тестовый файл `test_mk_label_parser.py` домена `ingestion`.
#
# Этот файл проверяет только:
# - проверку логики домена: ингест, parser-слой и нормализацию внешних меток;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `ingestion` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.ingestion.mk_label_parser import parse_mk_label


def test_parse_mk_label_parses_basic_g2v_label() -> None:
    result = parse_mk_label("G2V")

    assert result.spectral_class == "G"
    assert result.spectral_subclass == 2
    assert result.luminosity_class == "V"
    assert result.peculiarity_suffix is None
    assert result.parse_status == "parsed"


def test_parse_mk_label_parses_bare_supergiant_class() -> None:
    result = parse_mk_label("M2I")

    assert result.spectral_class == "M"
    assert result.spectral_subclass == 2
    assert result.luminosity_class == "I"
    assert result.parse_status == "parsed"


def test_parse_mk_label_parses_spaced_label() -> None:
    result = parse_mk_label("K7 III")

    assert result.spectral_class == "K"
    assert result.spectral_subclass == 7
    assert result.luminosity_class == "III"
    assert result.parse_status == "parsed"


def test_parse_mk_label_keeps_suffix_as_optional_tail() -> None:
    result = parse_mk_label("G2VFE-1")

    assert result.spectral_class == "G"
    assert result.spectral_subclass == 2
    assert result.luminosity_class == "V"
    assert result.peculiarity_suffix == "FE-1"
    assert result.parse_status == "parsed"


def test_parse_mk_label_marks_fractional_subclass_as_partial() -> None:
    result = parse_mk_label("B9.5V")

    assert result.spectral_class == "B"
    assert result.spectral_subclass is None
    assert result.luminosity_class == "V"
    assert result.parse_status == "partial"
    assert result.parse_note == "fractional_subclass_requires_separate_policy"


def test_parse_mk_label_marks_missing_luminosity_class_as_partial() -> None:
    result = parse_mk_label("G2")

    assert result.spectral_class == "G"
    assert result.spectral_subclass == 2
    assert result.luminosity_class is None
    assert result.parse_status == "partial"
    assert result.parse_note == "missing_luminosity_class"


def test_parse_mk_label_marks_ambiguous_ob_boundary_as_partial_ob() -> None:
    result = parse_mk_label("OB-")

    assert result.spectral_class == "OB"
    assert result.spectral_subclass is None
    assert result.luminosity_class is None
    assert result.peculiarity_suffix == "OB-"
    assert result.parse_status == "partial"
    assert result.parse_note == "ambiguous_ob_boundary_label"


def test_parse_mk_label_marks_o_slash_b_boundary_as_partial_ob() -> None:
    result = parse_mk_label("O9.5/B0IV/V")

    assert result.spectral_class == "OB"
    assert result.spectral_subclass is None
    assert result.luminosity_class is None
    assert result.peculiarity_suffix == "O9.5/B0IV/V"
    assert result.parse_status == "partial"
    assert result.parse_note == "ambiguous_ob_boundary_label"


def test_parse_mk_label_marks_empty_value() -> None:
    result = parse_mk_label("   ")

    assert result.parse_status == "empty"
    assert result.parse_note == "empty_label"


def test_parse_mk_label_marks_unsupported_value() -> None:
    result = parse_mk_label("WD")

    assert result.spectral_class is None
    assert result.parse_status == "unsupported"
    assert result.parse_note == "missing_supported_spectral_class"
