# Файл `mk_label_parser.py` слоя `ingestion`.
#
# Этот файл отвечает только за:
# - разбор и нормализацию внешних B/mk-меток;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `ingestion` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

type ParseStatus = Literal["parsed", "partial", "unsupported", "empty"]
type LuminosityClass = Literal["0", "Ia+", "Ia", "Iab", "Ib", "I", "II", "III", "IV", "V", "VI", "VII"]

SPECTRAL_CLASS_PATTERN = re.compile(r"[OBAFGKM]")
INTEGER_SUBCLASS_PATTERN = re.compile(r"^([OBAFGKM])([0-9])(.*)$")
FRACTIONAL_SUBCLASS_PATTERN = re.compile(r"^([OBAFGKM])([0-9]\.[0-9]+)(.*)$")
AMBIGUOUS_OB_BOUNDARY_PATTERN = re.compile(
    r"^(OB|O/B|O[0-9](?:\.[0-9]+)?/B)",
)
LUMINOSITY_CLASSES: tuple[LuminosityClass, ...] = (
    "Ia+",
    "Iab",
    "Ia",
    "Ib",
    "III",
    "VII",
    "VI",
    "IV",
    "II",
    "I",
    "V",
    "0",
)


@dataclass(frozen=True)
class MkLabelParseResult:
    # Результат консервативного разбора сырой MK-строки.
    raw_value: str
    normalized_value: str
    spectral_class: str | None
    spectral_subclass: int | None
    luminosity_class: str | None
    peculiarity_suffix: str | None
    parse_status: ParseStatus
    parse_note: str | None


def normalize_mk_label(value: str) -> str:
    # Убираем пробелы и приводим к верхнему регистру без лишней пунктуации.
    collapsed_value = " ".join(value.strip().split())
    return collapsed_value.upper()


def parse_mk_label(value: str) -> MkLabelParseResult:
    # Разбираем только базовые MK-компоненты и не гадаем там, где строка двусмысленна.
    normalized_value = normalize_mk_label(value)
    if not normalized_value:
        return MkLabelParseResult(
            raw_value=value,
            normalized_value=normalized_value,
            spectral_class=None,
            spectral_subclass=None,
            luminosity_class=None,
            peculiarity_suffix=None,
            parse_status="empty",
            parse_note="empty_label",
        )

    compact_value = normalized_value.replace(" ", "")
    if _is_ambiguous_ob_boundary_label(compact_value):
        return MkLabelParseResult(
            raw_value=value,
            normalized_value=normalized_value,
            spectral_class="OB",
            spectral_subclass=None,
            luminosity_class=None,
            peculiarity_suffix=compact_value,
            parse_status="partial",
            parse_note="ambiguous_ob_boundary_label",
        )

    leading_match = SPECTRAL_CLASS_PATTERN.match(compact_value)
    if leading_match is None:
        return MkLabelParseResult(
            raw_value=value,
            normalized_value=normalized_value,
            spectral_class=None,
            spectral_subclass=None,
            luminosity_class=None,
            peculiarity_suffix=compact_value or None,
            parse_status="unsupported",
            parse_note="missing_supported_spectral_class",
        )

    spectral_class = leading_match.group(0)
    fractional_match = FRACTIONAL_SUBCLASS_PATTERN.match(compact_value)
    if fractional_match is not None:
        luminosity_class, peculiarity_suffix = _extract_luminosity_class(
            fractional_match.group(3),
        )
        return MkLabelParseResult(
            raw_value=value,
            normalized_value=normalized_value,
            spectral_class=fractional_match.group(1),
            spectral_subclass=None,
            luminosity_class=luminosity_class,
            peculiarity_suffix=peculiarity_suffix,
            parse_status="partial",
            parse_note="fractional_subclass_requires_separate_policy",
        )

    subclass_match = INTEGER_SUBCLASS_PATTERN.match(compact_value)
    if subclass_match is None:
        trailing_value = compact_value[1:]
        luminosity_class, peculiarity_suffix = _extract_luminosity_class(trailing_value)
        return MkLabelParseResult(
            raw_value=value,
            normalized_value=normalized_value,
            spectral_class=spectral_class,
            spectral_subclass=None,
            luminosity_class=luminosity_class,
            peculiarity_suffix=peculiarity_suffix,
            parse_status="partial",
            parse_note="missing_integer_subclass",
        )

    spectral_subclass = int(subclass_match.group(2))
    trailing_value = subclass_match.group(3)
    luminosity_class, peculiarity_suffix = _extract_luminosity_class(trailing_value)
    parse_status: ParseStatus = "parsed"
    parse_note: str | None = None
    if luminosity_class is None:
        parse_status = "partial"
        parse_note = "missing_luminosity_class"

    return MkLabelParseResult(
        raw_value=value,
        normalized_value=normalized_value,
        spectral_class=spectral_class,
        spectral_subclass=spectral_subclass,
        luminosity_class=luminosity_class,
        peculiarity_suffix=peculiarity_suffix,
        parse_status=parse_status,
        parse_note=parse_note,
    )


def _extract_luminosity_class(value: str) -> tuple[str | None, str | None]:
    # Выделяем класс светимости и остаток особенностей без агрессивной нормализации.
    normalized_value = value.strip()
    if not normalized_value:
        return None, None

    for luminosity_class in LUMINOSITY_CLASSES:
        if normalized_value.startswith(luminosity_class.upper()):
            suffix = normalized_value[len(luminosity_class) :].strip() or None
            return luminosity_class, suffix

    return None, normalized_value or None


def _is_ambiguous_ob_boundary_label(compact_value: str) -> bool:
    # Не считаем `OB...` и `O/B...` clean `O`: это hot-boundary, а не явный MK class.
    return AMBIGUOUS_OB_BOUNDARY_PATTERN.match(compact_value) is not None
