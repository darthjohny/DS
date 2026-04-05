# Файл `normalization.py` слоя `ingestion`.
#
# Этот файл отвечает только за:
# - разбор и нормализацию внешних B/mk-меток;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `ingestion` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

import math
from typing import Any, cast

from astropy.table import Row
from astropy.utils.masked.core import Masked

from exohost.ingestion.bmk.contracts import B_MK_CATALOG_NAME


def build_bmk_base_record(row: Row, external_row_id: int) -> dict[str, object]:
    # Собираем единый нормализованный base-record, не отбрасывая строку молча.
    raw_sptype = normalize_optional_text(row["SpType"])
    coordinate_pair = build_icrs_degree_pair(row)
    ra_deg = coordinate_pair[0] if coordinate_pair is not None else None
    dec_deg = coordinate_pair[1] if coordinate_pair is not None else None

    return {
        "external_row_id": external_row_id,
        "external_catalog_name": B_MK_CATALOG_NAME,
        "external_object_id": normalize_optional_text(row["Name"]),
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "raw_sptype": raw_sptype,
        "raw_magnitude": normalize_optional_float(row["Mag"]),
        "raw_source_bibcode": normalize_optional_text(row["Bibcode"]),
        "raw_notes": normalize_optional_text(row["Remarks"]),
    }


def build_icrs_degree_pair(row: Row) -> tuple[float, float] | None:
    # Собираем координаты из компонент J2000 и переводим их в десятичные градусы.
    ra_hour = normalize_optional_int(row["RAh"])
    ra_minute = normalize_optional_int(row["RAm"])
    ra_second = normalize_optional_float(row["RAs"])
    dec_sign = normalize_optional_text(row["DE-"])
    dec_degree = normalize_optional_int(row["DEd"])
    dec_minute = normalize_optional_int(row["DEm"])
    dec_second = normalize_optional_float(row["DEs"])

    required_values = (
        ra_hour,
        ra_minute,
        ra_second,
        dec_sign,
        dec_degree,
        dec_minute,
        dec_second,
    )
    if any(value is None for value in required_values):
        return None

    assert ra_hour is not None
    assert ra_minute is not None
    assert ra_second is not None
    assert dec_sign is not None
    assert dec_degree is not None
    assert dec_minute is not None
    assert dec_second is not None

    ra_deg = 15.0 * (ra_hour + (ra_minute / 60.0) + (ra_second / 3600.0))
    dec_abs_deg = dec_degree + (dec_minute / 60.0) + (dec_second / 3600.0)
    dec_deg = -dec_abs_deg if dec_sign == "-" else dec_abs_deg
    return ra_deg, dec_deg


def normalize_optional_text(value: object) -> str | None:
    # Приводим каталожное значение к очищенной строке или None.
    python_value = to_python_scalar(value)
    if python_value is None:
        return None

    normalized_value = str(python_value).strip()
    if normalized_value == "":
        return None

    return normalized_value


def normalize_optional_int(value: object) -> int | None:
    # Приводим каталожное значение к int или None.
    python_value = to_python_scalar(value)
    if python_value is None:
        return None
    if isinstance(python_value, str):
        return int(python_value)
    return int(python_value)


def normalize_optional_float(value: object) -> float | None:
    # Приводим каталожное значение к float или None.
    python_value = to_python_scalar(value)
    if python_value is None:
        return None
    if isinstance(python_value, str):
        return float(python_value)
    return float(python_value)


def to_python_scalar(value: object) -> str | int | float | None:
    # Снимаем astropy/masked-обертки и переводим пустые значения в None.
    if isinstance(value, Masked):
        return None

    if hasattr(value, "mask") and bool(getattr(value, "mask", False)):
        return None

    if hasattr(value, "item"):
        item_value = cast(Any, value).item()
        if item_value is None:
            return None
        if isinstance(item_value, str):
            return item_value
        if isinstance(item_value, bool):
            return int(item_value)
        if isinstance(item_value, (int, float)):
            return item_value
        return str(item_value)

    if value is None:
        return None

    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    return str(value)


def csv_cell_value(value: object) -> object:
    # Для CSV явно пишем пустую строку вместо None.
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value
