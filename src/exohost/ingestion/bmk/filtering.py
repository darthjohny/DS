# Файл `filtering.py` слоя `ingestion`.
#
# Этот файл отвечает только за:
# - разбор и нормализацию внешних B/mk-меток;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `ingestion` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from typing import Any, cast

import pandas as pd
from astropy.table import Row, Table

from exohost.ingestion.bmk.contracts import (
    B_MK_FILTERED_COLUMNS,
    B_MK_RAW_COLUMNS,
    B_MK_REJECTED_COLUMNS,
    SUPPORTED_SPECTRAL_PREFIXES,
    BmkImportSummary,
    BmkPrimaryFilterSummary,
    BmkTransformBundle,
    RejectReason,
)
from exohost.ingestion.bmk.normalization import (
    build_bmk_base_record,
)
from exohost.ingestion.mk_label_parser import parse_mk_label


def build_bmk_raw_frame(table: Table) -> pd.DataFrame:
    # Формируем staging-таблицу только из строк с координатами и непустым SpType.
    return build_bmk_transform_bundle(table).raw_frame


def build_bmk_import_summary(table: Table, raw_frame: pd.DataFrame) -> BmkImportSummary:
    # Считаем сводку по сырому импорту до и после отбора в staging-таблицу.
    transform_bundle = build_bmk_transform_bundle(table)
    return BmkImportSummary(
        total_rows=transform_bundle.import_summary.total_rows,
        rows_with_coordinates=transform_bundle.import_summary.rows_with_coordinates,
        rows_with_raw_sptype=transform_bundle.import_summary.rows_with_raw_sptype,
        rows_with_supported_spectral_prefix=(
            transform_bundle.import_summary.rows_with_supported_spectral_prefix
        ),
        exported_rows=int(raw_frame.shape[0]),
    )


def build_bmk_primary_filter_frames(table: Table) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Делим исходный B/mk на первично пригодные строки и отбракованные строки с причиной.
    transform_bundle = build_bmk_transform_bundle(table)
    return transform_bundle.filtered_frame, transform_bundle.rejected_frame


def build_bmk_primary_filter_summary(
    table: Table,
    filtered_frame: pd.DataFrame,
    rejected_frame: pd.DataFrame,
) -> BmkPrimaryFilterSummary:
    # Считаем сводку по первичной фильтрации, чтобы видеть чистый слой и отбраковку.
    reject_reason_counts = (
        rejected_frame["reject_reason"].value_counts().to_dict()
        if "reject_reason" in rejected_frame.columns
        else {}
    )
    return BmkPrimaryFilterSummary(
        total_rows=len(table),
        filtered_rows=int(filtered_frame.shape[0]),
        rejected_rows=int(rejected_frame.shape[0]),
        rows_ready_for_gaia_crossmatch=_count_ready_for_gaia_crossmatch(filtered_frame),
        rejected_missing_coordinates=int(reject_reason_counts.get("missing_coordinates", 0)),
        rejected_missing_raw_sptype=int(reject_reason_counts.get("missing_raw_sptype", 0)),
        rejected_unsupported_spectral_prefix=int(
            reject_reason_counts.get("unsupported_spectral_prefix", 0)
        ),
    )


def build_bmk_transform_bundle(table: Table) -> BmkTransformBundle:
    # Строим все B/mk staging-слои и сводки за один проход по CDS-таблице.
    raw_rows: list[dict[str, object]] = []
    filtered_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    rows_with_coordinates = 0
    rows_with_raw_sptype = 0
    rows_with_supported_spectral_prefix = 0
    rejected_missing_coordinates = 0
    rejected_missing_raw_sptype = 0
    rejected_unsupported_spectral_prefix = 0

    for row_index, row in enumerate(cast(Any, table)):
        base_record = build_bmk_base_record(cast(Row, row), external_row_id=row_index)
        has_coordinates = (
            base_record["ra_deg"] is not None and base_record["dec_deg"] is not None
        )
        has_raw_sptype = base_record["raw_sptype"] is not None
        raw_sptype = cast(str | None, base_record["raw_sptype"])

        if has_coordinates:
            rows_with_coordinates += 1
        if has_raw_sptype:
            rows_with_raw_sptype += 1
            if raw_sptype is not None and raw_sptype.startswith(SUPPORTED_SPECTRAL_PREFIXES):
                rows_with_supported_spectral_prefix += 1
        if has_coordinates and has_raw_sptype:
            raw_rows.append(base_record)

        filtered_row, rejected_row = _classify_bmk_base_record(base_record)
        if filtered_row is not None:
            filtered_rows.append(filtered_row)
            continue

        if rejected_row is None:
            continue

        rejected_rows.append(rejected_row)
        reject_reason = cast(RejectReason, rejected_row["reject_reason"])
        if reject_reason == "missing_coordinates":
            rejected_missing_coordinates += 1
        elif reject_reason == "missing_raw_sptype":
            rejected_missing_raw_sptype += 1
        elif reject_reason == "unsupported_spectral_prefix":
            rejected_unsupported_spectral_prefix += 1

    raw_frame = pd.DataFrame(raw_rows, columns=B_MK_RAW_COLUMNS)
    filtered_frame = pd.DataFrame(filtered_rows, columns=B_MK_FILTERED_COLUMNS)
    rejected_frame = pd.DataFrame(rejected_rows, columns=B_MK_REJECTED_COLUMNS)
    return BmkTransformBundle(
        raw_frame=raw_frame,
        filtered_frame=filtered_frame,
        rejected_frame=rejected_frame,
        import_summary=BmkImportSummary(
            total_rows=len(table),
            rows_with_coordinates=rows_with_coordinates,
            rows_with_raw_sptype=rows_with_raw_sptype,
            rows_with_supported_spectral_prefix=rows_with_supported_spectral_prefix,
            exported_rows=int(raw_frame.shape[0]),
        ),
        primary_filter_summary=BmkPrimaryFilterSummary(
            total_rows=len(table),
            filtered_rows=int(filtered_frame.shape[0]),
            rejected_rows=int(rejected_frame.shape[0]),
            rows_ready_for_gaia_crossmatch=int(filtered_frame.shape[0]),
            rejected_missing_coordinates=rejected_missing_coordinates,
            rejected_missing_raw_sptype=rejected_missing_raw_sptype,
            rejected_unsupported_spectral_prefix=rejected_unsupported_spectral_prefix,
        ),
    )


def _classify_bmk_base_record(
    base_record: dict[str, object],
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    # Отделяем первично пригодные строки от брака с явной причиной.
    has_coordinates = base_record["ra_deg"] is not None and base_record["dec_deg"] is not None
    has_raw_sptype = base_record["raw_sptype"] is not None

    if not has_coordinates:
        return None, _build_rejected_bmk_record(
            base_record,
            spectral_prefix=None,
            reject_reason="missing_coordinates",
        )
    if not has_raw_sptype:
        return None, _build_rejected_bmk_record(
            base_record,
            spectral_prefix=None,
            reject_reason="missing_raw_sptype",
        )

    raw_sptype = cast(str, base_record["raw_sptype"])
    parse_result = parse_mk_label(raw_sptype)
    spectral_prefix = parse_result.spectral_class
    has_supported_prefix = spectral_prefix in SUPPORTED_SPECTRAL_PREFIXES
    if not has_supported_prefix:
        return None, _build_rejected_bmk_record(
            base_record,
            spectral_prefix=spectral_prefix,
            reject_reason="unsupported_spectral_prefix",
        )

    return (
        {
            **base_record,
            "spectral_prefix": spectral_prefix,
            "spectral_class": parse_result.spectral_class,
            "spectral_subclass": parse_result.spectral_subclass,
            "luminosity_class": parse_result.luminosity_class,
            "parse_status": parse_result.parse_status,
            "parse_note": parse_result.parse_note,
            "has_supported_prefix": has_supported_prefix,
            "has_coordinates": has_coordinates,
            "has_raw_sptype": has_raw_sptype,
            "ready_for_gaia_crossmatch": True,
        },
        None,
    )


def _build_rejected_bmk_record(
    base_record: dict[str, object],
    spectral_prefix: str | None,
    reject_reason: RejectReason,
) -> dict[str, object]:
    # Формируем отбракованную строку с минимальным набором полей и понятной причиной.
    return {
        **base_record,
        "spectral_prefix": spectral_prefix,
        "reject_reason": reject_reason,
    }


def _count_ready_for_gaia_crossmatch(filtered_frame: pd.DataFrame) -> int:
    # Считаем готовые к Gaia строки без неявной арифметики по pandas Series.
    if "ready_for_gaia_crossmatch" not in filtered_frame.columns:
        return 0

    ready_values = filtered_frame["ready_for_gaia_crossmatch"].tolist()
    return sum(1 for value in ready_values if bool(value))
