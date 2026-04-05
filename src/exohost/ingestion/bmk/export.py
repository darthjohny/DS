# Файл `export.py` слоя `ingestion`.
#
# Этот файл отвечает только за:
# - разбор и нормализацию внешних B/mk-меток;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `ingestion` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from exohost.ingestion.bmk.contracts import (
    B_MK_FILTERED_COLUMNS,
    B_MK_FILTERED_CSV_FILENAME,
    B_MK_RAW_COLUMNS,
    B_MK_RAW_CSV_FILENAME,
    B_MK_REJECTED_COLUMNS,
    B_MK_REJECTED_CSV_FILENAME,
    BmkExportPaths,
)

NULLABLE_INTEGER_EXPORT_COLUMNS: tuple[str, ...] = ("spectral_subclass",)


def write_bmk_raw_csv(raw_frame: pd.DataFrame, output_path: Path) -> Path:
    # Записываем raw CSV с явным порядком колонок parser-а.
    return _write_bmk_frame_csv(raw_frame, output_path=output_path, columns=B_MK_RAW_COLUMNS)


def write_bmk_filtered_csv(filtered_frame: pd.DataFrame, output_path: Path) -> Path:
    # Записываем filtered CSV с явным порядком колонок parser-а.
    return _write_bmk_frame_csv(
        filtered_frame,
        output_path=output_path,
        columns=B_MK_FILTERED_COLUMNS,
    )


def write_bmk_rejected_csv(rejected_frame: pd.DataFrame, output_path: Path) -> Path:
    # Записываем rejected CSV с явным порядком колонок parser-а.
    return _write_bmk_frame_csv(
        rejected_frame,
        output_path=output_path,
        columns=B_MK_REJECTED_COLUMNS,
    )


def write_bmk_csv_bundle(
    raw_frame: pd.DataFrame,
    filtered_frame: pd.DataFrame,
    rejected_frame: pd.DataFrame,
    *,
    output_dir: Path,
) -> BmkExportPaths:
    # Сохраняем все три staging CSV одного B/mk прогона.
    return BmkExportPaths(
        raw_csv_path=write_bmk_raw_csv(raw_frame, output_dir / B_MK_RAW_CSV_FILENAME),
        filtered_csv_path=write_bmk_filtered_csv(
            filtered_frame,
            output_dir / B_MK_FILTERED_CSV_FILENAME,
        ),
        rejected_csv_path=write_bmk_rejected_csv(
            rejected_frame,
            output_dir / B_MK_REJECTED_CSV_FILENAME,
        ),
    )


def _write_bmk_frame_csv(
    frame: pd.DataFrame,
    *,
    output_path: Path,
    columns: tuple[str, ...],
) -> Path:
    # Записываем B/mk frame напрямую через pandas без materialize в records-list.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_frame = _prepare_bmk_export_frame(frame, columns=columns)
    export_frame.to_csv(
        output_path,
        index=False,
        na_rep="",
        encoding="utf-8",
        lineterminator="\n",
    )
    return output_path


def _prepare_bmk_export_frame(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    # Подготавливаем export frame и фиксируем nullable integer-колонки до CSV.
    export_frame = frame.reindex(columns=list(columns)).copy()
    for column_name in NULLABLE_INTEGER_EXPORT_COLUMNS:
        if column_name not in export_frame.columns:
            continue
        integer_series = pd.Series(
            pd.to_numeric(
                export_frame.loc[:, column_name],
                errors="coerce",
            ),
            index=export_frame.index,
        )
        export_frame[column_name] = integer_series.astype("Int64")
    return export_frame
