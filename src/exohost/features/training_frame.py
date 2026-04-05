# Файл `training_frame.py` слоя `features`.
#
# Этот файл отвечает только за:
# - подготовку признаков и training frame-слой;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `features` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

import pandas as pd

from exohost.contracts.dataset_contracts import (
    HOST_TRAINING_CONTRACT,
    ROUTER_TRAINING_CONTRACT,
)
from exohost.contracts.feature_contract import unique_columns
from exohost.contracts.label_contract import (
    is_supported_evolution_stage,
    is_supported_spectral_class,
    is_valid_spectral_subclass,
    normalize_evolution_stage,
    normalize_spectral_subclass,
)

ROUTER_NUMERIC_COLUMNS: tuple[str, ...] = (
    "ra",
    "dec",
    "teff_gspphot",
    "logg_gspphot",
    "radius_gspphot",
    "parallax",
    "parallax_over_error",
    "ruwe",
    "bp_rp",
    "mh_gspphot",
)

HOST_NUMERIC_COLUMNS: tuple[str, ...] = (
    "teff_gspphot",
    "logg_gspphot",
    "radius_flame",
    "radius_gspphot",
    "lum_flame",
    "dist_arcsec",
    "parallax",
    "parallax_over_error",
    "ruwe",
    "phot_g_mean_mag",
    "bp_rp",
    "mh_gspphot",
    "classprob_dsc_combmod_star",
    "validation_factor",
)


def require_columns(
    df: pd.DataFrame,
    required_columns: tuple[str, ...],
    *,
    frame_name: str,
) -> None:
    # Проверяем, что входной DataFrame содержит все обязательные поля.
    missing_columns = [name for name in required_columns if name not in df.columns]
    if missing_columns:
        missing_columns_sql = ", ".join(missing_columns)
        raise ValueError(
            f"{frame_name} is missing required columns: {missing_columns_sql}"
        )


def normalize_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Приводим label-поля к каноническому текстовому виду.
    result = df.copy()
    result["spec_class"] = result["spec_class"].astype(str).str.strip().str.upper()
    result["evolution_stage"] = result["evolution_stage"].map(
        lambda value: normalize_evolution_stage(str(value))
    )
    if "spec_subclass" in result.columns:
        result["spec_subclass"] = result["spec_subclass"].map(
            lambda value: (
                normalize_spectral_subclass(str(value))
                if pd.notna(value)
                else pd.NA
            )
        )
    return result


def validate_label_columns(df: pd.DataFrame, *, frame_name: str) -> None:
    # Отлавливаем некорректные метки до обучения и split-логики.
    invalid_classes = sorted(
        {
            str(value)
            for value in df["spec_class"].dropna().unique().tolist()
            if not is_supported_spectral_class(str(value))
        }
    )
    if invalid_classes:
        sample = ", ".join(invalid_classes[:5])
        raise ValueError(f"{frame_name} contains unsupported spec_class values: {sample}")

    invalid_stages = sorted(
        {
            str(value)
            for value in df["evolution_stage"].dropna().unique().tolist()
            if not is_supported_evolution_stage(str(value))
        }
    )
    if invalid_stages:
        sample = ", ".join(invalid_stages[:5])
        raise ValueError(
            f"{frame_name} contains unsupported evolution_stage values: {sample}"
        )

    if "spec_subclass" in df.columns:
        invalid_subclasses = sorted(
            {
                str(value)
                for value in df["spec_subclass"].dropna().unique().tolist()
                if not is_valid_spectral_subclass(str(value))
            }
        )
        if invalid_subclasses:
            sample = ", ".join(invalid_subclasses[:5])
            raise ValueError(
                f"{frame_name} contains unsupported spec_subclass values: {sample}"
            )


def coerce_noncanonical_subclass_to_missing(df: pd.DataFrame) -> pd.DataFrame:
    # Сбрасываем нестандартные subclass-значения в NA,
    # если источник не следует нашему каноническому формату.
    if "spec_subclass" not in df.columns:
        return df

    result = df.copy()
    result["spec_subclass"] = result["spec_subclass"].map(
        lambda value: value if pd.isna(value) or is_valid_spectral_subclass(str(value)) else pd.NA
    )
    return result


def cast_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: tuple[str, ...],
) -> pd.DataFrame:
    # Явно приводим числовые признаки к float, но только если колонка есть.
    result = df.copy()
    for column_name in numeric_columns:
        if column_name not in result.columns:
            continue
        numeric_series = pd.Series(
            pd.to_numeric(
                result.loc[:, column_name],
                errors="coerce",
            ),
            index=result.index,
        )
        result[column_name] = numeric_series.astype(float)
    return result


def ensure_unique_source_id(df: pd.DataFrame, *, frame_name: str) -> None:
    # Для первой волны запрещаем дубликаты source_id, чтобы не получить leakage.
    duplicate_mask = df["source_id"].astype(str).duplicated(keep=False)
    if not bool(duplicate_mask.any()):
        return

    sample_ids = (
        df.loc[duplicate_mask, "source_id"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    sample_sql = ", ".join(sample_ids[:5])
    raise ValueError(
        f"{frame_name} contains duplicate source_id values. Sample ids: {sample_sql}"
    )


def sort_by_source_id(df: pd.DataFrame) -> pd.DataFrame:
    # Делаем детерминированный порядок строк по source_id.
    return (
        df.assign(_source_sort_key=df["source_id"].astype(str))
        .sort_values("_source_sort_key", kind="mergesort", ignore_index=True)
        .drop(columns="_source_sort_key")
    )


def prepare_router_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Нормализуем router training source до канонического табличного вида.
    required_columns = ROUTER_TRAINING_CONTRACT.required_columns
    optional_columns = ROUTER_TRAINING_CONTRACT.optional_columns
    selected_columns = unique_columns(required_columns, optional_columns)

    require_columns(df, required_columns, frame_name="router training frame")

    result = df.loc[:, [name for name in selected_columns if name in df.columns]].copy()
    result = normalize_label_columns(result)
    result = cast_numeric_columns(result, ROUTER_NUMERIC_COLUMNS)
    result = result.dropna(subset=required_columns).reset_index(drop=True)
    validate_label_columns(result, frame_name="router training frame")
    ensure_unique_source_id(result, frame_name="router training frame")
    return sort_by_source_id(result)


def prepare_host_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Нормализуем host training source до канонического вида.
    required_columns = HOST_TRAINING_CONTRACT.required_columns
    optional_columns = HOST_TRAINING_CONTRACT.optional_columns
    selected_columns = unique_columns(required_columns, optional_columns)

    require_columns(df, required_columns, frame_name="host training frame")

    result = df.loc[:, [name for name in selected_columns if name in df.columns]].copy()
    result = normalize_label_columns(result)
    result = coerce_noncanonical_subclass_to_missing(result)
    result = cast_numeric_columns(result, HOST_NUMERIC_COLUMNS)
    result = apply_host_radius_compatibility_alias(result)
    result = result.dropna(subset=required_columns).reset_index(drop=True)
    validate_label_columns(result, frame_name="host training frame")
    ensure_unique_source_id(result, frame_name="host training frame")
    return sort_by_source_id(result)


def apply_host_radius_compatibility_alias(df: pd.DataFrame) -> pd.DataFrame:
    # Новая clean host-wave считает canonical radius только по radius_flame.
    # Для существующих model-feature names сохраняем явный compatibility alias.
    if "radius_flame" not in df.columns:
        return df

    result = df.copy()
    if "radius_gspphot" in result.columns:
        result["radius_gspphot_legacy"] = result["radius_gspphot"]
    result["radius_gspphot"] = result["radius_flame"]
    return result
