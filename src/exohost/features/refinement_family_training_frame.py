# Файл `refinement_family_training_frame.py` слоя `features`.
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

from exohost.contracts.dataset_contracts import select_contract_columns
from exohost.contracts.label_contract import normalize_spectral_subclass
from exohost.contracts.refinement_family_dataset_contracts import (
    build_gaia_mk_refinement_family_training_contract,
    validate_refinement_family_class,
)
from exohost.features.training_frame import (
    cast_numeric_columns,
    ensure_unique_source_id,
    require_columns,
    sort_by_source_id,
)

REFINEMENT_FAMILY_NUMERIC_COLUMNS: tuple[str, ...] = (
    "ra",
    "dec",
    "phot_g_mean_mag",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "bp_rp",
    "parallax",
    "parallax_over_error",
    "ruwe",
    "radius_flame",
    "lum_flame",
    "evolstage_flame",
    "xmatch_separation_arcsec",
)

SPECTRAL_SUBCLASS_DIGITS: tuple[str, ...] = tuple(str(index) for index in range(10))


def prepare_gaia_mk_refinement_family_training_frame(
    df: pd.DataFrame,
    *,
    spectral_class: str,
) -> pd.DataFrame:
    # Нормализуем одну family-view relation до train-ready frame.
    normalized_class = validate_refinement_family_class(spectral_class)
    contract = build_gaia_mk_refinement_family_training_contract(normalized_class)
    require_columns(
        df,
        contract.required_columns,
        frame_name="gaia mk refinement family training frame",
    )

    result = df.loc[
        :,
        [
            column_name
            for column_name in select_contract_columns(contract, set(str(name) for name in df.columns))
            if column_name in df.columns
        ],
    ].copy()
    result["spectral_class"] = (
        result["spectral_class"].astype(str).str.strip().str.upper()
    )
    result["spectral_subclass"] = result["spectral_subclass"].map(
        lambda value: _normalize_family_subclass(normalized_class, value)
    )
    if "full_subclass_label" in result.columns:
        result["full_subclass_label"] = (
            result["spectral_class"] + result["spectral_subclass"].astype(str)
        )

    result = cast_numeric_columns(result, REFINEMENT_FAMILY_NUMERIC_COLUMNS)
    result = result.dropna(
        subset=(
            "source_id",
            "spectral_class",
            "spectral_subclass",
            "teff_gspphot",
            "logg_gspphot",
            "mh_gspphot",
            "bp_rp",
            "parallax",
            "parallax_over_error",
            "ruwe",
            "radius_flame",
            "lum_flame",
            "evolstage_flame",
            "phot_g_mean_mag",
        )
    ).reset_index(drop=True)
    _validate_single_family_class(
        result,
        spectral_class=normalized_class,
    )
    ensure_unique_source_id(
        result,
        frame_name="gaia mk refinement family training frame",
    )
    return sort_by_source_id(result)


def _normalize_family_subclass(
    spectral_class: str,
    subclass_value: object,
) -> object:
    # Приводим family target к digit-only form внутри фиксированного coarse class.
    if subclass_value is None or subclass_value is pd.NA:
        return pd.NA
    if isinstance(subclass_value, int):
        return str(subclass_value) if 0 <= subclass_value <= 9 else pd.NA
    if isinstance(subclass_value, float):
        if pd.isna(subclass_value) or not float(subclass_value).is_integer():
            return pd.NA
        integer_value = int(subclass_value)
        return str(integer_value) if 0 <= integer_value <= 9 else pd.NA

    normalized_value = str(subclass_value).strip().upper()
    if not normalized_value:
        return pd.NA
    if normalized_value in SPECTRAL_SUBCLASS_DIGITS:
        return normalized_value

    normalized_full_label = normalize_spectral_subclass(normalized_value)
    if not normalized_full_label.startswith(spectral_class):
        return pd.NA

    subclass_digit = normalized_full_label.removeprefix(spectral_class)
    return subclass_digit if subclass_digit in SPECTRAL_SUBCLASS_DIGITS else pd.NA


def _validate_single_family_class(
    df: pd.DataFrame,
    *,
    spectral_class: str,
) -> None:
    invalid_classes = sorted(
        {
            str(value)
            for value in df["spectral_class"].dropna().unique().tolist()
            if str(value) != spectral_class
        }
    )
    if invalid_classes:
        sample = ", ".join(invalid_classes[:5])
        raise ValueError(
            "gaia mk refinement family training frame contains rows from "
            f"unexpected spectral classes: {sample}"
        )
