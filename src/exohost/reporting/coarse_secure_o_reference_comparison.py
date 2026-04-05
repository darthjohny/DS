# Файл `coarse_secure_o_reference_comparison.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

import math

import pandas as pd

from exohost.db.engine import make_read_only_engine

REFERENCE_COMPARISON_COLUMNS: tuple[str, ...] = (
    "comparison_group",
    "source_id",
    "teff_gspphot",
    "logg_gspphot",
    "radius_gspphot",
    "parallax",
)


def build_secure_o_reference_comparison_query() -> str:
    # Собираем общий comparison-frame для надежного O-хвоста и чистых reference-групп.
    return """
SELECT
    'secure_o_tail' AS "comparison_group",
    secure_o."source_id",
    secure_o."teff_gspphot",
    secure_o."logg_gspphot",
    secure_o."radius_gspphot",
    secure_o."parallax"
FROM "lab"."gaia_ob_secure_o_like_subset" AS secure_o
WHERE secure_o."spectral_class" = 'O'

UNION ALL

SELECT
    'reference_o' AS "comparison_group",
    ref_o."source_id",
    ref_o."teff_gspphot",
    ref_o."logg_gspphot",
    ref_o."radius_gspphot",
    ref_o."parallax"
FROM "public"."gaia_ref_class_o" AS ref_o

UNION ALL

SELECT
    'reference_b' AS "comparison_group",
    ref_b."source_id",
    ref_b."teff_gspphot",
    ref_b."logg_gspphot",
    ref_b."radius_gspphot",
    ref_b."parallax"
FROM "public"."gaia_ref_class_b" AS ref_b

UNION ALL

SELECT
    'reference_evolved_o' AS "comparison_group",
    ref_o."source_id",
    ref_o."teff_gspphot",
    ref_o."logg_gspphot",
    ref_o."radius_gspphot",
    ref_o."parallax"
FROM "public"."gaia_ref_evolved_class_o" AS ref_o

UNION ALL

SELECT
    'reference_evolved_b' AS "comparison_group",
    ref_b."source_id",
    ref_b."teff_gspphot",
    ref_b."logg_gspphot",
    ref_b."radius_gspphot",
    ref_b."parallax"
FROM "public"."gaia_ref_evolved_class_b" AS ref_b
ORDER BY "comparison_group" ASC, "source_id" ASC
""".strip()


def load_secure_o_reference_comparison_frame(
    *,
    dotenv_path: str = ".env",
) -> pd.DataFrame:
    # Загружаем единый comparison-frame для secure O и чистых reference-наборов.
    engine = make_read_only_engine(dotenv_path=dotenv_path)
    try:
        return pd.read_sql_query(build_secure_o_reference_comparison_query(), engine)
    finally:
        engine.dispose()


def build_secure_o_reference_group_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Считаем компактные медианы по каждой comparison-группе.
    rows: list[dict[str, object]] = []
    for group_name, group_df in df.groupby("comparison_group", sort=True, dropna=False):
        rows.append(
            {
                "comparison_group": str(group_name),
                "n_rows": int(group_df.shape[0]),
                "median_teff_gspphot": _median_or_na(group_df, "teff_gspphot"),
                "median_logg_gspphot": _median_or_na(group_df, "logg_gspphot"),
                "median_radius_gspphot": _median_or_na(group_df, "radius_gspphot"),
                "median_parallax": _median_or_na(group_df, "parallax"),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_secure_o_reference_distance_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Сравниваем, к какой reference-группе надежный O-хвост ближе по медианам.
    summary_df = build_secure_o_reference_group_summary_frame(df)
    secure_row = summary_df.loc[
        summary_df["comparison_group"] == "secure_o_tail"
    ].reset_index(drop=True)
    if secure_row.empty:
        return pd.DataFrame(
            columns=(
                "comparison_group",
                "abs_diff_teff_gspphot",
                "abs_diff_logg_gspphot",
                "abs_diff_radius_gspphot",
                "abs_diff_parallax",
            )
        )

    secure_payload = secure_row.loc[0]
    reference_df = summary_df.loc[
        summary_df["comparison_group"] != "secure_o_tail"
    ].copy()
    secure_median_teff = secure_payload["median_teff_gspphot"]
    secure_median_logg = secure_payload["median_logg_gspphot"]
    secure_median_radius = secure_payload["median_radius_gspphot"]
    secure_median_parallax = secure_payload["median_parallax"]

    reference_df["abs_diff_teff_gspphot"] = _build_abs_diff_series(
        _require_series_column(reference_df, "median_teff_gspphot"),
        reference_value=secure_median_teff,
    )
    reference_df["abs_diff_logg_gspphot"] = _build_abs_diff_series(
        _require_series_column(reference_df, "median_logg_gspphot"),
        reference_value=secure_median_logg,
    )
    reference_df["abs_diff_radius_gspphot"] = _build_abs_diff_series(
        _require_series_column(reference_df, "median_radius_gspphot"),
        reference_value=secure_median_radius,
    )
    reference_df["abs_diff_parallax"] = _build_abs_diff_series(
        _require_series_column(reference_df, "median_parallax"),
        reference_value=secure_median_parallax,
    )
    return reference_df.loc[
        :,
        [
            "comparison_group",
            "abs_diff_teff_gspphot",
            "abs_diff_logg_gspphot",
            "abs_diff_radius_gspphot",
            "abs_diff_parallax",
        ],
    ].sort_values(
        ["abs_diff_teff_gspphot", "abs_diff_logg_gspphot", "comparison_group"],
        ascending=[True, True, True],
        kind="mergesort",
        ignore_index=True,
    )


def _median_or_na(df: pd.DataFrame, column_name: str) -> object:
    series = _require_series_column(df, column_name)
    numeric_series = pd.Series(
        pd.to_numeric(series, errors="coerce"),
        index=series.index,
        dtype="float64",
    ).dropna()
    if numeric_series.empty:
        return pd.NA
    return float(numeric_series.median())


def _abs_diff(left: object, right: object) -> object:
    if left is pd.NA or right is pd.NA:
        return pd.NA
    if _is_missing_scalar(left) or _is_missing_scalar(right):
        return pd.NA
    left_value = _to_float(left)
    right_value = _to_float(right)
    return abs(left_value - right_value)


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _to_float(value: object) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value.strip())
    raise TypeError(f"Unable to convert comparison value to float: {value!r}")


def _require_series_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    column = df.loc[:, column_name]
    if not isinstance(column, pd.Series):
        raise TypeError(f"{column_name} must resolve to a pandas Series.")
    return column


def _build_abs_diff_series(series: pd.Series, *, reference_value: object) -> pd.Series:
    return pd.Series(
        [_abs_diff(value, reference_value) for value in series.tolist()],
        index=series.index,
        dtype="object",
    )
