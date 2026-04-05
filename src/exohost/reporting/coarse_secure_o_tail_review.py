# Файл `coarse_secure_o_tail_review.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

import pandas as pd

from exohost.db.engine import make_read_only_engine

SECURE_O_TAIL_STAR_COLUMNS: tuple[str, ...] = (
    "source_id",
    "raw_sptype",
    "spectral_subclass",
    "spectraltype_esphs",
    "esphs_class_letter",
    "flags_esphs",
    "in_gold_sample_oba_stars",
    "teff_gspphot",
    "teff_esphs",
    "logg_gspphot",
    "logg_esphs",
    "radius_flame",
    "lum_flame",
)


def build_secure_o_tail_review_query(*, limit: int | None = None) -> str:
    # Собираем SQL только для локального надежного хвоста класса O.
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided.")

    limit_sql = ""
    if limit is not None:
        limit_sql = f"\nLIMIT {limit:d}"

    return (
        """
SELECT
    secure_o."source_id",
    labeled."raw_sptype",
    secure_o."spectral_class",
    secure_o."spectral_subclass",
    secure_o."spectraltype_esphs",
    secure_o."esphs_class_letter",
    secure_o."flags_esphs",
    secure_o."in_gold_sample_oba_stars",
    secure_o."teff_gspphot",
    secure_o."teff_esphs",
    secure_o."logg_gspphot",
    secure_o."logg_esphs",
    secure_o."radius_flame",
    secure_o."lum_flame"
FROM "lab"."gaia_ob_secure_o_like_subset" AS secure_o
JOIN "lab"."gaia_mk_external_labeled" AS labeled
  ON labeled."source_id" = secure_o."source_id"
WHERE secure_o."spectral_class" = 'O'
ORDER BY secure_o."source_id" ASC
""".strip()
        + limit_sql
    )


def load_secure_o_tail_review_frame(
    *,
    limit: int | None = None,
    dotenv_path: str = ".env",
) -> pd.DataFrame:
    # Загружаем только небольшой надежный O-хвост для review.
    query = build_secure_o_tail_review_query(limit=limit)
    engine = make_read_only_engine(dotenv_path=dotenv_path)
    try:
        return pd.read_sql_query(query, engine)
    finally:
        engine.dispose()


def build_secure_o_tail_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Собираем one-row summary по надежному хвосту класса O.
    teff_gspphot = _require_series_column(df, "teff_gspphot")
    teff_esphs = _require_series_column(df, "teff_esphs")
    lum_flame = _require_series_column(df, "lum_flame")
    spectral_subclass = _require_series_column(df, "spectral_subclass")
    in_gold_sample = _require_series_column(df, "in_gold_sample_oba_stars")

    return pd.DataFrame(
        [
            {
                "n_rows": int(df.shape[0]),
                "n_with_numeric_subclass": int(spectral_subclass.notna().sum()),
                "n_with_teff_esphs": int(teff_esphs.notna().sum()),
                "n_in_gold_sample_oba": int(in_gold_sample.fillna(False).astype(bool).sum()),
                "median_teff_gspphot": _median_or_na(teff_gspphot),
                "median_teff_esphs": _median_or_na(teff_esphs),
                "median_lum_flame": _median_or_na(lum_flame),
            }
        ]
    )


def build_secure_o_tail_raw_label_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Показываем, какие именно исходные O-метки дошли до надежного хвоста.
    raw_sptype = _require_series_column(df, "raw_sptype")
    counts = raw_sptype.value_counts(dropna=False).rename_axis("raw_sptype").reset_index(
        name="n_rows"
    )
    total_rows = int(df.shape[0])
    counts["share"] = counts["n_rows"].astype(float) / total_rows if total_rows > 0 else 0.0
    return counts


def build_secure_o_tail_esphs_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Компактный breakdown по Gaia hot-star сигналу и наличию численной температуры.
    working_df = df.loc[:, ["spectraltype_esphs", "flags_esphs", "teff_esphs"]].copy()
    working_df["has_teff_esphs"] = working_df["teff_esphs"].notna()
    summary_df = (
        working_df.groupby(
            ["spectraltype_esphs", "flags_esphs", "has_teff_esphs"],
            dropna=False,
        )
        .size()
        .reset_index(name="n_rows")
        .sort_values(
            ["n_rows", "spectraltype_esphs", "flags_esphs"],
            ascending=[False, True, True],
            kind="mergesort",
            ignore_index=True,
        )
    )
    return summary_df


def build_secure_o_tail_star_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Возвращаем star-level таблицу для notebook-review.
    return df.loc[:, SECURE_O_TAIL_STAR_COLUMNS].copy()


def _median_or_na(series: pd.Series) -> object:
    numeric_series = pd.Series(
        pd.to_numeric(series, errors="coerce"),
        index=series.index,
        dtype="float64",
    ).dropna()
    if numeric_series.empty:
        return pd.NA
    return float(numeric_series.median())


def _require_series_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    column = df.loc[:, column_name]
    if not isinstance(column, pd.Series):
        raise TypeError(f"{column_name} must resolve to a pandas Series.")
    return column
