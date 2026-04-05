# Review-хелперы для train/test support true `O` rows в coarse training source.

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import pandas as pd

from exohost.db.engine import make_read_only_engine
from exohost.evaluation.hierarchical_tasks import GAIA_ID_COARSE_CLASSIFICATION_TASK
from exohost.evaluation.protocol import SplitConfig
from exohost.evaluation.split import DatasetSplit, split_dataset
from exohost.reporting.model_pipeline_review import (
    BenchmarkReviewBundle,
    load_benchmark_review_bundle,
)
from exohost.training.hierarchical_source import load_hierarchical_prepared_training_frame


@dataclass(frozen=True, slots=True)
class CoarseOTrainSupportReviewConfig:
    # Project baseline: повторяем benchmark split без изменения coarse pipeline.
    split_config: SplitConfig = SplitConfig()
    hot_teff_min_k: float = 10_000.0
    teff_band_edges_k: tuple[float, ...] = (10_000.0, 15_000.0, 20_000.0, 25_000.0)


@dataclass(frozen=True, slots=True)
class CoarseOTrainSupportReviewBundle:
    # Полный пакет train-time review для `O` support в coarse source.
    config: CoarseOTrainSupportReviewConfig
    source_df: pd.DataFrame
    split: DatasetSplit
    benchmark_bundle: BenchmarkReviewBundle | None


_DEFAULT_SUPPORT_CONFIG = CoarseOTrainSupportReviewConfig()


def load_coarse_o_train_support_review_bundle_from_env(
    *,
    benchmark_run_dir: str | Path | None = None,
    config: CoarseOTrainSupportReviewConfig = _DEFAULT_SUPPORT_CONFIG,
    source_limit: int | None = None,
    dotenv_path: str = ".env",
    connect_timeout: int | None = 10,
) -> CoarseOTrainSupportReviewBundle:
    # Загружаем live coarse source и восстанавливаем тот же split, что и в benchmark.
    engine = make_read_only_engine(
        dotenv_path=dotenv_path,
        connect_timeout=connect_timeout,
    )
    try:
        source_df = load_hierarchical_prepared_training_frame(
            engine,
            task_name=GAIA_ID_COARSE_CLASSIFICATION_TASK.name,
            limit=source_limit,
        )
    finally:
        engine.dispose()

    return build_coarse_o_train_support_review_bundle(
        source_df,
        benchmark_run_dir=benchmark_run_dir,
        config=config,
    )


def build_coarse_o_train_support_review_bundle(
    source_df: pd.DataFrame,
    *,
    benchmark_run_dir: str | Path | None = None,
    config: CoarseOTrainSupportReviewConfig = _DEFAULT_SUPPORT_CONFIG,
) -> CoarseOTrainSupportReviewBundle:
    # Собираем reproducible train/test support review bundle для coarse source.
    split = split_dataset(
        source_df,
        split_config=config.split_config,
        stratify_columns=GAIA_ID_COARSE_CLASSIFICATION_TASK.stratify_columns,
    )
    benchmark_bundle = (
        load_benchmark_review_bundle(benchmark_run_dir)
        if benchmark_run_dir is not None
        else None
    )
    return CoarseOTrainSupportReviewBundle(
        config=config,
        source_df=source_df,
        split=split,
        benchmark_bundle=benchmark_bundle,
    )


def build_benchmark_alignment_frame(
    bundle: CoarseOTrainSupportReviewBundle,
) -> pd.DataFrame:
    # Проверяем, совпадает ли reconstructed split с сохраненным benchmark run.
    benchmark_counts = _build_benchmark_row_count_map(bundle.benchmark_bundle)
    rows: list[dict[str, object]] = []

    for split_name, frame in _iter_split_frames(bundle.split):
        benchmark_count = benchmark_counts.get(split_name)
        rows.append(
            {
                "split_name": split_name,
                "n_rows_reconstructed": int(frame.shape[0]),
                "n_rows_benchmark": benchmark_count if benchmark_count is not None else pd.NA,
                "is_match": (
                    pd.NA
                    if benchmark_count is None
                    else bool(int(frame.shape[0]) == benchmark_count)
                ),
            }
        )

    return pd.DataFrame.from_records(rows)


def build_split_membership_summary_frame(
    bundle: CoarseOTrainSupportReviewBundle,
) -> pd.DataFrame:
    # Совместимое имя для notebook-level summary reconstructed split.
    return build_benchmark_alignment_frame(bundle)


def build_true_o_split_support_frame(
    bundle: CoarseOTrainSupportReviewBundle,
) -> pd.DataFrame:
    # Сколько true `O` строк есть в full/train/test и какова их доля.
    total_true_o = int(_build_true_o_frame(bundle.source_df).shape[0])
    rows: list[dict[str, object]] = []

    for split_name, frame in _iter_split_frames(bundle.split):
        true_o_df = _build_true_o_frame(frame)
        n_rows_split = int(frame.shape[0])
        n_rows_true_o = int(true_o_df.shape[0])
        rows.append(
            {
                "split_name": split_name,
                "n_rows_split": n_rows_split,
                "n_rows_true_o": n_rows_true_o,
                "share_true_o_in_split": (
                    float(n_rows_true_o / n_rows_split) if n_rows_split > 0 else 0.0
                ),
                "share_true_o_of_total_o": (
                    float(n_rows_true_o / total_true_o) if total_true_o > 0 else 0.0
                ),
            }
        )

    return pd.DataFrame.from_records(rows)


def build_true_o_stage_support_frame(
    bundle: CoarseOTrainSupportReviewBundle,
) -> pd.DataFrame:
    # Как true `O` раскладывается по `evolution_stage` в full/train/test.
    rows: list[dict[str, object]] = []

    for split_name, frame in _iter_split_frames(bundle.split):
        true_o_df = _build_true_o_frame(frame)
        if true_o_df.empty:
            continue
        grouped_df = (
            true_o_df.groupby("evolution_stage", dropna=False, sort=True)
            .agg(n_rows=("source_id", "size"))
            .reset_index()
        )
        total_rows = _sum_int_column(grouped_df, "n_rows")
        for row in grouped_df.to_dict(orient="records"):
            n_rows = _require_int_like(row["n_rows"])
            rows.append(
                {
                    "split_name": split_name,
                    "evolution_stage": row["evolution_stage"],
                    "n_rows": n_rows,
                    "share_within_split_true_o": (
                        float(n_rows / total_rows) if total_rows > 0 else 0.0
                    ),
                }
            )

    return pd.DataFrame.from_records(rows)


def build_true_o_teff_band_support_frame(
    bundle: CoarseOTrainSupportReviewBundle,
) -> pd.DataFrame:
    # Показываем, как распределен true `O` по temperature bands в full/train/test.
    rows: list[dict[str, object]] = []

    for split_name, frame in _iter_split_frames(bundle.split):
        true_o_df = _build_true_o_frame(frame)
        if true_o_df.empty:
            continue
        teff_band_df = true_o_df.copy()
        teff_band_df["teff_band"] = teff_band_df["teff_gspphot"].map(
            lambda value: _format_teff_band_label(
                value,
                band_edges_k=bundle.config.teff_band_edges_k,
            )
        )
        grouped_df = (
            teff_band_df.groupby("teff_band", dropna=False, sort=False)
            .agg(n_rows=("source_id", "size"))
            .reset_index()
        )
        total_rows = _sum_int_column(grouped_df, "n_rows")
        for row in grouped_df.to_dict(orient="records"):
            n_rows = _require_int_like(row["n_rows"])
            rows.append(
                {
                    "split_name": split_name,
                    "teff_band": row["teff_band"],
                    "n_rows": n_rows,
                    "share_within_split_true_o": (
                        float(n_rows / total_rows) if total_rows > 0 else 0.0
                    ),
                }
            )

    result = pd.DataFrame.from_records(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["split_name", "teff_band"],
        ascending=[True, True],
        kind="mergesort",
        ignore_index=True,
    )


def build_hot_ob_boundary_support_frame(
    bundle: CoarseOTrainSupportReviewBundle,
) -> pd.DataFrame:
    # Смотрим поддержку hot `O/B` boundary slice в full/train/test еще до inference.
    rows: list[dict[str, object]] = []

    for split_name, frame in _iter_split_frames(bundle.split):
        hot_ob_df = _build_hot_ob_frame(frame, config=bundle.config)
        if hot_ob_df.empty:
            continue
        grouped_df = (
            hot_ob_df.groupby("spec_class", dropna=False, sort=True)
            .agg(n_rows=("source_id", "size"))
            .reset_index()
        )
        total_rows = _sum_int_column(grouped_df, "n_rows")
        for row in grouped_df.to_dict(orient="records"):
            n_rows = _require_int_like(row["n_rows"])
            rows.append(
                {
                    "split_name": split_name,
                    "spec_class": row["spec_class"],
                    "n_rows": n_rows,
                    "share_within_hot_boundary": (
                        float(n_rows / total_rows) if total_rows > 0 else 0.0
                    ),
                }
            )

    return pd.DataFrame.from_records(rows)


def build_true_o_physics_summary_frame(
    bundle: CoarseOTrainSupportReviewBundle,
) -> pd.DataFrame:
    # Сравниваем медианную физику true `O` между full/train/test.
    rows: list[dict[str, object]] = []

    for split_name, frame in _iter_split_frames(bundle.split):
        true_o_df = _build_true_o_frame(frame)
        if true_o_df.empty:
            continue
        rows.append(
            {
                "split_name": split_name,
                "n_rows_true_o": int(true_o_df.shape[0]),
                "median_teff_gspphot": _median_numeric_value(true_o_df, "teff_gspphot"),
                "median_logg_gspphot": _median_numeric_value(true_o_df, "logg_gspphot"),
                "median_bp_rp": _median_numeric_value(true_o_df, "bp_rp"),
                "median_radius_feature": _median_or_na(true_o_df, "radius_feature"),
                "median_radius_flame": _median_or_na(true_o_df, "radius_flame"),
            }
        )

    return pd.DataFrame.from_records(rows)


def build_hottest_true_o_preview_frame(
    bundle: CoarseOTrainSupportReviewBundle,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    # Показываем самые горячие true `O` rows и их фактический train/test membership.
    rows: list[dict[str, object]] = []
    preview_columns = [
        name
        for name in (
            "source_id",
            "spec_class",
            "evolution_stage",
            "teff_gspphot",
            "logg_gspphot",
            "mh_gspphot",
            "bp_rp",
            "parallax",
            "parallax_over_error",
            "ruwe",
            "radius_feature",
            "radius_flame",
        )
        if name in bundle.source_df.columns
    ]

    for split_name, frame in (
        ("train", bundle.split.train_df),
        ("test", bundle.split.test_df),
    ):
        true_o_df = _build_true_o_frame(frame).loc[:, preview_columns].copy()
        if true_o_df.empty:
            continue
        true_o_df["split_name"] = split_name
        rows.extend(true_o_df.to_dict(orient="records"))

    preview_df = pd.DataFrame.from_records(rows)
    if preview_df.empty:
        return preview_df
    return preview_df.sort_values(
        ["teff_gspphot", "split_name", "source_id"],
        ascending=[False, True, True],
        kind="mergesort",
        ignore_index=True,
    ).head(top_n)


def _iter_split_frames(split: DatasetSplit) -> tuple[tuple[str, pd.DataFrame], ...]:
    return (
        ("full", split.full_df),
        ("train", split.train_df),
        ("test", split.test_df),
    )


def _build_benchmark_row_count_map(
    benchmark_bundle: BenchmarkReviewBundle | None,
) -> dict[str, int]:
    if benchmark_bundle is None:
        return {}
    metadata = benchmark_bundle.metadata
    return {
        "full": int(metadata["n_rows_full"]),
        "train": int(metadata["n_rows_train"]),
        "test": int(metadata["n_rows_test"]),
    }


def _build_true_o_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["spec_class"].astype("string").str.upper() == "O"].copy()


def _build_hot_ob_frame(
    df: pd.DataFrame,
    *,
    config: CoarseOTrainSupportReviewConfig,
) -> pd.DataFrame:
    spec_class_series = df["spec_class"].astype("string").str.upper()
    hot_mask = df["teff_gspphot"].map(
        lambda value: (_to_optional_float(value) or float("-inf")) >= config.hot_teff_min_k
    )
    boundary_mask = spec_class_series.isin({"O", "B"}) & hot_mask
    return df.loc[boundary_mask].copy()


def _format_teff_band_label(
    value: object,
    *,
    band_edges_k: tuple[float, ...],
) -> str:
    numeric_value = _to_optional_float(value)
    if numeric_value is None:
        return "missing"

    sorted_edges = tuple(sorted(float(edge) for edge in band_edges_k))
    if numeric_value < sorted_edges[0]:
        return f"< {int(sorted_edges[0])} K"

    for left_edge, right_edge in zip(sorted_edges[:-1], sorted_edges[1:], strict=True):
        if left_edge <= numeric_value < right_edge:
            return f"{int(left_edge)}-{int(right_edge) - 1} K"

    return f">= {int(sorted_edges[-1])} K"


def _median_or_na(df: pd.DataFrame, column_name: str) -> float | object:
    if column_name not in df.columns:
        return pd.NA
    median_value = _median_numeric_value(df, column_name)
    return median_value if median_value is not pd.NA else pd.NA


def _median_numeric_value(df: pd.DataFrame, column_name: str) -> float | object:
    if column_name not in df.columns:
        return pd.NA
    numeric_values = [
        numeric_value
        for numeric_value in (_to_optional_float(value) for value in df[column_name].tolist())
        if numeric_value is not None
    ]
    if not numeric_values:
        return pd.NA
    median_value = pd.Series(numeric_values, dtype="float64").median()
    return float(median_value) if pd.notna(median_value) else pd.NA


def _sum_int_column(df: pd.DataFrame, column_name: str) -> int:
    if column_name not in df.columns:
        return 0
    return sum(_require_int_like(value) for value in df[column_name].tolist())


def _require_int_like(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Expected integer-like scalar, got {type(value).__name__}.")
    return int(value)


def _to_optional_float(value: object) -> float | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return None if pd.isna(value) else float(value)
    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            return None
        try:
            return float(stripped_value)
        except ValueError:
            return None
    return None
