# Файл `coarse_ob_domain_shift_bundle.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from exohost.datasets.load_coarse_ob_downstream_boundary_dataset import (
    load_coarse_ob_downstream_boundary_dataset,
)
from exohost.db.engine import make_read_only_engine
from exohost.evaluation.hierarchical_tasks import GAIA_ID_COARSE_CLASSIFICATION_TASK
from exohost.reporting.coarse_ob_domain_shift_contracts import (
    DEFAULT_COARSE_OB_DOMAIN_SHIFT_CONFIG,
    CoarseOBDomainShiftConfig,
    CoarseOBDomainShiftReviewBundle,
)
from exohost.reporting.coarse_ob_domain_shift_scoring import (
    build_coarse_ob_domain_scored_frame,
)
from exohost.training.hierarchical_source import load_hierarchical_prepared_training_frame


def load_coarse_ob_domain_shift_review_bundle_from_env(
    *,
    coarse_model_run_dir: str | Path,
    config: CoarseOBDomainShiftConfig = DEFAULT_COARSE_OB_DOMAIN_SHIFT_CONFIG,
    source_limit: int | None = None,
    dotenv_path: str = ".env",
    connect_timeout: int | None = 10,
) -> CoarseOBDomainShiftReviewBundle:
    # Загружаем оба `O/B` domains и собираем единый bundle для domain-shift review.
    engine = make_read_only_engine(
        dotenv_path=dotenv_path,
        connect_timeout=connect_timeout,
    )
    try:
        train_source_df = load_hierarchical_prepared_training_frame(
            engine,
            task_name=GAIA_ID_COARSE_CLASSIFICATION_TASK.name,
            limit=source_limit,
        )
        downstream_boundary_df = load_coarse_ob_downstream_boundary_dataset(
            engine,
            quality_state=config.quality_state,
            teff_min_k=config.hot_teff_min_k,
            limit=source_limit,
        )
    finally:
        engine.dispose()

    return build_coarse_ob_domain_shift_review_bundle(
        train_source_df,
        downstream_boundary_df=downstream_boundary_df,
        coarse_model_run_dir=coarse_model_run_dir,
        config=config,
    )


def build_coarse_ob_domain_shift_review_bundle(
    train_source_df: pd.DataFrame,
    *,
    downstream_boundary_df: pd.DataFrame,
    coarse_model_run_dir: str | Path,
    config: CoarseOBDomainShiftConfig = DEFAULT_COARSE_OB_DOMAIN_SHIFT_CONFIG,
) -> CoarseOBDomainShiftReviewBundle:
    # Собираем typed bundle для train/downstream `O/B` domain comparison.
    train_boundary_df = build_train_time_ob_boundary_frame(train_source_df, config=config)
    downstream_boundary_df = build_downstream_ob_boundary_frame(
        downstream_boundary_df,
        config=config,
    )
    train_scored_df = build_coarse_ob_domain_scored_frame(
        train_boundary_df,
        coarse_model_run_dir=coarse_model_run_dir,
        domain_name="train_time",
    )
    downstream_scored_df = build_coarse_ob_domain_scored_frame(
        downstream_boundary_df,
        coarse_model_run_dir=coarse_model_run_dir,
        domain_name="downstream_pass",
    )
    return CoarseOBDomainShiftReviewBundle(
        config=config,
        train_boundary_df=train_boundary_df,
        downstream_boundary_df=downstream_boundary_df,
        train_scored_df=train_scored_df,
        downstream_scored_df=downstream_scored_df,
    )


def build_train_time_ob_boundary_frame(
    train_source_df: pd.DataFrame,
    *,
    config: CoarseOBDomainShiftConfig = DEFAULT_COARSE_OB_DOMAIN_SHIFT_CONFIG,
) -> pd.DataFrame:
    # Выделяем train-time hot `O/B` boundary из coarse training source.
    spec_class_series = train_source_df["spec_class"].astype("string").str.upper()
    teff_series = pd.to_numeric(train_source_df["teff_gspphot"], errors="coerce")
    if not isinstance(teff_series, pd.Series):
        raise TypeError("Expected pandas Series after teff_gspphot normalization.")
    hot_mask = teff_series.ge(config.hot_teff_min_k)
    result = train_source_df.loc[spec_class_series.isin(["O", "B"]) & hot_mask].copy()
    result["spectral_class"] = spec_class_series.loc[result.index]
    result["domain_name"] = "train_time"
    return result.reset_index(drop=True)


def build_downstream_ob_boundary_frame(
    downstream_boundary_df: pd.DataFrame,
    *,
    config: CoarseOBDomainShiftConfig = DEFAULT_COARSE_OB_DOMAIN_SHIFT_CONFIG,
) -> pd.DataFrame:
    # Нормализуем downstream hot pass boundary под общий compare-contract.
    result = downstream_boundary_df.copy()
    result["spectral_class"] = result["spectral_class"].astype("string").str.upper()
    if "radius_feature" not in result.columns and "radius_flame" in result.columns:
        result["radius_feature"] = result["radius_flame"]
    result["domain_name"] = "downstream_pass"
    return result.reset_index(drop=True)
