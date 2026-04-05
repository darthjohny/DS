# Bundle/load слой для review feature separability `O vs B`.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from exohost.db.engine import make_read_only_engine
from exohost.evaluation.hierarchical_tasks import GAIA_ID_COARSE_CLASSIFICATION_TASK
from exohost.reporting.archive_research.coarse_ob_feature_separability_contracts import (
    DEFAULT_COARSE_OB_SEPARABILITY_CONFIG,
    CoarseOBFeatureSeparabilityConfig,
    CoarseOBFeatureSeparabilityReviewBundle,
)
from exohost.reporting.archive_research.coarse_ob_feature_separability_frames import (
    build_train_time_ob_boundary_frame,
)
from exohost.reporting.archive_research.coarse_ob_feature_separability_scoring import (
    build_boundary_permutation_importance_frame,
    build_coarse_ob_boundary_scored_frame,
)
from exohost.training.hierarchical_source import load_hierarchical_prepared_training_frame


def load_coarse_ob_feature_separability_review_bundle_from_env(
    *,
    coarse_model_run_dir: str | Path,
    config: CoarseOBFeatureSeparabilityConfig = DEFAULT_COARSE_OB_SEPARABILITY_CONFIG,
    source_limit: int | None = None,
    dotenv_path: str = ".env",
    connect_timeout: int | None = 10,
) -> CoarseOBFeatureSeparabilityReviewBundle:
    # Загружаем live coarse source и строим narrow `O/B` separability bundle.
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

    return build_coarse_ob_feature_separability_review_bundle(
        source_df,
        coarse_model_run_dir=coarse_model_run_dir,
        config=config,
    )


def build_coarse_ob_feature_separability_review_bundle(
    source_df: pd.DataFrame,
    *,
    coarse_model_run_dir: str | Path,
    config: CoarseOBFeatureSeparabilityConfig = DEFAULT_COARSE_OB_SEPARABILITY_CONFIG,
) -> CoarseOBFeatureSeparabilityReviewBundle:
    # Собираем bundle для train-time `O/B` feature separability review.
    boundary_df = build_train_time_ob_boundary_frame(source_df, config=config)
    scored_boundary_df = build_coarse_ob_boundary_scored_frame(
        boundary_df,
        coarse_model_run_dir=coarse_model_run_dir,
    )
    permutation_importance_df = build_boundary_permutation_importance_frame(
        boundary_df,
        coarse_model_run_dir=coarse_model_run_dir,
        config=config,
    )
    return CoarseOBFeatureSeparabilityReviewBundle(
        config=config,
        source_df=source_df,
        boundary_df=boundary_df,
        scored_boundary_df=scored_boundary_df,
        permutation_importance_df=permutation_importance_df,
    )
