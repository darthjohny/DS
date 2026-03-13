"""Тесты для legacy baseline в comparison-layer."""

from __future__ import annotations

import pandas as pd
from analysis.model_comparison import (
    LEGACY_GAUSSIAN_MODEL_NAME,
    CrossValidationConfig,
    SearchConfig,
    SplitConfig,
    run_legacy_gaussian_baseline,
    split_benchmark_dataset,
)


def make_legacy_benchmark_df() -> pd.DataFrame:
    """Собрать синтетический benchmark dataset для legacy baseline."""
    rows: list[dict[str, object]] = []
    source_id = 5000
    feature_templates = {
        "M": [
            (3150.0, 4.90, 0.25),
            (3180.0, 4.85, 0.27),
            (3195.0, 4.80, 0.29),
            (3170.0, 4.88, 0.26),
            (3160.0, 4.83, 0.28),
            (3210.0, 4.87, 0.30),
        ],
        "K": [
            (4500.0, 4.65, 0.70),
            (4600.0, 4.60, 0.75),
            (4700.0, 4.58, 0.80),
            (4550.0, 4.62, 0.72),
            (4650.0, 4.57, 0.78),
            (4750.0, 4.55, 0.83),
        ],
        "G": [
            (5400.0, 4.52, 0.95),
            (5500.0, 4.48, 1.00),
            (5600.0, 4.45, 1.05),
            (5450.0, 4.50, 0.97),
            (5550.0, 4.46, 1.02),
            (5650.0, 4.43, 1.08),
        ],
        "F": [
            (6200.0, 4.35, 1.20),
            (6350.0, 4.30, 1.30),
            (6500.0, 4.25, 1.40),
            (6250.0, 4.33, 1.24),
            (6400.0, 4.28, 1.34),
            (6550.0, 4.22, 1.44),
        ],
    }
    for spec_class, feature_rows in feature_templates.items():
        for index, (teff, logg, radius) in enumerate(feature_rows):
            rows.append(
                {
                    "source_id": source_id,
                    "spec_class": spec_class,
                    "is_host": True,
                    "teff_gspphot": teff,
                    "logg_gspphot": logg,
                    "radius_gspphot": radius,
                }
            )
            source_id += 1
            rows.append(
                {
                    "source_id": source_id,
                    "spec_class": spec_class,
                    "is_host": False,
                    "teff_gspphot": teff + 180.0 + index * 5.0,
                    "logg_gspphot": logg - 0.20,
                    "radius_gspphot": radius + 0.25,
                }
            )
            source_id += 1
    return pd.DataFrame(rows)


def test_run_legacy_gaussian_baseline_returns_common_score_contract() -> None:
    """Legacy wrapper должен возвращать единый score contract."""
    df_benchmark = make_legacy_benchmark_df()
    split = split_benchmark_dataset(
        df_benchmark,
        split_config=SplitConfig(test_size=0.25, random_state=11),
    )

    run = run_legacy_gaussian_baseline(
        split,
        cv_config=CrossValidationConfig(n_splits=3),
        search_config=SearchConfig(refit_metric="roc_auc", precision_k=5),
    )

    assert run.scored_split.model_name == LEGACY_GAUSSIAN_MODEL_NAME
    assert run.search_summary.model_name == LEGACY_GAUSSIAN_MODEL_NAME
    assert run.search_summary.refit_metric == "roc_auc"
    assert run.search_summary.precision_k == 5
    assert run.search_summary.cv_folds == 3
    assert run.search_summary.n_train_rows == len(split.train_df)
    assert run.search_summary.n_host > 0
    assert run.search_summary.n_field > 0
    assert run.search_summary.candidate_count == 6
    assert 0.0 <= run.search_summary.best_cv_score <= 1.0
    assert sorted(run.search_summary.best_params.keys()) == [
        "shrink_alpha",
        "use_m_subclasses",
    ]
    assert set(run.scored_split.train_scored_df["source_id"]) == set(split.train_df["source_id"])
    assert set(run.scored_split.test_scored_df["source_id"]) == set(split.test_df["source_id"])

    for scored_df in (run.scored_split.train_scored_df, run.scored_split.test_scored_df):
        assert "model_name" in scored_df.columns
        assert "model_score" in scored_df.columns
        assert "gauss_label" in scored_df.columns
        assert "d_mahal" in scored_df.columns
        assert "similarity" in scored_df.columns
        assert scored_df["model_name"].eq(LEGACY_GAUSSIAN_MODEL_NAME).all()
        assert scored_df["model_score"].equals(scored_df["similarity"])
        assert scored_df["model_score"].between(0.0, 1.0).all()
