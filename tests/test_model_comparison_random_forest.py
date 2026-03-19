"""Тесты для RandomForest baseline в comparison-layer."""

from __future__ import annotations

import pandas as pd
from analysis.model_comparison import (
    RANDOM_FOREST_MODEL_NAME,
    CrossValidationConfig,
    SearchConfig,
    SplitConfig,
    run_random_forest_baseline,
    split_benchmark_dataset,
)


def make_random_forest_benchmark_df() -> pd.DataFrame:
    """Собрать синтетический benchmark dataset для RF baseline."""
    rows: list[dict[str, object]] = []
    source_id = 9000
    feature_templates = {
        "M": [
            (3120.0, 4.90, 0.24),
            (3160.0, 4.86, 0.27),
            (3200.0, 4.82, 0.31),
            (3240.0, 4.78, 0.34),
            (3280.0, 4.74, 0.37),
            (3320.0, 4.70, 0.41),
        ],
        "K": [
            (4400.0, 4.68, 0.66),
            (4520.0, 4.64, 0.72),
            (4640.0, 4.60, 0.78),
            (4760.0, 4.56, 0.84),
            (4880.0, 4.52, 0.90),
            (5000.0, 4.48, 0.96),
        ],
        "G": [
            (5280.0, 4.56, 0.92),
            (5400.0, 4.52, 0.99),
            (5520.0, 4.48, 1.06),
            (5640.0, 4.44, 1.13),
            (5760.0, 4.40, 1.20),
            (5880.0, 4.36, 1.27),
        ],
        "F": [
            (6080.0, 4.38, 1.14),
            (6240.0, 4.33, 1.24),
            (6400.0, 4.28, 1.34),
            (6560.0, 4.23, 1.44),
            (6720.0, 4.18, 1.54),
            (6880.0, 4.13, 1.64),
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
                    "teff_gspphot": teff + 260.0 + index * 8.0,
                    "logg_gspphot": logg - 0.28,
                    "radius_gspphot": radius + 0.30,
                }
            )
            source_id += 1
    return pd.DataFrame(rows)


def test_run_random_forest_baseline_returns_common_score_contract() -> None:
    """RF wrapper должен возвращать единый score contract."""
    df_benchmark = make_random_forest_benchmark_df()
    split = split_benchmark_dataset(
        df_benchmark,
        split_config=SplitConfig(test_size=0.25, random_state=19),
    )

    run = run_random_forest_baseline(
        split,
        cv_config=CrossValidationConfig(n_splits=3),
        search_config=SearchConfig(refit_metric="roc_auc", precision_k=5),
    )

    assert run.scored_split.model_name == RANDOM_FOREST_MODEL_NAME
    assert sorted(run.models_by_class.keys()) == ["F", "G", "K", "M"]
    assert sorted(run.search_results_by_class.keys()) == ["F", "G", "K", "M"]
    assert set(run.scored_split.train_scored_df["source_id"]) == set(split.train_df["source_id"])
    assert set(run.scored_split.test_scored_df["source_id"]) == set(split.test_df["source_id"])

    for spec_class, search_summary in run.search_results_by_class.items():
        assert search_summary.model_name == RANDOM_FOREST_MODEL_NAME
        assert search_summary.spec_class == spec_class
        assert search_summary.refit_metric == "roc_auc"
        assert search_summary.precision_k == 5
        assert search_summary.cv_folds == 3
        assert search_summary.n_train_rows > 0
        assert search_summary.n_host > 0
        assert search_summary.n_field > 0
        assert search_summary.candidate_count == 6
        assert 0.0 <= search_summary.best_cv_score <= 1.0
        assert search_summary.cv_score_std >= 0.0
        assert search_summary.cv_score_min <= search_summary.best_cv_score
        assert search_summary.cv_score_max >= search_summary.best_cv_score
        assert sorted(search_summary.best_params.keys()) == [
            "min_samples_leaf",
            "n_estimators",
        ]

    for scored_df in (run.scored_split.train_scored_df, run.scored_split.test_scored_df):
        assert "model_name" in scored_df.columns
        assert "model_score" in scored_df.columns
        assert "rf_positive_proba" in scored_df.columns
        assert "rf_predicted_is_host" in scored_df.columns
        assert scored_df["model_name"].eq(RANDOM_FOREST_MODEL_NAME).all()
        assert scored_df["model_score"].equals(scored_df["rf_positive_proba"])
        assert scored_df["model_score"].between(0.0, 1.0).all()
        assert scored_df["rf_predicted_is_host"].isin([True, False]).all()
