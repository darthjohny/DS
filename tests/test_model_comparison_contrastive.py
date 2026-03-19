"""Тесты для wrapper-а основной contrastive-модели."""

from __future__ import annotations

import pandas as pd
from analysis.model_comparison import (
    MAIN_CONTRASTIVE_MODEL_NAME,
    CrossValidationConfig,
    SearchConfig,
    SplitConfig,
    run_main_contrastive_model,
    split_benchmark_dataset,
)


def make_contrastive_benchmark_df() -> pd.DataFrame:
    """Собрать синтетический benchmark dataset для contrastive wrapper."""
    rows: list[dict[str, object]] = []
    source_id = 12000
    feature_templates = {
        "M": [
            (3120.0, 4.92, 0.23),
            (3160.0, 4.88, 0.26),
            (3200.0, 4.84, 0.30),
            (3240.0, 4.80, 0.33),
            (3280.0, 4.76, 0.36),
            (3320.0, 4.72, 0.39),
        ],
        "K": [
            (4420.0, 4.68, 0.67),
            (4540.0, 4.64, 0.73),
            (4660.0, 4.60, 0.79),
            (4780.0, 4.56, 0.85),
            (4900.0, 4.52, 0.91),
            (5020.0, 4.48, 0.97),
        ],
        "G": [
            (5300.0, 4.56, 0.93),
            (5420.0, 4.52, 1.00),
            (5540.0, 4.48, 1.07),
            (5660.0, 4.44, 1.14),
            (5780.0, 4.40, 1.21),
            (5900.0, 4.36, 1.28),
        ],
        "F": [
            (6100.0, 4.38, 1.15),
            (6260.0, 4.33, 1.25),
            (6420.0, 4.28, 1.35),
            (6580.0, 4.23, 1.45),
            (6740.0, 4.18, 1.55),
            (6900.0, 4.13, 1.65),
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
                    "teff_gspphot": teff + 230.0 + index * 6.0,
                    "logg_gspphot": logg - 0.24,
                    "radius_gspphot": radius + 0.28,
                }
            )
            source_id += 1
    return pd.DataFrame(rows)


def test_run_main_contrastive_model_returns_common_score_contract() -> None:
    """Wrapper основной модели должен возвращать единый score contract."""
    df_benchmark = make_contrastive_benchmark_df()
    split = split_benchmark_dataset(
        df_benchmark,
        split_config=SplitConfig(test_size=0.25, random_state=23),
    )

    run = run_main_contrastive_model(
        split,
        cv_config=CrossValidationConfig(n_splits=3),
        search_config=SearchConfig(refit_metric="roc_auc", precision_k=5),
    )

    assert run.scored_split.model_name == MAIN_CONTRASTIVE_MODEL_NAME
    assert run.search_summary.model_name == MAIN_CONTRASTIVE_MODEL_NAME
    assert run.search_summary.refit_metric == "roc_auc"
    assert run.search_summary.precision_k == 5
    assert run.search_summary.cv_folds == 3
    assert run.search_summary.n_train_rows == len(split.train_df)
    assert run.search_summary.n_host > 0
    assert run.search_summary.n_field > 0
    assert run.search_summary.candidate_count == 8
    assert 0.0 <= run.search_summary.best_cv_score <= 1.0
    assert run.search_summary.cv_score_std >= 0.0
    assert run.search_summary.cv_score_min <= run.search_summary.best_cv_score
    assert run.search_summary.cv_score_max >= run.search_summary.best_cv_score
    assert sorted(run.search_summary.best_params.keys()) == [
        "min_population_size",
        "shrink_alpha",
        "use_m_subclasses",
    ]
    assert set(run.scored_split.train_scored_df["source_id"]) == set(split.train_df["source_id"])
    assert set(run.scored_split.test_scored_df["source_id"]) == set(split.test_df["source_id"])

    for scored_df in (run.scored_split.train_scored_df, run.scored_split.test_scored_df):
        assert "model_name" in scored_df.columns
        assert "model_score" in scored_df.columns
        assert "gauss_label" in scored_df.columns
        assert "host_log_likelihood" in scored_df.columns
        assert "field_log_likelihood" in scored_df.columns
        assert "host_log_lr" in scored_df.columns
        assert "host_posterior" in scored_df.columns
        assert scored_df["model_name"].eq(MAIN_CONTRASTIVE_MODEL_NAME).all()
        assert scored_df["model_score"].equals(scored_df["host_posterior"])
        assert scored_df["model_score"].between(0.0, 1.0).all()
