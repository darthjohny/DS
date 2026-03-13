"""Тесты для метрик и reporting comparison-layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from analysis.model_comparison import (
    LEGACY_GAUSSIAN_MODEL_NAME,
    RANDOM_FOREST_MODEL_NAME,
    ClassSearchSummary,
    ModelScoreFrames,
    ModelSearchSummary,
    build_comparison_markdown,
    build_comparison_summary_frame,
    build_metrics_frame,
    build_search_summary_frame,
    save_comparison_artifacts,
)


def make_perfect_scored_df(model_name: str) -> pd.DataFrame:
    """Собрать scored frame с идеальным разделением host и field."""
    return pd.DataFrame(
        [
            {
                "source_id": 1,
                "spec_class": "M",
                "is_host": True,
                "model_name": model_name,
                "model_score": 0.98,
            },
            {
                "source_id": 2,
                "spec_class": "M",
                "is_host": False,
                "model_name": model_name,
                "model_score": 0.10,
            },
            {
                "source_id": 3,
                "spec_class": "K",
                "is_host": True,
                "model_name": model_name,
                "model_score": 0.93,
            },
            {
                "source_id": 4,
                "spec_class": "K",
                "is_host": False,
                "model_name": model_name,
                "model_score": 0.08,
            },
        ]
    )


def make_search_summaries() -> list[ModelSearchSummary | ClassSearchSummary]:
    """Собрать минимальный набор search summary для reporting smoke tests."""
    return [
        ModelSearchSummary(
            model_name=LEGACY_GAUSSIAN_MODEL_NAME,
            refit_metric="roc_auc",
            precision_k=2,
            cv_folds=3,
            n_train_rows=20,
            n_host=10,
            n_field=10,
            candidate_count=6,
            best_cv_score=0.91,
            best_params={
                "shrink_alpha": 0.15,
                "use_m_subclasses": False,
            },
        ),
        ClassSearchSummary(
            model_name=RANDOM_FOREST_MODEL_NAME,
            spec_class="K",
            refit_metric="roc_auc",
            precision_k=2,
            cv_folds=3,
            n_train_rows=12,
            n_host=6,
            n_field=6,
            candidate_count=6,
            best_cv_score=0.95,
            best_params={
                "min_samples_leaf": 1,
                "n_estimators": 300,
            },
        ),
    ]


def test_build_metrics_frame_computes_expected_perfect_metrics() -> None:
    """Идеальный scored frame должен давать идеальные ranking-метрики."""
    scored_split = ModelScoreFrames(
        model_name=LEGACY_GAUSSIAN_MODEL_NAME,
        train_scored_df=make_perfect_scored_df(LEGACY_GAUSSIAN_MODEL_NAME),
        test_scored_df=make_perfect_scored_df(LEGACY_GAUSSIAN_MODEL_NAME),
    )

    metrics_df = build_metrics_frame(scored_split, precision_k=2)

    assert metrics_df["split_name"].tolist() == ["train", "test"]
    assert metrics_df["roc_auc"].eq(1.0).all()
    assert metrics_df["pr_auc"].eq(1.0).all()
    assert metrics_df["precision_at_k"].eq(1.0).all()


def test_build_comparison_markdown_mentions_models_and_tables() -> None:
    """Markdown report должен включать protocol summary и модели."""
    scored_splits = [
        ModelScoreFrames(
            model_name=LEGACY_GAUSSIAN_MODEL_NAME,
            train_scored_df=make_perfect_scored_df(LEGACY_GAUSSIAN_MODEL_NAME),
            test_scored_df=make_perfect_scored_df(LEGACY_GAUSSIAN_MODEL_NAME),
        ),
        ModelScoreFrames(
            model_name=RANDOM_FOREST_MODEL_NAME,
            train_scored_df=make_perfect_scored_df(RANDOM_FOREST_MODEL_NAME),
            test_scored_df=make_perfect_scored_df(RANDOM_FOREST_MODEL_NAME),
        ),
    ]

    summary_df = build_comparison_summary_frame(scored_splits, precision_k=2)
    search_summary_df = build_search_summary_frame(make_search_summaries())
    markdown = build_comparison_markdown(
        summary_df,
        pd.DataFrame(),
        search_summary_df=search_summary_df,
        precision_k=2,
        note="report smoke test",
    )

    assert "Model Comparison Report" in markdown
    assert LEGACY_GAUSSIAN_MODEL_NAME in markdown
    assert RANDOM_FOREST_MODEL_NAME in markdown
    assert "precision@k" in markdown
    assert "cv_folds" in markdown
    assert "search_refit_metric" in markdown
    assert "Hyperparameter Search" in markdown
    assert "best_params_json" in markdown
    assert "report smoke test" in markdown


def test_save_comparison_artifacts_writes_markdown_and_csv(tmp_path: Path) -> None:
    """save_comparison_artifacts должен создавать markdown и CSV артефакты."""
    scored_splits = [
        ModelScoreFrames(
            model_name=LEGACY_GAUSSIAN_MODEL_NAME,
            train_scored_df=make_perfect_scored_df(LEGACY_GAUSSIAN_MODEL_NAME),
            test_scored_df=make_perfect_scored_df(LEGACY_GAUSSIAN_MODEL_NAME),
        ),
        ModelScoreFrames(
            model_name=RANDOM_FOREST_MODEL_NAME,
            train_scored_df=make_perfect_scored_df(RANDOM_FOREST_MODEL_NAME),
            test_scored_df=make_perfect_scored_df(RANDOM_FOREST_MODEL_NAME),
        ),
    ]

    markdown_path = save_comparison_artifacts(
        "smoke_comparison",
        scored_splits,
        output_dir=tmp_path,
        precision_k=2,
        search_summaries=make_search_summaries(),
        note="saved from test",
    )

    assert markdown_path.exists()
    assert (tmp_path / "smoke_comparison_summary.csv").exists()
    assert (tmp_path / "smoke_comparison_classwise.csv").exists()
    assert (tmp_path / "smoke_comparison_search_summary.csv").exists()
    assert (
        tmp_path
        / f"smoke_comparison_{LEGACY_GAUSSIAN_MODEL_NAME}_test_scores.csv"
    ).exists()
    assert (
        tmp_path
        / f"smoke_comparison_{RANDOM_FOREST_MODEL_NAME}_test_scores.csv"
    ).exists()
