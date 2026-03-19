"""Smoke-тесты threshold-based quality comparison-layer."""

from __future__ import annotations

import pandas as pd
from analysis.model_comparison import (
    ModelScoreFrames,
    build_confusion_matrix_frame,
    build_quality_classwise_frame,
    build_quality_summary_frame,
    select_model_threshold,
)

TEST_MODEL_NAME = "baseline_test_model"


def make_train_scored_df() -> pd.DataFrame:
    """Собрать train scored frame с однозначно лучшим threshold."""
    return pd.DataFrame(
        [
            {
                "source_id": 1,
                "spec_class": "M",
                "is_host": True,
                "model_name": TEST_MODEL_NAME,
                "model_score": 0.95,
            },
            {
                "source_id": 2,
                "spec_class": "M",
                "is_host": False,
                "model_name": TEST_MODEL_NAME,
                "model_score": 0.20,
            },
            {
                "source_id": 3,
                "spec_class": "K",
                "is_host": True,
                "model_name": TEST_MODEL_NAME,
                "model_score": 0.80,
            },
            {
                "source_id": 4,
                "spec_class": "K",
                "is_host": False,
                "model_name": TEST_MODEL_NAME,
                "model_score": 0.70,
            },
        ]
    )


def make_test_scored_df() -> pd.DataFrame:
    """Собрать test scored frame для quality smoke-check."""
    return pd.DataFrame(
        [
            {
                "source_id": 11,
                "spec_class": "M",
                "is_host": True,
                "model_name": TEST_MODEL_NAME,
                "model_score": 0.92,
            },
            {
                "source_id": 12,
                "spec_class": "M",
                "is_host": False,
                "model_name": TEST_MODEL_NAME,
                "model_score": 0.85,
            },
            {
                "source_id": 13,
                "spec_class": "K",
                "is_host": True,
                "model_name": TEST_MODEL_NAME,
                "model_score": 0.81,
            },
            {
                "source_id": 14,
                "spec_class": "K",
                "is_host": False,
                "model_name": TEST_MODEL_NAME,
                "model_score": 0.10,
            },
        ]
    )


def make_scored_split() -> ModelScoreFrames:
    """Собрать pair train/test для quality-модуля."""
    return ModelScoreFrames(
        model_name=TEST_MODEL_NAME,
        train_scored_df=make_train_scored_df(),
        test_scored_df=make_test_scored_df(),
    )


def test_select_model_threshold_uses_train_split_and_max_f1() -> None:
    """Threshold должен выбираться на train split и давать лучший F1."""
    threshold_summary = select_model_threshold(make_train_scored_df())

    assert threshold_summary.model_name == TEST_MODEL_NAME
    assert threshold_summary.threshold_metric == "f1"
    assert threshold_summary.threshold_source_split == "train"
    assert threshold_summary.threshold_value == 0.80
    assert threshold_summary.threshold_score == 1.0


def test_build_quality_summary_frame_applies_train_threshold_to_test() -> None:
    """Quality summary должен использовать train-selected threshold на test."""
    quality_df = build_quality_summary_frame(make_scored_split())

    assert quality_df["split_name"].tolist() == ["train", "test"]
    assert quality_df["threshold_value"].tolist() == [0.80, 0.80]

    test_row = quality_df.loc[quality_df["split_name"] == "test"].iloc[0]
    assert test_row["tp"] == 2
    assert test_row["fp"] == 1
    assert test_row["tn"] == 1
    assert test_row["fn"] == 0
    assert test_row["precision"] == 2 / 3
    assert test_row["recall"] == 1.0
    assert test_row["f1"] == 0.8
    assert test_row["specificity"] == 0.5
    assert test_row["balanced_accuracy"] == 0.75
    assert test_row["accuracy"] == 0.75


def test_build_quality_classwise_frame_returns_class_rows_for_both_splits() -> None:
    """Class-wise quality должен строиться отдельно по `M` и `K`."""
    classwise_df = build_quality_classwise_frame(make_scored_split())

    assert set(classwise_df["split_name"].tolist()) == {"train", "test"}
    assert set(classwise_df["spec_class"].tolist()) == {"M", "K"}
    assert classwise_df["quality_scope"].eq("classwise").all()


def test_build_confusion_matrix_frame_preserves_total_rows_per_split() -> None:
    """Confusion matrix rows должны суммироваться в размер split."""
    confusion_df = build_confusion_matrix_frame(make_scored_split())

    assert confusion_df.shape[0] == 8
    train_total = int(
        confusion_df.loc[confusion_df["split_name"] == "train", "n_rows"].sum()
    )
    test_total = int(
        confusion_df.loc[confusion_df["split_name"] == "test", "n_rows"].sum()
    )
    assert train_total == 4
    assert test_total == 4
