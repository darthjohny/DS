# Тестовый файл `test_scoring_review.py` домена `reporting`.
#
# Этот файл проверяет только:
# - проверку логики домена: review-helper, сводки и notebook display-слой;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `reporting` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from exohost.models.inference import ModelScoringResult
from exohost.reporting.ranking_artifacts import save_ranking_artifacts
from exohost.reporting.scoring_artifacts import save_scoring_artifacts
from exohost.reporting.scoring_review import (
    build_goal_alignment_frame,
    build_observability_coverage_frame,
    build_prediction_distribution_frame,
    build_priority_distribution_frame,
    build_scoring_summary_frame,
    build_top_candidates_frame,
    load_scoring_review_bundle,
)


def build_scoring_result() -> ModelScoringResult:
    # Маленький scoring-result для review-слоя.
    return ModelScoringResult(
        task_name="host_field_classification",
        target_column="host_label",
        model_name="hist_gradient_boosting",
        n_rows=3,
        scored_df=pd.DataFrame(
            [
                {
                    "source_id": "1",
                    "spec_class": "G",
                    "evolution_stage": "dwarf",
                    "predicted_host_label": "host",
                    "predicted_host_label_confidence": 0.92,
                    "host_similarity_score": 0.92,
                },
                {
                    "source_id": "2",
                    "spec_class": "K",
                    "evolution_stage": "dwarf",
                    "predicted_host_label": "field",
                    "predicted_host_label_confidence": 0.71,
                    "host_similarity_score": 0.29,
                },
                {
                    "source_id": "3",
                    "spec_class": "A",
                    "evolution_stage": "evolved",
                    "predicted_host_label": "host",
                    "predicted_host_label_confidence": 0.68,
                    "host_similarity_score": 0.68,
                },
            ]
        ),
    )


def build_ranking_frame() -> pd.DataFrame:
    # Маленькая ranking-таблица для проверки alignment-метрик.
    return pd.DataFrame(
        [
            {
                "source_id": "1",
                "spec_class": "G",
                "evolution_stage": "dwarf",
                "priority_score": 0.90,
                "priority_label": "high",
                "host_similarity_score": 0.92,
                "observability_score": 0.84,
                "brightness_available": True,
                "distance_available": True,
                "astrometry_available": True,
                "observability_evidence_count": 3,
                "priority_reason": "сильный host-like сигнал",
            },
            {
                "source_id": "2",
                "spec_class": "K",
                "evolution_stage": "dwarf",
                "priority_score": 0.61,
                "priority_label": "medium",
                "host_similarity_score": 0.63,
                "observability_score": 0.70,
                "brightness_available": False,
                "distance_available": True,
                "astrometry_available": True,
                "observability_evidence_count": 2,
                "priority_reason": "хорошая наблюдательная пригодность",
            },
            {
                "source_id": "3",
                "spec_class": "A",
                "evolution_stage": "evolved",
                "priority_score": 0.31,
                "priority_label": "low",
                "host_similarity_score": 0.68,
                "observability_score": 0.46,
                "brightness_available": False,
                "distance_available": False,
                "astrometry_available": True,
                "observability_evidence_count": 1,
                "priority_reason": "упрощенная low-priority ветка по спектральному классу",
            },
        ]
    )


def test_load_scoring_review_bundle_reads_scoring_and_ranking(tmp_path: Path) -> None:
    # Проверяем, что bundle правильно подхватывает оба типа артефактов.
    scoring_paths = save_scoring_artifacts(build_scoring_result(), output_dir=tmp_path / "scoring")
    ranking_paths = save_ranking_artifacts(
        build_ranking_frame(),
        ranking_name="host_followup",
        output_dir=tmp_path / "ranking",
    )

    bundle = load_scoring_review_bundle(
        str(scoring_paths.run_dir),
        ranking_run_dir=str(ranking_paths.run_dir),
    )

    assert bundle.scoring_df.shape[0] == 3
    assert bundle.ranking_df is not None
    assert bundle.ranking_df.shape[0] == 3


def test_build_scoring_summary_frame_returns_one_row(tmp_path: Path) -> None:
    # Summary должен собирать верхнеуровневый контекст scoring-прогона.
    scoring_paths = save_scoring_artifacts(build_scoring_result(), output_dir=tmp_path)
    bundle = load_scoring_review_bundle(str(scoring_paths.run_dir))

    summary_df = build_scoring_summary_frame(bundle)

    assert summary_df.shape == (1, 7)
    assert summary_df.loc[0, "task_name"] == "host_field_classification"
    assert bool(summary_df.loc[0, "has_ranking"]) is False


def test_build_prediction_distribution_frame_counts_predictions(tmp_path: Path) -> None:
    # Проверяем распределение predicted_host_label в scored output.
    scoring_paths = save_scoring_artifacts(build_scoring_result(), output_dir=tmp_path)
    bundle = load_scoring_review_bundle(str(scoring_paths.run_dir))

    distribution_df = build_prediction_distribution_frame(bundle)

    assert list(distribution_df["prediction_label"]) == ["field", "host"]
    assert list(distribution_df["n_rows"]) == [1, 2]


def test_build_priority_distribution_and_goal_alignment_frames(tmp_path: Path) -> None:
    # Ranking review должен показывать и label-distribution, и goal-alignment метрики.
    scoring_paths = save_scoring_artifacts(build_scoring_result(), output_dir=tmp_path / "scoring")
    ranking_paths = save_ranking_artifacts(
        build_ranking_frame(),
        ranking_name="host_followup",
        output_dir=tmp_path / "ranking",
    )
    bundle = load_scoring_review_bundle(
        str(scoring_paths.run_dir),
        ranking_run_dir=str(ranking_paths.run_dir),
    )

    distribution_df = build_priority_distribution_frame(bundle)
    observability_coverage_df = build_observability_coverage_frame(bundle)
    alignment_df = build_goal_alignment_frame(bundle, top_n=3)
    top_candidates_df = build_top_candidates_frame(bundle, top_n=2)

    assert list(distribution_df["priority_label"]) == ["high", "low", "medium"]
    assert list(distribution_df["n_rows"]) == [1, 1, 1]
    assert top_candidates_df.shape[0] == 2
    coverage = dict(
        zip(
            observability_coverage_df["metric_name"].astype(str).tolist(),
            observability_coverage_df["metric_value"].tolist(),
            strict=False,
        )
    )
    assert coverage["brightness_available"] == pytest.approx(1 / 3)
    assert coverage["distance_available"] == pytest.approx(2 / 3)
    assert coverage["astrometry_available"] == pytest.approx(1.0)
    assert coverage["mean_observability_evidence_count"] == pytest.approx(2.0)

    alignment = dict(
        zip(
            alignment_df["metric_name"].astype(str).tolist(),
            alignment_df["metric_value"].tolist(),
            strict=False,
        )
    )
    assert alignment["top_n"] == 3
    assert alignment["target_class_share"] == pytest.approx(2 / 3)
    assert alignment["low_priority_class_share"] == pytest.approx(1 / 3)
    assert alignment["dwarf_share"] == pytest.approx(2 / 3)
    assert alignment["high_priority_share"] == pytest.approx(1 / 3)
    assert alignment["mean_observability_evidence_count"] == pytest.approx(2.0)


def test_build_goal_alignment_frame_rejects_missing_ranking_bundle(tmp_path: Path) -> None:
    # Alignment-метрики нельзя строить без ranking-артефактов.
    scoring_paths = save_scoring_artifacts(build_scoring_result(), output_dir=tmp_path)
    bundle = load_scoring_review_bundle(str(scoring_paths.run_dir))

    with pytest.raises(ValueError, match="does not include ranking artifacts"):
        build_goal_alignment_frame(bundle)
