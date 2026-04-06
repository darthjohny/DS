# Тестовый файл `test_final_decision_review_priority.py` домена `reporting`.
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

from exohost.reporting.final_decision_artifacts import save_final_decision_artifacts
from exohost.reporting.final_decision_review import (
    build_host_priority_status_frame,
    build_priority_by_coarse_class_frame,
    build_priority_component_quantiles_frame,
    build_priority_distribution_frame,
    build_priority_reason_frame,
    build_top_priority_candidates_frame,
    load_final_decision_review_bundle,
)

from .final_decision_review_testkit import (
    build_decision_input_df,
    build_final_decision_df,
    build_priority_input_df,
    build_priority_ranking_df,
)


def test_final_decision_review_priority_frames_cover_ranking_outputs(
    tmp_path: Path,
) -> None:
    paths = save_final_decision_artifacts(
        pipeline_name="hierarchical_final_decision",
        decision_input_df=build_decision_input_df(),
        final_decision_df=build_final_decision_df(),
        priority_input_df=build_priority_input_df(),
        priority_ranking_df=build_priority_ranking_df(),
        output_dir=tmp_path,
    )
    bundle = load_final_decision_review_bundle(paths.run_dir)

    priority_df = build_priority_distribution_frame(bundle)
    priority_reason_df = build_priority_reason_frame(bundle)
    priority_quantiles_df = build_priority_component_quantiles_frame(bundle)
    priority_by_class_df = build_priority_by_coarse_class_frame(bundle)
    priority_status_df = build_host_priority_status_frame(bundle)
    top_candidates_df = build_top_priority_candidates_frame(bundle)

    assert set(priority_df["priority_label"]) == {"high", "low"}
    assert "strong host-like signal" in set(priority_reason_df["priority_reason"])
    assert "priority_score" in priority_quantiles_df.columns
    assert "G" in set(priority_by_class_df["final_coarse_class"])
    assert bool(priority_status_df.loc[0, "host_signal_available"]) is True
    assert "host-сигнал" in str(priority_status_df.loc[0, "status_note"]).lower()
    assert list(top_candidates_df["source_id"]) == [1, 2]


def test_host_priority_status_detects_host_signal_outside_decision_input(
    tmp_path: Path,
) -> None:
    decision_input_df = build_decision_input_df().drop(columns=["host_similarity_score"])
    paths = save_final_decision_artifacts(
        pipeline_name="hierarchical_final_decision",
        decision_input_df=decision_input_df,
        final_decision_df=build_final_decision_df(),
        priority_input_df=build_priority_input_df(),
        priority_ranking_df=build_priority_ranking_df(),
        output_dir=tmp_path,
    )
    bundle = load_final_decision_review_bundle(paths.run_dir)

    priority_status_df = build_host_priority_status_frame(bundle)

    assert bool(priority_status_df.loc[0, "host_signal_available"]) is True
    assert int(priority_status_df.loc[0, "priority_output_rows"]) == 2
    assert "host-сигнал" in str(priority_status_df.loc[0, "status_note"]).lower()
