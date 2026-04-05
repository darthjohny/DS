# Тестовый файл `test_final_decision_review_priority_cohort.py` домена `reporting`.
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

import pytest

from exohost.reporting.final_decision_artifacts import save_final_decision_artifacts
from exohost.reporting.final_decision_review import (
    build_high_priority_candidate_physics_frame,
    build_high_priority_coarse_class_frame,
    build_high_priority_component_summary_frame,
    build_high_priority_refinement_label_frame,
    build_high_priority_summary_frame,
    load_final_decision_review_bundle,
)

from .final_decision_review_testkit import (
    build_decision_input_df,
    build_final_decision_df,
    build_priority_input_df,
    build_priority_ranking_df,
    require_int_scalar,
)


def test_high_priority_cohort_frames_build_expected_outputs(tmp_path: Path) -> None:
    paths = save_final_decision_artifacts(
        pipeline_name="hierarchical_final_decision",
        decision_input_df=build_decision_input_df(),
        final_decision_df=build_final_decision_df(),
        priority_input_df=build_priority_input_df(),
        priority_ranking_df=build_priority_ranking_df(),
        output_dir=tmp_path,
    )
    bundle = load_final_decision_review_bundle(paths.run_dir)

    summary_df = build_high_priority_summary_frame(bundle)
    coarse_df = build_high_priority_coarse_class_frame(bundle)
    refinement_df = build_high_priority_refinement_label_frame(bundle)
    component_df = build_high_priority_component_summary_frame(bundle)
    physics_df = build_high_priority_candidate_physics_frame(bundle)

    assert require_int_scalar(summary_df.loc[0, "n_rows"]) == 1
    assert float(summary_df.loc[0, "share_priority_ranking"]) == pytest.approx(0.5)
    assert float(summary_df.loc[0, "share_final_id"]) == pytest.approx(1.0)
    assert coarse_df.loc[0, "final_coarse_class"] == "G"
    assert require_int_scalar(coarse_df.loc[0, "n_rows"]) == 1
    assert refinement_df.loc[0, "final_refinement_label"] == "G2"
    assert "host_similarity_score" in set(component_df["metric_name"])
    assert list(physics_df["source_id"]) == [1]
    assert physics_df.loc[0, "final_refinement_label"] == "G2"
