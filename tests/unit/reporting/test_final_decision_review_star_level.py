# Тестовый файл `test_final_decision_review_star_level.py` домена `reporting`.
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
    build_final_coarse_class_frame,
    build_final_refinement_label_frame,
    build_numeric_state_summary_frame,
    build_star_level_result_frame,
    build_star_result_preview_frame,
    load_final_decision_review_bundle,
)

from .final_decision_review_testkit import (
    build_decision_input_df,
    build_final_decision_df,
    build_priority_input_df,
    build_priority_ranking_df,
)


def test_final_decision_review_star_level_frames_cover_main_views(tmp_path: Path) -> None:
    paths = save_final_decision_artifacts(
        pipeline_name="hierarchical_final_decision",
        decision_input_df=build_decision_input_df(),
        final_decision_df=build_final_decision_df(),
        priority_input_df=build_priority_input_df(),
        priority_ranking_df=build_priority_ranking_df(),
        output_dir=tmp_path,
    )
    bundle = load_final_decision_review_bundle(paths.run_dir)

    coarse_class_df = build_final_coarse_class_frame(bundle)
    refinement_label_df = build_final_refinement_label_frame(bundle)
    star_level_df = build_star_level_result_frame(bundle)
    id_preview_df = build_star_result_preview_frame(bundle, final_domain_state="id")
    unknown_preview_df = build_star_result_preview_frame(bundle, final_domain_state="unknown")
    numeric_summary_df = build_numeric_state_summary_frame(bundle)

    assert "G" in set(coarse_class_df["final_coarse_class"])
    assert "G2" in set(refinement_label_df["final_refinement_label"])
    assert "hostname" in star_level_df.columns
    assert list(id_preview_df["source_id"]) == [1]
    assert list(unknown_preview_df["source_id"]) == [3]
    assert "host_similarity_score" in set(numeric_summary_df["metric_name"])
