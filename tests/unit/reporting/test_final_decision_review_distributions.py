# Тестовый файл `test_final_decision_review_distributions.py` домена `reporting`.
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
    build_decision_reason_frame,
    build_domain_distribution_frame,
    build_domain_quality_crosstab_frame,
    build_quality_distribution_frame,
    build_quality_reason_frame,
    build_refinement_distribution_frame,
    build_review_bucket_frame,
    load_final_decision_review_bundle,
)

from .final_decision_review_testkit import (
    build_decision_input_df,
    build_final_decision_df,
    build_priority_input_df,
    build_priority_ranking_df,
    require_int_scalar,
)


def test_final_decision_review_distribution_frames_cover_main_outputs(
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

    domain_df = build_domain_distribution_frame(bundle)
    quality_df = build_quality_distribution_frame(bundle)
    refinement_df = build_refinement_distribution_frame(bundle)
    reason_df = build_decision_reason_frame(bundle)
    quality_reason_df = build_quality_reason_frame(bundle)
    review_bucket_df = build_review_bucket_frame(bundle)
    crosstab_df = build_domain_quality_crosstab_frame(bundle)

    assert set(domain_df["final_domain_state"]) == {"id", "candidate_ood", "unknown"}
    assert set(quality_df["final_quality_state"]) == {"pass", "reject"}
    assert "accepted" in set(refinement_df["final_refinement_state"])
    assert "refinement_accepted" in set(reason_df["final_decision_reason"])
    assert "clean" in set(quality_reason_df["quality_reason"])
    assert "review_high_ruwe" in set(review_bucket_df["review_bucket"])
    assert require_int_scalar(crosstab_df.loc["id", "pass"]) == 1
