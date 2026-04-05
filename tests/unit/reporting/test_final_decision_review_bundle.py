# Тестовый файл `test_final_decision_review_bundle.py` домена `reporting`.
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
    build_final_decision_summary_frame,
    load_final_decision_review_bundle,
)

from .final_decision_review_testkit import (
    build_decision_input_df,
    build_final_decision_df,
    build_priority_input_df,
    build_priority_ranking_df,
    require_int_scalar,
)


def test_final_decision_review_bundle_and_summary(tmp_path: Path) -> None:
    paths = save_final_decision_artifacts(
        pipeline_name="hierarchical_final_decision",
        decision_input_df=build_decision_input_df(),
        final_decision_df=build_final_decision_df(),
        priority_input_df=build_priority_input_df(),
        priority_ranking_df=build_priority_ranking_df(),
        output_dir=tmp_path,
    )

    bundle = load_final_decision_review_bundle(paths.run_dir)
    summary_df = build_final_decision_summary_frame(bundle)

    assert bundle.run_dir == paths.run_dir
    assert list(summary_df["pipeline_name"]) == ["hierarchical_final_decision"]
    assert require_int_scalar(summary_df.loc[0, "n_rows_final_decision"]) == 3
