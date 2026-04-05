# Тестовый файл `test_model_pipeline_review_artifacts.py` домена `reporting`.
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

from exohost.posthoc.id_ood_gate import build_id_ood_threshold_policy
from exohost.reporting.id_ood_threshold_artifacts import save_id_ood_threshold_artifact
from exohost.reporting.model_artifacts import save_model_artifacts
from exohost.reporting.model_pipeline_review import (
    build_model_artifact_summary_frame,
    build_threshold_artifact_summary_frame,
)

from .model_pipeline_review_testkit import (
    build_train_result,
    require_float_scalar,
    require_int_scalar,
)


def test_build_model_and_threshold_artifact_summary_frames(tmp_path: Path) -> None:
    model_paths = save_model_artifacts(
        build_train_result(),
        output_dir=tmp_path / "models",
    )
    threshold_paths = save_id_ood_threshold_artifact(
        build_id_ood_threshold_policy(
            tuned_threshold=0.42,
            threshold_policy_version="id_ood_threshold_v1",
            candidate_ood_threshold=0.21,
        ),
        task_name="gaia_id_ood_classification",
        model_name="hist_gradient_boosting",
        output_dir=tmp_path / "thresholds",
    )

    model_summary_df = build_model_artifact_summary_frame({"coarse": model_paths.run_dir})
    threshold_summary_df = build_threshold_artifact_summary_frame(threshold_paths.run_dir)

    assert model_summary_df["stage_name"].tolist() == ["coarse"]
    assert require_int_scalar(model_summary_df.loc[0, "n_features"]) == 2
    assert require_float_scalar(threshold_summary_df.loc[0, "threshold_value"]) == 0.42
