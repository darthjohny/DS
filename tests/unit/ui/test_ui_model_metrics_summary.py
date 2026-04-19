# Тестовый файл `test_ui_model_metrics_summary.py` домена `ui`.
#
# Этот файл проверяет только:
# - helper-слой trust-summary страницы качества моделей;
# - классификацию stage-level риска и верхнюю benchmark overview-сводку.
#
# Следующий слой:
# - визуальные компоненты страницы метрик;
# - smoke-проверки рендера benchmark overview.

from __future__ import annotations

import pandas as pd

from exohost.ui.model_metrics_summary import (
    build_ui_metric_stage_assessment_frame,
    build_ui_model_metrics_overview,
)


def test_build_ui_metric_stage_assessment_frame_assigns_expected_trust_levels() -> None:
    assessment_df = build_ui_metric_stage_assessment_frame(_build_metrics_df())

    trust_by_stage = {
        row["stage_name"]: row["trust_level"]
        for _, row in assessment_df.iterrows()
    }

    assert trust_by_stage == {
        "ID/OOD": "strong",
        "Coarse": "strong",
        "Host": "stable",
        "Refinement": "caution",
    }


def test_build_ui_model_metrics_overview_summarizes_best_and_weakest_stages() -> None:
    overview = build_ui_model_metrics_overview(
        build_ui_metric_stage_assessment_frame(_build_metrics_df())
    )

    assert overview.n_strong_stages == 2
    assert overview.n_stable_stages == 1
    assert overview.n_caution_stages == 1
    assert overview.best_stage_name == "Coarse"
    assert overview.weakest_stage_name == "Refinement"


def test_build_ui_metric_stage_assessment_frame_marks_missing_metrics() -> None:
    metrics_df = _build_metrics_df()
    metrics_df.loc[0, "test_macro_f1"] = pd.NA

    assessment_df = build_ui_metric_stage_assessment_frame(metrics_df)

    assert (
        assessment_df.loc[assessment_df["stage_name"] == "ID/OOD", "trust_level"].iloc[0]
        == "missing"
    )


def _build_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage_name": "ID/OOD",
                "benchmark_run_dir": "ood_run",
                "test_macro_f1": 0.945,
                "test_balanced_accuracy": 0.926,
                "test_roc_auc_ovr": 0.996,
                "cv_mean_macro_f1": 0.948,
                "n_rows_test": 100,
                "note": "ood note",
            },
            {
                "stage_name": "Coarse",
                "benchmark_run_dir": "coarse_run",
                "test_macro_f1": 0.993,
                "test_balanced_accuracy": 0.992,
                "test_roc_auc_ovr": 1.000,
                "cv_mean_macro_f1": 0.992,
                "n_rows_test": 100,
                "note": "coarse note",
            },
            {
                "stage_name": "Host",
                "benchmark_run_dir": "host_run",
                "test_macro_f1": 0.846,
                "test_balanced_accuracy": 0.812,
                "test_roc_auc_ovr": 0.931,
                "cv_mean_macro_f1": 0.836,
                "n_rows_test": 100,
                "note": "host note",
            },
            {
                "stage_name": "Refinement",
                "benchmark_run_dir": "refinement_run",
                "test_macro_f1": 0.190,
                "test_balanced_accuracy": 0.188,
                "test_roc_auc_ovr": 0.713,
                "cv_mean_macro_f1": 0.194,
                "n_rows_test": 100,
                "note": "refinement note",
            },
        ]
    )
