"""Тесты heavy validation слоя."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from analysis.model_comparison.contracts import ComparisonProtocol
from analysis.model_validation import (
    AVG_GAP_WARNING_THRESHOLD,
    CANONICAL_STAGE_ORDER,
    DEFAULT_MODEL_VALIDATION_OUTPUT_DIR,
    FAST_MODE_RANDOM_STATES,
    FULL_MODE_RANDOM_STATES,
    ModelValidationProtocol,
    ModelValidationRunRequest,
    ModelValidationSplitResult,
    RepeatedSplitConfig,
    build_gap_diagnostics_frame,
    build_generalization_stage_frame,
    build_generalization_summary_frame,
    build_model_risk_audit_frame,
    build_model_validation_artifact_paths,
    build_protocol_from_args,
    build_repeated_split_model_summary,
    build_repeated_splits_frame,
    build_split_protocol,
    build_validation_layout_markdown,
    initialize_model_validation_layout,
    parse_args,
    run_model_validation,
    run_model_validation_scaffold,
    run_repeated_split_evaluation,
)


def test_repeated_split_config_requires_unique_states() -> None:
    """Repeated split contract не должен принимать дубли random_state."""
    try:
        RepeatedSplitConfig(random_states=(11, 11))
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("Expected ValueError for duplicate random states.")


def test_build_model_validation_artifact_paths_uses_stable_names() -> None:
    """Artifact layout должен использовать стабильные canonical имена."""
    request = ModelValidationRunRequest(
        run_name="validation_smoke",
        output_dir=Path("experiments/model_validation"),
    )

    paths = build_model_validation_artifact_paths(request)

    assert paths.report_markdown_path.name == "validation_smoke_validation_report.md"
    assert paths.repeated_splits_csv_path.name == "validation_smoke_repeated_splits.csv"
    assert paths.model_summary_csv_path.name == "validation_smoke_model_summary.csv"
    assert (
        paths.generalization_summary_csv_path.name
        == "validation_smoke_generalization_summary.csv"
    )
    assert (
        paths.gap_diagnostics_csv_path.name == "validation_smoke_gap_diagnostics.csv"
    )
    assert paths.risk_audit_csv_path.name == "validation_smoke_risk_audit.csv"
    assert paths.plots_dir.name == "validation_smoke_plots"


def test_parse_args_builds_fast_mode_by_default() -> None:
    """CLI parser должен собирать fast-mode scaffold по умолчанию."""
    args = parse_args(["--run-name", "smoke"])
    protocol = build_protocol_from_args(args)

    assert args.mode == "fast"
    assert args.output_dir == DEFAULT_MODEL_VALIDATION_OUTPUT_DIR
    assert protocol.repeated_split.random_states == FAST_MODE_RANDOM_STATES


def test_build_protocol_from_args_uses_full_mode_defaults() -> None:
    """Full mode должен подставлять расширенный список repeated split seeds."""
    args = parse_args(["--run-name", "smoke", "--mode", "full"])
    protocol = build_protocol_from_args(args)

    assert protocol.repeated_split.random_states == FULL_MODE_RANDOM_STATES


def test_initialize_model_validation_layout_creates_output_dirs(
    tmp_path: Path,
) -> None:
    """Layout initializer должен создавать output dir и plots dir."""
    request = ModelValidationRunRequest(
        run_name="layout_smoke",
        output_dir=tmp_path,
    )

    layout = initialize_model_validation_layout(
        request,
        protocol=ModelValidationProtocol(),
    )

    assert layout.artifact_paths.output_dir.exists()
    assert layout.artifact_paths.plots_dir.exists()


def test_build_validation_layout_markdown_mentions_protocol_and_artifacts() -> None:
    """Markdown scaffold должен фиксировать protocol и planned artifacts."""
    request = ModelValidationRunRequest(
        run_name="layout_smoke",
        output_dir=Path("experiments/model_validation"),
        note="scaffold note",
    )
    layout = initialize_model_validation_layout(
        request,
        protocol=ModelValidationProtocol(),
    )

    markdown = build_validation_layout_markdown(layout)

    assert "model_generalization_validation_v1" in markdown
    assert "layout_initialized" in markdown
    assert "layout_smoke_validation_report.md" in markdown
    assert "layout_smoke_generalization_summary.csv" in markdown
    assert "layout_smoke_gap_diagnostics.csv" in markdown
    assert "scaffold note" in markdown


def test_run_model_validation_scaffold_writes_markdown_report(tmp_path: Path) -> None:
    """Scaffold-run должен писать markdown report в canonical layout."""
    request = ModelValidationRunRequest(
        run_name="run_smoke",
        output_dir=tmp_path,
        mode="full",
        include_optional_diagnostics=True,
    )

    result = run_model_validation_scaffold(
        ModelValidationProtocol(),
        request=request,
    )

    assert result.report_markdown_path.exists()
    content = result.report_markdown_path.read_text(encoding="utf-8")
    assert "run_smoke" in content
    assert "mode: `full`" in content
    assert "include_optional_diagnostics: `True`" in content


def make_split_result(
    random_state: int,
    *,
    roc_auc_test: float,
    brier_test: float,
    pr_auc_test: float = 0.78,
    precision_at_k_test: float = 0.88,
) -> ModelValidationSplitResult:
    """Собрать synthetic repeated split result для unit-тестов."""
    generalization_df = pd.DataFrame(
        [
            {
                "model_name": "baseline_random_forest",
                "metric_name": "roc_auc",
                "train_scope": "in_sample_refit",
                "test_scope": "holdout_test",
                "train_value": 0.98,
                "test_value": roc_auc_test,
                "train_minus_test": 0.98 - roc_auc_test,
                "abs_train_test_gap": abs(0.98 - roc_auc_test),
                "is_refit_metric": True,
                "cv_summary_scope": "class_weighted_mean",
                "cv_score_mean": 0.90,
                "cv_score_std": 0.03,
                "cv_score_min": 0.86,
                "cv_score_max": 0.94,
                "cv_minus_test": 0.90 - roc_auc_test,
            },
            {
                "model_name": "baseline_random_forest",
                "metric_name": "pr_auc",
                "train_scope": "in_sample_refit",
                "test_scope": "holdout_test",
                "train_value": 0.96,
                "test_value": pr_auc_test,
                "train_minus_test": 0.96 - pr_auc_test,
                "abs_train_test_gap": abs(0.96 - pr_auc_test),
                "is_refit_metric": False,
                "cv_summary_scope": None,
                "cv_score_mean": float("nan"),
                "cv_score_std": float("nan"),
                "cv_score_min": float("nan"),
                "cv_score_max": float("nan"),
                "cv_minus_test": float("nan"),
            },
            {
                "model_name": "baseline_random_forest",
                "metric_name": "precision_at_k",
                "train_scope": "in_sample_refit",
                "test_scope": "holdout_test",
                "train_value": 1.0,
                "test_value": precision_at_k_test,
                "train_minus_test": 1.0 - precision_at_k_test,
                "abs_train_test_gap": abs(1.0 - precision_at_k_test),
                "is_refit_metric": False,
                "cv_summary_scope": None,
                "cv_score_mean": float("nan"),
                "cv_score_std": float("nan"),
                "cv_score_min": float("nan"),
                "cv_score_max": float("nan"),
                "cv_minus_test": float("nan"),
            },
            {
                "model_name": "baseline_random_forest",
                "metric_name": "brier",
                "train_scope": "in_sample_refit",
                "test_scope": "holdout_test",
                "train_value": 0.08,
                "test_value": brier_test,
                "train_minus_test": 0.08 - brier_test,
                "abs_train_test_gap": abs(0.08 - brier_test),
                "is_refit_metric": False,
                "cv_summary_scope": None,
                "cv_score_mean": float("nan"),
                "cv_score_std": float("nan"),
                "cv_score_min": float("nan"),
                "cv_score_max": float("nan"),
                "cv_minus_test": float("nan"),
            },
        ]
    )
    return ModelValidationSplitResult(
        split_random_state=random_state,
        summary_df=pd.DataFrame(),
        search_summary_df=pd.DataFrame(),
        generalization_df=generalization_df,
    )


def test_build_split_protocol_replaces_random_state_only() -> None:
    """Repeated split protocol должен менять только split random_state."""
    protocol = ModelValidationProtocol()

    split_protocol = build_split_protocol(protocol, random_state=17)

    assert split_protocol.split.random_state == 17
    assert split_protocol.split.test_size == protocol.comparison_protocol.split.test_size
    assert split_protocol.sources == protocol.comparison_protocol.sources


def test_build_repeated_splits_frame_stacks_split_results() -> None:
    """Long-form repeated split frame должен включать random_state и protocol ids."""
    protocol = ModelValidationProtocol()
    repeated_df = build_repeated_splits_frame(
        [
            make_split_result(11, roc_auc_test=0.92, brier_test=0.11),
            make_split_result(17, roc_auc_test=0.87, brier_test=0.18),
        ],
        validation_protocol=protocol,
    )

    assert repeated_df.shape[0] == 8
    assert set(repeated_df["split_random_state"].tolist()) == {11, 17}
    assert repeated_df["validation_protocol_name"].nunique() == 1
    assert repeated_df["benchmark_protocol_name"].nunique() == 1


def test_build_repeated_split_model_summary_uses_metric_direction() -> None:
    """ROC-AUC должен максимизироваться, а Brier минимизироваться."""
    repeated_df = build_repeated_splits_frame(
        [
            make_split_result(11, roc_auc_test=0.92, brier_test=0.11),
            make_split_result(17, roc_auc_test=0.87, brier_test=0.18),
        ],
        validation_protocol=ModelValidationProtocol(),
    )

    summary_df = build_repeated_split_model_summary(repeated_df)
    roc_row = summary_df[summary_df["metric_name"] == "roc_auc"].iloc[0]
    brier_row = summary_df[summary_df["metric_name"] == "brier"].iloc[0]

    assert int(roc_row["best_split_random_state"]) == 11
    assert int(roc_row["worst_split_random_state"]) == 17
    assert int(brier_row["best_split_random_state"]) == 11
    assert int(brier_row["worst_split_random_state"]) == 17


def test_build_generalization_stage_frame_separates_train_cv_test() -> None:
    """Stage frame должен явно различать train/cv/test представления."""
    repeated_df = build_repeated_splits_frame(
        [make_split_result(11, roc_auc_test=0.92, brier_test=0.11)],
        validation_protocol=ModelValidationProtocol(),
    )

    stage_df = build_generalization_stage_frame(repeated_df)

    roc_stage_names = stage_df[stage_df["metric_name"] == "roc_auc"]["stage_name"].tolist()
    brier_stage_names = stage_df[stage_df["metric_name"] == "brier"]["stage_name"].tolist()
    assert roc_stage_names == list(CANONICAL_STAGE_ORDER)
    assert brier_stage_names == ["train_in_sample", "test_holdout"]


def test_build_generalization_summary_frame_aggregates_stage_scores() -> None:
    """Stage summary должен агрегировать repeated split значения по stage."""
    repeated_df = build_repeated_splits_frame(
        [
            make_split_result(11, roc_auc_test=0.92, brier_test=0.11),
            make_split_result(17, roc_auc_test=0.87, brier_test=0.18),
        ],
        validation_protocol=ModelValidationProtocol(),
    )

    stage_summary_df = build_generalization_summary_frame(
        build_generalization_stage_frame(repeated_df)
    )
    roc_cv_row = stage_summary_df[
        (stage_summary_df["metric_name"] == "roc_auc")
        & (stage_summary_df["stage_name"] == "cv_oof")
    ].iloc[0]

    assert int(roc_cv_row["split_count"]) == 2
    assert float(roc_cv_row["score_mean"]) == 0.90


def test_build_gap_diagnostics_frame_tracks_train_and_cv_gaps() -> None:
    """Gap diagnostics должен хранить separate train/test и cv/test сигналы."""
    repeated_df = build_repeated_splits_frame(
        [
            make_split_result(11, roc_auc_test=0.92, brier_test=0.11),
            make_split_result(17, roc_auc_test=0.87, brier_test=0.18),
        ],
        validation_protocol=ModelValidationProtocol(),
    )

    gap_df = build_gap_diagnostics_frame(repeated_df)
    roc_row = gap_df[gap_df["metric_name"] == "roc_auc"].iloc[0]
    brier_row = gap_df[gap_df["metric_name"] == "brier"].iloc[0]

    assert int(roc_row["cv_available_splits"]) == 2
    assert int(brier_row["cv_available_splits"]) == 0
    assert float(roc_row["abs_train_test_gap_max"]) > 0.0


def test_build_model_risk_audit_frame_scores_instability() -> None:
    """Risk audit должен собирать per-model verdict из heavy diagnostics."""
    repeated_df = build_repeated_splits_frame(
        [
            make_split_result(
                11,
                roc_auc_test=0.92,
                brier_test=0.11,
                pr_auc_test=0.79,
                precision_at_k_test=0.88,
            ),
            make_split_result(
                17,
                roc_auc_test=0.90,
                brier_test=0.12,
                pr_auc_test=0.42,
                precision_at_k_test=0.58,
            ),
        ],
        validation_protocol=ModelValidationProtocol(),
    )

    audit_df = build_model_risk_audit_frame(
        ModelValidationProtocol(),
        generalization_summary_df=build_generalization_summary_frame(
            build_generalization_stage_frame(repeated_df)
        ),
        gap_diagnostics_df=build_gap_diagnostics_frame(repeated_df),
    )

    row = audit_df.iloc[0]
    assert row["model_name"] == "baseline_random_forest"
    assert row["audit_metric"] == "roc_auc"
    assert row["risk_level"] in {"MODERATE", "HIGH"}
    assert float(row["avg_cv_test_gap"]) <= AVG_GAP_WARNING_THRESHOLD
    assert "pr_auc" in str(row["risk_reasons"])


def test_run_repeated_split_evaluation_aggregates_all_requested_states() -> None:
    """Runner должен пройти все requested random_state и собрать summary."""

    def fake_run_split(protocol: ComparisonProtocol) -> ModelValidationSplitResult:
        random_state = int(protocol.split.random_state)
        roc_auc_test = 0.90 if random_state == 11 else 0.85
        brier_test = 0.10 if random_state == 11 else 0.16
        return make_split_result(
            random_state,
            roc_auc_test=roc_auc_test,
            brier_test=brier_test,
        )

    result = run_repeated_split_evaluation(
        ModelValidationProtocol(
            repeated_split=RepeatedSplitConfig(random_states=(11, 17)),
        ),
        run_split=fake_run_split,
    )

    assert result.repeated_splits_df["split_random_state"].nunique() == 2
    assert set(result.model_summary_df["metric_name"]) == {
        "brier",
        "pr_auc",
        "precision_at_k",
        "roc_auc",
    }
    assert set(result.generalization_summary_df["stage_name"]) == {
        "train_in_sample",
        "cv_oof",
        "test_holdout",
    }
    assert "cv_minus_test_mean" in result.gap_diagnostics_df.columns
    assert "risk_level" in result.risk_audit_df.columns


def test_run_model_validation_writes_repeated_split_artifacts(tmp_path: Path) -> None:
    """Heavy validation run должен писать markdown и CSV артефакты."""
    request = ModelValidationRunRequest(
        run_name="heavy_smoke",
        output_dir=tmp_path,
        mode="fast",
    )

    def fake_run_split(protocol: ComparisonProtocol) -> ModelValidationSplitResult:
        random_state = int(protocol.split.random_state)
        return make_split_result(
            random_state,
            roc_auc_test=0.90,
            brier_test=0.10,
        )

    result = run_model_validation(
        ModelValidationProtocol(
            repeated_split=RepeatedSplitConfig(random_states=(11, 17)),
        ),
        request=request,
        run_split=fake_run_split,
    )

    assert result.report_markdown_path.exists()
    assert result.layout.artifact_paths.repeated_splits_csv_path.exists()
    assert result.layout.artifact_paths.model_summary_csv_path.exists()
    assert result.layout.artifact_paths.generalization_summary_csv_path.exists()
    assert result.layout.artifact_paths.gap_diagnostics_csv_path.exists()
    assert result.layout.artifact_paths.risk_audit_csv_path.exists()
    markdown = result.report_markdown_path.read_text(encoding="utf-8")
    assert "repeated_split_completed" in markdown
    assert "heavy_smoke_repeated_splits.csv" in markdown
    assert "heavy_smoke_model_summary.csv" in markdown
    assert "heavy_smoke_generalization_summary.csv" in markdown
    assert "heavy_smoke_gap_diagnostics.csv" in markdown
    assert "heavy_smoke_risk_audit.csv" in markdown
