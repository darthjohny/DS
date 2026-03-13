"""Тесты для CLI-оркестрации comparison-layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from analysis.model_comparison import cli
from analysis.model_comparison.contracts import (
    DEFAULT_COMPARISON_PROTOCOL,
    BenchmarkSources,
    ClassSearchSummary,
    ComparisonProtocol,
    ModelScoreFrames,
    ModelSearchSummary,
    SplitConfig,
)
from analysis.model_comparison.data import BenchmarkSplit
from analysis.model_comparison.snapshot import SnapshotComparisonResult

from priority_pipeline.branching import RouterBranchFrames


def test_parse_args_defaults_follow_canonical_protocol() -> None:
    """CLI defaults должны совпадать с каноническим protocol comparison-layer."""
    args = cli.parse_args([])

    assert args.test_size == DEFAULT_COMPARISON_PROTOCOL.split.test_size
    assert args.random_state == DEFAULT_COMPARISON_PROTOCOL.split.random_state
    assert args.cv_folds == DEFAULT_COMPARISON_PROTOCOL.cv.n_splits
    assert args.cv_random_state == DEFAULT_COMPARISON_PROTOCOL.cv.random_state
    assert (
        args.search_refit_metric
        == DEFAULT_COMPARISON_PROTOCOL.search.refit_metric
    )
    assert args.precision_k == DEFAULT_COMPARISON_PROTOCOL.search.precision_k


def test_build_protocol_from_args_uses_cli_overrides() -> None:
    """CLI overrides должны корректно попадать в ComparisonProtocol."""
    args = cli.parse_args(
        [
            "--host-view",
            "lab.custom_host_view",
            "--field-view",
            "lab.custom_field_view",
            "--test-size",
            "0.25",
            "--random-state",
            "7",
            "--cv-folds",
            "12",
            "--cv-random-state",
            "17",
            "--search-refit-metric",
            "pr_auc",
            "--snapshot-source",
            "public.custom_snapshot",
        ]
    )

    protocol = cli.build_protocol_from_args(args)

    assert protocol.sources.host_view == "lab.custom_host_view"
    assert protocol.sources.field_view == "lab.custom_field_view"
    assert protocol.split.test_size == 0.25
    assert protocol.split.random_state == 7
    assert protocol.cv.n_splits == 12
    assert protocol.cv.random_state == 17
    assert protocol.search.refit_metric == "pr_auc"
    assert protocol.search.precision_k == 50
    assert protocol.snapshot_relation == "public.custom_snapshot"


def test_run_model_comparison_orchestrates_four_model_runs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """CLI orchestration должна запускать benchmark и snapshot-артефакты."""
    protocol = ComparisonProtocol(
        sources=BenchmarkSources(),
        split=SplitConfig(),
    )
    split = BenchmarkSplit(
        full_df=pd.DataFrame(),
        train_df=pd.DataFrame(),
        test_df=pd.DataFrame(),
    )
    calls: list[str] = []

    def fake_load_and_split_benchmark_dataset(
        protocol: ComparisonProtocol,
    ) -> BenchmarkSplit:
        calls.append("load")
        return split

    def make_scored_split(model_name: str) -> ModelScoreFrames:
        frame = pd.DataFrame(
            [
                {
                    "source_id": 1,
                    "spec_class": "K",
                    "is_host": True,
                    "model_name": model_name,
                    "model_score": 0.9,
                }
            ]
        )
        return ModelScoreFrames(
            model_name=model_name,
            train_scored_df=frame,
            test_scored_df=frame,
        )

    class DummyGlobalRun:
        def __init__(self, model_name: str) -> None:
            self.scored_split = make_scored_split(model_name)
            self.search_summary = ModelSearchSummary(
                model_name=model_name,
                refit_metric=protocol.search.refit_metric,
                precision_k=protocol.search.precision_k,
                cv_folds=protocol.cv.n_splits,
                n_train_rows=4,
                n_host=2,
                n_field=2,
                candidate_count=2,
                best_cv_score=0.9,
                best_params={"shrink_alpha": 0.15},
            )

    class DummyClassRun:
        def __init__(self, model_name: str) -> None:
            self.scored_split = make_scored_split(model_name)
            self.search_results_by_class = {
                spec_class: ClassSearchSummary(
                    model_name=model_name,
                    spec_class=spec_class,
                    refit_metric=protocol.search.refit_metric,
                    precision_k=protocol.search.precision_k,
                    cv_folds=protocol.cv.n_splits,
                    n_train_rows=4,
                    n_host=2,
                    n_field=2,
                    candidate_count=2,
                    best_cv_score=0.9,
                    best_params={"alpha": 0.001},
                )
                for spec_class in protocol.sources.allowed_classes
            }

    snapshot_result = SnapshotComparisonResult(
        source_name=protocol.snapshot_relation,
        input_rows=1,
        router_df=pd.DataFrame([{"source_id": 1}]),
        branches=RouterBranchFrames(
            host_df=pd.DataFrame([{"source_id": 1}]),
            low_known_df=pd.DataFrame(),
            unknown_df=pd.DataFrame(),
        ),
        model_runs=[],
    )

    def fake_main(
        split_arg: BenchmarkSplit,
        sources: BenchmarkSources,
        cv_config,
        search_config,
    ) -> DummyGlobalRun:
        assert split_arg is split
        calls.append("main")
        assert cv_config == protocol.cv
        assert search_config == protocol.search
        return DummyGlobalRun("main_contrastive_v1")

    def fake_legacy(
        split_arg: BenchmarkSplit,
        sources: BenchmarkSources,
        cv_config,
        search_config,
    ) -> DummyGlobalRun:
        assert split_arg is split
        calls.append("legacy")
        assert cv_config == protocol.cv
        assert search_config == protocol.search
        return DummyGlobalRun("baseline_legacy_gaussian")

    def fake_mlp(
        split_arg: BenchmarkSplit,
        sources: BenchmarkSources,
        cv_config,
        search_config,
    ) -> DummyClassRun:
        assert split_arg is split
        calls.append("mlp")
        assert cv_config == protocol.cv
        assert search_config == protocol.search
        return DummyClassRun("baseline_mlp_small")

    def fake_rf(
        split_arg: BenchmarkSplit,
        sources: BenchmarkSources,
        cv_config,
        search_config,
    ) -> DummyClassRun:
        assert split_arg is split
        calls.append("rf")
        assert cv_config == protocol.cv
        assert search_config == protocol.search
        return DummyClassRun("baseline_random_forest")

    def fake_save(
        run_name: str,
        scored_splits: list[ModelScoreFrames],
        *,
        output_dir: Path,
        precision_k: int,
        search_summaries,
        protocol: ComparisonProtocol,
        note: str,
    ) -> Path:
        calls.append("save")
        assert run_name == "smoke_run"
        assert output_dir == tmp_path
        assert precision_k == 25
        assert note == "cli smoke"
        assert [item.model_name for item in scored_splits] == [
            "main_contrastive_v1",
            "baseline_legacy_gaussian",
            "baseline_mlp_small",
            "baseline_random_forest",
        ]
        assert len(search_summaries) == 10
        return tmp_path / "smoke_run.md"

    def fake_run_snapshot_comparison(
        *,
        protocol: ComparisonProtocol,
        source_name: str,
        limit: int | None,
        top_k: int,
    ) -> SnapshotComparisonResult:
        calls.append("snapshot")
        assert source_name == protocol.snapshot_relation
        assert limit is None
        assert top_k == 25
        return snapshot_result

    def fake_save_snapshot_artifacts(
        run_name: str,
        result: SnapshotComparisonResult,
        *,
        output_dir: Path,
        top_k: int,
        note: str,
    ) -> Path:
        calls.append("save_snapshot")
        assert run_name == "smoke_run"
        assert result is snapshot_result
        assert output_dir == tmp_path
        assert top_k == 25
        assert note == "cli smoke"
        return tmp_path / "smoke_run_snapshot.md"

    monkeypatch.setattr(
        cli,
        "load_and_split_benchmark_dataset",
        fake_load_and_split_benchmark_dataset,
    )
    monkeypatch.setattr(cli, "run_main_contrastive_model", fake_main)
    monkeypatch.setattr(cli, "run_legacy_gaussian_baseline", fake_legacy)
    monkeypatch.setattr(cli, "run_mlp_baseline", fake_mlp)
    monkeypatch.setattr(cli, "run_random_forest_baseline", fake_rf)
    monkeypatch.setattr(cli, "save_comparison_artifacts", fake_save)
    monkeypatch.setattr(cli, "run_snapshot_comparison", fake_run_snapshot_comparison)
    monkeypatch.setattr(cli, "save_snapshot_artifacts", fake_save_snapshot_artifacts)

    result = cli.run_model_comparison(
        protocol,
        run_name="smoke_run",
        output_dir=tmp_path,
        precision_k=25,
        note="cli smoke",
        snapshot_top_k=25,
    )

    assert result.markdown_path == tmp_path / "smoke_run.md"
    assert result.snapshot_markdown_path == tmp_path / "smoke_run_snapshot.md"
    assert result.snapshot_result is snapshot_result
    assert calls == [
        "load",
        "main",
        "legacy",
        "mlp",
        "rf",
        "save",
        "snapshot",
        "save_snapshot",
    ]


def test_run_model_comparison_can_skip_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """CLI orchestration должна уметь запускаться без snapshot-слоя."""
    protocol = ComparisonProtocol(
        sources=BenchmarkSources(),
        split=SplitConfig(),
    )
    split = BenchmarkSplit(
        full_df=pd.DataFrame(),
        train_df=pd.DataFrame(),
        test_df=pd.DataFrame(),
    )
    frame = pd.DataFrame(
        [
            {
                "source_id": 1,
                "spec_class": "K",
                "is_host": True,
                "model_name": "dummy",
                "model_score": 0.9,
            }
        ]
    )
    scored = ModelScoreFrames(
        model_name="dummy",
        train_scored_df=frame,
        test_scored_df=frame,
    )

    class DummyGlobalRun:
        def __init__(self) -> None:
            self.scored_split = scored
            self.search_summary = ModelSearchSummary(
                model_name="dummy",
                refit_metric="roc_auc",
                precision_k=10,
                cv_folds=10,
                n_train_rows=1,
                n_host=1,
                n_field=0,
                candidate_count=1,
                best_cv_score=1.0,
                best_params={},
            )

    class DummyClassRun:
        def __init__(self) -> None:
            self.scored_split = scored
            self.search_results_by_class = {
                spec_class: ClassSearchSummary(
                    model_name="dummy",
                    spec_class=spec_class,
                    refit_metric="roc_auc",
                    precision_k=10,
                    cv_folds=10,
                    n_train_rows=1,
                    n_host=1,
                    n_field=0,
                    candidate_count=1,
                    best_cv_score=1.0,
                    best_params={},
                )
                for spec_class in ("M", "K", "G", "F")
            }

    monkeypatch.setattr(
        cli,
        "load_and_split_benchmark_dataset",
        lambda protocol: split,
    )
    monkeypatch.setattr(
        cli,
        "run_main_contrastive_model",
        lambda split, sources, cv_config, search_config: DummyGlobalRun(),
    )
    monkeypatch.setattr(
        cli,
        "run_legacy_gaussian_baseline",
        lambda split, sources, cv_config, search_config: DummyGlobalRun(),
    )
    monkeypatch.setattr(
        cli,
        "run_mlp_baseline",
        lambda split, sources, cv_config, search_config: DummyClassRun(),
    )
    monkeypatch.setattr(
        cli,
        "run_random_forest_baseline",
        lambda split, sources, cv_config, search_config: DummyClassRun(),
    )
    monkeypatch.setattr(
        cli,
        "save_comparison_artifacts",
        lambda *args, **kwargs: tmp_path / "skip_snapshot.md",
    )

    result = cli.run_model_comparison(
        protocol,
        run_name="skip_snapshot",
        output_dir=tmp_path,
        precision_k=10,
        run_snapshot=False,
    )

    assert result.markdown_path == tmp_path / "skip_snapshot.md"
    assert result.snapshot_markdown_path is None
    assert result.snapshot_result is None
