"""CLI-точка входа для comparative benchmark baseline-слоя."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from analysis.model_comparison.contracts import (
    DEFAULT_COMPARISON_PROTOCOL,
    SEARCH_REFIT_METRICS,
    BenchmarkSources,
    ClassSearchSummary,
    ComparisonProtocol,
    CrossValidationConfig,
    ModelScoreFrames,
    ModelSearchSummary,
    SearchConfig,
    SearchRefitMetric,
    SplitConfig,
)
from analysis.model_comparison.contrastive import run_main_contrastive_model
from analysis.model_comparison.data import load_and_split_benchmark_dataset
from analysis.model_comparison.legacy_gaussian import run_legacy_gaussian_baseline
from analysis.model_comparison.mlp_baseline import run_mlp_baseline
from analysis.model_comparison.random_forest import run_random_forest_baseline
from analysis.model_comparison.reporting import (
    DEFAULT_MODEL_COMPARISON_OUTPUT_DIR,
    build_comparison_quality_summary_frame,
    build_comparison_summary_frame,
    build_comparison_thresholds_frame,
    save_comparison_artifacts,
)
from analysis.model_comparison.snapshot import (
    DEFAULT_SNAPSHOT_TOP_K,
    SnapshotComparisonResult,
    build_snapshot_summary_frame,
    run_snapshot_comparison,
    save_snapshot_artifacts,
)
from analysis.model_comparison.validation import (
    BenchmarkValidationResult,
    save_benchmark_validation_artifacts,
    validate_benchmark_split,
)


@dataclass(slots=True)
class ComparisonRunResult:
    """Результат одного запуска comparative benchmark."""

    markdown_path: Path
    scored_splits: list[ModelScoreFrames]
    validation_markdown_path: Path | None = None
    validation_result: BenchmarkValidationResult | None = None
    snapshot_markdown_path: Path | None = None
    snapshot_result: SnapshotComparisonResult | None = None


def default_run_name() -> str:
    """Собрать имя артефактов по текущему времени."""
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return f"model_comparison_{timestamp}"


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    """Разобрать аргументы CLI для comparative benchmark."""
    parser = ArgumentParser(
        description="Сравнение contrastive host-model и baseline-моделей.",
    )
    parser.add_argument(
        "--host-view",
        default="lab.v_nasa_gaia_train_dwarfs",
        help="Relation/view с host-population для benchmark.",
    )
    parser.add_argument(
        "--field-view",
        default="lab.v_gaia_ref_mkgf_dwarfs",
        help="Relation/view с field-population для benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MODEL_COMPARISON_OUTPUT_DIR,
        help="Каталог сохранения markdown и CSV артефактов.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Префикс имени артефактов. По умолчанию генерируется автоматически.",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Произвольное пояснение, которое попадёт в markdown report.",
    )
    parser.add_argument(
        "--precision-k",
        type=int,
        default=50,
        help="Порог k для precision@k в summary-метриках.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_COMPARISON_PROTOCOL.split.test_size,
        help="Доля benchmark dataset, уходящая в test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_COMPARISON_PROTOCOL.split.random_state,
        help="Seed для общего deterministic split.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_COMPARISON_PROTOCOL.cv.n_splits,
        help="Число stratified folds для tuning-контура benchmark.",
    )
    parser.add_argument(
        "--cv-random-state",
        type=int,
        default=DEFAULT_COMPARISON_PROTOCOL.cv.random_state,
        help="Seed для cross-validation внутри tuning-контура.",
    )
    parser.add_argument(
        "--search-refit-metric",
        default=DEFAULT_COMPARISON_PROTOCOL.search.refit_metric,
        choices=SEARCH_REFIT_METRICS,
        help="Основная метрика, по которой выбирается лучшая конфигурация модели.",
    )
    parser.add_argument(
        "--snapshot-source",
        default=DEFAULT_COMPARISON_PROTOCOL.snapshot_relation,
        help="Relation/view для live snapshot после общего router + OOD.",
    )
    parser.add_argument(
        "--snapshot-limit",
        type=int,
        default=None,
        help="Необязательный LIMIT для snapshot relation.",
    )
    parser.add_argument(
        "--snapshot-top-k",
        type=int,
        default=DEFAULT_SNAPSHOT_TOP_K,
        help="Сколько top-строк сохранять в snapshot-отчёте по каждой модели.",
    )
    parser.add_argument(
        "--skip-snapshot",
        action="store_true",
        help="Не запускать live snapshot на production relation.",
    )
    return parser.parse_args(argv)


def build_protocol_from_args(args: Namespace) -> ComparisonProtocol:
    """Построить comparison protocol из аргументов CLI."""
    return ComparisonProtocol(
        sources=BenchmarkSources(
            host_view=str(args.host_view),
            field_view=str(args.field_view),
        ),
        split=SplitConfig(
            test_size=float(args.test_size),
            random_state=int(args.random_state),
        ),
        cv=CrossValidationConfig(
            n_splits=int(args.cv_folds),
            random_state=int(args.cv_random_state),
        ),
        search=SearchConfig(
            refit_metric=cast(
                SearchRefitMetric,
                str(args.search_refit_metric),
            ),
            precision_k=int(args.precision_k),
        ),
        snapshot_relation=str(args.snapshot_source),
    )


def run_model_comparison(
    protocol: ComparisonProtocol,
    *,
    run_name: str,
    output_dir: Path,
    precision_k: int,
    note: str = "",
    run_snapshot: bool = True,
    snapshot_source: str | None = None,
    snapshot_limit: int | None = None,
    snapshot_top_k: int = DEFAULT_SNAPSHOT_TOP_K,
) -> ComparisonRunResult:
    """Запустить весь comparative benchmark и сохранить артефакты."""
    split = load_and_split_benchmark_dataset(protocol=protocol)
    validation_result = validate_benchmark_split(
        split,
        protocol=protocol,
    )
    if validation_result.has_errors:
        joined_errors = "; ".join(validation_result.errors)
        raise ValueError(f"Benchmark dataset validation failed: {joined_errors}")
    validation_markdown_path = save_benchmark_validation_artifacts(
        run_name,
        validation_result,
        output_dir=output_dir,
        protocol=protocol,
        note=note,
    )

    main_run = run_main_contrastive_model(
        split,
        sources=protocol.sources,
        cv_config=protocol.cv,
        search_config=protocol.search,
    )
    legacy_run = run_legacy_gaussian_baseline(
        split,
        sources=protocol.sources,
        cv_config=protocol.cv,
        search_config=protocol.search,
    )
    mlp_run = run_mlp_baseline(
        split,
        sources=protocol.sources,
        cv_config=protocol.cv,
        search_config=protocol.search,
    )
    random_forest_run = run_random_forest_baseline(
        split,
        sources=protocol.sources,
        cv_config=protocol.cv,
        search_config=protocol.search,
    )

    scored_splits = [
        main_run.scored_split,
        legacy_run.scored_split,
        mlp_run.scored_split,
        random_forest_run.scored_split,
    ]
    search_summaries: list[ClassSearchSummary | ModelSearchSummary] = [
        main_run.search_summary,
        legacy_run.search_summary,
        *mlp_run.search_results_by_class.values(),
        *random_forest_run.search_results_by_class.values(),
    ]
    markdown_path = save_comparison_artifacts(
        run_name,
        scored_splits,
        output_dir=output_dir,
        precision_k=precision_k,
        search_summaries=search_summaries,
        protocol=protocol,
        note=note,
    )
    snapshot_result: SnapshotComparisonResult | None = None
    snapshot_markdown_path: Path | None = None
    if run_snapshot:
        snapshot_result = run_snapshot_comparison(
            protocol=protocol,
            source_name=snapshot_source or protocol.snapshot_relation,
            limit=snapshot_limit,
            top_k=snapshot_top_k,
        )
        snapshot_markdown_path = save_snapshot_artifacts(
            run_name,
            snapshot_result,
            output_dir=output_dir,
            top_k=snapshot_top_k,
            note=note,
        )

    return ComparisonRunResult(
        markdown_path=markdown_path,
        scored_splits=scored_splits,
        validation_markdown_path=validation_markdown_path,
        validation_result=validation_result,
        snapshot_markdown_path=snapshot_markdown_path,
        snapshot_result=snapshot_result,
    )


def print_summary(
    result: ComparisonRunResult,
    protocol: ComparisonProtocol,
    precision_k: int,
) -> None:
    """Напечатать короткую summary после успешного comparative run."""
    summary_df = build_comparison_summary_frame(
        result.scored_splits,
        precision_k=precision_k,
        protocol=protocol,
    )
    thresholds_df = build_comparison_thresholds_frame(
        result.scored_splits,
        protocol=protocol,
    )
    quality_summary_df = build_comparison_quality_summary_frame(
        result.scored_splits,
        protocol=protocol,
    )
    print("\n=== MODEL COMPARISON ===")
    print("Protocol:", protocol.name)
    print("Host view:", protocol.sources.host_view)
    print("Field view:", protocol.sources.field_view)
    print("Markdown report:", result.markdown_path)
    if result.validation_markdown_path is not None:
        print("Validation report:", result.validation_markdown_path)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    if not thresholds_df.empty:
        print("\n=== QUALITY THRESHOLDS ===")
        print("Threshold metric:", protocol.quality.refit_metric)
        print(thresholds_df.to_string(index=False))
    if not quality_summary_df.empty:
        print("\n=== TEST QUALITY ===")
        print(
            quality_summary_df.loc[
                quality_summary_df["split_name"] == "test"
            ].to_string(index=False)
        )
    if result.snapshot_markdown_path is not None:
        print("Snapshot report:", result.snapshot_markdown_path)
    if result.snapshot_result is not None:
        snapshot_summary_df = build_snapshot_summary_frame(result.snapshot_result)
        if not snapshot_summary_df.empty:
            print("\n=== SNAPSHOT ===")
            print(snapshot_summary_df.to_string(index=False))


def main(argv: Sequence[str] | None = None) -> None:
    """Запустить comparative benchmark end-to-end."""
    args = parse_args(argv)
    protocol = build_protocol_from_args(args)
    run_name = str(args.run_name).strip() or default_run_name()
    result = run_model_comparison(
        protocol,
        run_name=run_name,
        output_dir=Path(args.output_dir),
        precision_k=int(args.precision_k),
        note=str(args.note),
        run_snapshot=not bool(args.skip_snapshot),
        snapshot_source=str(args.snapshot_source),
        snapshot_limit=args.snapshot_limit,
        snapshot_top_k=int(args.snapshot_top_k),
    )
    print_summary(
        result=result,
        protocol=protocol,
        precision_k=int(args.precision_k),
    )


if __name__ == "__main__":
    main()
