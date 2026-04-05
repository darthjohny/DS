# Файл `command.py` слоя `cli`.
#
# Этот файл отвечает только за:
# - CLI-команды и orchestration entrypoints;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - CLI-команды или support-модули этого же домена;
# - пользовательский запуск через `python -m exohost.cli.main`.

from __future__ import annotations

import argparse

from exohost.contracts.label_contract import HOST_FIELD_TARGET_COLUMN
from exohost.datasets.load_model_scoring_dataset import load_model_scoring_dataset
from exohost.db.engine import make_read_only_engine
from exohost.models.inference import ModelScoringResult, score_with_model
from exohost.ranking.priority_score import (
    DEFAULT_HOST_SCORE_COLUMN,
    build_priority_ranking_frame,
)
from exohost.reporting.model_artifacts import (
    LoadedModelArtifact,
    load_model_artifact,
)
from exohost.reporting.ranking_artifacts import save_ranking_artifacts
from exohost.reporting.scoring_artifacts import save_scoring_artifacts

from .support import (
    apply_router_predictions_for_ranking,
    build_feature_union,
    build_prioritize_context,
    build_source_name,
    format_frame_preview,
    load_candidate_frame,
    print_prioritize_stage,
    print_ranking_artifact_paths,
    print_scoring_artifact_paths,
    require_target_column,
    resolve_ranking_output_dir,
    resolve_scoring_output_dir,
)


def register_prioritize_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    # Регистрируем сквозную prioritize-команду для реальных кандидатных выборок.
    prioritize_parser = subparsers.add_parser(
        "prioritize",
        help="Сквозной candidate scoring и ranking через router + host модели.",
    )
    input_group = prioritize_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-csv",
        help="CSV с кандидатами для сквозного priority-контура.",
    )
    input_group.add_argument(
        "--relation-name",
        default=None,
        help="Relation в БД для DB-backed candidate scoring.",
    )
    prioritize_parser.add_argument(
        "--router-model-run-dir",
        required=True,
        help="Каталог сохраненной spectral-class модели.",
    )
    prioritize_parser.add_argument(
        "--host-model-run-dir",
        required=True,
        help="Каталог сохраненной host-vs-field модели.",
    )
    prioritize_parser.add_argument(
        "--stage-model-run-dir",
        default=None,
        help="Необязательный каталог сохраненной stage-модели.",
    )
    prioritize_parser.add_argument(
        "--host-score-column",
        default=DEFAULT_HOST_SCORE_COLUMN,
        help="Имя колонки с host-like score.",
    )
    prioritize_parser.add_argument(
        "--output-dir",
        default=None,
        help="Необязательный каталог для scoring-артефактов сквозного контура.",
    )
    prioritize_parser.add_argument(
        "--ranking-output-dir",
        default=None,
        help="Необязательный каталог для ranking-артефактов.",
    )
    prioritize_parser.add_argument(
        "--preview-rows",
        type=int,
        default=10,
        help="Сколько верхних строк печатать в консоль.",
    )
    prioritize_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Необязательный лимит строк для DB-backed input source.",
    )
    prioritize_parser.add_argument(
        "--dotenv-path",
        default=".env",
        help="Путь к .env файлу с параметрами подключения.",
    )
    prioritize_parser.add_argument(
        "--connect-timeout",
        type=int,
        default=10,
        help="Таймаут подключения к БД в секундах.",
    )
    prioritize_parser.set_defaults(handler=handle_prioritize)
def handle_prioritize(namespace: argparse.Namespace) -> int:
    # Выполняем сквозной candidate scoring через router + host модели и строим ranking.
    print_prioritize_stage(f"load router_model={namespace.router_model_run_dir}")
    router_artifact = load_model_artifact(namespace.router_model_run_dir)
    require_target_column(
        router_artifact,
        expected_target_column="spec_class",
        flag_name="--router-model-run-dir",
    )

    print_prioritize_stage(f"load host_model={namespace.host_model_run_dir}")
    host_artifact = load_model_artifact(namespace.host_model_run_dir)
    require_target_column(
        host_artifact,
        expected_target_column=HOST_FIELD_TARGET_COLUMN,
        flag_name="--host-model-run-dir",
    )

    stage_artifact: LoadedModelArtifact | None = None
    if namespace.stage_model_run_dir is not None:
        print_prioritize_stage(f"load stage_model={namespace.stage_model_run_dir}")
        stage_artifact = load_model_artifact(namespace.stage_model_run_dir)
        require_target_column(
            stage_artifact,
            expected_target_column="evolution_stage",
            flag_name="--stage-model-run-dir",
        )

    feature_columns = build_feature_union(
        router_artifact=router_artifact,
        host_artifact=host_artifact,
        stage_artifact=stage_artifact,
    )
    candidate_frame = load_candidate_frame(
        namespace,
        feature_columns=feature_columns,
        engine_factory=make_read_only_engine,
        dataset_loader=load_model_scoring_dataset,
    )

    print_prioritize_stage("score router")
    router_scoring_result = score_with_model(
        candidate_frame,
        estimator=router_artifact.estimator,
        task_name=router_artifact.task_name,
        target_column=router_artifact.target_column,
        feature_columns=router_artifact.feature_columns,
        model_name=router_artifact.model_name,
        host_score_column=namespace.host_score_column,
    )
    enriched_frame = router_scoring_result.scored_df

    if stage_artifact is not None:
        print_prioritize_stage("score stage")
        stage_scoring_result = score_with_model(
            enriched_frame,
            estimator=stage_artifact.estimator,
            task_name=stage_artifact.task_name,
            target_column=stage_artifact.target_column,
            feature_columns=stage_artifact.feature_columns,
            model_name=stage_artifact.model_name,
            host_score_column=namespace.host_score_column,
        )
        enriched_frame = stage_scoring_result.scored_df

    print_prioritize_stage("score host")
    host_scoring_result = score_with_model(
        enriched_frame,
        estimator=host_artifact.estimator,
        task_name=host_artifact.task_name,
        target_column=host_artifact.target_column,
        feature_columns=host_artifact.feature_columns,
        model_name=host_artifact.model_name,
        host_score_column=namespace.host_score_column,
    )
    enriched_frame = apply_router_predictions_for_ranking(
        host_scoring_result.scored_df,
        router_target_column=router_artifact.target_column,
        stage_target_column=None if stage_artifact is None else stage_artifact.target_column,
    )

    pipeline_model_name_parts = [router_artifact.model_name, host_artifact.model_name]
    if stage_artifact is not None:
        pipeline_model_name_parts.insert(1, stage_artifact.model_name)
    pipeline_scoring_result = ModelScoringResult(
        task_name="candidate_prioritization",
        target_column=host_artifact.target_column,
        model_name="__".join(pipeline_model_name_parts),
        n_rows=int(enriched_frame.shape[0]),
        scored_df=enriched_frame,
    )

    scoring_output_dir = resolve_scoring_output_dir(namespace)
    ranking_output_dir = resolve_ranking_output_dir(namespace)
    context = build_prioritize_context(
        namespace,
        scoring_output_dir=scoring_output_dir,
        ranking_output_dir=ranking_output_dir,
        router_artifact=router_artifact,
        host_artifact=host_artifact,
        stage_artifact=stage_artifact,
    )

    print_prioritize_stage("save scoring artifacts")
    scoring_artifact_paths = save_scoring_artifacts(
        pipeline_scoring_result,
        output_dir=scoring_output_dir,
        extra_metadata=context,
    )
    print("[prioritize] === scoring_preview ===")
    print(format_frame_preview(enriched_frame, preview_rows=namespace.preview_rows))
    print_scoring_artifact_paths(scoring_artifact_paths)

    ranking_frame = build_priority_ranking_frame(
        enriched_frame,
        host_score_column=namespace.host_score_column,
    )
    print_prioritize_stage("save ranking artifacts")
    ranking_artifact_paths = save_ranking_artifacts(
        ranking_frame,
        ranking_name=f"{build_source_name(namespace)}__candidate_prioritization",
        output_dir=ranking_output_dir,
        extra_metadata=context,
    )
    print("[prioritize] === ranking_preview ===")
    print(format_frame_preview(ranking_frame, preview_rows=namespace.preview_rows))
    print_ranking_artifact_paths(ranking_artifact_paths)
    return 0
