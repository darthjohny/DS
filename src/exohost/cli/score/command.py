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
from pathlib import Path

from exohost.datasets.load_model_scoring_dataset import load_model_scoring_dataset
from exohost.db.engine import make_read_only_engine
from exohost.models.inference import score_with_model
from exohost.ranking.priority_score import (
    DEFAULT_HOST_SCORE_COLUMN,
    build_priority_ranking_frame,
)
from exohost.reporting.model_artifacts import load_model_artifact
from exohost.reporting.ranking_artifacts import save_ranking_artifacts
from exohost.reporting.scoring_artifacts import save_scoring_artifacts

from .support import (
    build_ranking_name_from_scoring,
    build_score_context,
    format_frame_preview,
    load_score_input_frame,
    print_ranking_artifact_paths,
    print_score_stage,
    print_scoring_artifact_paths,
    resolve_host_score_column,
    resolve_ranking_output_dir,
    resolve_score_output_dir,
)


def register_score_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    # Регистрируем score-команду для ranking и model-scoring режимов.
    score_parser = subparsers.add_parser(
        "score",
        help="Скоринг и ранжирование целей V2.",
    )
    input_group = score_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-csv",
        help="CSV с колонками ranking-слоя или признаками сохраненной модели.",
    )
    input_group.add_argument(
        "--relation-name",
        default=None,
        help="Relation в БД для DB-backed model-scoring режима.",
    )
    score_parser.add_argument(
        "--host-score-column",
        default=DEFAULT_HOST_SCORE_COLUMN,
        help="Имя колонки с host-like score или вероятностью.",
    )
    score_parser.add_argument(
        "--model-run-dir",
        default=None,
        help="Каталог сохраненного model artifact для режима model-scoring.",
    )
    score_parser.add_argument(
        "--output-dir",
        default=None,
        help="Необязательный каталог для сохранения ranking или scoring артефактов.",
    )
    score_parser.add_argument(
        "--with-ranking",
        action="store_true",
        help="После model-scoring дополнительно построить ranking по scored output.",
    )
    score_parser.add_argument(
        "--ranking-output-dir",
        default=None,
        help="Необязательный каталог для ranking-артефактов после model-scoring.",
    )
    score_parser.add_argument(
        "--preview-rows",
        type=int,
        default=10,
        help="Сколько верхних строк итоговой таблицы печатать в консоль.",
    )
    score_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Необязательный лимит строк для DB-backed input source.",
    )
    score_parser.add_argument(
        "--dotenv-path",
        default=".env",
        help="Путь к .env файлу с параметрами подключения.",
    )
    score_parser.add_argument(
        "--connect-timeout",
        type=int,
        default=10,
        help="Таймаут подключения к БД в секундах.",
    )
    score_parser.set_defaults(handler=handle_score)
def handle_score(namespace: argparse.Namespace) -> int:
    # Выполняем score-команду в ranking или model-scoring режиме.
    if namespace.model_run_dir is None:
        if namespace.with_ranking:
            raise ValueError("--with-ranking requires --model-run-dir.")
        if namespace.relation_name is not None:
            raise ValueError("DB-backed score currently requires --model-run-dir.")

        candidate_frame = load_score_input_frame(
            namespace,
            feature_columns=None,
            engine_factory=make_read_only_engine,
            dataset_loader=load_model_scoring_dataset,
        )
        output_dir = resolve_score_output_dir(namespace, score_mode="ranking")
        ranking_frame = build_priority_ranking_frame(
            candidate_frame,
            host_score_column=resolve_host_score_column(namespace),
        )
        print_score_stage("save artifacts")
        ranking_artifact_paths = save_ranking_artifacts(
            ranking_frame,
            ranking_name=Path(namespace.input_csv).stem,
            output_dir=output_dir,
            extra_metadata=build_score_context(
                namespace,
                score_mode="ranking",
                output_dir=output_dir,
            ),
        )
        print("[score] === ranking_preview ===")
        print(format_frame_preview(ranking_frame, preview_rows=namespace.preview_rows))
        print_ranking_artifact_paths(ranking_artifact_paths)
        return 0

    print_score_stage(f"load model={namespace.model_run_dir}")
    loaded_artifact = load_model_artifact(namespace.model_run_dir)
    candidate_frame = load_score_input_frame(
        namespace,
        feature_columns=loaded_artifact.feature_columns,
        engine_factory=make_read_only_engine,
        dataset_loader=load_model_scoring_dataset,
    )
    scoring_result = score_with_model(
        candidate_frame,
        estimator=loaded_artifact.estimator,
        task_name=loaded_artifact.task_name,
        target_column=loaded_artifact.target_column,
        feature_columns=loaded_artifact.feature_columns,
        model_name=loaded_artifact.model_name,
        host_score_column=resolve_host_score_column(namespace),
    )
    output_dir = resolve_score_output_dir(namespace, score_mode="model_scoring")
    print_score_stage("save artifacts")
    scoring_artifact_paths = save_scoring_artifacts(
        scoring_result,
        output_dir=output_dir,
        extra_metadata=build_score_context(
            namespace,
            score_mode="model_scoring",
            output_dir=output_dir,
        ),
    )
    print("[score] === scoring_preview ===")
    print(format_frame_preview(scoring_result.scored_df, preview_rows=namespace.preview_rows))
    print_scoring_artifact_paths(scoring_artifact_paths)

    if namespace.with_ranking:
        ranking_output_dir = resolve_ranking_output_dir(namespace)
        ranking_frame = build_priority_ranking_frame(
            scoring_result.scored_df,
            host_score_column=resolve_host_score_column(namespace),
        )
        print("[score] === ranking_preview ===")
        print(format_frame_preview(ranking_frame, preview_rows=namespace.preview_rows))
        print_score_stage("save ranking artifacts")
        ranking_artifact_paths = save_ranking_artifacts(
            ranking_frame,
            ranking_name=build_ranking_name_from_scoring(namespace),
            output_dir=ranking_output_dir,
            extra_metadata=build_score_context(
                namespace,
                score_mode="model_scoring_with_ranking",
                output_dir=ranking_output_dir,
            ),
        )
        print_ranking_artifact_paths(ranking_artifact_paths)

    return 0
