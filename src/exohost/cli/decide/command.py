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

from exohost.datasets.load_final_decision_input_dataset import (
    load_final_decision_input_dataset,
)
from exohost.db.engine import make_read_only_engine
from exohost.posthoc.candidate_ood_policy import (
    CANDIDATE_OOD_KEEP_DISPOSITION,
    CANDIDATE_OOD_MAP_TO_UNKNOWN_DISPOSITION,
)
from exohost.posthoc.decision_model_bundle import load_final_decision_model_bundle
from exohost.posthoc.final_decision_artifact_runner import (
    run_final_decision_with_artifacts,
)
from exohost.posthoc.quality_gate_tuning import apply_quality_gate_tuning
from exohost.ranking.priority_score import DEFAULT_HOST_SCORE_COLUMN
from exohost.reporting.final_decision_artifacts import save_final_decision_artifacts

from .priority_support import build_priority_integration_config_from_namespace
from .quality_gate_support import build_quality_gate_tuning_config_from_namespace
from .support import (
    build_decide_context,
    build_final_decision_policy_from_namespace,
    format_frame_preview,
    load_decision_input_frame,
    print_decide_stage,
    print_decision_artifact_paths,
    resolve_decision_output_dir,
)


def register_decide_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    # Регистрируем decide-команду поверх saved model/threshold artifacts.
    decide_parser = subparsers.add_parser(
        "decide",
        help="Сквозной final decision pipeline поверх saved artifacts.",
    )
    input_group = decide_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-csv",
        help="CSV с input rows для final decision pipeline.",
    )
    input_group.add_argument(
        "--relation-name",
        default=None,
        help="Relation в БД для DB-backed decision pipeline.",
    )
    decide_parser.add_argument(
        "--ood-model-run-dir",
        required=True,
        help="Каталог saved model artifact для ID/OOD classifier.",
    )
    decide_parser.add_argument(
        "--ood-threshold-run-dir",
        required=True,
        help="Каталог saved threshold-policy artifact для ID/OOD gate.",
    )
    decide_parser.add_argument(
        "--coarse-model-run-dir",
        required=True,
        help="Каталог saved model artifact для coarse classifier.",
    )
    decide_parser.add_argument(
        "--refinement-model-run-dir",
        action="append",
        default=[],
        help="Необязательный saved model artifact одной refinement family. Повторяемый флаг.",
    )
    decide_parser.add_argument(
        "--host-model-run-dir",
        default=None,
        help="Необязательный saved model artifact для host scoring и priority integration.",
    )
    decide_parser.add_argument(
        "--decision-policy-version",
        default="final_decision_v2",
        help="Версия final decision policy.",
    )
    decide_parser.add_argument(
        "--candidate-ood-disposition",
        choices=(
            CANDIDATE_OOD_KEEP_DISPOSITION,
            CANDIDATE_OOD_MAP_TO_UNKNOWN_DISPOSITION,
        ),
        default=CANDIDATE_OOD_KEEP_DISPOSITION,
        help="Как маршрутизировать candidate_ood после binary gate.",
    )
    decide_parser.add_argument(
        "--min-refinement-confidence",
        type=float,
        default=None,
        help="Необязательный confidence threshold для accepted refinement.",
    )
    decide_parser.add_argument(
        "--min-coarse-probability",
        type=float,
        default=None,
        help="Необязательный threshold coarse_probability_max для handoff в refinement.",
    )
    decide_parser.add_argument(
        "--min-coarse-margin",
        type=float,
        default=None,
        help="Необязательный threshold coarse_probability_margin для handoff в refinement.",
    )
    decide_parser.add_argument(
        "--host-score-column",
        default=DEFAULT_HOST_SCORE_COLUMN,
        help="Имя host-like score колонки для priority integration.",
    )
    decide_parser.add_argument(
        "--quality-ruwe-unknown-threshold",
        type=float,
        default=None,
        help="Необязательный override для RUWE-порога quality-gate.",
    )
    decide_parser.add_argument(
        "--quality-parallax-snr-unknown-threshold",
        type=float,
        default=None,
        help="Необязательный override для parallax_over_error quality-gate.",
    )
    decide_parser.add_argument(
        "--quality-require-flame-for-pass",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Нужно ли требовать FLAME-признаки для pass-состояния quality-gate. "
            "Используйте --no-quality-require-flame-for-pass для relaxed policy."
        ),
    )
    decide_parser.add_argument(
        "--priority-high-min",
        type=float,
        default=None,
        help="Необязательный high-threshold для priority label.",
    )
    decide_parser.add_argument(
        "--priority-medium-min",
        type=float,
        default=None,
        help="Необязательный medium-threshold для priority label.",
    )
    decide_parser.add_argument(
        "--output-dir",
        default=None,
        help="Необязательный каталог для final decision artifacts.",
    )
    decide_parser.add_argument(
        "--preview-rows",
        type=int,
        default=10,
        help="Сколько верхних строк итоговой таблицы печатать в консоль.",
    )
    decide_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Необязательный лимит строк для DB-backed input source.",
    )
    decide_parser.add_argument(
        "--dotenv-path",
        default=".env",
        help="Путь к .env файлу с параметрами подключения.",
    )
    decide_parser.add_argument(
        "--connect-timeout",
        type=int,
        default=10,
        help="Таймаут подключения к БД в секундах.",
    )
    decide_parser.set_defaults(handler=handle_decide)


def handle_decide(namespace: argparse.Namespace) -> int:
    # Выполняем final decision pipeline через saved artifacts и сохраняем outputs.
    print_decide_stage("load saved artifacts")
    bundle = load_final_decision_model_bundle(
        ood_model_run_dir=namespace.ood_model_run_dir,
        ood_threshold_run_dir=namespace.ood_threshold_run_dir,
        coarse_model_run_dir=namespace.coarse_model_run_dir,
        refinement_model_run_dirs=namespace.refinement_model_run_dir,
        host_model_run_dir=namespace.host_model_run_dir,
    )
    quality_gate_config = build_quality_gate_tuning_config_from_namespace(namespace)
    base_df = load_decision_input_frame(
        namespace,
        bundle=bundle,
        engine_factory=make_read_only_engine,
        dataset_loader=load_final_decision_input_dataset,
    )
    base_df = apply_quality_gate_tuning(base_df, config=quality_gate_config)
    print_decide_stage("run decision pipeline")
    decision_result = run_final_decision_with_artifacts(
        base_df,
        bundle=bundle,
        final_decision_policy=build_final_decision_policy_from_namespace(namespace),
        priority_config=build_priority_integration_config_from_namespace(namespace),
    )
    output_dir = resolve_decision_output_dir(namespace)
    print_decide_stage("save artifacts")
    artifact_paths = save_final_decision_artifacts(
        pipeline_name="hierarchical_final_decision",
        decision_input_df=decision_result.decision_input_df,
        final_decision_df=decision_result.final_decision_df,
        priority_input_df=decision_result.priority_input_df,
        priority_ranking_df=decision_result.priority_ranking_df,
        output_dir=output_dir,
        extra_metadata=build_decide_context(
            namespace,
            output_dir=output_dir,
            bundle=bundle,
            quality_gate_policy_name=quality_gate_config.policy_name,
        ),
    )
    print("[decide] === final_decision_preview ===")
    print(
        format_frame_preview(
            decision_result.final_decision_df,
            preview_rows=namespace.preview_rows,
        )
    )
    if not decision_result.priority_ranking_df.empty:
        print("[decide] === priority_preview ===")
        print(
            format_frame_preview(
                decision_result.priority_ranking_df,
                preview_rows=namespace.preview_rows,
            )
        )
    print_decision_artifact_paths(artifact_paths)
    return 0
