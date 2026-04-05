# Файл `support.py` слоя `cli`.
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
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import pandas as pd

from exohost.datasets.load_final_decision_input_dataset import (
    apply_final_decision_feature_aliases,
)
from exohost.posthoc.candidate_ood_policy import (
    CandidateOodDisposition,
    CandidateOodPolicy,
)
from exohost.posthoc.decision_model_bundle import (
    FinalDecisionModelBundle,
    build_final_decision_feature_union,
)
from exohost.posthoc.final_decision import FinalDecisionPolicy
from exohost.posthoc.refinement_handoff import RefinementHandoffPolicy
from exohost.reporting.final_decision_artifacts import (
    DEFAULT_FINAL_DECISION_OUTPUT_DIR,
    FinalDecisionArtifactPaths,
)


class SupportsDispose(Protocol):
    # Минимальный контракт engine для безопасного закрытия DB-backed загрузки.

    def dispose(self) -> None:
        # Освобождаем ресурсы подключения после чтения relation.
        ...


EngineFactory = Callable[..., SupportsDispose]
DatasetLoader = Callable[..., pd.DataFrame]


def print_decide_stage(message: str) -> None:
    # Печатаем короткий статус decide-команды.
    print(f"[decide] {message}")


def print_decision_artifact_paths(paths: FinalDecisionArtifactPaths) -> None:
    # Печатаем каталог сохраненных final-decision artifacts.
    print(f"[artifacts] decision_saved_to={paths.run_dir}")


def format_frame_preview(frame: pd.DataFrame, *, preview_rows: int) -> str:
    # Собираем компактный preview итогового frame.
    return frame.head(preview_rows).to_string(index=False)


def build_decide_context(
    namespace: argparse.Namespace,
    *,
    output_dir: Path,
    bundle: FinalDecisionModelBundle,
    quality_gate_policy_name: str,
) -> dict[str, object]:
    # Собираем metadata-контекст saved-artifact decision run.
    context: dict[str, object] = {
        "input_csv": None if namespace.input_csv is None else str(namespace.input_csv),
        "relation_name": None if namespace.relation_name is None else str(namespace.relation_name),
        "output_dir": str(output_dir),
        "preview_rows": int(namespace.preview_rows),
        "dotenv_path": str(namespace.dotenv_path),
        "connect_timeout": int(namespace.connect_timeout),
        "ood_model_run_dir": str(namespace.ood_model_run_dir),
        "ood_threshold_run_dir": str(namespace.ood_threshold_run_dir),
        "coarse_model_run_dir": str(namespace.coarse_model_run_dir),
        "decision_policy_version": str(namespace.decision_policy_version),
        "candidate_ood_disposition": str(namespace.candidate_ood_disposition),
        "host_score_column": str(namespace.host_score_column),
        "quality_gate_policy_name": str(quality_gate_policy_name),
        "refinement_families": sorted(bundle.refinement_artifacts_by_family),
    }
    if namespace.host_model_run_dir is not None:
        context["host_model_run_dir"] = str(namespace.host_model_run_dir)
    if namespace.refinement_model_run_dir:
        context["refinement_model_run_dirs"] = [str(value) for value in namespace.refinement_model_run_dir]
    if namespace.limit is not None:
        context["limit"] = int(namespace.limit)
    if namespace.min_refinement_confidence is not None:
        context["min_refinement_confidence"] = float(namespace.min_refinement_confidence)
    if namespace.min_coarse_probability is not None:
        context["min_coarse_probability"] = float(namespace.min_coarse_probability)
    if namespace.min_coarse_margin is not None:
        context["min_coarse_margin"] = float(namespace.min_coarse_margin)
    if namespace.quality_ruwe_unknown_threshold is not None:
        context["quality_ruwe_unknown_threshold"] = float(
            namespace.quality_ruwe_unknown_threshold
        )
    if namespace.quality_parallax_snr_unknown_threshold is not None:
        context["quality_parallax_snr_unknown_threshold"] = float(
            namespace.quality_parallax_snr_unknown_threshold
        )
    if namespace.quality_require_flame_for_pass is not None:
        context["quality_require_flame_for_pass"] = bool(
            namespace.quality_require_flame_for_pass
        )
    if namespace.priority_high_min is not None:
        context["priority_high_min"] = float(namespace.priority_high_min)
    if namespace.priority_medium_min is not None:
        context["priority_medium_min"] = float(namespace.priority_medium_min)
    return context


def resolve_decision_output_dir(namespace: argparse.Namespace) -> Path:
    # Разрешаем output dir decision artifacts.
    if namespace.output_dir is not None:
        return Path(namespace.output_dir)
    return DEFAULT_FINAL_DECISION_OUTPUT_DIR


def load_decision_input_frame(
    namespace: argparse.Namespace,
    *,
    bundle: FinalDecisionModelBundle,
    engine_factory: EngineFactory,
    dataset_loader: DatasetLoader,
) -> pd.DataFrame:
    # Загружаем decision input source либо из CSV, либо из relation в БД.
    feature_columns = build_final_decision_feature_union(bundle)
    if namespace.input_csv is not None:
        print_decide_stage(f"load input={namespace.input_csv}")
        loaded_df = pd.read_csv(namespace.input_csv)
        return apply_final_decision_feature_aliases(
            loaded_df,
            feature_columns=feature_columns,
        )

    if namespace.relation_name is None:
        raise RuntimeError("Decision input source is not configured.")

    print_decide_stage(f"load relation={namespace.relation_name}")
    engine = engine_factory(
        dotenv_path=namespace.dotenv_path,
        connect_timeout=namespace.connect_timeout,
    )
    try:
        return dataset_loader(
            engine,
            relation_name=namespace.relation_name,
            feature_columns=feature_columns,
            limit=namespace.limit,
        )
    finally:
        engine.dispose()


def build_final_decision_policy_from_namespace(
    namespace: argparse.Namespace,
) -> FinalDecisionPolicy:
    # Собираем explicit final decision policy из CLI namespace.
    return FinalDecisionPolicy(
        decision_policy_version=str(namespace.decision_policy_version),
        refinement_handoff_policy=RefinementHandoffPolicy(
            min_coarse_probability=_coerce_optional_float(namespace.min_coarse_probability),
            min_coarse_margin=_coerce_optional_float(namespace.min_coarse_margin),
        ),
        candidate_ood_policy=CandidateOodPolicy(
            disposition=cast(
                CandidateOodDisposition,
                namespace.candidate_ood_disposition,
            ),
        ),
        min_refinement_confidence=_coerce_optional_float(namespace.min_refinement_confidence),
    )


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Expected optional numeric CLI value.")
    return float(value)
