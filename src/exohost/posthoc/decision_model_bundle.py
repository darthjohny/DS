# Файл `decision_model_bundle.py` слоя `posthoc`.
#
# Этот файл отвечает только за:
# - post-hoc scoring, routing и final decision policy;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `posthoc` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from exohost.contracts.feature_contract import unique_columns
from exohost.contracts.label_contract import HOST_FIELD_TARGET_COLUMN
from exohost.evaluation.hierarchical_tasks import (
    GAIA_ID_COARSE_CLASSIFICATION_TASK,
    GAIA_ID_OOD_CLASSIFICATION_TASK,
)
from exohost.evaluation.refinement_family_tasks import REFINEMENT_FAMILY_TASK_BY_NAME
from exohost.posthoc.final_decision import FinalDecisionPolicy
from exohost.posthoc.final_decision_runner import FinalDecisionRunnerConfig
from exohost.reporting.id_ood_threshold_artifacts import (
    LoadedIdOodThresholdArtifact,
    load_id_ood_threshold_artifact,
)
from exohost.reporting.model_artifacts import LoadedModelArtifact, load_model_artifact


@dataclass(frozen=True, slots=True)
class FinalDecisionModelBundle:
    # Полный validated bundle saved artifacts для final decision pipeline.
    ood_artifact: LoadedModelArtifact
    ood_threshold_artifact: LoadedIdOodThresholdArtifact
    coarse_artifact: LoadedModelArtifact
    refinement_artifacts_by_family: dict[str, LoadedModelArtifact]
    host_artifact: LoadedModelArtifact | None = None


def load_final_decision_model_bundle(
    *,
    ood_model_run_dir: str | Path,
    ood_threshold_run_dir: str | Path,
    coarse_model_run_dir: str | Path,
    refinement_model_run_dirs: Iterable[str | Path] = (),
    host_model_run_dir: str | Path | None = None,
) -> FinalDecisionModelBundle:
    # Загружаем и валидируем saved-artifact bundle для decision pipeline.
    ood_artifact = load_model_artifact(ood_model_run_dir)
    _require_ood_artifact(ood_artifact)
    ood_threshold_artifact = load_id_ood_threshold_artifact(ood_threshold_run_dir)
    _require_threshold_alignment(
        ood_artifact=ood_artifact,
        threshold_artifact=ood_threshold_artifact,
    )

    coarse_artifact = load_model_artifact(coarse_model_run_dir)
    _require_coarse_artifact(coarse_artifact)

    refinement_artifacts_by_family = _load_refinement_artifacts_by_family(
        refinement_model_run_dirs
    )
    host_artifact = _load_host_artifact(host_model_run_dir)

    return FinalDecisionModelBundle(
        ood_artifact=ood_artifact,
        ood_threshold_artifact=ood_threshold_artifact,
        coarse_artifact=coarse_artifact,
        refinement_artifacts_by_family=refinement_artifacts_by_family,
        host_artifact=host_artifact,
    )


def build_final_decision_feature_union(
    bundle: FinalDecisionModelBundle,
) -> tuple[str, ...]:
    # Объединяем признаки всех saved models в один input contract.
    feature_groups: list[tuple[str, ...]] = [
        bundle.ood_artifact.feature_columns,
        bundle.coarse_artifact.feature_columns,
    ]
    feature_groups.extend(
        artifact.feature_columns
        for artifact in bundle.refinement_artifacts_by_family.values()
    )
    if bundle.host_artifact is not None:
        feature_groups.append(bundle.host_artifact.feature_columns)
    return unique_columns(*feature_groups)


def build_final_decision_runner_config_from_bundle(
    bundle: FinalDecisionModelBundle,
    *,
    final_decision_policy: FinalDecisionPolicy,
) -> FinalDecisionRunnerConfig:
    # Преобразуем validated bundle в узкий runner config.
    refinement_estimators_by_family: dict[str, object] | None = None
    refinement_model_names_by_family: dict[str, str] | None = None
    refinement_feature_columns: tuple[str, ...] | None = None
    if bundle.refinement_artifacts_by_family:
        refinement_estimators_by_family = {
            family_name: artifact.estimator
            for family_name, artifact in bundle.refinement_artifacts_by_family.items()
        }
        refinement_model_names_by_family = {
            family_name: artifact.model_name
            for family_name, artifact in bundle.refinement_artifacts_by_family.items()
        }
        refinement_feature_columns = _build_shared_refinement_feature_columns(bundle)

    return FinalDecisionRunnerConfig(
        ood_estimator=bundle.ood_artifact.estimator,
        ood_feature_columns=bundle.ood_artifact.feature_columns,
        ood_threshold_policy=bundle.ood_threshold_artifact.policy,
        coarse_estimator=bundle.coarse_artifact.estimator,
        coarse_feature_columns=bundle.coarse_artifact.feature_columns,
        final_decision_policy=final_decision_policy,
        refinement_estimators_by_family=refinement_estimators_by_family,
        refinement_feature_columns=refinement_feature_columns,
        coarse_model_name=bundle.coarse_artifact.model_name,
        refinement_model_names_by_family=refinement_model_names_by_family,
    )


def _require_ood_artifact(artifact: LoadedModelArtifact) -> None:
    if artifact.task_name != GAIA_ID_OOD_CLASSIFICATION_TASK.name:
        raise ValueError(
            "OOD model artifact must belong to task "
            f"{GAIA_ID_OOD_CLASSIFICATION_TASK.name}, got {artifact.task_name}."
        )
    if artifact.target_column != GAIA_ID_OOD_CLASSIFICATION_TASK.target_column:
        raise ValueError(
            "OOD model artifact must have target_column "
            f"{GAIA_ID_OOD_CLASSIFICATION_TASK.target_column}, got {artifact.target_column}."
        )


def _require_threshold_alignment(
    *,
    ood_artifact: LoadedModelArtifact,
    threshold_artifact: LoadedIdOodThresholdArtifact,
) -> None:
    if threshold_artifact.task_name != ood_artifact.task_name:
        raise ValueError(
            "ID/OOD threshold artifact task_name must match OOD model artifact."
        )
    if threshold_artifact.model_name != ood_artifact.model_name:
        raise ValueError(
            "ID/OOD threshold artifact model_name must match OOD model artifact."
        )


def _require_coarse_artifact(artifact: LoadedModelArtifact) -> None:
    if artifact.task_name != GAIA_ID_COARSE_CLASSIFICATION_TASK.name:
        raise ValueError(
            "Coarse model artifact must belong to task "
            f"{GAIA_ID_COARSE_CLASSIFICATION_TASK.name}, got {artifact.task_name}."
        )
    if artifact.target_column != GAIA_ID_COARSE_CLASSIFICATION_TASK.target_column:
        raise ValueError(
            "Coarse model artifact must have target_column "
            f"{GAIA_ID_COARSE_CLASSIFICATION_TASK.target_column}, got {artifact.target_column}."
        )


def _load_refinement_artifacts_by_family(
    run_dirs: Iterable[str | Path],
) -> dict[str, LoadedModelArtifact]:
    artifacts_by_family: dict[str, LoadedModelArtifact] = {}
    for run_dir in run_dirs:
        artifact = load_model_artifact(run_dir)
        family_name = _resolve_refinement_family_name(artifact)
        if family_name in artifacts_by_family:
            raise ValueError(
                f"Duplicate refinement artifact for family {family_name}."
            )
        artifacts_by_family[family_name] = artifact
    return artifacts_by_family


def _resolve_refinement_family_name(artifact: LoadedModelArtifact) -> str:
    task_definition = REFINEMENT_FAMILY_TASK_BY_NAME.get(artifact.task_name)
    if task_definition is None:
        raise ValueError(
            "Refinement model artifact must belong to a registered refinement family task, "
            f"got {artifact.task_name}."
        )
    if artifact.target_column != task_definition.task.target_column:
        raise ValueError(
            "Refinement model artifact must have target_column "
            f"{task_definition.task.target_column}, got {artifact.target_column}."
        )
    return task_definition.spectral_class


def _load_host_artifact(run_dir: str | Path | None) -> LoadedModelArtifact | None:
    if run_dir is None:
        return None
    artifact = load_model_artifact(run_dir)
    if artifact.target_column != HOST_FIELD_TARGET_COLUMN:
        raise ValueError(
            "Host model artifact must have target_column "
            f"{HOST_FIELD_TARGET_COLUMN}, got {artifact.target_column}."
        )
    return artifact


def _build_shared_refinement_feature_columns(
    bundle: FinalDecisionModelBundle,
) -> tuple[str, ...]:
    refinement_artifacts = tuple(bundle.refinement_artifacts_by_family.values())
    if not refinement_artifacts:
        return ()

    shared_columns = refinement_artifacts[0].feature_columns
    for artifact in refinement_artifacts[1:]:
        if artifact.feature_columns != shared_columns:
            raise ValueError(
                "All refinement family artifacts must share the same feature_columns contract."
            )
    return shared_columns
