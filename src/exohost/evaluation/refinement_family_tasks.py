# Файл `refinement_family_tasks.py` слоя `evaluation`.
#
# Этот файл отвечает только за:
# - метрики, split-логику и benchmark-task contracts;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `evaluation` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from dataclasses import dataclass

from exohost.contracts.refinement_family_dataset_contracts import (
    REFINEMENT_ENABLED_SPECTRAL_CLASSES,
    REFINEMENT_FAMILY_TARGET_CARDINALITY,
    validate_refinement_family_class,
)
from exohost.contracts.refinement_family_feature_contract import (
    REFINEMENT_FAMILY_FEATURES,
)
from exohost.evaluation.protocol import ClassificationTask


@dataclass(frozen=True, slots=True)
class RefinementFamilyTaskDefinition:
    # Полный contract одной coarse-conditioned refinement family task.
    spectral_class: str
    target_cardinality: int
    task: ClassificationTask


def build_refinement_family_task_name(spectral_class: str) -> str:
    # Собираем каноническое task name для одной second-wave family.
    normalized_class = validate_refinement_family_class(spectral_class)
    return f"gaia_mk_refinement_{normalized_class.lower()}_classification"


def build_refinement_family_task_definition(
    spectral_class: str,
) -> RefinementFamilyTaskDefinition:
    # Строим полное task-definition для одной refinement family.
    normalized_class = validate_refinement_family_class(spectral_class)
    task = ClassificationTask(
        name=build_refinement_family_task_name(normalized_class),
        target_column="spectral_subclass",
        feature_columns=REFINEMENT_FAMILY_FEATURES,
        stratify_columns=("spectral_subclass",),
    )
    return RefinementFamilyTaskDefinition(
        spectral_class=normalized_class,
        target_cardinality=REFINEMENT_FAMILY_TARGET_CARDINALITY[normalized_class],
        task=task,
    )


REFINEMENT_FAMILY_TASK_DEFINITIONS: tuple[RefinementFamilyTaskDefinition, ...] = tuple(
    build_refinement_family_task_definition(spectral_class)
    for spectral_class in REFINEMENT_ENABLED_SPECTRAL_CLASSES
)

REFINEMENT_FAMILY_TASKS: tuple[ClassificationTask, ...] = tuple(
    definition.task for definition in REFINEMENT_FAMILY_TASK_DEFINITIONS
)

REFINEMENT_FAMILY_TASK_BY_NAME: dict[str, RefinementFamilyTaskDefinition] = {
    definition.task.name: definition
    for definition in REFINEMENT_FAMILY_TASK_DEFINITIONS
}
