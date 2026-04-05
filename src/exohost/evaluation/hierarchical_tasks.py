# Файл `hierarchical_tasks.py` слоя `evaluation`.
#
# Этот файл отвечает только за:
# - метрики, split-логику и benchmark-task contracts;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `evaluation` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from exohost.contracts.hierarchical_feature_contract import (
    GAIA_ID_COARSE_FEATURES,
    GAIA_ID_OOD_FEATURES,
    GAIA_MK_REFINEMENT_FEATURES,
)
from exohost.evaluation.protocol import ClassificationTask

GAIA_ID_COARSE_CLASSIFICATION_TASK = ClassificationTask(
    name="gaia_id_coarse_classification",
    target_column="spec_class",
    feature_columns=GAIA_ID_COARSE_FEATURES,
    stratify_columns=("spec_class", "evolution_stage"),
)

GAIA_MK_REFINEMENT_CLASSIFICATION_TASK = ClassificationTask(
    name="gaia_mk_refinement_classification",
    target_column="spec_subclass",
    feature_columns=GAIA_MK_REFINEMENT_FEATURES,
    stratify_columns=("spec_subclass",),
)

GAIA_ID_OOD_CLASSIFICATION_TASK = ClassificationTask(
    name="gaia_id_ood_classification",
    target_column="domain_target",
    feature_columns=GAIA_ID_OOD_FEATURES,
    stratify_columns=("domain_target",),
)

HIERARCHICAL_BENCHMARK_TASKS: tuple[ClassificationTask, ...] = (
    GAIA_ID_COARSE_CLASSIFICATION_TASK,
    GAIA_MK_REFINEMENT_CLASSIFICATION_TASK,
    GAIA_ID_OOD_CLASSIFICATION_TASK,
)

HIERARCHICAL_TASK_BY_NAME: dict[str, ClassificationTask] = {
    task.name: task for task in HIERARCHICAL_BENCHMARK_TASKS
}
