# Тестовый файл `test_hierarchical_tasks.py` домена `evaluation`.
#
# Этот файл проверяет только:
# - проверку логики домена: метрики, split-логику и benchmark contracts;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `evaluation` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.evaluation.hierarchical_tasks import (
    GAIA_ID_COARSE_CLASSIFICATION_TASK,
    GAIA_ID_OOD_CLASSIFICATION_TASK,
    GAIA_MK_REFINEMENT_CLASSIFICATION_TASK,
    HIERARCHICAL_TASK_BY_NAME,
)


def test_hierarchical_task_registry_exposes_expected_names() -> None:
    assert tuple(sorted(HIERARCHICAL_TASK_BY_NAME)) == (
        "gaia_id_coarse_classification",
        "gaia_id_ood_classification",
        "gaia_mk_refinement_classification",
    )


def test_hierarchical_tasks_have_expected_targets() -> None:
    assert GAIA_ID_COARSE_CLASSIFICATION_TASK.target_column == "spec_class"
    assert GAIA_MK_REFINEMENT_CLASSIFICATION_TASK.target_column == "spec_subclass"
    assert GAIA_ID_OOD_CLASSIFICATION_TASK.target_column == "domain_target"
