# Файл `task_catalog.py` слоя `cli`.
#
# Этот файл отвечает только за:
# - CLI-команды и orchestration entrypoints;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - CLI-команды или support-модули этого же домена;
# - пользовательский запуск через `python -m exohost.cli.main`.

from __future__ import annotations

from exohost.evaluation.hierarchical_tasks import HIERARCHICAL_TASK_BY_NAME
from exohost.evaluation.refinement_family_tasks import REFINEMENT_FAMILY_TASK_BY_NAME
from exohost.training.run_host_benchmark import HOST_TASK_BY_NAME
from exohost.training.run_router_benchmark import ROUTER_TASK_BY_NAME

HIDDEN_ROUTER_TASK_NAMES: tuple[str, ...] = ("spectral_subclass_classification",)

PUBLIC_ROUTER_TASK_NAMES: tuple[str, ...] = tuple(
    sorted(
        task_name
        for task_name in ROUTER_TASK_BY_NAME
        if task_name not in HIDDEN_ROUTER_TASK_NAMES
    )
)
PUBLIC_HOST_TASK_NAMES: tuple[str, ...] = tuple(sorted(HOST_TASK_BY_NAME))
PUBLIC_HIERARCHICAL_TASK_NAMES: tuple[str, ...] = tuple(sorted(HIERARCHICAL_TASK_BY_NAME))
PUBLIC_REFINEMENT_FAMILY_TASK_NAMES: tuple[str, ...] = tuple(
    sorted(REFINEMENT_FAMILY_TASK_BY_NAME)
)
PUBLIC_BENCHMARK_TASK_NAMES: tuple[str, ...] = tuple(
    sorted(
        {
            *PUBLIC_ROUTER_TASK_NAMES,
            *PUBLIC_HOST_TASK_NAMES,
            *PUBLIC_HIERARCHICAL_TASK_NAMES,
            *PUBLIC_REFINEMENT_FAMILY_TASK_NAMES,
        }
    )
)
