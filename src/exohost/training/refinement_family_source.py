# Файл `refinement_family_source.py` слоя `training`.
#
# Этот файл отвечает только за:
# - оркестрацию обучения и benchmark-прогонов;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `training` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from exohost.datasets.load_gaia_mk_refinement_family_training_dataset import (
    load_gaia_mk_refinement_family_training_dataset,
)
from exohost.evaluation.refinement_family_tasks import REFINEMENT_FAMILY_TASK_BY_NAME
from exohost.features.refinement_family_training_frame import (
    prepare_gaia_mk_refinement_family_training_frame,
)


def load_refinement_family_prepared_training_frame(
    engine: Engine,
    *,
    task_name: str,
    limit: int | None = None,
) -> pd.DataFrame:
    # Загружаем и нормализуем source для одной second-wave refinement family task.
    try:
        task_definition = REFINEMENT_FAMILY_TASK_BY_NAME[task_name]
    except KeyError as error:
        supported_tasks = ", ".join(sorted(REFINEMENT_FAMILY_TASK_BY_NAME))
        raise ValueError(
            f"Unsupported refinement family task source: {task_name}. "
            f"Supported tasks: {supported_tasks}"
        ) from error

    raw_frame = load_gaia_mk_refinement_family_training_dataset(
        engine,
        spectral_class=task_definition.spectral_class,
        limit=limit,
    )
    return prepare_gaia_mk_refinement_family_training_frame(
        raw_frame,
        spectral_class=task_definition.spectral_class,
    )
