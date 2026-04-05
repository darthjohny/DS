# Файл `hierarchical_source.py` слоя `training`.
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

from exohost.datasets.load_gaia_id_coarse_training_dataset import (
    load_gaia_id_coarse_training_dataset,
)
from exohost.datasets.load_gaia_id_ood_training_dataset import (
    load_gaia_id_ood_training_dataset,
)
from exohost.datasets.load_gaia_mk_refinement_training_dataset import (
    load_gaia_mk_refinement_training_dataset,
)
from exohost.evaluation.hierarchical_tasks import (
    GAIA_ID_COARSE_CLASSIFICATION_TASK,
    GAIA_ID_OOD_CLASSIFICATION_TASK,
    GAIA_MK_REFINEMENT_CLASSIFICATION_TASK,
)
from exohost.features.hierarchical_training_frame import (
    prepare_gaia_id_coarse_training_frame,
    prepare_gaia_id_ood_training_frame,
    prepare_gaia_mk_refinement_training_frame,
)


def load_hierarchical_prepared_training_frame(
    engine: Engine,
    *,
    task_name: str,
    limit: int | None = None,
) -> pd.DataFrame:
    # Загружаем и нормализуем source для одной hierarchical-задачи.
    if task_name == GAIA_ID_COARSE_CLASSIFICATION_TASK.name:
        raw_frame = load_gaia_id_coarse_training_dataset(engine, limit=limit)
        return prepare_gaia_id_coarse_training_frame(raw_frame)

    if task_name == GAIA_MK_REFINEMENT_CLASSIFICATION_TASK.name:
        raw_frame = load_gaia_mk_refinement_training_dataset(engine, limit=limit)
        return prepare_gaia_mk_refinement_training_frame(raw_frame)

    if task_name == GAIA_ID_OOD_CLASSIFICATION_TASK.name:
        raw_frame = load_gaia_id_ood_training_dataset(engine, limit=limit)
        return prepare_gaia_id_ood_training_frame(raw_frame)

    raise ValueError(f"Unsupported hierarchical task source: {task_name}")
