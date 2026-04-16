# Файл `hierarchical_training_frame.py` слоя `features`.
#
# Этот файл отвечает только за:
# - подготовку признаков и training frame-слой;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `features` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from exohost.features.hierarchical_training_frame_coarse import (
    prepare_gaia_id_coarse_training_frame,
)
from exohost.features.hierarchical_training_frame_ood import (
    collapse_multi_membership_ood_rows,
    prepare_gaia_id_ood_training_frame,
)
from exohost.features.hierarchical_training_frame_refinement import (
    prepare_gaia_mk_refinement_training_frame,
)

# Публичный фасад training-frame слоя держит три независимых контура:
# coarse, ID/OOD и refinement. Это упрощает импорт на уровне обучения и benchmark.
__all__ = [
    "collapse_multi_membership_ood_rows",
    "prepare_gaia_id_coarse_training_frame",
    "prepare_gaia_id_ood_training_frame",
    "prepare_gaia_mk_refinement_training_frame",
]
