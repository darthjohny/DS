# Файл `benchmark_orchestration.py` слоя `training`.
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


def build_task_ready_frame(
    df: pd.DataFrame,
    *,
    target_column: str,
    frame_name: str,
) -> pd.DataFrame:
    # Готовим frame под конкретную benchmark-задачу.
    if df.empty:
        raise RuntimeError(
            f"{frame_name} is empty after applying the current data contract."
        )

    if target_column not in df.columns:
        raise RuntimeError(
            f"{frame_name} does not contain target column: {target_column}"
        )

    task_frame = df.dropna(subset=[target_column]).reset_index(drop=True)
    if task_frame.empty:
        raise RuntimeError(
            f"{frame_name} does not contain usable target labels for task: {target_column}"
        )

    return task_frame


def raise_small_sample_error(
    *,
    sample_name: str,
    limit_hint: str,
    cause: ValueError,
) -> None:
    # Поднимаем единое сообщение для случаев,
    # когда ограниченная выборка ломает стратификацию benchmark.
    raise RuntimeError(
        f"{sample_name} sample is too small for the current split or "
        f"stratification rules. {limit_hint}"
    ) from cause
