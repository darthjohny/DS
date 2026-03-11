"""Типизированные контракты боевого pipeline приоритизации."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PipelineRunResult:
    """Результат одного полного запуска боевого pipeline.

    Хранит `run_id`, табличный результат router-слоя и итоговую таблицу
    приоритизации после объединения host-ветки и low-priority ветки.
    """

    run_id: str
    router_results: pd.DataFrame
    priority_results: pd.DataFrame


__all__ = ["PipelineRunResult"]
