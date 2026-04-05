# Файл `model_pipeline_review_contracts.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class BenchmarkReviewBundle:
    # Полный пакет benchmark-артефактов одного stage/run для notebook review.
    run_dir: Path
    metrics_df: pd.DataFrame
    cv_summary_df: pd.DataFrame
    target_distribution_df: pd.DataFrame
    metadata: dict[str, Any]


__all__ = ["BenchmarkReviewBundle"]
