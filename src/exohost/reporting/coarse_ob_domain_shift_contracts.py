# Файл `coarse_ob_domain_shift_contracts.py` слоя `reporting`.
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

import pandas as pd


@dataclass(frozen=True, slots=True)
class CoarseOBDomainShiftConfig:
    # Project baseline: сравниваем train-time и downstream hot pass `O/B` domains.
    quality_state: str = "pass"
    hot_teff_min_k: float = 10_000.0


@dataclass(frozen=True, slots=True)
class CoarseOBDomainShiftReviewBundle:
    # Typed bundle для сравнения train-time и downstream hot pass `O/B` boundary.
    config: CoarseOBDomainShiftConfig
    train_boundary_df: pd.DataFrame
    downstream_boundary_df: pd.DataFrame
    train_scored_df: pd.DataFrame
    downstream_scored_df: pd.DataFrame


DEFAULT_COARSE_OB_DOMAIN_SHIFT_CONFIG = CoarseOBDomainShiftConfig()
