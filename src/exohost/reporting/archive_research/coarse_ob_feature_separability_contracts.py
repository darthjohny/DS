# Contracts для review feature separability `O vs B`.

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class CoarseOBFeatureSeparabilityConfig:
    # Project baseline: narrow hot `O/B` slice внутри coarse training source.
    hot_teff_min_k: float = 10_000.0
    permutation_n_repeats: int = 20
    permutation_random_state: int = 42


@dataclass(frozen=True, slots=True)
class CoarseOBFeatureSeparabilityReviewBundle:
    # Полный пакет feature-separability review для train-time `O/B` boundary.
    config: CoarseOBFeatureSeparabilityConfig
    source_df: pd.DataFrame
    boundary_df: pd.DataFrame
    scored_boundary_df: pd.DataFrame
    permutation_importance_df: pd.DataFrame


DEFAULT_COARSE_OB_SEPARABILITY_CONFIG = CoarseOBFeatureSeparabilityConfig()
