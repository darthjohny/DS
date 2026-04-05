# Файл `model_pipeline_review_stage_frames.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

import pandas as pd

from exohost.reporting.model_pipeline_review_contracts import BenchmarkReviewBundle


def build_split_metrics_frame(bundle: BenchmarkReviewBundle) -> pd.DataFrame:
    # Возвращаем stage-level metrics в стабильном порядке для notebook display.
    split_order = pd.CategoricalDtype(categories=["train", "test"], ordered=True)
    result = bundle.metrics_df.copy()
    if "split_name" in result.columns:
        result["split_name"] = result["split_name"].astype(split_order)
        result = result.sort_values(
            ["split_name", "accuracy", "macro_f1", "balanced_accuracy"],
            ascending=[True, False, False, False],
            kind="mergesort",
            ignore_index=True,
        )
        result["split_name"] = result["split_name"].astype(str)
    return result


def build_target_distribution_frame(
    bundle: BenchmarkReviewBundle,
    *,
    split_name: str = "full",
) -> pd.DataFrame:
    # Возвращаем distribution target-лейблов для выбранного split.
    distribution_df = bundle.target_distribution_df.copy()
    if "split_name" not in distribution_df.columns:
        return distribution_df

    filtered_df = distribution_df.loc[distribution_df["split_name"] == split_name].copy()
    if filtered_df.empty:
        return filtered_df
    return filtered_df.sort_values(
        ["n_rows", "target_label"],
        ascending=[False, True],
        kind="mergesort",
        ignore_index=True,
    )


__all__ = [
    "build_split_metrics_frame",
    "build_target_distribution_frame",
]
