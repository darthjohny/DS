# Файл `final_decision_review_distributions.py` слоя `reporting`.
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

from exohost.reporting.final_decision_review_contracts import FinalDecisionReviewBundle


def build_domain_distribution_frame(bundle: FinalDecisionReviewBundle) -> pd.DataFrame:
    # Показываем итоговое распределение final_domain_state.
    return build_distribution_frame(
        bundle.final_decision_df,
        column_name="final_domain_state",
        label_name="final_domain_state",
    )


def build_quality_distribution_frame(bundle: FinalDecisionReviewBundle) -> pd.DataFrame:
    # Показываем итоговое распределение final_quality_state.
    return build_distribution_frame(
        bundle.final_decision_df,
        column_name="final_quality_state",
        label_name="final_quality_state",
    )


def build_refinement_distribution_frame(bundle: FinalDecisionReviewBundle) -> pd.DataFrame:
    # Показываем, как refinement закончился после final routing.
    return build_distribution_frame(
        bundle.final_decision_df,
        column_name="final_refinement_state",
        label_name="final_refinement_state",
    )


def build_decision_reason_frame(
    bundle: FinalDecisionReviewBundle,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    # Возвращаем top decision reasons для quick review.
    distribution_df = build_distribution_frame(
        bundle.final_decision_df,
        column_name="final_decision_reason",
        label_name="final_decision_reason",
    )
    return distribution_df.head(top_n).copy()


def build_quality_reason_frame(
    bundle: FinalDecisionReviewBundle,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    # Возвращаем top quality reasons из decision-input слоя.
    distribution_df = build_distribution_frame(
        bundle.decision_input_df,
        column_name="quality_reason",
        label_name="quality_reason",
    )
    return distribution_df.head(top_n).copy()


def build_review_bucket_frame(
    bundle: FinalDecisionReviewBundle,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    # Возвращаем top review buckets после quality/OOD routing.
    distribution_df = build_distribution_frame(
        bundle.decision_input_df,
        column_name="review_bucket",
        label_name="review_bucket",
    )
    return distribution_df.head(top_n).copy()


def build_domain_quality_crosstab_frame(bundle: FinalDecisionReviewBundle) -> pd.DataFrame:
    # Удобный crosstab для проверки routing и quality-gate одновременно.
    return pd.crosstab(
        bundle.final_decision_df["final_domain_state"],
        bundle.final_decision_df["final_quality_state"],
        dropna=False,
    )


def build_distribution_frame(
    df: pd.DataFrame,
    *,
    column_name: str,
    label_name: str,
) -> pd.DataFrame:
    if column_name not in df.columns:
        return pd.DataFrame(columns=[label_name, "n_rows", "share"])

    counts = df.loc[:, column_name].astype(str).value_counts(dropna=False)
    total_rows = int(counts.sum())
    rows = [
        {
            label_name: str(label_value),
            "n_rows": int(n_rows),
            "share": float(n_rows / total_rows),
        }
        for label_value, n_rows in counts.items()
    ]
    return pd.DataFrame.from_records(rows)


__all__ = [
    "build_decision_reason_frame",
    "build_distribution_frame",
    "build_domain_distribution_frame",
    "build_domain_quality_crosstab_frame",
    "build_quality_distribution_frame",
    "build_quality_reason_frame",
    "build_refinement_distribution_frame",
    "build_review_bucket_frame",
]
