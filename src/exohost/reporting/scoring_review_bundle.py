# Файл `scoring_review_bundle.py` слоя `reporting`.
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

from exohost.reporting.ranking_artifacts import load_ranking_artifacts
from exohost.reporting.scoring_artifacts import load_scoring_artifacts
from exohost.reporting.scoring_review_contracts import ScoringReviewBundle


def load_scoring_review_bundle(
    scoring_run_dir: str,
    *,
    ranking_run_dir: str | None = None,
) -> ScoringReviewBundle:
    # Загружаем scoring-артефакты и при необходимости ranking-артефакты.
    scoring_df, scoring_metadata = load_scoring_artifacts(scoring_run_dir)
    if ranking_run_dir is None:
        return ScoringReviewBundle(
            scoring_df=scoring_df,
            scoring_metadata=scoring_metadata,
        )

    ranking_df, ranking_metadata = load_ranking_artifacts(ranking_run_dir)
    return ScoringReviewBundle(
        scoring_df=scoring_df,
        scoring_metadata=scoring_metadata,
        ranking_df=ranking_df,
        ranking_metadata=ranking_metadata,
    )


def require_prediction_column(
    scoring_df: pd.DataFrame,
    *,
    target_column: str,
) -> str:
    # Проверяем наличие канонической prediction-колонки в scored DataFrame.
    prediction_column = f"predicted_{target_column}"
    if prediction_column not in scoring_df.columns:
        raise ValueError(
            "Scoring review expected prediction column: "
            f"{prediction_column}"
        )
    return prediction_column


def require_ranking_frame(bundle: ScoringReviewBundle) -> pd.DataFrame:
    # Для ranking-review нужен реально загруженный ranking DataFrame.
    if bundle.ranking_df is None:
        raise ValueError("Scoring review bundle does not include ranking artifacts.")
    return bundle.ranking_df


__all__ = [
    "load_scoring_review_bundle",
    "require_prediction_column",
    "require_ranking_frame",
]
