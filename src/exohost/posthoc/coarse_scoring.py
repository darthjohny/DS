# Файл `coarse_scoring.py` слоя `posthoc`.
#
# Этот файл отвечает только за:
# - post-hoc scoring, routing и final decision policy;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `posthoc` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from typing import cast

import pandas as pd

from exohost.models.inference import build_probability_frame, require_feature_columns
from exohost.models.protocol import ClassifierModel
from exohost.posthoc.probability_summary import build_probability_summary_frame


def build_coarse_scored_frame(
    df: pd.DataFrame,
    *,
    estimator: object,
    feature_columns: tuple[str, ...],
    model_name: str | None = None,
) -> pd.DataFrame:
    # Строим compact coarse stage output для final routing.
    _require_source_id_column(df)
    require_feature_columns(df, feature_columns=feature_columns)
    model = cast(ClassifierModel, estimator)
    probability_frame = build_probability_frame(
        model,
        df.loc[:, list(feature_columns)].copy(),
    )
    if probability_frame is None:
        raise ValueError("Coarse estimator must expose predict_proba and classes_.")

    summary_frame = build_probability_summary_frame(
        probability_frame,
        prediction_column_name="coarse_predicted_label",
        confidence_column_name="coarse_probability_max",
        margin_column_name="coarse_probability_margin",
    )
    result = df.loc[:, ["source_id"]].copy().join(summary_frame)
    if model_name is not None:
        result["coarse_model_name"] = model_name
    return result


def _require_source_id_column(df: pd.DataFrame) -> None:
    if "source_id" not in df.columns:
        raise ValueError("Coarse scoring frame is missing required column: source_id")
